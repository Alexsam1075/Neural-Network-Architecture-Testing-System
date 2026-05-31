import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .aion_lite import AIONLite


class AppendOnlySQLiteMemory:
    """
    Disk-backed verbatim memory.

    This is deliberately outside the neural weights: the model learns reusable
    processing skill, while factual/user/session knowledge can be appended and
    retrieved without retraining.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                term TEXT NOT NULL,
                message_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY(term, message_id)
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)")
        self.conn.commit()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)

    def append(self, text: str, role: str = "user", block: str = "user") -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO messages(block, role, text, ts) VALUES (?, ?, ?, ?)",
            (block, role, text, now),
        )
        message_id = int(cur.lastrowid)
        counts = Counter(self.tokenize(text))
        self.conn.executemany(
            "INSERT OR REPLACE INTO terms(term, message_id, count) VALUES (?, ?, ?)",
            [(term, message_id, count) for term, count in counts.items()],
        )
        self.conn.commit()
        return message_id

    def search(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        terms = self.tokenize(query)
        if not terms:
            return []
        scores: Dict[int, float] = defaultdict(float)
        now = time.time()
        total_docs = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 1
        for term in terms:
            rows = self.conn.execute(
                """
                SELECT t.message_id, t.count, m.ts
                FROM terms t JOIN messages m ON m.id = t.message_id
                WHERE t.term = ?
                """,
                (term,),
            ).fetchall()
            doc_freq = max(1, len(rows))
            idf = math.log((total_docs + 1) / doc_freq)
            for message_id, count, ts in rows:
                freshness = 1.0 / (1.0 + max(0.0, now - ts) / 86_400.0)
                scores[int(message_id)] += (count * idf) + 0.05 * freshness
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        if not ranked:
            return []
        placeholders = ",".join("?" for _ in ranked)
        rows = self.conn.execute(
            f"SELECT id, block, role, text, ts FROM messages WHERE id IN ({placeholders})",
            [message_id for message_id, _ in ranked],
        ).fetchall()
        by_id = {
            int(row[0]): {
                "id": int(row[0]),
                "block": row[1],
                "role": row[2],
                "text": row[3],
                "ts": row[4],
            }
            for row in rows
        }
        return [{**by_id[message_id], "score": score} for message_id, score in ranked if message_id in by_id]


class ModularOntology:
    """Small modular fact graph with coherence scoring."""

    def __init__(self):
        self.blocks: Dict[str, Dict[str, Any]] = {
            "static": {},
            "semi_dynamic": {},
            "dynamic": {},
            "scientific": {},
            "language": {},
            "user": {},
            "inter_user": {},
            "custom": {},
        }

    def put(self, block: str, subject: str, predicate: str, obj: Any, confidence: float = 1.0) -> None:
        block_store = self.blocks.setdefault(block, {})
        block_store.setdefault(subject, {})[predicate] = {"object": obj, "confidence": float(confidence)}

    def query(self, subject: str, predicate: Optional[str] = None) -> List[Dict[str, Any]]:
        hits = []
        for block, store in self.blocks.items():
            facts = store.get(subject, {})
            for pred, data in facts.items():
                if predicate is None or predicate == pred:
                    hits.append({"block": block, "subject": subject, "predicate": pred, **data})
        return hits

    def coherence_score(self, claims: Iterable[Tuple[str, str, Any]]) -> float:
        checked = 0
        consistent = 0
        for subject, predicate, obj in claims:
            known = self.query(subject, predicate)
            if not known:
                continue
            checked += 1
            consistent += int(any(item["object"] == obj for item in known))
        if checked == 0:
            return 0.5
        return consistent / checked


class AIONLongMemoryCore(AIONLite):
    """
    Long-context neural core plus external append-only memory/ontology hooks.

    The forward path remains benchmark-compatible. Knowledge/memory APIs are
    separate from weights, so facts and user history can be updated without
    retraining the neural processor.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "AIONLongMemoryCore"
        self.ontology = ModularOntology()
        memory_path = config.get("memory_path")
        self.memory = AppendOnlySQLiteMemory(memory_path) if memory_path else None

    def remember(self, text: str, role: str = "user", block: str = "user") -> Optional[int]:
        if self.memory is None:
            return None
        return self.memory.append(text, role=role, block=block)

    def recall(self, query: str, limit: int = 8) -> List[Dict[str, Any]]:
        if self.memory is None:
            return []
        return self.memory.search(query, limit=limit)

    def assert_fact(
        self,
        subject: str,
        predicate: str,
        obj: Any,
        block: str = "custom",
        confidence: float = 1.0,
    ) -> None:
        self.ontology.put(block, subject, predicate, obj, confidence)

    def coherence_score(self, claims: Iterable[Tuple[str, str, Any]]) -> float:
        return self.ontology.coherence_score(claims)

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": self.name,
                "long_context_safe": False,
                "external_memory": "SQLite append-only WAL, optional memory_path",
                "ontology": "modular blocks with coherence_score",
                "knowledge_weight_separation": True,
                "pattern_logits": False,
                "suffix_induction": False,
                "sentinel_repair": False,
                "hypothesis": "reasoning core separated from mutable disk memory and modular ontology",
            }
        )
        return info


class AIONLongMemoryCoreV2(AIONLongMemoryCore):
    """Honest AION long-memory core variant without handcrafted logit repair."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "AIONLongMemoryCoreV2"


class AIONLongMemoryCoreV3(AIONLongMemoryCore):
    """Honest AION long-memory core variant without suffix induction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "AIONLongMemoryCoreV3"

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": self.name,
                "causal_suffix_induction": False,
                "sentinel_repair": False,
            }
        )
        return info


class AIONLongMemoryCoreV4(AIONLongMemoryCore):
    """Honest AION long-memory core variant without sentinel repair."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "AIONLongMemoryCoreV4"

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": self.name,
                "sentinel_repair": False,
            }
        )
        return info
