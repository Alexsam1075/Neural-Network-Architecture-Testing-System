"""Needle-in-haystack key-value retrieval benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .anti_cheat import remap_tokens
from .base_test import BaseTest


class NeedleInHaystackTest(BaseTest):
    """Retrieve the value paired with a queried key from many distractor pairs."""

    PAD = 0
    QUERY = 242

    def __init__(self, config: Dict[str, Any], name: str = "NeedleInHaystackTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.seq_length = config.get("seq_length", 64)
        self.num_pairs = min(config.get("num_pairs", 24), max(3, (self.seq_length - 3) // 2))
        self.num_samples = config.get("num_samples", config.get("num_sequences", 512))
        self.num_eval_samples = config.get("num_eval_samples", max(128, self.num_samples // 4))
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 6)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))
        self.anti_cheat_token_permutation = config.get("anti_cheat_token_permutation", True)
        self.anti_cheat_seed = config.get("anti_cheat_seed", 1337)

    def _make_dataset(self, count: int, offset: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed) + int(offset))
        inputs, targets, needle_buckets = [], [], []
        for i in range(count):
            keys = (10 + torch.randperm(80, generator=generator)[: self.num_pairs]).tolist()
            values = (100 + torch.randperm(80, generator=generator)[: self.num_pairs]).tolist()
            target_index = int(torch.randint(0, self.num_pairs, (1,), generator=generator).item())
            pairs = list(zip(keys, values))
            if offset:
                forced_bucket = i % 3
                target_index = [0, self.num_pairs // 2, self.num_pairs - 1][forced_bucket]
            seq = []
            for key, value in pairs:
                seq.extend([key, value])
            seq.extend([self.QUERY, keys[target_index]])
            seq = (seq + [self.PAD] * self.seq_length)[: self.seq_length]
            inputs.append(torch.tensor(seq, dtype=torch.long))
            targets.append(values[target_index])
            needle_buckets.append(0 if target_index < self.num_pairs // 3 else (2 if target_index >= 2 * self.num_pairs // 3 else 1))
        x_tensor = torch.stack(inputs).to(device)
        y_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        bucket_tensor = torch.tensor(needle_buckets, dtype=torch.long, device=device)
        if self.anti_cheat_token_permutation:
            x_tensor, y_tensor = remap_tokens(
                (x_tensor, y_tensor),
                self.vocab_size,
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.PAD, self.QUERY),
            )
        return x_tensor, y_tensor, bucket_tensor

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = self._make_dataset(self.num_samples, 0, self.device)
        return x, y

    def _evaluate(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, buckets: torch.Tensor) -> Dict[str, Any]:
        correct = total = 0
        bucket_correct = {0: 0, 1: 0, 2: 0}
        bucket_total = {0: 0, 1: 0, 2: 0}
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb, yb = x[start : start + self.batch_size], y[start : start + self.batch_size]
                bb = buckets[start : start + self.batch_size]
                pred = model(xb)[:, -1, :].argmax(dim=-1)
                matches = pred == yb
                correct += matches.sum().item()
                total += len(yb)
                for bucket, ok in zip(bb.tolist(), matches.tolist()):
                    bucket_total[bucket] += 1
                    bucket_correct[bucket] += int(ok)
        names = ["start", "middle", "end"]
        by_position = {
            names[bucket]: bucket_correct[bucket] / bucket_total[bucket]
            for bucket in bucket_total
            if bucket_total[bucket]
        }
        return {
            "accuracy": correct / total if total else 0.0,
            "correct_predictions": correct,
            "total_predictions": total,
            "position_accuracy": by_position,
            "lost_in_middle_gap": max(by_position.get("start", 0.0), by_position.get("end", 0.0)) - by_position.get("middle", 0.0),
        }

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        train_x, train_y, _ = self._make_dataset(self.num_samples, 0, device)
        eval_x, eval_y, eval_buckets = self._make_dataset(self.num_eval_samples, 10_000, device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        losses = []
        start_time = time.time()
        model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb)[:, -1, :], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, len(loader)))
        training_time = time.time() - start_time

        model.eval()
        eval_start = time.time()
        metrics = self._evaluate(model, eval_x, eval_y, eval_buckets)
        eval_time = time.time() - eval_start
        result = {
            **metrics,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": self.num_eval_samples / eval_time if eval_time > 0 else 0,
            "train_loss_initial": losses[0] if losses else 0.0,
            "train_loss_final": losses[-1] if losses else 0.0,
            "num_pairs": self.num_pairs,
            "sequence_length": self.seq_length,
            "num_samples": self.num_samples,
            "num_eval_samples": self.num_eval_samples,
            "epochs": self.epochs,
        }
        self.results = result
        self.metrics = {"accuracy": result["accuracy"], "inference_speed": result["inference_speed"]}
        return result
