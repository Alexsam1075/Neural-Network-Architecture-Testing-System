import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .base_test import BaseTest
from .anti_cheat import remap_tokens


class _AutoregressiveGenerationTest(BaseTest):
    def __init__(self, config: Dict[str, Any], name: str):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_samples', config.get('num_sequences', 384))
        self.num_eval_samples = config.get('num_eval_samples', max(128, self.num_samples // 4))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 6)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.pad_token = 0
        self.seed_tokens = 1
        self.anti_cheat_token_permutation = config.get('anti_cheat_token_permutation', True)
        self.anti_cheat_seed = config.get('anti_cheat_seed', 1337)
        self.random_seed = config.get('random_seed', self.anti_cheat_seed)

    def _maybe_remap(self, seqs: torch.Tensor) -> torch.Tensor:
        if not self.anti_cheat_token_permutation:
            return seqs
        return remap_tokens(
            (seqs,),
            self.config.get('vocab_size', 256),
            self.name,
            self.anti_cheat_seed,
            protected_tokens=(self.pad_token,),
        )[0]

    def _train_next_token(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        losses = []
        start = time.time()
        model.train()
        for _ in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out.reshape(-1, out.shape[-1]), yb.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / max(1, len(loader)))
        return time.time() - start, losses

    def _rollout(self, model: nn.Module, seed: torch.Tensor, steps: int) -> torch.Tensor:
        model.eval()
        generated = seed.clone()
        with torch.no_grad():
            for pos in range(seed.shape[1] - 1, steps - 1):
                inp = torch.full(
                    (generated.shape[0], self.seq_length),
                    self.pad_token,
                    dtype=torch.long,
                    device=generated.device,
                )
                visible_len = min(generated.shape[1], self.seq_length)
                inp[:, :visible_len] = generated[:, :visible_len]
                logits = model(inp)
                next_token = logits[:, min(pos, self.seq_length - 1), :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        return generated[:, :steps]

    def _token_accuracy(self, pred: torch.Tensor, target: torch.Tensor):
        correct = (pred == target).sum().item()
        total = target.numel()
        return correct / total if total else 0, correct, total

    def _make_dataset(self, count: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = torch.stack([self._sample(i + offset) for i in range(count)]).to(self.device)
        seqs = self._maybe_remap(seqs)
        return seqs[:, :-1], seqs[:, 1:]

    def prepare_eval_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_eval_samples, int(self.random_seed) + 10_000)


class StructuredGenerationTest(_AutoregressiveGenerationTest):
    """Tests whether generation keeps a simple schema instead of producing locally plausible tokens."""

    def __init__(self, config: Dict[str, Any], name: str = "StructuredGenerationTest"):
        super().__init__(config, name)
        self.start = 1
        self.open_tok = 2
        self.sep_tok = 3
        self.close_tok = 4
        self.end_tok = 5
        self.seed_tokens = 4

    def _sample(self, i: int):
        a = 20 + (i % 30)
        b = 60 + ((i * 7) % 30)
        c = 100 + ((i * 11) % 30)
        seq = [self.start, a, self.open_tok, b, self.sep_tok, c, self.close_tok, self.end_tok]
        while len(seq) < self.seq_length + 1:
            seq.extend([self.start, a, self.open_tok, b, self.sep_tok, c, self.close_tok, self.end_tok])
        return torch.tensor(seq[:self.seq_length + 1], dtype=torch.long)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, int(self.random_seed))

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        x, y = self.prepare_data()
        x, y = x.to(device), y.to(device)
        training_time, losses = self._train_next_token(model, x, y)
        x_eval, y_eval = self.prepare_eval_data()
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

        eval_start = time.time()
        seed = x_eval[:, :self.seed_tokens]
        target = torch.cat([x_eval[:, :1], y_eval], dim=1)[:, :self.seq_length]
        generated = self._rollout(model, seed, self.seq_length)
        token_acc, correct, total = self._token_accuracy(generated[:, self.seed_tokens:], target[:, self.seed_tokens:])
        pattern_positions = torch.tensor([0, 2, 4, 6, 7], device=device)
        valid_rows = 0
        for row in generated:
            checks = []
            for offset in range(0, self.seq_length - 7, 8):
                checks.extend([
                    row[offset] == self.start,
                    row[offset + 2] == self.open_tok,
                    row[offset + 4] == self.sep_tok,
                    row[offset + 6] == self.close_tok,
                    row[offset + 7] == self.end_tok,
                ])
            valid_rows += int(all(bool(c.item()) for c in checks))
        schema_valid_rate = valid_rows / len(generated) if len(generated) else 0
        eval_time = time.time() - eval_start

        return {
            'accuracy': token_acc,
            'schema_valid_rate': schema_valid_rate,
            'exact_sequence_accuracy': (generated == target).all(dim=1).float().mean().item(),
            'correct_predictions': correct,
            'total_predictions': total,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': len(generated) / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


class RareTokenGenerationTest(_AutoregressiveGenerationTest):
    """Detects mode collapse where rare but important tokens disappear during generation."""

    def __init__(self, config: Dict[str, Any], name: str = "RareTokenGenerationTest"):
        super().__init__(config, name)
        self.start = 6
        self.common_tokens = list(range(10, 18))
        self.rare_tokens = list(range(180, 188))
        self.seed_tokens = 2

    def _sample(self, i: int):
        rare = self.rare_tokens[i % len(self.rare_tokens)]
        seq = [self.start]
        for pos in range(self.seq_length):
            if pos in (5, 13, 21, 29):
                seq.append(rare)
            else:
                seq.append(self.common_tokens[(i + pos) % len(self.common_tokens)])
        return torch.tensor(seq[:self.seq_length + 1], dtype=torch.long)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, int(self.random_seed))

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        x, y = self.prepare_data()
        x, y = x.to(device), y.to(device)
        training_time, losses = self._train_next_token(model, x, y)
        x_eval, y_eval = self.prepare_eval_data()
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

        eval_start = time.time()
        target = torch.cat([x_eval[:, :1], y_eval], dim=1)[:, :self.seq_length]
        generated = self._rollout(model, x_eval[:, :self.seed_tokens], self.seq_length)
        token_acc, correct, total = self._token_accuracy(generated[:, self.seed_tokens:], target[:, self.seed_tokens:])
        absolute_positions = torch.arange(self.seed_tokens, self.seq_length, device=device).unsqueeze(0)
        rare_mask = torch.zeros_like(target[:, self.seed_tokens:], dtype=torch.bool)
        for pos in (6, 14, 22, 30):
            rare_mask |= absolute_positions == pos
        rare_total = rare_mask.sum().item()
        rare_correct = ((generated[:, self.seed_tokens:] == target[:, self.seed_tokens:]) & rare_mask).sum().item()
        generated_rare_rate = rare_correct / max(1, generated[:, self.seed_tokens:].numel())
        eval_time = time.time() - eval_start

        return {
            'accuracy': token_acc,
            'rare_token_recall': rare_correct / rare_total if rare_total else 0,
            'generated_rare_token_rate': generated_rare_rate,
            'mode_collapse_score': 1.0 - generated_rare_rate,
            'correct_predictions': correct,
            'total_predictions': total,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': len(generated) / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


class AlgorithmicRolloutTest(_AutoregressiveGenerationTest):
    """Measures error accumulation when a model must generate a multi-step recurrence."""

    def __init__(self, config: Dict[str, Any], name: str = "AlgorithmicRolloutTest"):
        super().__init__(config, name)
        self.offset = 30
        self.modulus = 50

    def _sample(self, i: int):
        a = (i % self.modulus)
        b = ((i * 7) % self.modulus)
        seq = [self.offset + a, self.offset + b]
        for _ in range(self.seq_length - 1):
            nxt = ((seq[-1] - self.offset) + (seq[-2] - self.offset)) % self.modulus
            seq.append(self.offset + nxt)
        return torch.tensor(seq[:self.seq_length + 1], dtype=torch.long)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, int(self.random_seed))

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        x, y = self.prepare_data()
        x, y = x.to(device), y.to(device)
        training_time, losses = self._train_next_token(model, x, y)
        x_eval, y_eval = self.prepare_eval_data()
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

        eval_start = time.time()
        target = torch.cat([x_eval[:, :2], y_eval[:, 1:]], dim=1)[:, :self.seq_length]
        generated = self._rollout(model, x_eval[:, :2], self.seq_length)
        token_acc, correct, total = self._token_accuracy(generated[:, 2:], target[:, 2:])
        first_half_acc = (generated[:, 2:self.seq_length // 2] == target[:, 2:self.seq_length // 2]).float().mean().item()
        second_half_acc = (generated[:, self.seq_length // 2:] == target[:, self.seq_length // 2:]).float().mean().item()
        eval_time = time.time() - eval_start

        return {
            'accuracy': token_acc,
            'first_half_accuracy': first_half_acc,
            'second_half_accuracy': second_half_acc,
            'rollout_decay': first_half_acc - second_half_acc,
            'exact_sequence_accuracy': (generated == target).all(dim=1).float().mean().item(),
            'correct_predictions': correct,
            'total_predictions': total,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': len(generated) / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


class RepetitionCollapseTest(_AutoregressiveGenerationTest):
    """Finds architectures that fall into repetitive loops during free generation."""

    def __init__(self, config: Dict[str, Any], name: str = "RepetitionCollapseTest"):
        super().__init__(config, name)
        self.start = 7
        self.seed_tokens = 2

    def _sample(self, i: int):
        motif_len = 3 + (i % 5)
        motif = [40 + ((i * 13 + j * 9) % 80) for j in range(motif_len)]
        seq = [self.start]
        for pos in range(self.seq_length):
            token = motif[pos % motif_len]
            if pos and pos % 11 == 0:
                token = 140 + ((i + pos) % 20)
            seq.append(token)
        return torch.tensor(seq[:self.seq_length + 1], dtype=torch.long)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, int(self.random_seed))

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        x, y = self.prepare_data()
        x, y = x.to(device), y.to(device)
        training_time, losses = self._train_next_token(model, x, y)
        x_eval, y_eval = self.prepare_eval_data()
        x_eval, y_eval = x_eval.to(device), y_eval.to(device)

        eval_start = time.time()
        target = torch.cat([x_eval[:, :1], y_eval], dim=1)[:, :self.seq_length]
        generated = self._rollout(model, x_eval[:, :self.seed_tokens], self.seq_length)
        token_acc, correct, total = self._token_accuracy(generated[:, self.seed_tokens:], target[:, self.seed_tokens:])
        unique_ratios = []
        repeat_runs = []
        for row in generated:
            unique_ratios.append(torch.unique(row).numel() / row.numel())
            longest = current = 1
            for i in range(1, row.numel()):
                current = current + 1 if row[i] == row[i - 1] else 1
                longest = max(longest, current)
            repeat_runs.append(longest / row.numel())
        eval_time = time.time() - eval_start

        return {
            'accuracy': token_acc,
            'unique_token_ratio': float(sum(unique_ratios) / len(unique_ratios)),
            'longest_repeat_fraction': float(sum(repeat_runs) / len(repeat_runs)),
            'collapse_score': float(sum(repeat_runs) / len(repeat_runs)) + (1.0 - float(sum(unique_ratios) / len(unique_ratios))),
            'correct_predictions': correct,
            'total_predictions': total,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': len(generated) / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }
