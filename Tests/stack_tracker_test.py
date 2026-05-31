"""Stack state tracking benchmark with real push/pop supervision."""

import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .anti_cheat import remap_tokens
from .base_test import BaseTest


class StackTrackerTest(BaseTest):
    """
    Predict the current top of a symbolic stack after a sequence of PUSH/POP ops.

    This is not a pattern-continuation shortcut: train and eval streams are
    generated from different seeds, include distractor pops, and report accuracy
    by true final stack depth.
    """

    PAD = 0
    EMPTY = 1
    PUSH = 240
    POP = 241
    QUERY = 242

    def __init__(self, config: Dict[str, Any], name: str = "StackTrackerTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.seq_length = config.get("seq_length", 40)
        self.num_samples = config.get("num_samples", config.get("num_sequences", 512))
        self.num_eval_samples = config.get("num_eval_samples", max(128, self.num_samples // 4))
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 6)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.max_symbol = min(config.get("stack_symbols", 24), 80)
        self.max_depth = max(2, config.get("max_stack_depth", 8))
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))
        self.anti_cheat_token_permutation = config.get("anti_cheat_token_permutation", True)
        self.anti_cheat_seed = config.get("anti_cheat_seed", 1337)

    def _generator(self, offset: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed) + int(offset) * 1_000_003)
        return generator

    def _randint(self, generator: torch.Generator, low: int, high: int) -> int:
        return int(torch.randint(low, high, (1,), generator=generator).item())

    def _make_sample(self, sample_id: int, *, eval_split: bool) -> Tuple[torch.Tensor, int, int, int]:
        generator = self._generator(sample_id + (100_000 if eval_split else 0))
        target_depth = 1 + (sample_id % self.max_depth)
        operations_budget = max(4, (self.seq_length - 2) // 2)
        stack: List[int] = []
        sequence: List[int] = []
        pop_count = 0

        for step in range(operations_budget):
            should_push = len(stack) < target_depth or self._randint(generator, 0, 100) < 58
            if should_push and len(sequence) + 2 <= self.seq_length - 1:
                symbol = 10 + self._randint(generator, 0, self.max_symbol)
                stack.append(symbol)
                if len(stack) > self.max_depth:
                    stack.pop(0)
                sequence.extend([self.PUSH, symbol])
            elif stack and len(sequence) + 1 <= self.seq_length - 1:
                stack.pop()
                pop_count += 1
                sequence.append(self.POP)
            elif len(sequence) + 2 <= self.seq_length - 1:
                symbol = 10 + self._randint(generator, 0, self.max_symbol)
                stack.append(symbol)
                sequence.extend([self.PUSH, symbol])

            if len(sequence) >= self.seq_length - 1:
                break

        while eval_split and len(stack) > target_depth and len(sequence) < self.seq_length - 1:
            stack.pop()
            pop_count += 1
            sequence.append(self.POP)

        target = stack[-1] if stack else self.EMPTY
        final_depth = len(stack)
        sequence = (sequence + [self.QUERY] + [self.PAD] * self.seq_length)[: self.seq_length]
        return torch.tensor(sequence, dtype=torch.long), target, final_depth, pop_count

    def _make_dataset(self, count: int, *, eval_split: bool, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets, depths, pops = [], [], [], []
        for i in range(count):
            x, y, depth, pop_count = self._make_sample(i, eval_split=eval_split)
            inputs.append(x)
            targets.append(y)
            depths.append(depth)
            pops.append(pop_count)

        x_tensor = torch.stack(inputs).to(device)
        y_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        depth_tensor = torch.tensor(depths, dtype=torch.long, device=device)
        pop_tensor = torch.tensor(pops, dtype=torch.long, device=device)

        if self.anti_cheat_token_permutation:
            x_tensor, y_tensor = remap_tokens(
                (x_tensor, y_tensor),
                self.vocab_size,
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.PAD, self.EMPTY, self.PUSH, self.POP, self.QUERY),
            )

        return x_tensor, y_tensor, depth_tensor, pop_tensor

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, _, _ = self._make_dataset(self.num_samples, eval_split=False, device=self.device)
        return x, y

    def _evaluate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        depths: torch.Tensor,
        pops: torch.Tensor,
        device,
    ) -> Dict[str, Any]:
        correct = total = top5_correct = pop_heavy_correct = pop_heavy_total = 0
        depth_totals: Dict[int, int] = {}
        depth_correct: Dict[int, int] = {}

        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = x[start : start + self.batch_size]
                yb = y[start : start + self.batch_size]
                db = depths[start : start + self.batch_size]
                pb = pops[start : start + self.batch_size]
                logits = model(xb)[:, -1, :]
                pred = logits.argmax(dim=-1)
                top5 = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices
                matches = pred == yb

                correct += matches.sum().item()
                top5_correct += (top5 == yb.unsqueeze(1)).any(dim=1).sum().item()
                total += len(yb)

                heavy_mask = pb >= 2
                if heavy_mask.any():
                    pop_heavy_correct += matches[heavy_mask].sum().item()
                    pop_heavy_total += heavy_mask.sum().item()

                for depth, ok in zip(db.tolist(), matches.tolist()):
                    depth_totals[depth] = depth_totals.get(depth, 0) + 1
                    depth_correct[depth] = depth_correct.get(depth, 0) + int(ok)

        depth_accuracy = {
            str(depth): depth_correct.get(depth, 0) / count
            for depth, count in sorted(depth_totals.items())
            if count
        }
        solved_depths = [depth for depth, acc in ((int(k), v) for k, v in depth_accuracy.items()) if acc >= 0.90]

        return {
            "accuracy": correct / total if total else 0.0,
            "top5_accuracy": top5_correct / total if total else 0.0,
            "pop_heavy_accuracy": pop_heavy_correct / pop_heavy_total if pop_heavy_total else 0.0,
            "correct_predictions": correct,
            "total_predictions": total,
            "depth_accuracy": depth_accuracy,
            "max_depth_at_90": max(solved_depths) if solved_depths else 0,
        }

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        train_x, train_y, _, _ = self._make_dataset(self.num_samples, eval_split=False, device=device)
        eval_x, eval_y, eval_depths, eval_pops = self._make_dataset(
            self.num_eval_samples,
            eval_split=True,
            device=device,
        )

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        start_time = time.time()
        losses = []
        model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)[:, -1, :]
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, len(loader)))
        training_time = time.time() - start_time

        model.eval()
        eval_start = time.time()
        metrics = self._evaluate(model, eval_x, eval_y, eval_depths, eval_pops, device)
        eval_time = time.time() - eval_start

        results = {
            **metrics,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": self.num_eval_samples / eval_time if eval_time > 0 else 0,
            "train_loss_initial": losses[0] if losses else 0.0,
            "train_loss_final": losses[-1] if losses else 0.0,
            "num_samples": self.num_samples,
            "num_eval_samples": self.num_eval_samples,
            "sequence_length": self.seq_length,
            "max_stack_depth": self.max_depth,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        self.results = results
        self.metrics = {
            "accuracy": results["accuracy"],
            "top5_accuracy": results["top5_accuracy"],
            "max_depth_at_90": results["max_depth_at_90"],
            "training_speed": self.num_samples * self.epochs / training_time if training_time > 0 else 0,
            "inference_speed": results["inference_speed"],
        }
        return results
