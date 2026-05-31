"""Exact sequence copying benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .anti_cheat import remap_tokens
from .base_test import BaseTest


class ExactCopyTest(BaseTest):
    """Copy a random token sequence exactly, measuring token accuracy and exact match."""

    PAD = 0
    SEP = 241

    def __init__(self, config: Dict[str, Any], name: str = "ExactCopyTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.seq_length = config.get("seq_length", 48)
        self.copy_length = min(config.get("copy_length", self.seq_length // 2), self.seq_length - 2)
        self.num_samples = config.get("num_samples", config.get("num_sequences", 512))
        self.num_eval_samples = config.get("num_eval_samples", max(128, self.num_samples // 4))
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 6)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))
        self.anti_cheat_token_permutation = config.get("anti_cheat_token_permutation", True)
        self.anti_cheat_seed = config.get("anti_cheat_seed", 1337)

    def _make_dataset(self, count: int, offset: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed) + int(offset))
        inputs, targets = [], []
        for _ in range(count):
            payload = torch.randint(3, min(self.vocab_size, 220), (self.copy_length,), generator=generator)
            x = torch.full((self.seq_length,), self.PAD, dtype=torch.long)
            y = torch.full((self.seq_length,), self.PAD, dtype=torch.long)
            x[: self.copy_length] = payload
            x[self.copy_length] = self.SEP
            y[: self.copy_length] = payload
            inputs.append(x)
            targets.append(y)
        x_tensor = torch.stack(inputs).to(device)
        y_tensor = torch.stack(targets).to(device)
        if self.anti_cheat_token_permutation:
            x_tensor, y_tensor = remap_tokens(
                (x_tensor, y_tensor),
                self.vocab_size,
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.PAD, self.SEP),
            )
        return x_tensor, y_tensor

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, 0, self.device)

    def _accuracy(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, device) -> Dict[str, float]:
        correct = total = exact = 0
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = x[start : start + self.batch_size]
                yb = y[start : start + self.batch_size]
                pred = model(xb)[:, : self.copy_length, :].argmax(dim=-1)
                target = yb[:, : self.copy_length]
                matches = pred == target
                correct += matches.sum().item()
                total += matches.numel()
                exact += matches.all(dim=1).sum().item()
        return {
            "accuracy": correct / total if total else 0.0,
            "exact_match": exact / len(x) if len(x) else 0.0,
            "correct_predictions": correct,
            "total_predictions": total,
        }

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        train_x, train_y = self._make_dataset(self.num_samples, 0, device)
        eval_x, eval_y = self._make_dataset(self.num_eval_samples, 10_000, device)
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
                logits = model(xb)[:, : self.copy_length, :]
                loss = criterion(logits.reshape(-1, logits.shape[-1]), yb[:, : self.copy_length].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, len(loader)))
        training_time = time.time() - start_time

        model.eval()
        eval_start = time.time()
        metrics = self._accuracy(model, eval_x, eval_y, device)
        eval_time = time.time() - eval_start
        result = {
            **metrics,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": self.num_eval_samples / eval_time if eval_time > 0 else 0,
            "train_loss_initial": losses[0] if losses else 0.0,
            "train_loss_final": losses[-1] if losses else 0.0,
            "copy_length": self.copy_length,
            "sequence_length": self.seq_length,
            "num_samples": self.num_samples,
            "num_eval_samples": self.num_eval_samples,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        self.results = result
        self.metrics = {
            "accuracy": result["accuracy"],
            "exact_match": result["exact_match"],
            "training_speed": self.num_samples * self.epochs / training_time if training_time > 0 else 0,
            "inference_speed": result["inference_speed"],
        }
        return result
