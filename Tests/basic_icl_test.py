"""Basic in-context learning benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .anti_cheat import remap_tokens
from .base_test import BaseTest


class BasicICLTest(BaseTest):
    """Infer a per-sample affine rule from examples in the prompt."""

    PAD = 0
    SEP = 241
    QUERY = 242

    def __init__(self, config: Dict[str, Any], name: str = "BasicICLTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.seq_length = config.get("seq_length", 40)
        self.num_shots = min(config.get("num_shots", 5), max(2, (self.seq_length - 3) // 3))
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
        xs, ys = [], []
        for _ in range(count):
            a = int(torch.randint(1, 6, (1,), generator=generator).item())
            b = int(torch.randint(0, 17, (1,), generator=generator).item())
            seq = []
            used = set()
            for _shot in range(self.num_shots):
                x_val = int(torch.randint(0, 24, (1,), generator=generator).item())
                used.add(x_val)
                y_val = (a * x_val + b) % 96 + 20
                seq.extend([x_val + 20, y_val, self.SEP])
            query = int(torch.randint(0, 24, (1,), generator=generator).item())
            while query in used:
                query = int(torch.randint(0, 24, (1,), generator=generator).item())
            target = (a * query + b) % 96 + 20
            seq.extend([self.QUERY, query + 20])
            seq = (seq + [self.PAD] * self.seq_length)[: self.seq_length]
            xs.append(torch.tensor(seq, dtype=torch.long))
            ys.append(target)
        x_tensor = torch.stack(xs).to(device)
        y_tensor = torch.tensor(ys, dtype=torch.long, device=device)
        if self.anti_cheat_token_permutation:
            x_tensor, y_tensor = remap_tokens(
                (x_tensor, y_tensor),
                self.vocab_size,
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.PAD, self.SEP, self.QUERY),
            )
        return x_tensor, y_tensor

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, 0, self.device)

    def _accuracy(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb, yb = x[start : start + self.batch_size], y[start : start + self.batch_size]
                pred = model(xb)[:, -1, :].argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        return correct / total if total else 0.0

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
                loss = criterion(model(xb)[:, -1, :], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, len(loader)))
        training_time = time.time() - start_time

        model.eval()
        eval_start = time.time()
        accuracy = self._accuracy(model, eval_x, eval_y)
        eval_time = time.time() - eval_start
        result = {
            "accuracy": accuracy,
            "num_shots": self.num_shots,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": self.num_eval_samples / eval_time if eval_time > 0 else 0,
            "train_loss_initial": losses[0] if losses else 0.0,
            "train_loss_final": losses[-1] if losses else 0.0,
            "sequence_length": self.seq_length,
            "num_samples": self.num_samples,
            "num_eval_samples": self.num_eval_samples,
            "epochs": self.epochs,
        }
        self.results = result
        self.metrics = {"accuracy": accuracy, "inference_speed": result["inference_speed"]}
        return result
