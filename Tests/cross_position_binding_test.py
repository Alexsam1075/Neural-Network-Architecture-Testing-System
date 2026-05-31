"""Cross-position binding benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .anti_cheat import remap_tokens
from .base_test import BaseTest


class CrossPositionBindingTest(BaseTest):
    """Bind an entity token at one position to an attribute token far away."""

    PAD = 0
    QUERY = 242
    FILL = 243

    def __init__(self, config: Dict[str, Any], name: str = "CrossPositionBindingTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.seq_length = config.get("seq_length", 56)
        self.num_bindings = min(config.get("num_bindings", 12), max(3, (self.seq_length - 4) // 3))
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
            entities = (10 + torch.randperm(70, generator=generator)[: self.num_bindings]).tolist()
            attrs = (110 + torch.randperm(70, generator=generator)[: self.num_bindings]).tolist()
            order_entities = torch.randperm(self.num_bindings, generator=generator).tolist()
            order_attrs = torch.randperm(self.num_bindings, generator=generator).tolist()
            target = int(torch.randint(0, self.num_bindings, (1,), generator=generator).item())
            seq = []
            for idx in order_entities:
                seq.extend([entities[idx], self.FILL])
            for idx in order_attrs:
                seq.extend([attrs[idx], entities[idx]])
            seq.extend([self.QUERY, entities[target]])
            seq = (seq + [self.PAD] * self.seq_length)[: self.seq_length]
            xs.append(torch.tensor(seq, dtype=torch.long))
            ys.append(attrs[target])
        x_tensor = torch.stack(xs).to(device)
        y_tensor = torch.tensor(ys, dtype=torch.long, device=device)
        if self.anti_cheat_token_permutation:
            x_tensor, y_tensor = remap_tokens(
                (x_tensor, y_tensor),
                self.vocab_size,
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.PAD, self.QUERY, self.FILL),
            )
        return x_tensor, y_tensor

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, 0, self.device)

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

        start_time = time.time()
        losses = []
        model.train()
        for _ in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb)[:, -1, :], yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / max(1, len(loader)))
        training_time = time.time() - start_time

        model.eval()
        eval_start = time.time()
        correct = top5_correct = total = 0
        with torch.no_grad():
            for start in range(0, len(eval_x), self.batch_size):
                xb, yb = eval_x[start : start + self.batch_size], eval_y[start : start + self.batch_size]
                logits = model(xb)[:, -1, :]
                pred = logits.argmax(dim=-1)
                top5 = torch.topk(logits, min(5, logits.shape[-1]), dim=-1).indices
                correct += (pred == yb).sum().item()
                top5_correct += (top5 == yb.unsqueeze(1)).any(dim=1).sum().item()
                total += len(yb)
        eval_time = time.time() - eval_start
        result = {
            "accuracy": correct / total if total else 0.0,
            "top5_accuracy": top5_correct / total if total else 0.0,
            "correct_predictions": correct,
            "total_predictions": total,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": self.num_eval_samples / eval_time if eval_time > 0 else 0,
            "train_loss_initial": losses[0] if losses else 0.0,
            "train_loss_final": losses[-1] if losses else 0.0,
            "num_bindings": self.num_bindings,
            "sequence_length": self.seq_length,
            "num_samples": self.num_samples,
            "num_eval_samples": self.num_eval_samples,
            "epochs": self.epochs,
        }
        self.results = result
        self.metrics = {"accuracy": result["accuracy"], "top5_accuracy": result["top5_accuracy"]}
        return result
