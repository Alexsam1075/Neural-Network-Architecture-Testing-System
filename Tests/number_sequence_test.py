"""Number sequence benchmark with seeded train/eval splits."""

import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .base_test import BaseTest


class NumberSequenceTest(BaseTest):
    """Predict the next token of randomized numeric sequences."""

    def __init__(self, config: Dict[str, Any], name: str = "NumberSequenceTest"):
        super().__init__(config, name)

        self.seq_length = config.get("seq_length", 32)
        self.num_sequences = config.get("num_sequences", 1000)
        self.num_eval_sequences = config.get("num_eval_sequences", max(128, self.num_sequences // 4))
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 10)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))

    def generate_arithmetic_sequence(self, start: int = 0, diff: int = 1, length: int = None) -> torch.Tensor:
        length = length or self.seq_length
        seq = torch.arange(start, start + length * diff, diff, dtype=torch.long)
        return torch.clamp(seq[:length], min=0, max=255)

    def generate_geometric_sequence(self, start: int = 1, ratio: int = 2, length: int = None) -> torch.Tensor:
        length = length or self.seq_length
        seq = []
        value = start
        for _ in range(length):
            seq.append(min(value, 255))
            value = min(value * ratio, 255)
        seq = torch.tensor(seq, dtype=torch.long)
        return torch.clamp(seq, min=0, max=255)

    def generate_fibonacci_sequence(self, first: int = 1, second: int = 1, length: int = None) -> torch.Tensor:
        length = length or self.seq_length
        seq = [first, second]
        for _ in range(length - 2):
            seq.append(min(seq[-1] + seq[-2], 255))
        return torch.clamp(torch.tensor(seq[:length], dtype=torch.long), min=0, max=255)

    def generate_square_sequence(self, offset: int = 1, length: int = None) -> torch.Tensor:
        length = length or self.seq_length
        seq = torch.tensor([(i + offset) ** 2 for i in range(length)], dtype=torch.long)
        return torch.clamp(seq, min=0, max=255)

    def generate_polynomial_sequence(self, coeffs: List[float], length: int = None) -> torch.Tensor:
        length = length or self.seq_length
        seq = []
        for x in range(length):
            val = sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))
            seq.append(int(val))
        return torch.clamp(torch.tensor(seq, dtype=torch.long), min=0, max=255)

    def _make_dataset(self, count: int, seed_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed) + int(seed_offset))
        full_length = self.seq_length + 1

        for _ in range(count):
            seq_type = int(torch.randint(0, 5, (1,), generator=generator).item())

            if seq_type == 0:
                start = int(torch.randint(0, 180, (1,), generator=generator).item())
                diff = int(torch.randint(1, 11, (1,), generator=generator).item())
                seq = self.generate_arithmetic_sequence(start=start, diff=diff, length=full_length)
            elif seq_type == 1:
                start = int(torch.randint(1, 8, (1,), generator=generator).item())
                ratio = int(torch.randint(2, 5, (1,), generator=generator).item())
                seq = self.generate_geometric_sequence(start=start, ratio=ratio, length=full_length)
            elif seq_type == 2:
                first = int(torch.randint(1, 8, (1,), generator=generator).item())
                second = int(torch.randint(1, 8, (1,), generator=generator).item())
                seq = self.generate_fibonacci_sequence(first=first, second=second, length=full_length)
            elif seq_type == 3:
                offset = int(torch.randint(0, 8, (1,), generator=generator).item())
                seq = self.generate_square_sequence(offset=offset, length=full_length)
            else:
                coeffs = [
                    float(torch.randint(0, 10, (1,), generator=generator).item()) / 10,
                    float(torch.randint(0, 10, (1,), generator=generator).item()) / 10,
                    float(torch.randint(1, 4, (1,), generator=generator).item()),
                ]
                seq = self.generate_polynomial_sequence(coeffs, length=full_length)

            sequences.append(seq[:-1])
            targets.append(seq[-1])

        x = torch.stack(sequences)
        y = torch.stack(targets)
        return x.to(self.device), y.to(self.device)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_sequences, seed_offset=0)

    def prepare_eval_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_eval_sequences, seed_offset=10_000)

    def run(self, model: nn.Module) -> Dict[str, Any]:
        model.to(self.device)

        print(f"[{self.name}] Preparing data...")
        x_train, y_train = self.prepare_data()

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        print(f"[{self.name}] Training...")
        model.train()

        start_time = time.time()
        train_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i : i + self.batch_size]
                y_batch = y_train[i : i + self.batch_size]

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits[:, -1, :], y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_loss)

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        training_time = time.time() - start_time

        print(f"[{self.name}] Evaluating...")
        model.eval()
        x_eval, y_eval = self.prepare_eval_data()

        with torch.no_grad():
            eval_start = time.time()
            correct = 0
            total = 0

            for i in range(0, len(x_eval), self.batch_size):
                x_batch = x_eval[i : i + self.batch_size]
                y_batch = y_eval[i : i + self.batch_size]

                logits = model(x_batch)
                predictions = torch.argmax(logits[:, -1, :], dim=-1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

            eval_time = time.time() - eval_start

        accuracy = correct / total if total > 0 else 0.0

        results = {
            "accuracy": accuracy,
            "correct_predictions": correct,
            "total_predictions": total,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": total / eval_time if eval_time > 0 else 0,
            "training_speed": self.num_sequences * self.epochs / training_time if training_time > 0 else 0,
            "train_loss_final": train_losses[-1] if train_losses else 0,
            "train_loss_initial": train_losses[0] if train_losses else 0,
            "num_sequences": self.num_sequences,
            "num_eval_sequences": self.num_eval_sequences,
            "sequence_length": self.seq_length,
            "random_seed": self.random_seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        self.results = results
        self.metrics = {
            "accuracy": accuracy,
            "training_speed": results["training_speed"],
            "inference_speed": results["inference_speed"],
        }

        print(f"[{self.name}] Completed")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Inference speed: {self.metrics['inference_speed']:.1f} samples/sec")

        return results
