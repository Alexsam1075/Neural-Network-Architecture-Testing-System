"""Next-token benchmark with seeded randomized train/eval splits."""

import random
import string
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .base_test import BaseTest


class NextTokenPredictionTest(BaseTest):
    """Predict the next character after a synthetic text prefix."""

    def __init__(self, config: Dict[str, Any], name: str = "NextTokenPredictionTest"):
        super().__init__(config, name)

        self.seq_length = config.get("seq_length", 32)
        self.num_samples = config.get("num_samples", 1000)
        self.num_eval_samples = config.get("num_eval_samples", max(128, self.num_samples // 4))
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 10)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))

        self.charset = string.ascii_lowercase + string.digits + ".,!? "
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

    def _repeat_to_length(self, pattern: str, length: int) -> str:
        text = ""
        while len(text) < length:
            text += pattern
        return text[:length]

    def generate_repeating_pattern(self, pattern: str, length: int) -> str:
        return self._repeat_to_length(pattern, length)

    def generate_alternating_pattern(self, char1: str, char2: str, length: int) -> str:
        return "".join(char1 if i % 2 == 0 else char2 for i in range(length))

    def generate_incrementing_pattern(self, start_char: str, step: int, length: int) -> str:
        idx = self.char_to_idx.get(start_char, 0)
        return "".join(self.idx_to_char[(idx + i * step) % len(self.charset)] for i in range(length))

    def generate_word_sequence(self, words: List[str], length: int) -> str:
        text = ""
        pos = 0
        while len(text) < length:
            text += ("" if not text else " ") + words[pos % len(words)]
            pos += 1
        return text[:length]

    def generate_random_text(self, rng: random.Random, length: int) -> str:
        return "".join(rng.choice(self.charset) for _ in range(length))

    def text_to_indices(self, text: str) -> torch.Tensor:
        return torch.tensor([self.char_to_idx.get(c, 0) for c in text], dtype=torch.long)

    def _make_dataset(self, count: int, seed_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        rng = random.Random(int(self.random_seed) + int(seed_offset))
        common_words = ["the", "cat", "dog", "run", "jump", "walk", "talk", "code", "data", "loop"]
        full_length = self.seq_length + 1

        for _ in range(count):
            pattern_type = rng.randrange(6)

            if pattern_type == 0:
                pattern = rng.choice(common_words)
                text = self.generate_repeating_pattern(pattern, full_length)
            elif pattern_type == 1:
                char1 = rng.choice(self.charset)
                char2 = rng.choice(self.charset)
                text = self.generate_alternating_pattern(char1, char2, full_length)
            elif pattern_type == 2:
                start_char = rng.choice(self.charset)
                step = rng.randrange(1, 7)
                text = self.generate_incrementing_pattern(start_char, step, full_length)
            elif pattern_type == 3:
                width = rng.randrange(1, 5)
                words = [rng.choice(common_words) for _ in range(width)]
                text = self.generate_word_sequence(words, full_length)
            elif pattern_type == 4:
                pattern = rng.choice(common_words)
                text = self._repeat_to_length(pattern + pattern[::-1], full_length)
            else:
                text = self.generate_random_text(rng, full_length)

            indices = self.text_to_indices(text)
            sequences.append(indices[:-1])
            targets.append(indices[-1])

        x = torch.stack(sequences)
        y = torch.stack(targets)
        return x.to(self.device), y.to(self.device)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_samples, seed_offset=0)

    def prepare_eval_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._make_dataset(self.num_eval_samples, seed_offset=10_000)

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
            top5_correct = 0
            total = 0

            for i in range(0, len(x_eval), self.batch_size):
                x_batch = x_eval[i : i + self.batch_size]
                y_batch = y_eval[i : i + self.batch_size]

                logits = model(x_batch)
                last_logits = logits[:, -1, :]
                predictions = torch.argmax(last_logits, dim=-1)
                correct += (predictions == y_batch).sum().item()
                top5 = torch.topk(last_logits, min(5, last_logits.shape[-1]), dim=-1)[1]
                top5_correct += (top5 == y_batch.unsqueeze(1)).any(dim=1).sum().item()
                total += y_batch.size(0)

            eval_time = time.time() - eval_start

        accuracy = correct / total if total else 0.0
        top5_accuracy = top5_correct / total if total else 0.0

        results = {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "correct_predictions": correct,
            "total_predictions": total,
            "training_time_seconds": training_time,
            "evaluation_time_seconds": eval_time,
            "inference_speed": total / eval_time if eval_time > 0 else 0,
            "training_speed": self.num_samples * self.epochs / training_time if training_time > 0 else 0,
            "train_loss_final": train_losses[-1] if train_losses else 0,
            "train_loss_initial": train_losses[0] if train_losses else 0,
            "num_samples": self.num_samples,
            "num_eval_samples": self.num_eval_samples,
            "sequence_length": self.seq_length,
            "vocab_size": len(self.charset),
            "random_seed": self.random_seed,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        self.results = results
        self.metrics = {
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "training_speed": results["training_speed"],
            "inference_speed": results["inference_speed"],
        }

        print(f"[{self.name}] Completed")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Inference speed: {self.metrics['inference_speed']:.1f} samples/sec")

        return results
