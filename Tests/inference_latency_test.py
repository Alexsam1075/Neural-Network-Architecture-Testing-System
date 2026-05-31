"""Inference latency benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base_test import BaseTest


class InferenceLatencyTest(BaseTest):
    """Measure eval-mode latency across short, medium, and long synthetic inputs."""

    def __init__(self, config: Dict[str, Any], name: str = "InferenceLatencyTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.batch_size = config.get("batch_size", 8)
        self.lengths = config.get("latency_lengths", [16, 64, 128])
        self.repeats = config.get("latency_repeats", 8)
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length = max(self.lengths)
        x = torch.arange(length, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1) % self.vocab_size
        y = torch.zeros(self.batch_size, dtype=torch.long)
        return x.to(self.device), y.to(self.device)

    def _input(self, length: int, device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed) + int(length))
        x = torch.randint(0, self.vocab_size, (self.batch_size, length), generator=generator)
        return x.to(device)

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()
        per_length = {}
        total_tokens = 0
        total_time = 0.0

        with torch.no_grad():
            for length in self.lengths:
                x = self._input(int(length), device)
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                for _ in range(self.repeats):
                    model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                tokens = self.batch_size * int(length) * self.repeats
                total_tokens += tokens
                total_time += elapsed
                per_length[str(length)] = {
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
                    "milliseconds_per_forward": (elapsed / self.repeats) * 1000 if self.repeats else 0,
                }

        result = {
            "accuracy": total_tokens / total_time if total_time > 0 else 0,
            "latency_score": total_tokens / total_time if total_time > 0 else 0,
            "inference_speed": total_tokens / total_time if total_time > 0 else 0,
            "training_time_seconds": 0.0,
            "evaluation_time_seconds": total_time,
            "batch_size": self.batch_size,
            "repeats": self.repeats,
            "per_length": per_length,
        }
        self.results = result
        self.metrics = {
            "accuracy": result["accuracy"],
            "inference_speed": result["inference_speed"],
            "latency_score": result["latency_score"],
        }
        return result
