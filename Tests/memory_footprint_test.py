"""Memory footprint benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base_test import BaseTest


class MemoryFootprintTest(BaseTest):
    """Estimate parameter, activation, and optional CUDA peak memory use."""

    def __init__(self, config: Dict[str, Any], name: str = "MemoryFootprintTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.batch_size = config.get("batch_size", 8)
        self.lengths = config.get("memory_lengths", [32, 128, 512])
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length = max(self.lengths)
        x = torch.arange(length, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1) % self.vocab_size
        y = torch.zeros(self.batch_size, dtype=torch.long)
        return x.to(self.device), y.to(self.device)

    def _input(self, length: int, device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed) + int(length))
        return torch.randint(0, self.vocab_size, (self.batch_size, length), generator=generator).to(device)

    def _declared_limit(self, model: nn.Module) -> int:
        info = {}
        if hasattr(model, "get_architecture_info"):
            try:
                info = model.get_architecture_info()
            except Exception:
                info = {}
        if info.get("long_context_safe", False):
            return 0
        config = getattr(model, "config", {}) or {}
        for key in ("max_seq_len", "seq_length", "context_length"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value
        for attr in ("max_seq_len", "seq_length"):
            value = getattr(model, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return 0

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        parameter_memory_mb = total_parameters * 4 / (1024 * 1024)
        per_length = {}
        skipped_lengths = []
        peak_mb = None
        total_time = 0.0
        declared_limit = self._declared_limit(model)

        with torch.no_grad():
            for length in self.lengths:
                if declared_limit and int(length) > declared_limit:
                    skipped_lengths.append(
                        {
                            "length": int(length),
                            "reason": "declared_position_limit",
                            "declared_limit": declared_limit,
                        }
                    )
                    continue
                x = self._input(int(length), device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.time()
                out = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                elapsed = time.time() - start
                total_time += elapsed
                output_memory_mb = out.numel() * out.element_size() / (1024 * 1024)
                per_length[str(length)] = {
                    "elapsed_seconds": elapsed,
                    "output_memory_mb": output_memory_mb,
                    "cuda_peak_memory_mb": peak_mb,
                    "tokens_per_second": (self.batch_size * int(length)) / elapsed if elapsed > 0 else 0,
                }

        largest = max(self.lengths) if self.lengths else 0
        measured_lengths = [int(length) for length in per_length.keys()]
        largest_measured = max(measured_lengths) if measured_lengths else 0
        largest_output_mb = per_length.get(str(largest_measured), {}).get("output_memory_mb", 0.0)
        score_length = largest_measured or largest
        measured_tokens = self.batch_size * sum(measured_lengths)
        memory_efficiency = score_length / max(1.0, parameter_memory_mb + largest_output_mb)
        result = {
            "accuracy": memory_efficiency,
            "memory_efficiency_score": memory_efficiency,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "parameter_memory_mb": parameter_memory_mb,
            "largest_output_memory_mb": largest_output_mb,
            "cuda_peak_memory_mb": peak_mb,
            "training_time_seconds": 0.0,
            "evaluation_time_seconds": total_time,
            "inference_speed": measured_tokens / total_time if total_time > 0 else 0,
            "batch_size": self.batch_size,
            "declared_context_limit": declared_limit,
            "largest_measured_length": largest_measured,
            "skipped_lengths": skipped_lengths,
            "per_length": per_length,
        }
        self.results = result
        self.metrics = {
            "accuracy": result["accuracy"],
            "memory_efficiency_score": memory_efficiency,
            "inference_speed": result["inference_speed"],
        }
        return result
