"""Autoregressive efficiency benchmark."""

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base_test import BaseTest


class AutoregressiveEfficiencyTest(BaseTest):
    """Measure repeated prefix-extension cost during naive autoregressive decoding."""

    def __init__(self, config: Dict[str, Any], name: str = "AutoregressiveEfficiencyTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.prompt_length = config.get("prompt_length", 24)
        self.generate_steps = config.get("generate_steps", 32)
        self.batch_size = config.get("batch_size", 4)
        self.random_seed = config.get("random_seed", config.get("anti_cheat_seed", 1337))

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.arange(self.prompt_length, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1) % self.vocab_size
        y = torch.zeros(self.batch_size, dtype=torch.long)
        return x.to(self.device), y.to(self.device)

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.random_seed))
        sequence = torch.randint(0, self.vocab_size, (self.batch_size, self.prompt_length), generator=generator).to(device)
        model.eval()

        step_times = []
        produced = []
        with torch.no_grad():
            model(sequence)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_total = time.time()
            for _ in range(self.generate_steps):
                step_start = time.time()
                logits = model(sequence)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.time() - step_start)
                produced.append(next_token)
                sequence = torch.cat([sequence, next_token], dim=1)
            total_time = time.time() - start_total

        tokens = self.batch_size * self.generate_steps
        first_half = sum(step_times[: max(1, len(step_times) // 2)])
        second_half = sum(step_times[len(step_times) // 2 :])
        growth_ratio = second_half / first_half if first_half > 0 else 0.0
        generated = torch.cat(produced, dim=1) if produced else torch.empty(0, device=device)
        unique_rate = generated.unique().numel() / max(1, generated.numel()) if generated.numel() else 0.0
        result = {
            "accuracy": tokens / total_time if total_time > 0 else 0,
            "autoregressive_tokens_per_second": tokens / total_time if total_time > 0 else 0,
            "inference_speed": tokens / total_time if total_time > 0 else 0,
            "training_time_seconds": 0.0,
            "evaluation_time_seconds": total_time,
            "step_time_mean": sum(step_times) / len(step_times) if step_times else 0,
            "step_time_growth_ratio": growth_ratio,
            "generated_unique_rate": float(unique_rate),
            "prompt_length": self.prompt_length,
            "generate_steps": self.generate_steps,
            "batch_size": self.batch_size,
        }
        self.results = result
        self.metrics = {
            "accuracy": result["accuracy"],
            "inference_speed": result["inference_speed"],
            "step_time_growth_ratio": growth_ratio,
        }
        return result
