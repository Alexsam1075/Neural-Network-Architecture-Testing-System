import math
import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base_test import BaseTest


class UltraLongContextTest(BaseTest):
    """
    Tests long-context survivability at 1k, 10k, 100k, 1M, and 1B tokens.

    The first lengths are executed when they fit the configured memory/runtime
    budget. Very large lengths are reported as estimates/skips so the benchmark
    can expose scaling limits without trying to allocate impossible tensors.
    """

    def __init__(self, config: Dict[str, Any], name: str = "UltraLongContextTest"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.target_lengths = config.get(
            "target_lengths",
            [1_000, 10_000, 100_000, 1_000_000, 1_000_000_000],
        )
        self.batch_size = config.get("batch_size", 1)
        self.exact_token_limit = config.get("exact_token_limit", 10_000)
        self.max_exact_seconds = config.get("max_exact_seconds", 5.0)
        self.memory_budget_mb = config.get("memory_budget_mb", 1024)
        self.quadratic_exact_limit = config.get("quadratic_exact_limit", 2_048)
        self.context_probe_length = config.get("context_probe_length", 512)

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        length = min(self.target_lengths)
        x = torch.arange(length, dtype=torch.long).unsqueeze(0) % self.vocab_size
        y = torch.roll(x, shifts=-1, dims=1)
        return x.to(self.device), y.to(self.device)

    def _looks_quadratic(self, model: nn.Module) -> bool:
        name = getattr(model, "name", model.__class__.__name__).lower()
        if any(isinstance(module, nn.MultiheadAttention) for module in model.modules()):
            return True
        if hasattr(model, "get_architecture_info"):
            try:
                info = model.get_architecture_info()
                if bool(info.get("use_attention", False)):
                    return True
            except Exception:
                pass
        markers = ("transformer", "multihead")
        return any(marker in name for marker in markers)

    def _estimate_output_memory_mb(self, length: int) -> float:
        bytes_per_logit = 4
        return self.batch_size * length * self.vocab_size * bytes_per_logit / (1024 * 1024)

    def _declared_max_seq_len(self, model: nn.Module) -> int:
        config = getattr(model, "config", {}) or {}
        for key in ("max_seq_len", "seq_length", "context_length"):
            value = config.get(key)
            if isinstance(value, int) and value > 0:
                return value
        value = getattr(model, "max_seq_len", None)
        if isinstance(value, int) and value > 0:
            return value
        return 0

    def _long_context_safe(self, model: nn.Module) -> bool:
        if hasattr(model, "get_architecture_info"):
            try:
                return bool(model.get_architecture_info().get("long_context_safe", False))
            except Exception:
                return False
        return False

    def _make_input(self, length: int, device) -> torch.Tensor:
        base = torch.arange(length, dtype=torch.long, device=device) % self.vocab_size
        return base.unsqueeze(0).expand(self.batch_size, -1).contiguous()

    def _run_exact(self, model: nn.Module, length: int, device) -> Dict[str, Any]:
        x = self._make_input(length, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.no_grad():
            out = model(x)
            pred = out[:, -1, :].argmax(dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_mb = None
        elapsed = time.time() - start
        return {
            "status": "passed",
            "elapsed_seconds": elapsed,
            "tokens_per_second": (self.batch_size * length) / elapsed if elapsed > 0 else 0,
            "output_shape": list(out.shape),
            "peak_memory_mb": peak_mb,
            "last_token_prediction": int(pred[0].item()),
        }

    def _test_context_retention(self, model: nn.Module, length: int, device) -> Dict[str, Any]:
        probe_length = max(2, min(length, self.context_probe_length))
        x1 = self._make_input(probe_length, device)
        x2 = x1.clone()
        x2[:, 0] = (x2[:, 0] + max(1, self.vocab_size // 3)) % self.vocab_size

        with torch.no_grad():
            y1 = model(x1)[:, -1, :]
            y2 = model(x2)[:, -1, :]

        delta = (y1 - y2).abs()
        max_delta = float(delta.max().item())
        mean_delta = float(delta.mean().item())
        context_used = max_delta > 1e-4
        return {
            "context_probe_length": probe_length,
            "context_used": context_used,
            "context_output_max_delta": max_delta,
            "context_output_mean_delta": mean_delta,
        }

    def _run_cpu_probe(self, model: nn.Module, length: int) -> Dict[str, Any]:
        original_device = next(model.parameters()).device
        try:
            model.to("cpu")
            return self._run_exact(model, length, torch.device("cpu"))
        finally:
            if original_device.type == "cuda":
                model.to(original_device)

    def _run_safe_exact(
        self,
        model: nn.Module,
        length: int,
        device,
        *,
        long_context_safe: bool,
    ) -> Dict[str, Any]:
        if long_context_safe:
            return self._run_exact(model, length, device)
        return self._run_cpu_probe(model, length)

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        model.eval()
        quadratic = self._looks_quadratic(model)
        declared_max_seq_len = self._declared_max_seq_len(model)
        long_context_safe = self._long_context_safe(model)
        per_length: Dict[str, Dict[str, Any]] = {}
        exact_passed = 0
        exact_attempted = 0
        largest_exact_passed = 0
        declared_probe = None

        if declared_max_seq_len > 0:
            probe_length = min(declared_max_seq_len, self.exact_token_limit)
            try:
                declared_probe = self._run_safe_exact(
                    model,
                    probe_length,
                    device,
                    long_context_safe=long_context_safe,
                )
                declared_probe["probe_length"] = probe_length
            except Exception as exc:
                declared_probe = {
                    "status": "failed",
                    "probe_length": probe_length,
                    "error": str(exc)[:500],
                }

        for length in self.target_lengths:
            key = str(length)
            estimated_output_mb = self._estimate_output_memory_mb(length)
            entry = {
                "length": length,
                "estimated_output_memory_mb": estimated_output_mb,
                "quadratic_risk": quadratic,
                "declared_max_seq_len": declared_max_seq_len,
                "long_context_safe": long_context_safe,
            }

            should_run = length <= self.exact_token_limit
            should_run = should_run and estimated_output_mb <= self.memory_budget_mb
            should_run = should_run and not (quadratic and length > self.quadratic_exact_limit)
            should_run = should_run and (
                long_context_safe or declared_max_seq_len <= 0 or length <= declared_max_seq_len
            )

            if not should_run:
                reason = "above_exact_token_limit"
                if estimated_output_mb > self.memory_budget_mb:
                    reason = "estimated_output_memory_over_budget"
                if quadratic and length > self.quadratic_exact_limit:
                    reason = "quadratic_attention_risk"
                if declared_max_seq_len > 0 and length > declared_max_seq_len and not long_context_safe:
                    reason = "declared_position_limit"
                entry.update(
                    {
                        "status": "estimated_only",
                        "skip_reason": reason,
                        "estimated_min_output_gb": estimated_output_mb / 1024,
                    }
                )
                per_length[key] = entry
                continue

            exact_attempted += 1
            try:
                exact = self._run_safe_exact(
                    model,
                    length,
                    device,
                    long_context_safe=long_context_safe,
                )
                try:
                    exact.update(self._test_context_retention(model, length, device))
                except Exception as exc:
                    exact["context_probe_error"] = str(exc)[:500]
                if exact["elapsed_seconds"] > self.max_exact_seconds:
                    exact["warning"] = "exact_run_exceeded_time_budget"
                entry.update(exact)
                exact_passed += 1
                largest_exact_passed = max(largest_exact_passed, length)
            except RuntimeError as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                entry.update({"status": "failed", "error": str(exc)[:500]})
            except Exception as exc:
                entry.update({"status": "failed", "error": str(exc)[:500]})
            per_length[key] = entry

        target_count = len(self.target_lengths)
        valid_estimate_reasons = {
            "above_exact_token_limit",
            "quadratic_attention_risk",
            "estimated_output_memory_over_budget",
        }
        exact_supported = sum(1 for v in per_length.values() if v["status"] == "passed")
        estimated_supported = sum(
            1
            for v in per_length.values()
            if v["status"] == "passed"
            or (v["status"] == "estimated_only" and v.get("skip_reason") in valid_estimate_reasons)
        )
        context_checked = [v for v in per_length.values() if v.get("context_used") is not None]
        context_used_count = sum(1 for v in context_checked if v.get("context_used"))
        context_retention_rate = context_used_count / len(context_checked) if context_checked else 0.0
        support_score = estimated_supported / target_count if target_count else 0.0
        context_multiplier = context_retention_rate if context_checked else 1.0
        accuracy = support_score * context_multiplier
        exact_pass_rate = exact_passed / exact_attempted if exact_attempted else 0.0

        result = {
            "accuracy": accuracy,
            "long_context_score": accuracy,
            "support_score": support_score,
            "context_multiplier": context_multiplier,
            "exact_supported_count": exact_supported,
            "estimated_supported_count": estimated_supported,
            "context_retention_rate": context_retention_rate,
            "context_used_count": context_used_count,
            "context_checked_count": len(context_checked),
            "exact_pass_rate": exact_pass_rate,
            "exact_attempted": exact_attempted,
            "exact_passed": exact_passed,
            "largest_exact_passed": largest_exact_passed,
            "declared_limit_probe": declared_probe,
            "declared_limit_supported": bool(declared_probe and declared_probe.get("status") == "passed"),
            "target_lengths": self.target_lengths,
            "batch_size": self.batch_size,
            "exact_token_limit": self.exact_token_limit,
            "memory_budget_mb": self.memory_budget_mb,
            "quadratic_detected": quadratic,
            "declared_max_seq_len": declared_max_seq_len,
            "long_context_safe": long_context_safe,
            "per_length": per_length,
        }
        self.results = result
        self.metrics = {
            "accuracy": accuracy,
            "support_score": support_score,
            "exact_pass_rate": exact_pass_rate,
            "largest_exact_passed": largest_exact_passed,
            "declared_limit_supported": result["declared_limit_supported"],
            "context_retention_rate": context_retention_rate,
        }
        return result
