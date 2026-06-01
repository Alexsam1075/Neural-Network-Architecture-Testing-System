from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .honest_sequence_core import HonestSequenceCore
from .pointer_infinity import LastOccurrencePointer


class _ConvSprintGeneratorBase(HonestSequenceCore):
    """Non-quadratic convolutional generator with explicit local logits."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        conv_kernel: int,
        local_scale: float,
        context_scale: float,
        pointer_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(
            config,
            name,
            dim=dim,
            layers=layers,
            use_attention=False,
            use_recurrent=False,
            conv_kernel=conv_kernel,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.local_head = nn.Linear(self.dim * 4, self.vocab_size)
        self.context_head = nn.Linear(self.dim, self.vocab_size)
        self.pointer = LastOccurrencePointer(self.vocab_size)
        self.local_scale = nn.Parameter(torch.tensor(local_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))
        self.pointer_scale = nn.Parameter(torch.tensor(pointer_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_emb(x)
        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta = token_h - prev_h
        steps = torch.arange(1, length + 1, device=x.device, dtype=token_h.dtype).view(1, -1, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps

        base_logits = super().forward(x)
        local_logits = self.local_head(torch.cat([token_h, prev_h, delta, prefix_mean], dim=-1))
        context_logits = torch.tanh(self.context_head(prefix_mean))
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=base_logits.dtype)
        return (
            base_logits
            + self.local_scale * local_logits
            + self.context_scale * context_logits
            + self.pointer_scale * self.pointer(x, base_logits.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": arch_type,
                "max_context_len": self.max_context_len,
                "non_quadratic": True,
                "local_generation_head": True,
                "last_occurrence_pointer": True,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "complexity": "O(sequence * conv_kernel * dim + sequence * vocab), no QK^T matrix",
                "formula": (
                    "h=ConvSequenceCore(x); p_t=mean(E[x_<=t]); "
                    "local_logits=W_l[E[x_t],E[x_{t-1}],E[x_t]-E[x_{t-1}],p_t]; "
                    "P_t=one_hot(x_{j+1}) where j<t and x_j=x_t; "
                    "logits=W_o LN(h)+alpha local_logits+beta tanh(W_c p_t)+eta P_t-rho one_hot(x_t)"
                ),
            }
        )
        return info


class ConvSprintGeneratorLite(_ConvSprintGeneratorBase):
    """Fast non-quadratic generator."""

    def __init__(self, config: Dict[str, Any], name: str = "ConvSprintGeneratorLite"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=2,
            conv_kernel=7,
            local_scale=0.18,
            context_scale=0.08,
            pointer_scale=0.12,
            anti_repeat_scale=0.055,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("ConvSprintGeneratorLite")


class ConvSprintGeneratorWide(_ConvSprintGeneratorBase):
    """Wider non-quadratic generator for quality experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "ConvSprintGeneratorWide"):
        super().__init__(
            config,
            name,
            dim=112,
            layers=2,
            conv_kernel=9,
            local_scale=0.18,
            context_scale=0.08,
            pointer_scale=0.12,
            anti_repeat_scale=0.055,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("ConvSprintGeneratorWide")
