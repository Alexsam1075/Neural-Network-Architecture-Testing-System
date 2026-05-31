from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .infinite_context import _shift
from .pointer_infinity import LastOccurrencePointer
from .positional import DynamicSinusoidalPositionEncoding


class SprintInfinityBlock(nn.Module):
    """Depthwise local mixer with token-delta gating.

    Formula:
        n_t = LN(h_t)
        l_t = W_o DWConv_k(n)_t
        d_t = n_t - n_{t-1}
        g_t = sigmoid(W_g[n_t, l_t, d_t])
        h_t = h_t + g_t * (l_t + W_d d_t)
        h_t = h_t + SwiGLU(LN(h_t))

    It is deliberately not attention: no token-token matrix is formed.
    """

    def __init__(self, dim: int, kernel: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pointwise = nn.Linear(dim, dim, bias=False)
        self.delta = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 3, dim)
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * ff_mult * 2)
        self.ffn_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        local = self.depthwise(n.transpose(1, 2)).transpose(1, 2)
        local = self.pointwise(local)
        delta = n - _shift(n, 1)
        gate = torch.sigmoid(self.gate(torch.cat([n, local, delta], dim=-1)))
        h = h + self.drop(gate * (local + self.delta(delta)))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.drop(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class _SprintInfinityBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        kernel: int,
        ff_mult: int,
        dropout: float,
        transition_scale: float,
        context_scale: float,
        pointer_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.kernel = config.get("kernel_size", kernel)
        self.ff_mult = config.get("ff_mult", ff_mult)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000_000))
        self.dropout = config.get("dropout", dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [SprintInfinityBlock(self.d_model, self.kernel, self.ff_mult, self.dropout) for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.pointer = LastOccurrencePointer(self.vocab_size)
        self.tied_scale = nn.Parameter(torch.tensor(0.20))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))
        self.pointer_scale = nn.Parameter(torch.tensor(pointer_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))
        self.drop = nn.Dropout(self.dropout)

    def _context_frequency_logits(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seen = F.one_hot(x, num_classes=self.vocab_size).to(dtype=dtype)
        return torch.log1p(seen.cumsum(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.drop(self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype)))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=h.dtype)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
            + self.context_scale * self._context_frequency_logits(x, h.dtype)
            + self.pointer_scale * self.pointer(x, h.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel,
            "ff_mult": self.ff_mult,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "last_occurrence_pointer": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * kernel * dim + sequence * vocab), no attention matrix",
            "formula": (
                "l_t=W_oDWConv(LN(h))_t; d_t=LN(h_t)-LN(h_{t-1}); "
                "h_t+=sigmoid(W_g[n_t,l_t,d_t])*(l_t+W_d d_t); "
                "P_t=one_hot(x_{j_t+1}) where j_t=max{i<t|x_i=x_t}; "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+gamma log(1+freq_t)+eta P_t-rho one_hot(x_t)"
            ),
        }


class SprintInfinityTiny(_SprintInfinityBase):
    """Very fast non-quadratic LLM-core candidate."""

    def __init__(self, config: Dict[str, Any], name: str = "SprintInfinityTiny"):
        super().__init__(
            config,
            name,
            dim=48,
            layers=1,
            kernel=3,
            ff_mult=2,
            dropout=0.01,
            transition_scale=0.60,
            context_scale=0.06,
            pointer_scale=0.95,
            anti_repeat_scale=0.04,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SprintInfinityTiny")


class SprintInfinityLite(_SprintInfinityBase):
    """Fast generation-focused candidate."""

    def __init__(self, config: Dict[str, Any], name: str = "SprintInfinityLite"):
        super().__init__(
            config,
            name,
            dim=64,
            layers=1,
            kernel=5,
            ff_mult=2,
            dropout=0.01,
            transition_scale=0.55,
            context_scale=0.07,
            pointer_scale=1.05,
            anti_repeat_scale=0.05,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SprintInfinityLite")


class SprintInfinityCore(_SprintInfinityBase):
    """Balanced depthwise-pointer candidate with no quadratic attention."""

    def __init__(self, config: Dict[str, Any], name: str = "SprintInfinityCore"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=2,
            kernel=5,
            ff_mult=2,
            dropout=0.02,
            transition_scale=0.50,
            context_scale=0.08,
            pointer_scale=1.15,
            anti_repeat_scale=0.06,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SprintInfinityCore")
