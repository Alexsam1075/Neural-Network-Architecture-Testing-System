from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .flash_infinity import LowRankPrefixAttention
from .infinite_context import _shift
from .pointer_infinity import LastOccurrencePointer
from .positional import DynamicSinusoidalPositionEncoding


class LinearSprintBlock(nn.Module):
    """Linear prefix memory fused with local shift features.

    Formula:
        p_t = phi(Q_t) S_t / (phi(Q_t) Z_t)
        S_t = S_{t-1} + phi(K_t) V_t^T
        l_t = W_l[n_t, n_{t-1}, n_{t-2}, n_{t-4}]
        h_t = h_t + sigmoid(W_g[p_t, l_t, m_t]) * p_t
                    + (1 - sigmoid(...)) * (l_t + m_t)

    It keeps the useful long prefix path but removes slot memory and every
    quadratic token-token attention matrix.
    """

    def __init__(self, dim: int, rank: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.prefix = LowRankPrefixAttention(dim, rank)
        self.local = nn.Linear(dim * 4, dim, bias=False)
        self.fuse = nn.Linear(dim * 3, dim)
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * ff_mult * 2)
        self.ffn_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        prefix = self.prefix(n)
        local = self.local(torch.cat([n, _shift(n, 1), _shift(n, 2), _shift(n, 4)], dim=-1))
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        mean = n.cumsum(dim=1) / steps
        gate = torch.sigmoid(self.fuse(torch.cat([prefix, local, mean], dim=-1)))
        h = h + self.drop(gate * prefix + (1.0 - gate) * (local + mean))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.drop(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class _LinearSprintBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        rank: int,
        ff_mult: int,
        dropout: float,
        local_rank: int,
        transition_scale: float,
        tied_scale: float,
        local_scale: float,
        context_scale: float,
        pointer_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.rank = config.get("rank", rank)
        self.ff_mult = config.get("ff_mult", ff_mult)
        self.local_rank = config.get("local_rank", local_rank)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000_000))
        self.dropout = config.get("dropout", dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [LinearSprintBlock(self.d_model, self.rank, self.ff_mult, self.dropout) for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.local_left = nn.Linear(self.d_model * 4, self.local_rank)
        self.local_right = nn.Linear(self.local_rank, self.vocab_size, bias=False)
        self.context_head = nn.Linear(self.d_model, self.vocab_size)
        self.pointer = LastOccurrencePointer(self.vocab_size)

        self.tied_scale = nn.Parameter(torch.tensor(tied_scale))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.local_scale = nn.Parameter(torch.tensor(local_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))
        self.pointer_scale = nn.Parameter(torch.tensor(pointer_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.drop(self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype)))

        for block in self.blocks:
            h = block(h)

        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta = token_h - prev_h
        steps = torch.arange(1, length + 1, device=x.device, dtype=token_h.dtype).view(1, -1, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps
        local_state = F.gelu(self.local_left(torch.cat([token_h, prev_h, delta, prefix_mean], dim=-1)))
        local_logits = self.local_right(local_state)
        context_logits = torch.tanh(self.context_head(prefix_mean))

        h = self.output_norm(h)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=h.dtype)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
            + self.local_scale * local_logits
            + self.context_scale * context_logits
            + self.pointer_scale * self.pointer(x, h.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "ff_mult": self.ff_mult,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "linear_prefix_memory": True,
            "last_occurrence_pointer": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * rank * dim + sequence * local_rank * vocab + sequence * vocab), no QK^T matrix",
            "formula": (
                "S_t=S_{t-1}+phi(K_t)V_t^T; Z_t=Z_{t-1}+phi(K_t); "
                "p_t=phi(Q_t)S_t/(phi(Q_t)Z_t); "
                "l_t=W_l[n_t,n_{t-1},n_{t-2},n_{t-4}]; "
                "h_t+=Gate(p_t,l_t,mean_t); "
                "g_t=B GELU(A[E[x_t],E[x_{t-1}],delta_t,mean(E[x_<=t])]); "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+gamma g_t+eta P_t-rho one_hot(x_t)"
            ),
        }


class LinearSprintTiny(_LinearSprintBase):
    """Small non-quadratic candidate focused on speed."""

    def __init__(self, config: Dict[str, Any], name: str = "LinearSprintTiny"):
        super().__init__(
            config,
            name,
            dim=64,
            layers=1,
            rank=12,
            ff_mult=2,
            dropout=0.01,
            local_rank=32,
            transition_scale=0.45,
            tied_scale=0.12,
            local_scale=0.14,
            context_scale=0.05,
            pointer_scale=0.10,
            anti_repeat_scale=0.045,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("LinearSprintTiny")


class LinearSprintCore(_LinearSprintBase):
    """Balanced non-quadratic candidate for LLM-style generation."""

    def __init__(self, config: Dict[str, Any], name: str = "LinearSprintCore"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            rank=16,
            ff_mult=2,
            dropout=0.02,
            local_rank=48,
            transition_scale=0.42,
            tied_scale=0.14,
            local_scale=0.16,
            context_scale=0.06,
            pointer_scale=0.11,
            anti_repeat_scale=0.05,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("LinearSprintCore")


class LinearSprintPro(_LinearSprintBase):
    """Wider non-quadratic candidate when quality matters more."""

    def __init__(self, config: Dict[str, Any], name: str = "LinearSprintPro"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=1,
            rank=20,
            ff_mult=2,
            dropout=0.02,
            local_rank=64,
            transition_scale=0.38,
            tied_scale=0.16,
            local_scale=0.17,
            context_scale=0.07,
            pointer_scale=0.12,
            anti_repeat_scale=0.055,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("LinearSprintPro")
