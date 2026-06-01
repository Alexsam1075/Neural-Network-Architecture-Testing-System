from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .linear_sprint import LinearSprintBlock
from .pointer_infinity import LastOccurrencePointer
from .positional import DynamicSinusoidalPositionEncoding


class PrefixTransitionPointer(nn.Module):
    """Causal token-transition memory built from the visible prefix only.

    Formula:
        C_t[a, b] = C_{t-1}[a, b] + 1[x_{t-1}=a and x_t=b]
        R_t[b] = C_t[x_t, b] / sum_b C_t[x_t, b]

    This is a general in-context memory for repeated words, identifiers and
    local coding conventions. Every forward pass starts with an empty table.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, length = x.shape
        device = x.device
        counts = torch.zeros(batch_size, self.vocab_size, self.vocab_size, device=device, dtype=dtype)
        outputs = []
        batch_index = torch.arange(batch_size, device=device)

        for t in range(length):
            if t > 0:
                counts[batch_index, x[:, t - 1], x[:, t]] += 1.0
            row = counts[batch_index, x[:, t], :]
            outputs.append(row / row.sum(dim=-1, keepdim=True).clamp_min(1.0))

        return torch.stack(outputs, dim=1)


class _TransitionSprintBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        rank: int,
        ff_mult: int,
        local_rank: int,
        dropout: float,
        transition_scale: float,
        tied_scale: float,
        local_scale: float,
        unigram_scale: float,
        table_scale: float,
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
        self.unigram_pointer = LastOccurrencePointer(self.vocab_size)
        self.transition_pointer = PrefixTransitionPointer(self.vocab_size)

        self.tied_scale = nn.Parameter(torch.tensor(tied_scale))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.local_scale = nn.Parameter(torch.tensor(local_scale))
        self.unigram_scale = nn.Parameter(torch.tensor(unigram_scale))
        self.table_scale = nn.Parameter(torch.tensor(table_scale))
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

        h = self.output_norm(h)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=h.dtype)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
            + self.local_scale * local_logits
            + self.unigram_scale * self.unigram_pointer(x, h.dtype)
            + self.table_scale * self.transition_pointer(x, h.dtype)
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
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "linear_prefix_memory": True,
            "prefix_transition_table": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * rank * dim + sequence * vocab), plus O(batch * vocab^2) temporary prefix table",
            "formula": (
                "S_t=S_{t-1}+phi(K_t)V_t^T; p_t=phi(Q_t)S_t/(phi(Q_t)Z_t); "
                "C_t[a,b]=C_{t-1}[a,b]+1[x_{t-1}=a,x_t=b]; R_t=C_t[x_t]/sum(C_t[x_t]); "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+gamma Local_t+eta LastOccur_t+mu R_t-rho one_hot(x_t)"
            ),
        }


class TransitionSprintCore(_TransitionSprintBase):
    """Balanced transition-memory sprint architecture."""

    def __init__(self, config: Dict[str, Any], name: str = "TransitionSprintCore"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            rank=16,
            ff_mult=2,
            local_rank=48,
            dropout=0.02,
            transition_scale=0.42,
            tied_scale=0.14,
            local_scale=0.16,
            unigram_scale=0.08,
            table_scale=0.90,
            anti_repeat_scale=0.05,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("TransitionSprintCore")


class TransitionSprintPro(_TransitionSprintBase):
    """Wider transition-memory sprint architecture."""

    def __init__(self, config: Dict[str, Any], name: str = "TransitionSprintPro"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=1,
            rank=20,
            ff_mult=2,
            local_rank=64,
            dropout=0.02,
            transition_scale=0.38,
            tied_scale=0.16,
            local_scale=0.17,
            unigram_scale=0.09,
            table_scale=1.10,
            anti_repeat_scale=0.055,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("TransitionSprintPro")
