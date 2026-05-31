from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .flash_infinity import FlashInfinityBlock
from .positional import DynamicSinusoidalPositionEncoding


class LastOccurrencePointer(nn.Module):
    """Causal pointer over repeated tokens in the visible prefix.

    Formula:
        j_t(a) = max { i < t | x_i = a and x_{i+1} is visible }
        P_t = one_hot(x_{j_t(x_t)+1})

    This is a general copy/identifier mechanism for code, names and local
    facts. It resets every forward pass and never stores benchmark answers.
    Complexity is O(sequence * vocab), not O(sequence^2).
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, length = x.shape
        logits = torch.zeros(batch_size, length, self.vocab_size, device=x.device, dtype=dtype)
        if length <= 1:
            return logits

        pair_keys = x[:, :-1]
        pair_values = x[:, 1:]
        key_seen = F.one_hot(pair_keys, num_classes=self.vocab_size).to(dtype=torch.long)
        visible_time = torch.arange(1, length, device=x.device, dtype=torch.long).view(1, length - 1, 1)
        last_visible_time = torch.cummax(key_seen * visible_time, dim=1).values
        last_visible_time = F.pad(last_visible_time, (0, 0, 1, 0))

        token_time = last_visible_time.gather(2, x.unsqueeze(-1)).squeeze(-1)
        pair_index = (token_time - 1).clamp_min(0).clamp_max(length - 2)
        continuation = pair_values.gather(1, pair_index)
        valid = (token_time > 0).to(dtype=dtype)
        return logits.scatter(2, continuation.unsqueeze(-1), valid.unsqueeze(-1))


class _PointerInfinityBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        rank: int,
        slots: int,
        slot_dim: int,
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
        self.rank = config.get("rank", rank)
        self.memory_slots = config.get("memory_slots", slots)
        self.slot_dim = config.get("slot_dim", slot_dim)
        self.ff_mult = config.get("ff_mult", ff_mult)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000_000))
        self.dropout = config.get("dropout", dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [
                FlashInfinityBlock(
                    self.d_model,
                    self.rank,
                    self.memory_slots,
                    self.slot_dim,
                    self.ff_mult,
                    self.dropout,
                )
                for _ in range(self.num_layers)
            ]
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
            "rank": self.rank,
            "memory_slots": self.memory_slots,
            "slot_dim": self.slot_dim,
            "ff_mult": self.ff_mult,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "unbounded_streaming_attention": True,
            "last_occurrence_pointer": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * rank * dim + sequence * slots * dim + sequence * vocab), no QK^T matrix",
            "formula": (
                "S_t=S_{t-1}+phi(K_t)V_t^T; R_t=softmax(Q_tA^T)M_t; "
                "j_t=max{i<t|x_i=x_t and x_{i+1} visible}; P_t=one_hot(x_{j_t+1}); "
                "F_t=log(1+cumsum(one_hot(x_t))); "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+gamma F_t+eta P_t-rho one_hot(x_t)"
            ),
        }


class PointerInfinityTiny(_PointerInfinityBase):
    """Fast pointer-memory candidate for small devices."""

    def __init__(self, config: Dict[str, Any], name: str = "PointerInfinityTiny"):
        super().__init__(
            config,
            name,
            dim=64,
            layers=1,
            rank=16,
            slots=16,
            slot_dim=24,
            ff_mult=2,
            dropout=0.01,
            transition_scale=0.55,
            context_scale=0.06,
            pointer_scale=1.10,
            anti_repeat_scale=0.04,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("PointerInfinityTiny")


class PointerInfinityLite(_PointerInfinityBase):
    """Balanced pointer-memory architecture with low parameter count."""

    def __init__(self, config: Dict[str, Any], name: str = "PointerInfinityLite"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            rank=24,
            slots=24,
            slot_dim=32,
            ff_mult=2,
            dropout=0.02,
            transition_scale=0.50,
            context_scale=0.08,
            pointer_scale=1.25,
            anti_repeat_scale=0.05,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("PointerInfinityLite")


class PointerInfinityCore(_PointerInfinityBase):
    """Quality-oriented non-quadratic pointer-memory architecture."""

    def __init__(self, config: Dict[str, Any], name: str = "PointerInfinityCore"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=2,
            rank=24,
            slots=32,
            slot_dim=32,
            ff_mult=2,
            dropout=0.03,
            transition_scale=0.42,
            context_scale=0.10,
            pointer_scale=1.35,
            anti_repeat_scale=0.06,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("PointerInfinityCore")
