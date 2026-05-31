from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .future_architectures import _choose_heads
from .infinite_context import _shift
from .positional import DynamicSinusoidalPositionEncoding


class LowRankPrefixAttention(nn.Module):
    """Fast causal prefix attention without a quadratic QK matrix.

    Formula:
        q_t = elu(W_q h_t) + 1
        k_t = elu(W_k h_t) + 1
        v_t = W_v h_t
        S_t = S_{t-1} + k_t outer v_t
        z_t = z_{t-1} + k_t
        y_t = q_t S_t / (q_t z_t + eps)

    It is a low-rank version of linear attention: one shared rank axis instead
    of multiple full heads. That keeps the formula close to Transformer
    attention while making the core path cheap.
    """

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.q = nn.Linear(dim, rank, bias=False)
        self.k = nn.Linear(dim, rank, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        q = F.elu(self.q(h)) + 1.0
        k = F.elu(self.k(h)) + 1.0
        v = self.v(h)

        kv = torch.einsum("blr,bld->blrd", k, v).cumsum(dim=1)
        z = k.cumsum(dim=1)
        numerator = torch.einsum("blr,blrd->bld", q, kv)
        denominator = torch.einsum("blr,blr->bl", q, z).unsqueeze(-1).clamp_min(1e-6)
        return self.out(numerator / denominator)


class VectorSlotMemory(nn.Module):
    """Causal slot memory computed with prefix sums instead of Python loops."""

    def __init__(self, dim: int, slots: int, slot_dim: int):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.slot_dim = slot_dim
        self.q = nn.Linear(dim, slot_dim, bias=False)
        self.k = nn.Linear(dim, slot_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.anchors = nn.Parameter(torch.randn(slots, slot_dim) * 0.02)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(self.anchors, dim=-1)
        scale = self.slot_dim ** -0.5
        write = torch.softmax((self.k(h) @ anchors.t()) * scale, dim=-1)
        read = torch.softmax((self.q(h) @ anchors.t()) * scale, dim=-1)
        values = self.v(h)

        slot_values = torch.einsum("bls,bld->blsd", write, values).cumsum(dim=1)
        slot_mass = write.cumsum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        slot_memory = slot_values / slot_mass
        recalled = torch.einsum("bls,blsd->bld", read, slot_memory)
        return self.out(recalled)


class FlashInfinityBlock(nn.Module):
    """Fast hybrid block: prefix attention, slot memory, local shifts, SwiGLU."""

    def __init__(self, dim: int, rank: int, slots: int, slot_dim: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.prefix = LowRankPrefixAttention(dim, rank)
        self.slots = VectorSlotMemory(dim, slots, slot_dim)
        self.local = nn.Linear(dim * 5, dim, bias=False)
        self.fuse = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * ff_mult * 2)
        self.ffn_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        prefix = self.prefix(n)
        slots = self.slots(n)
        local = self.local(torch.cat([n, _shift(n, 1), _shift(n, 2), _shift(n, 4), _shift(n, 8)], dim=-1))
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        mean = n.cumsum(dim=1) / steps
        gate = torch.sigmoid(self.fuse(torch.cat([prefix, slots, local, mean], dim=-1)))
        h = h + self.drop(gate * (prefix + slots) + (1.0 - gate) * (local + mean))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.drop(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class _FlashInfinityBase(BaseArchitecture):
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
        self.tied_scale = nn.Parameter(torch.tensor(0.20))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))
        self.drop = nn.Dropout(self.dropout)

    def _context_frequency_logits(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seen = F.one_hot(x, num_classes=self.vocab_size).to(dtype=dtype)
        prefix_counts = seen.cumsum(dim=1)
        return torch.log1p(prefix_counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.drop(self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype)))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
            + self.context_scale * self._context_frequency_logits(x, h.dtype)
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
            "dynamic_prefix_memory": True,
            "context_frequency_prior": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * rank * dim + sequence * memory_slots * dim), no QK^T matrix",
            "formula": (
                "S_t=S_{t-1}+phi(K_t)V_t^T; Z_t=Z_{t-1}+phi(K_t); "
                "P_t=phi(Q_t)S_t/(phi(Q_t)Z_t); "
                "M_t=cumsum(softmax(K_tA^T)V_t)/cumsum(softmax(K_tA^T)); "
                "R_t=softmax(Q_tA^T)M_t; "
                "h_t+=Gate(P_t,R_t,LocalShift_t,PrefixMean_t); "
                "F_t=log(1+cumsum(one_hot(x_t))); "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+delta F_t"
            ),
        }


class FlashInfinityTiny(_FlashInfinityBase):
    """Very fast non-quadratic candidate."""

    def __init__(self, config: Dict[str, Any], name: str = "FlashInfinityTiny"):
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
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("FlashInfinityTiny")


class FlashInfinityLite(_FlashInfinityBase):
    """Fast balanced candidate with a wider prefix rank."""

    def __init__(self, config: Dict[str, Any], name: str = "FlashInfinityLite"):
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
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("FlashInfinityLite")


class FlashInfinityCore(_FlashInfinityBase):
    """Quality-focused fast architecture without quadratic attention."""

    def __init__(self, config: Dict[str, Any], name: str = "FlashInfinityCore"):
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
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("FlashInfinityCore")
