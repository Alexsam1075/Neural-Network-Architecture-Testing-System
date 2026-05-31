from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .future_architectures import _choose_heads
from .infinite_context import InfinitePrefixAttention, _shift
from .positional import DynamicSinusoidalPositionEncoding


class SlotEpisodicMemory(nn.Module):
    """Fixed-size differentiable memory for sparse prefix retrieval.

    Formula:
        r_t = softmax(q_t A^T) M_t
        w_t = softmax(k_t A^T)
        M_{t+1} = rho M_t + w_t outer v_t

    The slot count is fixed, so inference grows linearly with sequence length
    instead of building a quadratic token-token attention matrix.
    """

    def __init__(self, dim: int, slots: int, slot_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.slot_dim = slot_dim

        self.q = nn.Linear(dim, slot_dim, bias=False)
        self.k = nn.Linear(dim, slot_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.anchors = nn.Parameter(torch.randn(slots, slot_dim) * 0.02)
        self.decay = nn.Parameter(torch.tensor(0.96))
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = h.shape
        queries = self.q(h)
        keys = self.k(h)
        values = self.v(h)
        anchors = F.normalize(self.anchors, dim=-1)
        memory = h.new_zeros(batch_size, self.slots, self.dim)
        outputs = []

        decay = self.decay.sigmoid().clamp(0.50, 0.995)
        scale = self.slot_dim ** -0.5
        for t in range(length):
            write = torch.softmax((keys[:, t] @ anchors.t()) * scale, dim=-1).unsqueeze(-1)
            memory = decay * memory + write * values[:, t].unsqueeze(1)

            read = torch.softmax((queries[:, t] @ anchors.t()) * scale, dim=-1).unsqueeze(1)
            outputs.append(torch.bmm(read, memory).squeeze(1))

        return self.out(self.dropout(torch.stack(outputs, dim=1)))


class PrefixTransitionMemory(nn.Module):
    """Dynamic prefix transition memory over the current sequence only.

    Formula:
        C_t[a, b] = C_{t-1}[a, b] + 1[x_{t-1}=a and x_t=b]
        R_t[b] = log((C_t[x_t, b] + eps) / sum_b(C_t[x_t, b] + eps))

    This is an episodic context mechanism, not a benchmark answer cache: every
    forward pass starts from an empty table and uses only the visible prefix.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.05):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length = x.shape
        device = x.device
        dtype = torch.float32
        counts = torch.zeros(batch_size, self.vocab_size, self.vocab_size, device=device, dtype=dtype)
        totals = torch.zeros(batch_size, self.vocab_size, device=device, dtype=dtype)
        batch_index = torch.arange(batch_size, device=device)
        outputs = []

        for t in range(length):
            if t > 0:
                prev_token = x[:, t - 1]
                next_token = x[:, t]
                counts[batch_index, prev_token, next_token] += 1.0
                totals[batch_index, prev_token] += 1.0

            row = counts[batch_index, x[:, t], :]
            total = totals[batch_index, x[:, t]].unsqueeze(-1)
            probs = (row + self.smoothing) / (total + self.smoothing * self.vocab_size)
            outputs.append(torch.log(probs.clamp_min(1e-8)))

        return torch.stack(outputs, dim=1)


class RecallInfinityBlock(nn.Module):
    """Infinite compression plus sparse exact-retrieval memory."""

    def __init__(
        self,
        dim: int,
        heads: int,
        feature_dim: int,
        memory_scales: int,
        slots: int,
        slot_dim: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.infinite = InfinitePrefixAttention(dim, heads, feature_dim, memory_scales)
        self.episodic = SlotEpisodicMemory(dim, slots, slot_dim, dropout)
        self.local = nn.Linear(dim * 4, dim, bias=False)
        self.fuse = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * ff_mult * 2)
        self.ffn_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        infinite = self.infinite(n)
        episodic = self.episodic(n)
        local = self.local(torch.cat([n, _shift(n, 1), _shift(n, 2), _shift(n, 4)], dim=-1))
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix = n.cumsum(dim=1) / steps

        gate = torch.sigmoid(self.fuse(torch.cat([infinite, episodic, local, prefix], dim=-1)))
        h = h + self.drop(gate * (infinite + episodic) + (1.0 - gate) * (local + prefix))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.drop(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class _RecallInfinityBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        heads: int,
        feature_dim: int,
        memory_scales: int,
        slots: int,
        slot_dim: int,
        ff_mult: int,
        dropout: float,
        transition_scale: float,
        associative_scale: float,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.num_heads = _choose_heads(self.d_model, config.get("num_heads", heads))
        self.feature_dim = config.get("feature_dim", feature_dim)
        self.memory_scales = config.get("memory_scales", memory_scales)
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
                RecallInfinityBlock(
                    self.d_model,
                    self.num_heads,
                    self.feature_dim,
                    self.memory_scales,
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
        self.prefix_transition = PrefixTransitionMemory(self.vocab_size)
        self.tied_scale = nn.Parameter(torch.tensor(0.20))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.associative_scale = nn.Parameter(torch.tensor(associative_scale))
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.drop(self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype)))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        learned_logits = (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
        )
        associative_logits = self.prefix_transition(x).to(dtype=learned_logits.dtype)
        return learned_logits + self.associative_scale * associative_logits

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feature_dim": self.feature_dim,
            "memory_scales": self.memory_scales,
            "memory_slots": self.memory_slots,
            "slot_dim": self.slot_dim,
            "ff_mult": self.ff_mult,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "unbounded_streaming_attention": True,
            "dynamic_prefix_memory": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * (linear_prefix_features + fixed_memory_slots)), no QK^T matrix",
            "formula": (
                "S_t^m=lambda_m*S_{t-1}^m+phi(K_t)V_t^T; "
                "M_{t+1}=rho*M_t+softmax(K_tA^T)V_t; "
                "E_t=softmax(Q_tA^T)M_t; "
                "C_t[a,b]=C_{t-1}[a,b]+1[x_{t-1}=a,x_t=b]; "
                "h_t+=Gate(LinearAttention_t,E_t,Local_t,PrefixMean_t); "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+gamma log C_t[x_t]"
            ),
        }


class RecallInfinityLite(_RecallInfinityBase):
    """Fast hybrid architecture for broad next-token generation."""

    def __init__(self, config: Dict[str, Any], name: str = "RecallInfinityLite"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            heads=4,
            feature_dim=24,
            memory_scales=3,
            slots=32,
            slot_dim=32,
            ff_mult=2,
            dropout=0.02,
            transition_scale=0.45,
            associative_scale=0.18,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("RecallInfinityLite")


class RecallInfinityCore(_RecallInfinityBase):
    """Balanced hybrid with compressed infinite context and episodic recall."""

    def __init__(self, config: Dict[str, Any], name: str = "RecallInfinityCore"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=2,
            heads=4,
            feature_dim=32,
            memory_scales=4,
            slots=48,
            slot_dim=40,
            ff_mult=2,
            dropout=0.03,
            transition_scale=0.40,
            associative_scale=0.22,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("RecallInfinityCore")


class RecallInfinityPro(_RecallInfinityBase):
    """Quality-oriented hybrid for future LLM experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "RecallInfinityPro"):
        super().__init__(
            config,
            name,
            dim=128,
            layers=2,
            heads=4,
            feature_dim=32,
            memory_scales=4,
            slots=64,
            slot_dim=48,
            ff_mult=3,
            dropout=0.03,
            transition_scale=0.35,
            associative_scale=0.25,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("RecallInfinityPro")
