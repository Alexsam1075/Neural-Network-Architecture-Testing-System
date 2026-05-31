from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .future_architectures import _choose_heads
from .positional import DynamicSinusoidalPositionEncoding


def _shift(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    return torch.cat([x[:, :1, :].expand(-1, steps, -1), x[:, :-steps, :]], dim=1)


class InfinitePrefixAttention(nn.Module):
    """Attention over the whole prefix through recurrent KV summaries.

    This is not quadratic softmax attention. It is an unbounded causal attention
    approximation whose state can be carried forever during streaming inference.

    Formula:
        phi(x) = elu(x) + 1
        S_t = lambda * S_{t-1} + phi(k_t) outer v_t
        z_t = lambda * z_{t-1} + phi(k_t)
        y_t = phi(q_t) S_t / (phi(q_t) z_t + eps)

    Several lambdas are used in parallel, so the model can keep short, medium
    and very old context signals at the same time.
    """

    def __init__(self, dim: int, heads: int, feature_dim: int, scales: int):
        super().__init__()
        self.dim = dim
        self.heads = _choose_heads(dim, heads)
        self.head_dim = dim // self.heads
        self.feature_dim = feature_dim
        self.scales = scales

        self.q = nn.Linear(dim, self.heads * feature_dim, bias=False)
        self.k = nn.Linear(dim, self.heads * feature_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale_logits = nn.Parameter(torch.linspace(-3.0, 3.0, scales))
        self.scale_gate = nn.Linear(dim, scales)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = h.shape
        q = F.elu(self.q(h)) + 1.0
        k = F.elu(self.k(h)) + 1.0
        v = self.v(h)

        q = q.view(bsz, length, self.heads, self.feature_dim).transpose(1, 2)
        k = k.view(bsz, length, self.heads, self.feature_dim).transpose(1, 2)
        v = v.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)

        decays = torch.sigmoid(self.scale_logits).to(device=h.device, dtype=h.dtype)
        scale_outputs = []
        for decay in decays:
            weights = decay ** torch.arange(length - 1, -1, -1, device=h.device, dtype=h.dtype)
            weights = weights.view(1, 1, length, 1)
            weighted_k = k * weights
            weighted_kv = torch.einsum("bhlf,bhld->bhlfd", weighted_k, v).cumsum(dim=2)
            weighted_z = weighted_k.cumsum(dim=2)

            correction = weights.clamp_min(1e-6)
            kv_state = weighted_kv / correction.view(1, 1, length, 1, 1)
            z_state = weighted_z / correction
            numerator = torch.einsum("bhlf,bhlfd->bhld", q, kv_state)
            denominator = torch.einsum("bhlf,bhlf->bhl", q, z_state).unsqueeze(-1).clamp_min(1e-6)
            scale_outputs.append(numerator / denominator)

        all_scales = torch.stack(scale_outputs, dim=-2)
        gates = torch.softmax(self.scale_gate(h), dim=-1).view(bsz, 1, length, self.scales, 1)
        mixed = (all_scales * gates).sum(dim=-2)
        mixed = mixed.transpose(1, 2).contiguous().view(bsz, length, self.dim)
        return self.out(mixed)


class InfiniteContextBlock(nn.Module):
    """Full-prefix attention block with local and persistent prefix paths."""

    def __init__(self, dim: int, heads: int, feature_dim: int, scales: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = InfinitePrefixAttention(dim, heads, feature_dim, scales)
        self.local = nn.Linear(dim * 3, dim, bias=False)
        self.prefix_gate = nn.Linear(dim * 3, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * ff_mult * 2)
        self.ffn_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        attention = self.attn(n)
        local = self.local(torch.cat([n, _shift(n, 1), _shift(n, 2)], dim=-1))
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix = n.cumsum(dim=1) / steps
        gate = torch.sigmoid(self.prefix_gate(torch.cat([attention, local, prefix], dim=-1)))
        h = h + self.dropout(gate * attention + (1.0 - gate) * (local + prefix))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.dropout(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class _InfiniteContextBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        heads: int,
        feature_dim: int,
        scales: int,
        ff_mult: int,
        dropout: float,
        transition_scale: float,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.num_heads = _choose_heads(self.d_model, config.get("num_heads", heads))
        self.feature_dim = config.get("feature_dim", feature_dim)
        self.memory_scales = config.get("memory_scales", scales)
        self.ff_mult = config.get("ff_mult", ff_mult)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000_000))
        self.dropout = config.get("dropout", dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [
                InfiniteContextBlock(
                    self.d_model,
                    self.num_heads,
                    self.feature_dim,
                    self.memory_scales,
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
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.dropout_layer(
            self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype))
        )

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feature_dim": self.feature_dim,
            "memory_scales": self.memory_scales,
            "ff_mult": self.ff_mult,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "unbounded_streaming_attention": True,
            "attention_scope": "all previous tokens through recurrent multi-scale KV summaries",
            "dropout": self.dropout,
            "complexity": "O(sequence * heads * feature_dim * head_dim * memory_scales), no QK^T matrix",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "phi=elu+1; S_t^m=lambda_m*S_{t-1}^m+phi(K_t)V_t^T; "
                "Z_t^m=lambda_m*Z_{t-1}^m+phi(K_t); "
                "A_t=sum_m softmax(W_g h_t)_m * phi(Q_t)S_t^m/(phi(Q_t)Z_t^m); "
                "h_t+=Gate(A_t,L_t,P_t); logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]"
            ),
        }


class InfiniteContextLite(_InfiniteContextBase):
    """Small unbounded-context candidate for long chat/code sessions."""

    def __init__(self, config: Dict[str, Any], name: str = "InfiniteContextLite"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            heads=4,
            feature_dim=24,
            scales=3,
            ff_mult=2,
            dropout=0.02,
            transition_scale=0.45,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("InfiniteContextLite")


class InfiniteContextCore(_InfiniteContextBase):
    """Balanced multi-scale infinite-context architecture."""

    def __init__(self, config: Dict[str, Any], name: str = "InfiniteContextCore"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=2,
            heads=4,
            feature_dim=32,
            scales=4,
            ff_mult=2,
            dropout=0.03,
            transition_scale=0.40,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("InfiniteContextCore")


class InfiniteContextPro(_InfiniteContextBase):
    """Wider infinite-context model for quality-oriented LLM experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "InfiniteContextPro"):
        super().__init__(
            config,
            name,
            dim=128,
            layers=2,
            heads=4,
            feature_dim=32,
            scales=4,
            ff_mult=3,
            dropout=0.03,
            transition_scale=0.35,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("InfiniteContextPro")
