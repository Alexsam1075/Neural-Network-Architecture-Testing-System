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


class CoolTrainLinearAttention(nn.Module):
    """Transformer attention rewritten as causal linear memory.

    Formula:
        Q = W_q h, K = W_k h, V = W_v h
        phi(x) = elu(x) + 1
        S_t = sum_{i<=t} phi(K_i) outer V_i
        Z_t = sum_{i<=t} phi(K_i)
        A_t = phi(Q_t) S_t / (phi(Q_t) Z_t + eps)
    """

    def __init__(self, dim: int, heads: int, feature_dim: int):
        super().__init__()
        self.dim = dim
        self.heads = _choose_heads(dim, heads)
        self.head_dim = dim // self.heads
        self.feature_dim = feature_dim
        self.q = nn.Linear(dim, self.heads * feature_dim, bias=False)
        self.k = nn.Linear(dim, self.heads * feature_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = h.shape
        q = F.elu(self.q(h)) + 1.0
        k = F.elu(self.k(h)) + 1.0
        v = self.v(h)

        q = q.view(bsz, length, self.heads, self.feature_dim).transpose(1, 2)
        k = k.view(bsz, length, self.heads, self.feature_dim).transpose(1, 2)
        v = v.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)

        kv = torch.einsum("bhlf,bhld->bhlfd", k, v).cumsum(dim=2)
        kz = k.cumsum(dim=2)
        numerator = torch.einsum("bhlf,bhlfd->bhld", q, kv)
        denominator = torch.einsum("bhlf,bhlf->bhl", q, kz).unsqueeze(-1).clamp_min(1e-6)
        out = (numerator / denominator).transpose(1, 2).contiguous().view(bsz, length, self.dim)
        return self.out(out)


class CoolTrainBlock(nn.Module):
    """Transformer-like block without quadratic attention.

    Formula:
        n_t = LN(h_t)
        a_t = LinearAttention(n)_t
        l_t = W_l[n_t, n_{t-1}, n_{t-2}]
        h'_t = h_t + W_g[n_t, a_t, l_t] * (a_t + l_t)
        h''_t = h'_t + SwiGLU(LN(h'_t))
    """

    def __init__(self, dim: int, heads: int, feature_dim: int, ff_mult: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = CoolTrainLinearAttention(dim, heads, feature_dim)
        self.local = nn.Linear(dim * 3, dim, bias=False)
        self.fuse_gate = nn.Linear(dim * 3, dim)
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * ff_mult * 2)
        self.ffn_down = nn.Linear(dim * ff_mult, dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        attn = self.attn(n)
        local = self.local(torch.cat([n, _shift(n, 1), _shift(n, 2)], dim=-1))
        gate = torch.sigmoid(self.fuse_gate(torch.cat([n, attn, local], dim=-1)))
        h = h + self.drop(gate * (attn + local))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.drop(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class _CoolTrainBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        heads: int,
        feature_dim: int,
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
        self.ff_mult = config.get("ff_mult", ff_mult)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [
                CoolTrainBlock(self.d_model, self.num_heads, self.feature_dim, self.ff_mult, self.dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.20))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.drop = nn.Dropout(self.dropout)

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
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feature_dim": self.feature_dim,
            "ff_mult": self.ff_mult,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "complexity": "O(sequence * heads * feature_dim * head_dim), no QK^T matrix",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "Q=W_qh,K=W_kh,V=W_vh; phi=elu+1; "
                "S_t=sum_{i<=t}phi(K_i)V_i^T; Z_t=sum_{i<=t}phi(K_i); "
                "A_t=phi(Q_t)S_t/(phi(Q_t)Z_t); "
                "h_t+=sigmoid(W_g[h_t,A_t,L_t])*(A_t+L_t); h_t+=SwiGLU(LN(h_t))"
            ),
        }


class CoolTrainLite(_CoolTrainBase):
    """Small CoolTrain: Transformer-like quality path with fewer parameters."""

    def __init__(self, config: Dict[str, Any], name: str = "CoolTrainLite"):
        super().__init__(
            config,
            name,
            dim=72,
            layers=2,
            heads=4,
            feature_dim=24,
            ff_mult=2,
            dropout=0.03,
            transition_scale=0.35,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("CoolTrainLite")


class CoolTrainBalanced(_CoolTrainBase):
    """Balanced CoolTrain: larger linear attention for generation quality."""

    def __init__(self, config: Dict[str, Any], name: str = "CoolTrainBalanced"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=2,
            heads=4,
            feature_dim=32,
            ff_mult=2,
            dropout=0.03,
            transition_scale=0.40,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("CoolTrainBalanced")


class CoolTrainExpress(_CoolTrainBase):
    """Fast generation-oriented CoolTrain with a stronger transition path."""

    def __init__(self, config: Dict[str, Any], name: str = "CoolTrainExpress"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            heads=4,
            feature_dim=24,
            ff_mult=2,
            dropout=0.02,
            transition_scale=0.55,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("CoolTrainExpress")
