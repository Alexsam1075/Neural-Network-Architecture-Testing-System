from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .positional import DynamicSinusoidalPositionEncoding


def _choose_heads(dim: int, requested: int) -> int:
    heads = max(1, min(requested, dim))
    while dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution that never reads future positions."""

    def __init__(self, dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.dilation * (self.kernel_size - 1)
        y = F.pad(x.transpose(1, 2), (pad, 0))
        return self.conv(y).transpose(1, 2)


class GatedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.up = nn.Linear(dim, hidden * 2)
        self.down = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


class CausalLinearMemoryBlock(nn.Module):
    """Linear causal attention plus local convolution.

    Formula:
        phi(x) = elu(x) + 1
        S_t = sum_{i<=t} phi(k_i) outer v_i
        z_t = sum_{i<=t} phi(k_i)
        r_t = phi(q_t) S_t / (phi(q_t) z_t + eps)
        h'_t = h_t + W_o r_t + W_c Conv_causal(h)_t
    """

    def __init__(self, dim: int, heads: int, feature_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.heads = _choose_heads(dim, heads)
        self.head_dim = dim // self.heads
        self.feature_dim = feature_dim

        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, self.heads * feature_dim, bias=False)
        self.k = nn.Linear(dim, self.heads * feature_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.local = CausalDepthwiseConv1d(dim, kernel_size=5)
        self.local_mix = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = GatedMLP(dim, dim * 3, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        n = self.norm(x)
        bsz, length, _ = n.shape

        q = F.elu(self.q(n)) + 1.0
        k = F.elu(self.k(n)) + 1.0
        v = self.v(n)

        q = q.view(bsz, length, self.heads, self.feature_dim).transpose(1, 2)
        k = k.view(bsz, length, self.heads, self.feature_dim).transpose(1, 2)
        v = v.view(bsz, length, self.heads, self.head_dim).transpose(1, 2)

        kv = torch.einsum("bhlf,bhld->bhlfd", k, v).cumsum(dim=2)
        kz = k.cumsum(dim=2)
        num = torch.einsum("bhlf,bhlfd->bhld", q, kv)
        den = torch.einsum("bhlf,bhlf->bhl", q, kz).unsqueeze(-1).clamp_min(1e-6)
        memory = (num / den).transpose(1, 2).contiguous().view(bsz, length, self.dim)

        local = self.local_mix(self.local(n))
        x = residual + self.dropout(self.out(memory) + local)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class RetentionBlock(nn.Module):
    """Multi-scale recurrent retention with learned decay rates.

    Formula:
        a_m = sigmoid(theta_m)
        s_{m,t} = a_m s_{m,t-1} + (1-a_m) W_v h_t
        r_t = sum_m sigmoid(W_g h_t)_m * (W_q h_t * s_{m,t})
        h'_t = h_t + W_o r_t + FFN(h_t)
    """

    def __init__(self, dim: int, scales: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.scales = scales
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim, scales)
        self.decay_logits = nn.Parameter(torch.linspace(-2.2, 2.2, scales))
        self.out = nn.Linear(dim, dim, bias=False)
        self.local = CausalDepthwiseConv1d(dim, kernel_size=3)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = GatedMLP(dim, dim * 3, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        q = self.q(n)
        v = self.v(n)
        weights = torch.softmax(self.gate(n), dim=-1)
        decays = torch.sigmoid(self.decay_logits).to(dtype=x.dtype, device=x.device)

        state = v.new_zeros(v.shape[0], self.scales, self.dim)
        outputs = []
        for t in range(v.shape[1]):
            vt = v[:, t, :].unsqueeze(1)
            state = decays.view(1, self.scales, 1) * state + (
                1.0 - decays.view(1, self.scales, 1)
            ) * vt
            mixed = (weights[:, t, :].unsqueeze(-1) * state).sum(dim=1)
            outputs.append(q[:, t, :] * mixed)

        retention = torch.stack(outputs, dim=1)
        x = x + self.dropout(self.out(retention) + self.local(n))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class RareAwareMixerBlock(nn.Module):
    """Local-global gated mixer for stable rare-token logits.

    Formula:
        c_t = W_c Conv_causal(LN(h))_t
        g_t = sigmoid(W_g [h_t, mean_{i<=t} h_i])
        m_t = g_t * c_t + (1-g_t) * mean_{i<=t} h_i
        h'_t = h_t + W_m m_t + FFN(h_t)
    """

    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.local = CausalDepthwiseConv1d(dim, kernel, dilation=dilation)
        self.local_mix = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = GatedMLP(dim, dim * 2, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        local = self.local_mix(self.local(n))
        prefix = n.cumsum(dim=1)
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        global_mean = prefix / steps
        gate = torch.sigmoid(self.gate(torch.cat([n, global_mean], dim=-1)))
        mixed = gate * local + (1.0 - gate) * global_mean
        x = x + self.dropout(self.out(mixed))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class _FutureBase(BaseArchitecture):
    def __init__(self, config: Dict[str, Any], name: str, dim: int, layers: int, dropout: float):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", dropout)
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.dropout_layer = nn.Dropout(self.dropout)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        bsz, length = x.shape
        token_h = self.token_embedding(x)
        pos_h = self.position(bsz, length, x.device, token_h.dtype)
        return self.dropout_layer(self.input_norm(token_h + pos_h))

    def _tied_logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(self.output_norm(h), self.token_embedding.weight, self.output_bias)

    def _base_info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "handcrafted_solver": False,
            "cached_answers": False,
        }


class FutureLinearMemory(_FutureBase):
    """Fast causal linear-memory architecture intended as a Transformer alternative."""

    def __init__(self, config: Dict[str, Any], name: str = "FutureLinearMemory"):
        super().__init__(config, name, dim=96, layers=2, dropout=0.05)
        heads = config.get("num_heads", 4)
        feature_dim = config.get("feature_dim", 32)
        self.blocks = nn.ModuleList(
            [CausalLinearMemoryBlock(self.d_model, heads, feature_dim, self.dropout) for _ in range(self.num_layers)]
        )
        self.feature_dim = feature_dim
        self.num_heads = _choose_heads(self.d_model, heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._embed(x)
        for block in self.blocks:
            h = block(h)
        return self._tied_logits(h)

    def get_architecture_info(self) -> Dict[str, Any]:
        info = self._base_info("FutureLinearMemory")
        info.update(
            {
                "num_heads": self.num_heads,
                "feature_dim": self.feature_dim,
                "complexity": "O(sequence * heads * feature_dim * head_dim)",
                "formula": (
                    "phi=elu(x)+1; S_t=sum_{i<=t} phi(k_i)v_i^T; "
                    "z_t=sum_{i<=t}phi(k_i); r_t=phi(q_t)S_t/(phi(q_t)z_t); "
                    "h_t+=W_o r_t + W_c Conv_causal(h)_t"
                ),
            }
        )
        return info


class FutureRetentionNet(_FutureBase):
    """Multi-scale retention model with O(L) recurrent memory."""

    def __init__(self, config: Dict[str, Any], name: str = "FutureRetentionNet"):
        super().__init__(config, name, dim=80, layers=2, dropout=0.05)
        self.scales = config.get("retention_scales", 4)
        self.blocks = nn.ModuleList([RetentionBlock(self.d_model, self.scales, self.dropout) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._embed(x)
        for block in self.blocks:
            h = block(h)
        return self._tied_logits(h)

    def get_architecture_info(self) -> Dict[str, Any]:
        info = self._base_info("FutureRetentionNet")
        info.update(
            {
                "retention_scales": self.scales,
                "complexity": "O(sequence * scales * d_model)",
                "formula": (
                    "s_{m,t}=a_m s_{m,t-1}+(1-a_m)W_v h_t; "
                    "r_t=sum_m softmax(W_g h_t)_m * (W_q h_t * s_{m,t}); "
                    "h_t+=W_o r_t + FFN(h_t)"
                ),
            }
        )
        return info


class FutureRareTokenMixer(_FutureBase):
    """Tied-embedding local/global mixer designed to preserve rare-token evidence."""

    def __init__(self, config: Dict[str, Any], name: str = "FutureRareTokenMixer"):
        super().__init__(config, name, dim=88, layers=3, dropout=0.05)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                RareAwareMixerBlock(self.d_model, kernel=5, dilation=dilations[i % len(dilations)], dropout=self.dropout)
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._embed(x)
        for block in self.blocks:
            h = block(h)
        return self._tied_logits(h)

    def get_architecture_info(self) -> Dict[str, Any]:
        info = self._base_info("FutureRareTokenMixer")
        info.update(
            {
                "weight_tying": "token_embedding == output_projection",
                "complexity": "O(sequence * d_model * kernel)",
                "formula": (
                    "c_t=W_c Conv_causal(LN(h))_t; u_t=mean_{i<=t}h_i; "
                    "g_t=sigmoid(W_g[h_t,u_t]); h_t+=W_m(g_t*c_t+(1-g_t)*u_t); "
                    "logits_t=h_t E^T+b"
                ),
            }
        )
        return info
