from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .future_architectures import CausalDepthwiseConv1d
from .positional import DynamicSinusoidalPositionEncoding


class SuperKvasLiteBlock(nn.Module):
    """Single-kernel version of SuperKvas for better speed/quality balance.

    Formula:
        n_t = LN(h_t)
        p_t = mean_{i<=t}(n_i)
        d_t = n_t - n_{t-1}
        c_t = ConvCausal_5(n)_t
        g_t = sigmoid(W_g[n_t, p_t, d_t, c_t])
        h'_t = h_t + W_o(g_t*c_t + (1-g_t)*(p_t+d_t))
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.local = CausalDepthwiseConv1d(dim, kernel_size=5)
        self.local_mix = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 4, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * 2)
        self.ffn_down = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        prev = torch.cat([n[:, :1, :], n[:, :-1, :]], dim=1)
        delta = n - prev
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix = n.cumsum(dim=1) / steps
        local = self.local_mix(self.local(n))

        gate = torch.sigmoid(self.gate(torch.cat([n, prefix, delta, local], dim=-1)))
        mixed = gate * local + (1.0 - gate) * (prefix + delta)
        h = h + self.dropout(self.out(mixed))

        value, ffn_gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.dropout(self.ffn_down(value * F.silu(ffn_gate)))
        return h


class SuperKvasLite(BaseArchitecture):
    """Compact causal local-prefix architecture for practical next-token experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "SuperKvasLite"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", 80)
        self.num_layers = config.get("num_layers", 2)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", 0.03)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList([SuperKvasLiteBlock(self.d_model, self.dropout) for _ in range(self.num_layers)])
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.20))
        self.transition_scale = nn.Parameter(torch.tensor(0.55))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": "SuperKvasLite",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "complexity": "O(sequence * d_model * kernel), no attention, no recurrent loop",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "p_t=mean_{i<=t}LN(h_i); d_t=LN(h_t)-LN(h_{t-1}); c_t=ConvCausal_5(LN(h))_t; "
                "g_t=sigmoid(W_g[LN(h_t),p_t,d_t,c_t]); h_t+=W_o(g_t*c_t+(1-g_t)*(p_t+d_t)); "
                "logits=W_s h_t+alpha h_t E^T+beta W_tr E[x_t]"
            ),
        }
