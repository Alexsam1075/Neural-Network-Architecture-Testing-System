from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .future_architectures import CausalDepthwiseConv1d, GatedMLP
from .positional import DynamicSinusoidalPositionEncoding


class SuperKvasBlock(nn.Module):
    """Cycle-free causal mixer for fast next-token modeling.

    Formula:
        n_t = LN(h_t)
        d_t = n_t - n_{t-1}
        u_t = mean_{i<=t}(n_i)
        c_t = sum_j softmax(a)_j ConvCausal_j(n)_t
        g_t = sigmoid(W_g[n_t, u_t, d_t])
        m_t = g_t * c_t + (1 - g_t) * u_t + W_d d_t
        h'_t = h_t + W_m m_t + FFN(LN(h_t))
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv3 = CausalDepthwiseConv1d(dim, kernel_size=3)
        self.conv7 = CausalDepthwiseConv1d(dim, kernel_size=7)
        self.conv_dilated = CausalDepthwiseConv1d(dim, kernel_size=5, dilation=2)
        self.branch_logits = nn.Parameter(torch.tensor([0.2, 0.0, -0.1]))
        self.local_mix = nn.Linear(dim, dim, bias=False)
        self.delta_mix = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 3, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = GatedMLP(dim, dim * 2, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        prev = torch.cat([n[:, :1, :], n[:, :-1, :]], dim=1)
        delta = n - prev

        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix_mean = n.cumsum(dim=1) / steps

        weights = torch.softmax(self.branch_logits.to(dtype=n.dtype), dim=0)
        local = (
            weights[0] * self.conv3(n)
            + weights[1] * self.conv7(n)
            + weights[2] * self.conv_dilated(n)
        )
        local = self.local_mix(local)

        gate = torch.sigmoid(self.gate(torch.cat([n, prefix_mean, delta], dim=-1)))
        mixed = gate * local + (1.0 - gate) * prefix_mean + self.delta_mix(delta)
        h = h + self.dropout(self.out(mixed))
        h = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return h


class SuperKvas(BaseArchitecture):
    """Fast general next-token architecture with local, prefix and transition paths."""

    def __init__(self, config: Dict[str, Any], name: str = "SuperKvas"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", 96)
        self.num_layers = config.get("num_layers", 2)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", 0.04)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList([SuperKvasBlock(self.d_model, self.dropout) for _ in range(self.num_layers)])
        self.output_norm = nn.LayerNorm(self.d_model)

        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_logit_scale = nn.Parameter(torch.tensor(0.25))
        self.transition_scale = nn.Parameter(torch.tensor(0.35))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        pos_h = self.position(batch_size, length, x.device, token_h.dtype)
        h = self.dropout_layer(self.input_norm(token_h + pos_h))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        semantic_logits = self.semantic_head(h)
        tied_logits = F.linear(h, self.token_embedding.weight)
        transition_logits = self.transition_head(token_h)
        return semantic_logits + self.tied_logit_scale * tied_logits + self.transition_scale * transition_logits

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": "SuperKvas",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "complexity": "O(sequence * d_model * kernels), no quadratic attention",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "d_t=LN(h_t)-LN(h_{t-1}); u_t=mean_{i<=t}LN(h_i); "
                "c_t=sum_j softmax(a)_j ConvCausal_j(LN(h))_t; "
                "m_t=sigmoid(W_g[n_t,u_t,d_t])*c_t+(1-g_t)*u_t+W_d d_t; "
                "logits=W_s LN(h_t)+alpha LN(h_t)E^T+beta W_tr E[x_t]"
            ),
        }
