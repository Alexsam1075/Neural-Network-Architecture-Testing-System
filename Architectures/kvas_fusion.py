from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .future_architectures import CausalDepthwiseConv1d
from .positional import DynamicSinusoidalPositionEncoding


def _shift_right(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    return torch.cat([x[:, :1, :].expand(-1, steps, -1), x[:, :-steps, :]], dim=1)


class KvasFusionBlock(nn.Module):
    """Bottleneck causal local-prefix mixer.

    Formula:
        n_t = LN(h_t)
        p_t = mean_{i<=t}(n_i)
        d_t = n_t - n_{t-1}
        l_t = ConvCausal_3(n)_t + W_s[n_{t-1}, n_{t-2}]
        b_t = SiLU(W_b[n_t, p_t, d_t, l_t])
        g_t = sigmoid(W_g[n_t, p_t, d_t, l_t])
        h'_t = h_t + W_o(g_t * b_t)
    """

    def __init__(self, dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.local_conv = CausalDepthwiseConv1d(dim, kernel_size=3)
        self.shift_mix = nn.Linear(dim * 2, dim, bias=False)
        self.bottleneck = nn.Linear(dim * 4, bottleneck)
        self.gate = nn.Linear(dim * 4, bottleneck)
        self.out = nn.Linear(bottleneck, dim, bias=False)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * 2)
        self.ffn_down = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        n1 = _shift_right(n, 1)
        n2 = _shift_right(n, 2)
        delta = n - n1
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix = n.cumsum(dim=1) / steps
        local = self.local_conv(n) + self.shift_mix(torch.cat([n1, n2], dim=-1))

        state = torch.cat([n, prefix, delta, local], dim=-1)
        fused = F.silu(self.bottleneck(state)) * torch.sigmoid(self.gate(state))
        h = h + self.dropout(self.out(fused))

        value, gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.dropout(self.ffn_down(value * F.silu(gate)))
        return h


class KvasFusion(BaseArchitecture):
    """General causal LLM candidate combining local evidence, prefix memory and transition logits."""

    def __init__(self, config: Dict[str, Any], name: str = "KvasFusion"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", 88)
        self.bottleneck = config.get("bottleneck", 128)
        self.num_layers = config.get("num_layers", 2)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", 0.03)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [KvasFusionBlock(self.d_model, self.bottleneck, self.dropout) for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.22))
        self.transition_scale = nn.Parameter(torch.tensor(0.42))
        self.contrast_scale = nn.Parameter(torch.tensor(0.04))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        steps = torch.arange(1, length + 1, device=x.device, dtype=h.dtype).view(1, -1, 1)
        prefix_h = h.cumsum(dim=1) / steps

        semantic_logits = self.semantic_head(h)
        tied_logits = F.linear(h, self.token_embedding.weight)
        transition_logits = self.transition_head(token_h)
        repeat_logits = F.linear(prefix_h, self.token_embedding.weight)
        return (
            semantic_logits
            + self.tied_scale * tied_logits
            + self.transition_scale * transition_logits
            - F.softplus(self.contrast_scale) * repeat_logits
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": "KvasFusion",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "bottleneck": self.bottleneck,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "complexity": "O(sequence * d_model * bottleneck), no quadratic attention, no recurrent loop",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "p_t=mean_{i<=t}LN(h_i); d_t=LN(h_t)-LN(h_{t-1}); "
                "l_t=ConvCausal_3(LN(h))_t+W_s[LN(h_{t-1}),LN(h_{t-2})]; "
                "b_t=SiLU(W_b[n_t,p_t,d_t,l_t])*sigmoid(W_g[n_t,p_t,d_t,l_t]); "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]-softplus(rho) mean_{i<=t}(h_i)E^T"
            ),
        }
