from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .positional import DynamicSinusoidalPositionEncoding


def causal_shift(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    return torch.cat([x[:, :1, :].expand(-1, steps, -1), x[:, :-steps, :]], dim=1)


class KvasShiftBlock(nn.Module):
    """Dense causal shift mixer: local n-gram state without attention or Conv1d.

    Formula:
        n_t = LN(h_t)
        p_t = mean_{i<=t}(n_i)
        d_t = n_t - n_{t-1}
        l_t = W_l[n_t, n_{t-1}, n_{t-2}, p_t, d_t]
        h'_t = h_t + W_o(SiLU(l_t) * sigmoid(W_g l_t)) + FFN(h_t)
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.local = nn.Linear(dim * 5, dim * 2)
        self.gate = nn.Linear(dim * 2, dim * 2)
        self.out = nn.Linear(dim * 2, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_up = nn.Linear(dim, dim * 2)
        self.ffn_down = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        n1 = causal_shift(n, 1)
        n2 = causal_shift(n, 2)
        delta = n - n1
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix = n.cumsum(dim=1) / steps

        local = self.local(torch.cat([n, n1, n2, prefix, delta], dim=-1))
        mixed = F.silu(local) * torch.sigmoid(self.gate(local))
        h = h + self.dropout(self.out(mixed))

        value, gate = self.ffn_up(self.ffn_norm(h)).chunk(2, dim=-1)
        h = h + self.dropout(self.ffn_down(value * F.silu(gate)))
        return h


class KvasShift(BaseArchitecture):
    """Causal shift-prefix architecture for practical next-token research."""

    def __init__(self, config: Dict[str, Any], name: str = "KvasShift"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", 96)
        self.num_layers = config.get("num_layers", 2)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", 0.03)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList([KvasShiftBlock(self.d_model, self.dropout) for _ in range(self.num_layers)])
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.20))
        self.transition_scale = nn.Parameter(torch.tensor(0.50))

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
            "type": "KvasShift",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "complexity": "O(sequence * d_model^2), no attention, no Conv1d, no recurrent loop",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "p_t=mean_{i<=t}LN(h_i); d_t=LN(h_t)-LN(h_{t-1}); "
                "l_t=W_l[LN(h_t),LN(h_{t-1}),LN(h_{t-2}),p_t,d_t]; "
                "h_t+=W_o(SiLU(l_t)*sigmoid(W_g l_t))+FFN(h_t); "
                "logits=W_s h_t+alpha h_t E^T+beta W_tr E[x_t]"
            ),
        }
