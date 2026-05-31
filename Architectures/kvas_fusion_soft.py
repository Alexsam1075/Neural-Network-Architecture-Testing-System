from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .kvas_fusion import KvasFusionBlock
from .positional import DynamicSinusoidalPositionEncoding


class KvasFusionSoft(BaseArchitecture):
    """KvasFusion variant with hidden-state novelty instead of logit subtraction.

    Formula:
        h_L = FusionBlocks(E[x] + PE)
        p_t = mean_{i<=t}(h_i)
        q_t = h_t + sigmoid(W_n[h_t, p_t]) * W_v(h_t - p_t)
        logits = W_s q_t + alpha q_t E^T + beta W_tr E[x_t]
    """

    def __init__(self, config: Dict[str, Any], name: str = "KvasFusionSoft"):
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
        self.novelty_gate = nn.Linear(self.d_model * 2, self.d_model)
        self.novelty_value = nn.Linear(self.d_model, self.d_model, bias=False)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.24))
        self.transition_scale = nn.Parameter(torch.tensor(0.40))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        steps = torch.arange(1, length + 1, device=x.device, dtype=h.dtype).view(1, -1, 1)
        prefix = h.cumsum(dim=1) / steps
        novelty = torch.sigmoid(self.novelty_gate(torch.cat([h, prefix], dim=-1)))
        q = h + novelty * self.novelty_value(h - prefix)

        return (
            self.semantic_head(q)
            + self.tied_scale * F.linear(q, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": "KvasFusionSoft",
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
                "h=FusionBlocks(E[x]+PE); p_t=mean_{i<=t}h_i; "
                "q_t=h_t+sigmoid(W_n[h_t,p_t])*W_v(h_t-p_t); "
                "logits=W_s q_t+alpha q_tE^T+beta W_trE[x_t]"
            ),
        }
