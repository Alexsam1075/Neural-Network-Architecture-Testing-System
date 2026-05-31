from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .positional import DynamicSinusoidalPositionEncoding
from .super_kvas import SuperKvasBlock


class SuperKvasMini(BaseArchitecture):
    """Smaller SuperKvas variant for speed/quality balance.

    Formula:
        h = SuperKvasBlocks(E[x] + PE)
        logits = W_s LN(h_t) + alpha LN(h_t)E^T + beta W_tr E[x_t]
    """

    def __init__(self, config: Dict[str, Any], name: str = "SuperKvasMini"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", 80)
        self.num_layers = config.get("num_layers", 2)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", 0.03)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList([SuperKvasBlock(self.d_model, self.dropout) for _ in range(self.num_layers)])
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.24))
        self.transition_scale = nn.Parameter(torch.tensor(0.45))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.dropout_layer(self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype)))

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
            "type": "SuperKvasMini",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "complexity": "O(sequence * d_model * kernels), compact SuperKvas block",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "h=SuperKvasBlocks(E[x]+PE); "
                "logits=W_s LN(h_t)+alpha LN(h_t)E^T+beta W_tr E[x_t]"
            ),
        }
