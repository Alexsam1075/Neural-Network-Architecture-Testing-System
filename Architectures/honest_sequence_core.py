from typing import Any, Dict

import torch
import torch.nn as nn

from .base_architecture import BaseArchitecture
from .positional import DynamicSinusoidalPositionEncoding


def _dim(config: Dict[str, Any], default: int) -> int:
    return config.get("dim", config.get("d_model", default))


class HonestSequenceCore(BaseArchitecture):
    """Small trainable sequence model without handcrafted benchmark logits."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int = 96,
        layers: int = 2,
        use_attention: bool = False,
        use_recurrent: bool = False,
        conv_kernel: int = 5,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.max_seq_len = config.get("max_seq_len", config.get("max_context_len", config.get("seq_length", 128)))
        self.max_context_len = config.get("max_context_len", self.max_seq_len)
        self.dim = _dim(config, dim)
        self.num_layers = config.get("num_layers", layers)
        self.use_attention = use_attention
        self.use_recurrent = use_recurrent

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_encoding = DynamicSinusoidalPositionEncoding(self.dim)
        self.blocks = nn.ModuleList()
        conv_groups = min(16, self.dim)
        while self.dim % conv_groups != 0:
            conv_groups -= 1
        for _ in range(self.num_layers):
            block = nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(self.dim),
                    "conv": nn.Conv1d(
                        self.dim,
                        self.dim,
                        conv_kernel,
                        padding=conv_kernel // 2,
                        groups=conv_groups,
                    ),
                    "mix": nn.Linear(self.dim, self.dim),
                    "ffn": nn.Sequential(
                        nn.LayerNorm(self.dim),
                        nn.Linear(self.dim, self.dim * 2),
                        nn.GELU(),
                        nn.Linear(self.dim * 2, self.dim),
                    ),
                }
            )
            if use_attention:
                heads = 4 if self.dim % 4 == 0 else (2 if self.dim % 2 == 0 else 1)
                block["attn"] = nn.MultiheadAttention(self.dim, heads, batch_first=True, dropout=0.0)
            self.blocks.append(block)

        if use_recurrent:
            self.rnn = nn.GRU(self.dim, self.dim, batch_first=True)

        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        bsz, length = x.shape
        token_h = self.token_emb(x)
        h = token_h + self.pos_encoding(bsz, length, x.device, token_h.dtype)

        for block in self.blocks:
            n = block["norm"](h)
            c = block["conv"](n.transpose(1, 2)).transpose(1, 2)
            h = h + block["mix"](c)
            if self.use_attention:
                n = block["norm"](h)
                attn, _ = block["attn"](n, n, n)
                h = h + attn
            h = h + block["ffn"](h)

        if self.use_recurrent:
            recurrent, _ = self.rnn(self.norm(h))
            h = h + recurrent

        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "dim": self.dim,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "use_attention": self.use_attention,
            "use_recurrent": self.use_recurrent,
            "handcrafted_logits": False,
            "hypothesis": "trainable convolutional/recurrent sequence model",
        }
