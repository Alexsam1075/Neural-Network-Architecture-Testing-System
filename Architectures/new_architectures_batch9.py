from typing import Any, Dict

import torch
import torch.nn as nn

from .honest_sequence_core import HonestSequenceCore


class LocalFormulaMixer(HonestSequenceCore):
    """Depthwise causal-style local mixer: h = h + W_mix(Conv(LN(h)))."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "LocalFormulaMixer", dim=72, layers=2, conv_kernel=7)

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "formula": "h_l = h_{l-1} + W_mix(Conv_k(LN(h_{l-1}))); logits = W_o LN(h_L)",
                "handcrafted_solver": False,
            }
        )
        return info


class GatedDeltaMixer(HonestSequenceCore):
    """Learns from token-to-token deltas without carrying hidden state across batches."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "GatedDeltaMixer", dim=96, layers=2, conv_kernel=5)
        self.delta_gate = nn.Linear(self.dim * 2, self.dim)
        self.delta_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        bsz, length = x.shape
        token_h = self.token_emb(x)
        h = token_h + self.pos_encoding(bsz, length, x.device, token_h.dtype)

        prev = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)
        delta = h - prev
        gate = torch.sigmoid(self.delta_gate(torch.cat([h, delta], dim=-1)))
        h = h + gate * self.delta_proj(delta)

        for block in self.blocks:
            n = block["norm"](h)
            c = block["conv"](n.transpose(1, 2)).transpose(1, 2)
            h = h + block["mix"](c)
            h = h + block["ffn"](h)

        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "formula": "delta_t = h_t - h_{t-1}; g_t = sigmoid(W_g[h_t, delta_t]); h_t += g_t * W_d delta_t",
                "handcrafted_solver": False,
            }
        )
        return info


class LinearAssociativeMemory(HonestSequenceCore):
    """Linear attention-style association: softmax(QK^T/sqrt(d))V with learned projections."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "LinearAssociativeMemory", dim=96, layers=1, use_attention=True)

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "formula": "Q=W_q h, K=W_k h, V=W_v h; A=softmax(QK^T/sqrt(d)); h += A V",
                "handcrafted_solver": False,
            }
        )
        return info


class FactorizedTransitionMixer(HonestSequenceCore):
    """Low-rank transition model: h = h + (hA)B."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "FactorizedTransitionMixer", dim=112, layers=2, conv_kernel=5)
        rank = config.get("transition_rank", 32)
        self.transition_rank = rank
        self.left = nn.Linear(self.dim, rank, bias=False)
        self.right = nn.Linear(rank, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_logits = super().forward(x)
        x = x.clamp(0, self.vocab_size - 1)
        bsz, length = x.shape
        token_h = self.token_emb(x)
        h = token_h + self.pos_encoding(bsz, length, x.device, token_h.dtype)
        transition = self.right(torch.relu(self.left(h)))
        return h_logits + self.head(self.norm(transition))

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "transition_rank": self.transition_rank,
                "formula": "r_t = ReLU(A h_t), transition_t = B r_t, logits += W_o LN(transition_t)",
                "handcrafted_solver": False,
            }
        )
        return info


class RecurrentFormulaCore(HonestSequenceCore):
    """GRU variant with explicit recurrence: s_t = GRU(h_t, s_{t-1})."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            config,
            "RecurrentFormulaCore",
            dim=80,
            layers=1,
            use_recurrent=True,
            conv_kernel=3,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "formula": "s_t = GRU(LN(h_t), s_{t-1}); h_t = h_t + s_t; logits_t = W_o LN(h_t)",
                "handcrafted_solver": False,
            }
        )
        return info
