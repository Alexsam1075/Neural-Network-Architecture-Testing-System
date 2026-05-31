from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .positional import DynamicSinusoidalPositionEncoding


class RMSNorm(nn.Module):
    """LayerNorm alternative with fewer operations and no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


def _shift(x: torch.Tensor, steps: int) -> torch.Tensor:
    if steps <= 0:
        return x
    return torch.cat([x[:, :1, :].expand(-1, steps, -1), x[:, :-steps, :]], dim=1)


def hard_gate(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + 1.0, min=0.0, max=2.0) * 0.5


class SweetyCoreBlock(nn.Module):
    """Tiny causal low-rank mixer built from shift, prefix mean and delta.

    Formula:
        n_t = RMSNorm(h_t)
        p_t = mean_{i<=t}(n_i)
        d_t = n_t - n_{t-1}
        s_t = n_{t-2}
        r_t = ReLU(W_down[n_t, p_t, d_t, s_t])
        g_t = hard_gate(W_gate[n_t, p_t, d_t, s_t])
        h'_t = h_t + W_up(g_t * r_t)
    """

    def __init__(self, dim: int, rank: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.down = nn.Linear(dim * 4, rank)
        self.gate = nn.Linear(dim * 4, rank)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        n = self.norm(h)
        n1 = _shift(n, 1)
        n2 = _shift(n, 2)
        delta = n - n1
        steps = torch.arange(1, n.shape[1] + 1, device=n.device, dtype=n.dtype).view(1, -1, 1)
        prefix = n.cumsum(dim=1) / steps

        state = torch.cat([n, prefix, delta, n2], dim=-1)
        low_rank = F.relu(self.down(state))
        mixed = low_rank * hard_gate(self.gate(state))
        return h + self.dropout(self.up(mixed))


class _SweetyBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        rank: int,
        layers: int,
        dropout: float,
        transition: bool,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.rank = config.get("rank", rank)
        self.num_layers = config.get("num_layers", layers)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", dropout)
        self.use_transition = config.get("use_transition", transition)
        self.use_semantic_head = config.get("use_semantic_head", False)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = RMSNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [SweetyCoreBlock(self.d_model, self.rank, self.dropout) for _ in range(self.num_layers)]
        )
        self.output_norm = RMSNorm(self.d_model)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size) if self.use_semantic_head else None
        self.semantic_scale = nn.Parameter(torch.tensor(0.50)) if self.use_semantic_head else None

        if self.use_transition:
            self.transition_down = nn.Linear(self.d_model, self.rank, bias=False)
            self.transition_up = nn.Linear(self.rank, self.vocab_size, bias=False)
            self.transition_scale = nn.Parameter(torch.tensor(0.35))
        else:
            self.transition_down = None
            self.transition_up = None
            self.transition_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype))

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        logits = F.linear(h, self.token_embedding.weight, self.output_bias)
        if self.use_semantic_head:
            logits = logits + self.semantic_scale * self.semantic_head(h)
        if self.use_transition:
            transition = self.transition_up(F.relu(self.transition_down(token_h)))
            logits = logits + self.transition_scale * transition
        return logits

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "rank": self.rank,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "use_transition": self.use_transition,
            "use_semantic_head": self.use_semantic_head,
            "microcontroller_target": "ESP32-class friendly operations after int8 export",
            "complexity": "O(sequence * d_model * rank), no attention, no Conv1d, no recurrent loop",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "n_t=RMSNorm(h_t); p_t=mean_{i<=t}n_i; d_t=n_t-n_{t-1}; "
                "r_t=ReLU(W_down[n_t,p_t,d_t,n_{t-2}]); "
                "g_t=clamp(W_gate[n_t,p_t,d_t,n_{t-2}]+1,0,2)/2; "
                "h_t+=W_up(g_t*r_t); logits=h_tE^T+b(+ gamma W_s h_t)(+ beta W2 ReLU(W1 E[x_t]))"
            ),
        }


class SweetyCoreMicro(_SweetyBase):
    """Extreme tiny variant for cheapest-device experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "SweetyCoreMicro"):
        super().__init__(config, name, dim=24, rank=16, layers=1, dropout=0.0, transition=False)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SweetyCoreMicro")


class SweetyCoreTiny(_SweetyBase):
    """Small transition-aware variant still intended for int8 deployment."""

    def __init__(self, config: Dict[str, Any], name: str = "SweetyCoreTiny"):
        super().__init__(config, name, dim=32, rank=24, layers=1, dropout=0.01, transition=True)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SweetyCoreTiny")


class SweetyCoreLite(_SweetyBase):
    """Balanced tiny model with two low-rank causal blocks."""

    def __init__(self, config: Dict[str, Any], name: str = "SweetyCoreLite"):
        super().__init__(config, name, dim=48, rank=32, layers=2, dropout=0.02, transition=True)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SweetyCoreLite")


class SweetyCoreSmart(_SweetyBase):
    """Small generalization-focused SweetyCore with an extra semantic head."""

    def __init__(self, config: Dict[str, Any], name: str = "SweetyCoreSmart"):
        super().__init__(config, name, dim=64, rank=48, layers=2, dropout=0.02, transition=True)
        self.use_semantic_head = True
        if self.semantic_head is None:
            self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
            self.semantic_scale = nn.Parameter(torch.tensor(0.50))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SweetyCoreSmart")


class SweetyCoreStream(BaseArchitecture):
    """Streaming recurrent SweetyCore for microcontroller-style inference.

    Formula:
        e_t = RMSNorm(E[x_t] + PE_t)
        z_t = sigmoid(W_z[e_t, s_{t-1}])
        r_t = sigmoid(W_r[e_t, s_{t-1}])
        c_t = tanh(W_c[e_t, r_t * s_{t-1}])
        s_t = (1-z_t) * s_{t-1} + z_t * c_t
        logits_t = W_s RMSNorm(s_t) + alpha RMSNorm(s_t)E^T
    """

    def __init__(self, config: Dict[str, Any], name: str = "SweetyCoreStream"):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", 48)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000))
        self.dropout = config.get("dropout", 0.0)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = RMSNorm(self.d_model)
        self.z_gate = nn.Linear(self.d_model * 2, self.d_model)
        self.r_gate = nn.Linear(self.d_model * 2, self.d_model)
        self.candidate = nn.Linear(self.d_model * 2, self.d_model)
        self.output_norm = RMSNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.tied_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype))

        state = h.new_zeros(batch_size, self.d_model)
        states = []
        for t in range(length):
            inp = h[:, t, :]
            joined = torch.cat([inp, state], dim=-1)
            z = torch.sigmoid(self.z_gate(joined))
            r = torch.sigmoid(self.r_gate(joined))
            cand = torch.tanh(self.candidate(torch.cat([inp, r * state], dim=-1)))
            state = (1.0 - z) * state + z * cand
            states.append(state)

        out = self.output_norm(torch.stack(states, dim=1))
        return self.semantic_head(out) + self.tied_scale * F.linear(out, self.token_embedding.weight)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": "SweetyCoreStream",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "dropout": self.dropout,
            "streaming_state_dim": self.d_model,
            "microcontroller_target": "ESP32-class streaming inference; stores only one state vector",
            "complexity": "O(sequence * d_model^2), no attention and constant activation memory",
            "handcrafted_solver": False,
            "cached_answers": False,
            "formula": (
                "z_t=sigmoid(W_z[e_t,s_{t-1}]); r_t=sigmoid(W_r[e_t,s_{t-1}]); "
                "c_t=tanh(W_c[e_t,r_t*s_{t-1}]); s_t=(1-z_t)*s_{t-1}+z_t*c_t; "
                "logits=W_s RMSNorm(s_t)+alpha RMSNorm(s_t)E^T"
            ),
        }
