from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture


def _dim(config: Dict[str, Any], default: int = 128) -> int:
    return config.get("dim", config.get("d_model", default))


def _heads(dim: int, requested: int) -> int:
    for heads in (requested, 8, 4, 2, 1):
        if dim % heads == 0:
            return heads
    return 1


class _Batch5Core(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int = 128,
        layers: int = 3,
        heads: int = 4,
        memory_slots: int = 16,
        conv_kernel: int = 5,
        use_attention: bool = True,
        use_state: bool = True,
        use_memory: bool = True,
        rare_bias: bool = False,
        anti_collapse: bool = False,
    ):
        super().__init__(config, name)
        self.vocab_size = config["vocab_size"]
        self.seq_length = config.get("max_seq_len", config.get("seq_length", 128))
        self.dim = config.get("dim", config.get("d_model", dim))
        self.num_layers = config.get("num_layers", layers)
        self.num_heads = _heads(self.dim, config.get("num_heads", heads))
        self.memory_slots = config.get("memory_slots", memory_slots)
        self.use_attention = use_attention
        self.use_state = use_state
        self.use_memory = use_memory
        self.rare_bias_enabled = rare_bias
        self.anti_collapse = anti_collapse

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.first_gate = nn.Linear(self.dim * 2, self.dim)
        self.local_gate = nn.Linear(self.dim * 3, self.dim)

        if self.use_memory:
            self.memory = nn.Parameter(torch.randn(self.memory_slots, self.dim) * 0.02)
            self.memory_key = nn.Linear(self.dim, self.dim, bias=False)
            self.memory_gate = nn.Linear(self.dim * 2, self.dim)

        if self.rare_bias_enabled:
            self.rare_context = nn.Linear(self.dim, self.vocab_size)
            self.rare_temperature = nn.Parameter(torch.tensor(0.15))

        self.state_decay = nn.Parameter(torch.full((self.num_layers, self.dim), 0.78))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(self.dim),
                "conv3": nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=max(1, self.dim // 16)),
                "convk": nn.Conv1d(self.dim, self.dim, conv_kernel, padding=conv_kernel // 2, groups=max(1, self.dim // 16)),
                "mix": nn.Linear(self.dim * 3, self.dim),
                "attn": nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True, dropout=0.0),
                "state_proj": nn.Linear(self.dim, self.dim),
                "ffn": nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, int(self.dim * 2.25)),
                    nn.GELU(),
                    nn.Linear(int(self.dim * 2.25), self.dim),
                ),
            })
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def _scan(self, h: torch.Tensor, layer_idx: int) -> torch.Tensor:
        bsz, length, _ = h.shape
        decay = torch.sigmoid(self.state_decay[layer_idx]).view(1, 1, -1)
        state = torch.zeros(bsz, 1, self.dim, device=h.device)
        outputs = []
        for t in range(length):
            state = decay * state + (1.0 - decay) * h[:, t:t + 1, :]
            outputs.append(state)
        return torch.cat(outputs, dim=1)

    def _memory_read(self, h: torch.Tensor) -> torch.Tensor:
        bsz = h.size(0)
        mem = self.memory.unsqueeze(0).expand(bsz, -1, -1)
        scores = torch.matmul(self.memory_key(h), mem.transpose(1, 2)) / (self.dim ** 0.5)
        read = torch.matmul(F.softmax(scores, dim=-1), mem)
        gate = torch.sigmoid(self.memory_gate(torch.cat([h, read], dim=-1)))
        return gate * read

    def _shift_context(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prev = torch.roll(h, shifts=1, dims=1)
        prev[:, 0, :] = 0
        first = h[:, :1, :].expand(-1, h.size(1), -1)
        return prev, first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)

        prev, first = self._shift_context(h)
        h = h + torch.sigmoid(self.first_gate(torch.cat([h, first], dim=-1))) * first
        h = h + torch.sigmoid(self.local_gate(torch.cat([h, prev, first], dim=-1))) * prev

        for idx, layer in enumerate(self.layers):
            n = layer["norm"](h)
            c_in = n.transpose(1, 2)
            c3 = layer["conv3"](c_in).transpose(1, 2)
            ck = layer["convk"](c_in).transpose(1, 2)
            prev, first = self._shift_context(n)
            h = h + layer["mix"](torch.cat([c3, ck, prev], dim=-1))

            if self.use_attention:
                n = layer["norm"](h)
                attn, _ = layer["attn"](n, n, n)
                h = h + attn

            if self.use_state:
                h = h + self._scan(layer["state_proj"](layer["norm"](h)), idx)

            if self.use_memory:
                h = h + self._memory_read(layer["norm"](h))

            h = h + layer["ffn"](h)

        h = self.norm(h)
        logits = self.head(h)

        if self.rare_bias_enabled:
            context = h.mean(dim=1, keepdim=True)
            logits = logits + torch.tanh(self.rare_context(context)) * self.rare_temperature.clamp(0.0, 1.0)

        if self.anti_collapse and length > 1:
            repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).float() * 0.03
            logits = logits - repeat_penalty

        return logits

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            "type": self.name,
            "dim": self.dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "memory_slots": self.memory_slots if self.use_memory else 0,
            "features": {
                "attention": self.use_attention,
                "state_scan": self.use_state,
                "memory": self.use_memory,
                "rare_bias": self.rare_bias_enabled,
                "anti_collapse": self.anti_collapse,
            },
        }


class PrimeRecallNet(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "PrimeRecallNet", dim=128, layers=3, memory_slots=24, conv_kernel=7)


class MosaicMixer(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MosaicMixer", dim=112, layers=3, use_attention=False, memory_slots=16, conv_kernel=9)


class CausalDeltaMemory(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "CausalDeltaMemory", dim=128, layers=3, use_attention=False, use_state=True, memory_slots=20)


class RareTokenSentinel(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "RareTokenSentinel", dim=128, layers=3, memory_slots=32, rare_bias=True, conv_kernel=7)


class AntiCollapseTransformer(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AntiCollapseTransformer", dim=144, layers=3, memory_slots=24, rare_bias=True, anti_collapse=True)


class SwiftRecallConv(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "SwiftRecallConv", dim=96, layers=2, use_attention=False, use_state=False, memory_slots=12)


class HoloFractalLite(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "HoloFractalLite", dim=128, layers=4, memory_slots=16, conv_kernel=11)


class StateSpaceRecall(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "StateSpaceRecall", dim=128, layers=4, use_attention=False, use_state=True, memory_slots=28)


class OmniMemoryMixer(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "OmniMemoryMixer", dim=160, layers=3, memory_slots=32, rare_bias=True)


class RolloutResonator(_Batch5Core):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "RolloutResonator", dim=128, layers=4, memory_slots=24, conv_kernel=13, anti_collapse=True)
