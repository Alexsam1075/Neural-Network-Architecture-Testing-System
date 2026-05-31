from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture


def _dim(config: Dict[str, Any], default: int = 128) -> int:
    return config.get('dim', config.get('d_model', default))


class CausalRecallHybrid(BaseArchitecture):
    """RecallMixerPro + CausalMixer ideas for strong memory and stable generation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "CausalRecallHybrid")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = _dim(config, 128)
        self.num_layers = config.get('num_layers', 3)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.first_gate = nn.Linear(self.dim * 2, self.dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.dim),
                'shift_mix': nn.Sequential(
                    nn.Linear(self.dim * 2, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim),
                ),
                'conv': nn.Conv1d(self.dim, self.dim, 5, padding=2, groups=max(1, self.dim // 16)),
                'ffn': nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.dim * 2, self.dim),
                ),
            })
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        first = h[:, :1, :].expand(-1, length, -1)
        h = h + torch.sigmoid(self.first_gate(torch.cat([h, first], dim=-1))) * first

        for layer in self.layers:
            prev = torch.roll(h, shifts=1, dims=1)
            prev[:, 0, :] = 0
            mixed = layer['shift_mix'](torch.cat([h, prev], dim=-1))
            conv = layer['conv'](layer['norm'](h).transpose(1, 2)).transpose(1, 2)
            h = h + mixed + conv
            h = h + layer['ffn'](h)
        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'CausalRecallHybrid', 'dim': self.dim, 'num_layers': self.num_layers}


class HoloCausalMemory(BaseArchitecture):
    """Holographic-style running memory without FFT, plus causal local mixing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "HoloCausalMemory")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = _dim(config, 128)
        self.num_layers = config.get('num_layers', 3)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.write = nn.Linear(self.dim, self.dim)
        self.read_gate = nn.Linear(self.dim * 2, self.dim)
        self.decay = nn.Parameter(torch.full((self.dim,), 0.85))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.dim),
                'conv': nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=self.dim),
                'proj': nn.Linear(self.dim, self.dim),
                'ffn': nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim),
                ),
            })
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def _running_memory(self, h: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = h.shape
        decay = torch.sigmoid(self.decay).view(1, 1, -1)
        state = torch.zeros(bsz, 1, self.dim, device=h.device)
        states = []
        values = torch.tanh(self.write(h))
        for t in range(length):
            state = decay * state + (1.0 - decay) * values[:, t:t + 1, :]
            states.append(state)
        return torch.cat(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        mem = self._running_memory(h)
        h = h + torch.sigmoid(self.read_gate(torch.cat([h, mem], dim=-1))) * mem

        for layer in self.layers:
            conv = F.silu(layer['conv'](layer['norm'](h).transpose(1, 2)).transpose(1, 2))
            h = h + layer['proj'](conv)
            h = h + layer['ffn'](h)
        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'HoloCausalMemory', 'dim': self.dim, 'num_layers': self.num_layers}


class AttentiveRecallSSM(BaseArchitecture):
    """Transformer/AccuracyFirst attention with RecallMixer and lightweight SSM state."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AttentiveRecallSSM")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = _dim(config, 128)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 4)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.first_gate = nn.Linear(self.dim * 2, self.dim)
        self.state_decay = nn.Parameter(torch.full((self.num_layers, self.dim), 0.8))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(self.dim, self.num_heads, batch_first=True, dropout=0.0),
                'norm2': nn.LayerNorm(self.dim),
                'state_proj': nn.Linear(self.dim, self.dim),
                'ffn': nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, int(self.dim * 2.5)),
                    nn.GELU(),
                    nn.Linear(int(self.dim * 2.5), self.dim),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        first = h[:, :1, :].expand(-1, length, -1)
        h = h + torch.sigmoid(self.first_gate(torch.cat([h, first], dim=-1))) * first

        for idx, layer in enumerate(self.layers):
            n = layer['norm1'](h)
            attn, _ = layer['attn'](n, n, n)
            h = h + attn
            state = self._scan(layer['state_proj'](layer['norm2'](h)), idx)
            h = h + state
            h = h + layer['ffn'](h)
        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'AttentiveRecallSSM', 'dim': self.dim, 'num_layers': self.num_layers}
