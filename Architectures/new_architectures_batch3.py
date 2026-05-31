from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture


def _model_dim(config: Dict[str, Any], default: int = 128) -> int:
    return config.get('dim', config.get('d_model', default))


class RecallMixerPro(BaseArchitecture):
    """Fast mixer with first-token recall path and multi-scale convolutions."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "RecallMixerPro")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = _model_dim(config, 128)
        self.num_layers = config.get('num_layers', 3)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.first_gate = nn.Linear(self.dim * 2, self.dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.dim),
                'conv3': nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=max(1, self.dim // 16)),
                'conv7': nn.Conv1d(self.dim, self.dim, 7, padding=3, groups=max(1, self.dim // 16)),
                'merge': nn.Linear(self.dim * 2, self.dim),
                'ffn': nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim),
                ),
            })
            for _ in range(self.num_layers)
        ])
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        first = h[:, :1, :].expand(-1, length, -1)
        h = h + torch.sigmoid(self.first_gate(torch.cat([h, first], dim=-1))) * first

        for layer in self.layers:
            residual = h
            n = layer['norm'](h).transpose(1, 2)
            c3 = layer['conv3'](n).transpose(1, 2)
            c7 = layer['conv7'](n).transpose(1, 2)
            h = residual + layer['merge'](torch.cat([c3, c7], dim=-1))
            h = h + layer['ffn'](h)
        return self.head(h)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'RecallMixerPro', 'dim': self.dim, 'num_layers': self.num_layers}


class FractalMemoryPro(BaseArchitecture):
    """BalancedPro-style linear attention plus compact learned memory."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "FractalMemoryPro")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = _model_dim(config, 128)
        self.num_layers = config.get('num_layers', 3)
        self.memory_slots = config.get('memory_slots', 24)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.memory = nn.Parameter(torch.randn(self.memory_slots, self.dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(self.dim),
                'q': nn.Linear(self.dim, self.dim, bias=False),
                'k': nn.Linear(self.dim, self.dim, bias=False),
                'v': nn.Linear(self.dim, self.dim, bias=False),
                'out': nn.Linear(self.dim, self.dim),
                'mem_q': nn.Linear(self.dim, self.dim),
                'gate': nn.Linear(self.dim * 2, self.dim),
                'norm2': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, int(self.dim * 2.5)),
                    nn.SiLU(),
                    nn.Linear(int(self.dim * 2.5), self.dim),
                ),
            })
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def _linear_attention(self, q, k, v):
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv = torch.einsum('bld,blm->bdm', k, v)
        denom = torch.einsum('bld,bd->bl', q, k.sum(dim=1)).unsqueeze(-1) + 1e-6
        return torch.einsum('bld,bdm->blm', q, kv) / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)
        mem = self.memory.unsqueeze(0).expand(bsz, -1, -1)

        for layer in self.layers:
            n = layer['norm1'](h)
            attn = self._linear_attention(layer['q'](n), layer['k'](n), layer['v'](n))
            mq = layer['mem_q'](n)
            mem_scores = torch.matmul(mq, mem.transpose(1, 2)) / (self.dim ** 0.5)
            mem_out = torch.matmul(F.softmax(mem_scores, dim=-1), mem)
            gate = torch.sigmoid(layer['gate'](torch.cat([attn, mem_out], dim=-1)))
            h = h + layer['out'](attn + gate * mem_out)
            h = h + layer['ffn'](layer['norm2'](h))
        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'FractalMemoryPro', 'dim': self.dim, 'memory_slots': self.memory_slots}


class StableSSMTransformer(BaseArchitecture):
    """Small SSM-conv front end with linear attention for stable OOD behavior."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "StableSSMTransformer")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = _model_dim(config, 128)
        self.num_layers = config.get('num_layers', 3)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        self.decay = nn.Parameter(torch.full((self.num_layers, self.dim), 0.8))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=self.dim),
                'norm': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(self.dim, 4, batch_first=True, dropout=0.0),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length = x.shape
        pos = torch.arange(length, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.token_emb(x) + self.pos_emb(pos)

        for idx, layer in enumerate(self.layers):
            conv = F.silu(layer['conv'](h.transpose(1, 2)).transpose(1, 2))
            decay = torch.sigmoid(self.decay[idx]).view(1, 1, -1)
            state = torch.zeros(bsz, 1, self.dim, device=x.device)
            states = []
            for t in range(length):
                state = decay * state + (1.0 - decay) * conv[:, t:t + 1, :]
                states.append(state)
            h = h + torch.cat(states, dim=1)
            n = layer['norm'](h)
            attn, _ = layer['attn'](n, n, n)
            h = h + attn
            h = h + layer['ffn'](h)
        return self.head(self.norm(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'StableSSMTransformer', 'dim': self.dim, 'num_layers': self.num_layers}
