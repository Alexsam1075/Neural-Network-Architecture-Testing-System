"""
Batch 2: 10 новых архитектур на основе лучших
Вдохновлены: SpeedDemon, UltraFast, HybridMemory, AIONModular, BalancedPro, FractalNetV2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional
from .base_architecture import BaseArchitecture


# ============================================================================
# 1. HyperSpeed - SpeedDemon + HybridMemory fusion
# Цель: максимальная скорость при сохранении точности HybridMemory
# ============================================================================

class HyperSpeed(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "HyperSpeed")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 192)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        # Гибридные быстрые блоки: conv + gated linear unit
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'dw_conv': nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=self.dim),
                'pw_conv': nn.Conv1d(self.dim, self.dim * 2, 1),  # gate + value
                'norm': nn.LayerNorm(self.dim),
                'proj': nn.Linear(self.dim, self.dim),
            })
            for _ in range(4)
        ])

        # Быстрый memory slot
        self.memory = nn.Parameter(torch.randn(1, 8, self.dim) * 0.02)
        self.mem_gate = nn.Linear(self.dim, 8)

        self.head = nn.Linear(self.dim, self.vocab_size)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            res = x
            # Depthwise + gated pointwise conv
            xc = x.transpose(1, 2)
            xc = block['dw_conv'](xc)
            xc = block['pw_conv'](xc).transpose(1, 2)
            gate, val = xc.chunk(2, dim=-1)
            x = block['norm'](res + block['proj'](F.silu(gate) * val))

        # Memory attention (8 слотов - очень быстро)
        mem = self.memory.expand(B, -1, -1)
        gate = torch.sigmoid(self.mem_gate(x.mean(1)))  # B, 8
        mem_out = (gate.unsqueeze(-1) * mem).sum(1, keepdim=True)
        x = x + mem_out * 0.1

        return self.head(x)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'HyperSpeed', 'dim': self.dim, 'blocks': 4, 'memory_slots': 8}


# ============================================================================
# 2. FlashConv - Сверхбыстрые свёртки с selective state
# Цель: быстрее UltraFast, лучше точность
# ============================================================================

class FlashConv(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "FlashConv")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        # Multi-scale convolutions — разные receptive fields
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'conv3': nn.Conv1d(self.dim, self.dim // 4, 3, padding=1, groups=self.dim // 4),
                'conv5': nn.Conv1d(self.dim, self.dim // 4, 5, padding=2, groups=self.dim // 4),
                'conv7': nn.Conv1d(self.dim, self.dim // 4, 7, padding=3, groups=self.dim // 4),
                'conv1': nn.Conv1d(self.dim, self.dim // 4, 1),
                'merge': nn.Conv1d(self.dim, self.dim, 1),
                'norm': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim),
                )
            })
            for _ in range(3)
        ])

        # Selective state: учим веса для агрегации
        self.state_w = nn.Linear(self.dim, self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            res = x
            xc = x.transpose(1, 2)
            m = torch.cat([
                layer['conv3'](xc),
                layer['conv5'](xc),
                layer['conv7'](xc),
                layer['conv1'](xc),
            ], dim=1)
            m = F.gelu(layer['merge'](m)).transpose(1, 2)
            x = layer['norm'](res + m)
            x = x + layer['ffn'](x)

        # Selective state aggregation
        x = x + torch.sigmoid(self.state_w(x.mean(1, keepdim=True))) * x
        return self.head(x)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'FlashConv', 'dim': self.dim, 'scales': [3, 5, 7, 1]}


# ============================================================================
# 3. GatedMemoryNet - AIONModular + HybridMemory идеи с gating
# Цель: низкий лосс как AIONModular, скорость как HybridMemory
# ============================================================================

class GatedMemoryNet(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "GatedMemoryNet")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_slots = 16

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        # Обучаемые memory slots
        self.memory_keys = nn.Parameter(torch.randn(self.num_slots, self.dim) * 0.02)
        self.memory_vals = nn.Parameter(torch.randn(self.num_slots, self.dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(self.dim, self.dim),
                'norm1': nn.LayerNorm(self.dim),
                'norm2': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 3),
                    nn.SiLU(),
                    nn.Linear(self.dim * 3, self.dim),
                ),
                'gate': nn.Linear(self.dim * 2, self.dim),
            })
            for _ in range(4)
        ])

        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            res = x
            # Memory attention
            q = layer['q_proj'](x)  # B, L, D
            scores = torch.matmul(q, self.memory_keys.T) / math.sqrt(self.dim)  # B, L, slots
            attn = F.softmax(scores, dim=-1)
            mem_out = torch.matmul(attn, self.memory_vals)  # B, L, D
            # Gate memory + input
            gate = torch.sigmoid(layer['gate'](torch.cat([x, mem_out], dim=-1)))
            x = layer['norm1'](res + gate * mem_out)
            # FFN
            x = layer['norm2'](x + layer['ffn'](x))

        return self.head(x)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'GatedMemoryNet', 'dim': self.dim, 'memory_slots': self.num_slots}


# ============================================================================
# 4. ParallelFractal - FractalNetV2 с параллельными ветками разной глубины
# Цель: лучше FractalNetV2 за счёт адаптивной глубины
# ============================================================================

class ParallelFractal(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ParallelFractal")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        # 3 параллельных пути с разной сложностью
        self.shallow = nn.ModuleList([  # 2 слоя
            nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim * 2),
                nn.GELU(),
                nn.Linear(self.dim * 2, self.dim),
            ) for _ in range(2)
        ])

        self.medium = nn.ModuleList([  # 4 слоя conv
            nn.Sequential(
                nn.LayerNorm(self.dim),
            ) for _ in range(4)
        ])
        self.medium_conv = nn.ModuleList([
            nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=8)
            for _ in range(4)
        ])

        self.deep_attn = nn.MultiheadAttention(self.dim, 4, batch_first=True, dropout=0.0)
        self.deep_norm = nn.LayerNorm(self.dim)

        # Адаптивное взвешивание путей
        self.path_weights = nn.Parameter(torch.ones(3) / 3)

        self.norm_out = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        # Shallow path
        s = x
        for layer in self.shallow:
            s = s + layer(s)

        # Medium path
        m = x
        for layer, conv in zip(self.medium, self.medium_conv):
            m_n = layer(m)
            m = m + conv(m_n.transpose(1, 2)).transpose(1, 2)

        # Deep path
        d, _ = self.deep_attn(x, x, x)
        d = self.deep_norm(x + d)

        # Адаптивное объединение
        w = F.softmax(self.path_weights, dim=0)
        x = w[0] * s + w[1] * m + w[2] * d

        return self.head(self.norm_out(x))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'ParallelFractal', 'dim': self.dim, 'paths': ['shallow', 'medium', 'deep']}


# ============================================================================
# 5. MambaLite - упрощённая Mamba-like архитектура (selective scan)
# Цель: скорость SSM без квадратичной сложности
# ============================================================================

class SelectiveScan(nn.Module):
    def __init__(self, dim: int, d_state: int = 16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.x_proj = nn.Linear(dim, d_state * 2 + dim)
        self.A = nn.Parameter(-torch.ones(dim, d_state))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv(x_in.transpose(1, 2)).transpose(1, 2))  # B,L,D
        bdt = self.x_proj(x_conv)
        B_param = bdt[:, :, :self.d_state]          # B,L,d_state
        C = bdt[:, :, self.d_state:2*self.d_state]  # B,L,d_state
        dt = F.softplus(bdt[:, :, 2*self.d_state:]) # B,L,D
        # Simplified selective scan: h shape B,D,d_state
        A = self.A  # D, d_state
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            # dA: B,D,d_state
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # B,D,d_state
            # dB: B,D,d_state
            dB = dt[:, t, :].unsqueeze(-1) * B_param[:, t, :].unsqueeze(1)  # B,D,d_state
            h = h * dA + dB * x_conv[:, t, :].unsqueeze(-1)  # B,D,d_state
            # y: B,D
            y = (h * C[:, t, :].unsqueeze(1)).sum(-1)  # B,D
            ys.append(y)
        y = torch.stack(ys, dim=1)  # B,L,D
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLite(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MambaLite")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.d_state = 16

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'scan': SelectiveScan(self.dim, self.d_state),
                'norm': nn.LayerNorm(self.dim),
                'ffn_norm': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.dim * 2, self.dim),
                )
            })
            for _ in range(3)
        ])

        self.norm_out = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            x = x + block['scan'](block['norm'](x))
            x = x + block['ffn'](block['ffn_norm'](x))

        return self.head(self.norm_out(x))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'MambaLite', 'dim': self.dim, 'd_state': self.d_state}


# ============================================================================
# 6. TurboTransformer - Transformer с линейным attention и кэшем
# Цель: быстрее оригинального Transformer при той же точности
# ============================================================================

class LinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Linear attention: O(L) вместо O(L^2)
        q = F.elu(q) + 1  # B,H,L,d
        k = F.elu(k) + 1  # B,H,L,d
        # KV context: B,H,d,d
        kv = torch.einsum('bhld,bhlm->bhdm', k, v)
        # Query output: B,H,L,d
        qkv_out = torch.einsum('bhld,bhdm->bhlm', q, kv)
        # Normalizer: B,H,L
        k_sum = k.sum(dim=2)  # B,H,d
        denom = torch.einsum('bhld,bhd->bhl', q, k_sum).unsqueeze(-1) + 1e-6
        out = qkv_out / denom
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.proj(out)


class TurboTransformer(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "TurboTransformer")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = 4

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': LinearAttention(self.dim, 4),
                'norm1': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 3),
                    nn.GELU(),
                    nn.Linear(self.dim * 3, self.dim),
                ),
                'norm2': nn.LayerNorm(self.dim),
            })
            for _ in range(self.num_layers)
        ])

        self.norm_out = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))

        return self.head(self.norm_out(x))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'TurboTransformer', 'dim': self.dim, 'num_layers': self.num_layers, 'attention': 'linear O(L)'}


# ============================================================================
# 7. CausalMixer - быстрый mixing без attention, только MLPs и shifts
# Цель: сверхбыстрый, точный, простой
# ============================================================================

class CausalShift(nn.Module):
    """Causal temporal shift - без параметров, O(1)"""
    def __init__(self, dim: int, shift: int = 1):
        super().__init__()
        self.shift = shift
        self.dim = dim
        # Половина каналов сдвигается, половина нет
        self.half = dim // 2

    def forward(self, x):
        x1, x2 = x[:, :, :self.half], x[:, :, self.half:]
        x1_shifted = torch.roll(x1, self.shift, dims=1)
        x1_shifted[:, :self.shift, :] = 0  # Causal mask
        return torch.cat([x1_shifted, x2], dim=-1)


class CausalMixer(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "CausalMixer")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'shift': CausalShift(self.dim, shift=i + 1),
                'mix': nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim),
                ),
                'channel': nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim * 3),
                    nn.SiLU(),
                    nn.Linear(self.dim * 3, self.dim),
                )
            })
            for i in range(4)
        ])

        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            x = x + block['mix'](block['shift'](x))
            x = x + block['channel'](x)

        return self.head(self.norm(x))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'CausalMixer', 'dim': self.dim, 'mechanism': 'causal_shift+mlp'}


# ============================================================================
# 8. DeepResidual - глубокая сеть с product key memory
# Цель: самый низкий лосс, высокая точность
# ============================================================================

class ProductKeyMemory(nn.Module):
    """Learned key-value memory bank with simple attention"""
    def __init__(self, dim: int, num_keys: int = 64):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_keys, dim) * 0.02)
        self.vals = nn.Parameter(torch.randn(num_keys, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.num_keys = num_keys

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x)  # B,L,D
        scores = torch.matmul(q, self.keys.T) / math.sqrt(D)  # B,L,num_keys
        weights = F.softmax(scores, dim=-1)  # B,L,num_keys
        out = torch.matmul(weights, self.vals)  # B,L,D
        return out

class DeepResidual(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "DeepResidual")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = 6

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(self.dim, 8, batch_first=True, dropout=0.0),
                'norm1': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 4),
                    nn.GELU(),
                    nn.Linear(self.dim * 4, self.dim),
                ),
                'norm2': nn.LayerNorm(self.dim),
            })
            for _ in range(self.num_layers)
        ])

        self.pkm = ProductKeyMemory(self.dim, num_keys=64)
        self.pkm_norm = nn.LayerNorm(self.dim)

        self.norm_out = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for i, layer in enumerate(self.layers):
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            x = layer['norm2'](x + layer['ffn'](x))
            # PKM каждые 2 слоя
            if i % 2 == 1:
                x = self.pkm_norm(x + self.pkm(x))

        return self.head(self.norm_out(x))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'DeepResidual', 'dim': self.dim, 'num_layers': self.num_layers, 'pkm_keys': 64}


# ============================================================================
# 9. WaveNet2 - дилатированные причинные свёртки (быстро + точно)
# Цель: аналог WaveNet но оптимизированный для последовательностей
# ============================================================================

class WaveNet2(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "WaveNet2")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        # Дилатированные свёртки с exponentially растущей дилатацией
        dilations = [1, 2, 4, 8, 16, 1, 2, 4]
        self.convs = nn.ModuleList([
            nn.Conv1d(self.dim, self.dim * 2, kernel_size=3,
                      padding=d, dilation=d, groups=4)
            for d in dilations
        ])
        self.residual_convs = nn.ModuleList([
            nn.Conv1d(self.dim, self.dim, 1) for _ in dilations
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(self.dim, self.dim, 1) for _ in dilations
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in dilations])

        self.post = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
        )
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        skip_sum = 0
        xc = x.transpose(1, 2)

        for conv, res_conv, skip_conv, norm in zip(self.convs, self.residual_convs, self.skip_convs, self.norms):
            out = conv(xc)[:, :, :L]  # trim to L
            gate, filt = out.chunk(2, dim=1)
            out = torch.tanh(filt) * torch.sigmoid(gate)
            skip_sum = skip_sum + skip_conv(out)
            xc = xc + res_conv(out)

        x = skip_sum.transpose(1, 2)
        x = self.post(x)
        return self.head(x)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'WaveNet2', 'dim': self.dim, 'dilations': [1, 2, 4, 8, 16, 1, 2, 4]}


# ============================================================================
# 10. HybridSSMAttn - SSM + sparse attention для длинных зависимостей
# Цель: лучше SSM, быстрее чистого Transformer
# ============================================================================

class SimpleSSMBlock(nn.Module):
    """Simplified SSM using leaky integrator state"""
    def __init__(self, dim: int, d_state: int = 8):
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.decay = nn.Parameter(torch.full((dim,), 0.9))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(self.conv(x_in.transpose(1, 2)).transpose(1, 2))  # B,L,D
        decay = torch.sigmoid(self.decay)  # D
        h = torch.zeros(B, D, device=x.device)
        ys = []
        for t in range(L):
            h = decay * h + (1.0 - decay) * x_conv[:, t, :]
            ys.append(h)
        y = torch.stack(ys, dim=1)  # B,L,D
        y = y * F.silu(z)
        return self.out_proj(y)

class HybridSSMAttn(BaseArchitecture):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "HybridSSMAttn")
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)

        # Чередуем SSM и Attention блоки
        self.ssm_blocks = nn.ModuleList([SimpleSSMBlock(self.dim) for _ in range(2)])
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(self.dim, 4, batch_first=True, dropout=0.0) for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(4)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim * 3),
                nn.SiLU(),
                nn.Linear(self.dim * 3, self.dim),
            ) for _ in range(4)
        ])

        self.norm_out = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        for i in range(2):
            # SSM block
            x = x + self.ssm_blocks[i](self.norms[i*2](x))
            x = x + self.ffns[i*2](x)
            # Attention block
            attn_out, _ = self.attn_blocks[i](self.norms[i*2+1](x), self.norms[i*2+1](x), self.norms[i*2+1](x))
            x = x + attn_out
            x = x + self.ffns[i*2+1](x)

        return self.head(self.norm_out(x))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {'type': 'HybridSSMAttn', 'dim': self.dim, 'pattern': 'SSM-FFN-Attn-FFN x2'}