"""
Batch 1: Новые архитектуры на основе лучших
Включает: EchoState V2, HybridMemory, AdaptiveDepth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


# ============================================================================
# EchoState V2 - Улучшенный резервуар с обучаемыми параметрами
# ============================================================================

class AdaptiveReservoir(nn.Module):
    """Адаптивный резервуар с обучаемой динамикой"""
    def __init__(self, input_dim: int, reservoir_dim: int, spectral_radius: float = 0.9):
        super().__init__()
        self.reservoir_dim = reservoir_dim
        
        # Обучаемые веса входа
        self.W_in = nn.Linear(input_dim, reservoir_dim)
        
        # Резервуар с обучаемым spectral radius
        self.spectral_radius = nn.Parameter(torch.tensor(spectral_radius))
        W_res = torch.randn(reservoir_dim, reservoir_dim)
        # Нормализуем спектральный радиус
        eigenvalues = torch.linalg.eigvals(W_res).abs().max()
        W_res = W_res / eigenvalues
        self.register_buffer('W_reservoir', W_res)
        
        # Обучаемый leak rate
        self.leak_rate = nn.Parameter(torch.tensor(0.3))
        
        # State
        self.register_buffer('state', torch.zeros(1, reservoir_dim))
        
    def forward(self, x):
        B, L, D = x.shape
        
        outputs = []
        state = self.state.expand(B, -1)
        
        for t in range(L):
            # Input activation
            u = self.W_in(x[:, t, :])
            
            # Reservoir update с обучаемым leak rate
            preactivation = u + (state @ (self.W_reservoir * self.spectral_radius))
            new_state = torch.tanh(preactivation)
            state = (1 - torch.sigmoid(self.leak_rate)) * state + torch.sigmoid(self.leak_rate) * new_state
            
            outputs.append(state.unsqueeze(1))
        
        # Сохраняем последнее состояние
        self.state = state.detach()[:1]
        
        return torch.cat(outputs, dim=1)


class EchoStateV2(BaseArchitecture):
    """
    EchoState V2 - улучшенная версия с обучаемыми параметрами
    
    Проблема оригинала: фиксированный резервуар
    Решение: адаптивные параметры резервуара
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "EchoStateV2")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.reservoir_dim = config.get('reservoir_dim', 512)
        self.num_layers = config.get('num_layers', 3)
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Адаптивные резервуары
        self.reservoirs = nn.ModuleList([
            AdaptiveReservoir(self.dim, self.reservoir_dim)
            for _ in range(self.num_layers)
        ])
        
        # Проекции и нормализации
        self.projections = nn.ModuleList([
            nn.Linear(self.reservoir_dim, self.dim)
            for _ in range(self.num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.dim)
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for reservoir, proj, norm in zip(self.reservoirs, self.projections, self.norms):
            residual = x
            reservoir_out = reservoir(x)
            x = proj(reservoir_out)
            x = norm(x + residual)
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'EchoStateV2',
            'dim': self.dim,
            'reservoir_dim': self.reservoir_dim,
            'num_layers': self.num_layers,
            'improvements': ['Learnable spectral radius', 'Adaptive leak rate', 'Better parameter efficiency']
        }


# ============================================================================
# HybridMemory - Комбинация HyperMemory + FractalNet
# ============================================================================

class HybridMemory(BaseArchitecture):
    """
    HybridMemory - объединяет сильные стороны HyperMemory и FractalNet
    
    От HyperMemory: адаптивная память
    От FractalNet: многопутевая обработка
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "HybridMemory")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = config.get('num_layers', 4)
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'path1': nn.Conv1d(self.dim, self.dim, 3, padding=1, groups=self.dim),
                'path2': nn.MultiheadAttention(self.dim, 8, batch_first=True),
                'path_norm': nn.LayerNorm(self.dim),
                'memory_gate': nn.Linear(self.dim, self.dim),
                'memory_value': nn.Linear(self.dim, self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 3),
                    nn.GELU(),
                    nn.Linear(self.dim * 3, self.dim)
                ),
                'norm': nn.LayerNorm(self.dim)
            })
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        memory = torch.zeros(B, self.dim, device=x.device)
        
        for layer in self.layers:
            residual = x
            
            # Multi-path processing (от FractalNet)
            x_conv = layer['path1'](x.transpose(1, 2)).transpose(1, 2)
            x_attn, _ = layer['path2'](x, x, x)
            x = layer['path_norm'](x_conv + x_attn)
            x = residual + x
            
            # Memory update (от HyperMemory)
            x_mean = x.mean(dim=1)
            gate = torch.sigmoid(layer['memory_gate'](x_mean))
            value = layer['memory_value'](x_mean)
            memory = gate * memory + (1 - gate) * value
            
            # FFN с памятью
            residual = x
            x = layer['norm'](x + memory.unsqueeze(1))
            x = layer['ffn'](x)
            x = residual + x
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'HybridMemory',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'features': ['Multi-path processing', 'Adaptive memory', 'Best of both worlds']
        }


# ============================================================================
# AdaptiveDepth - Динамическая глубина на основе сложности
# ============================================================================

class AdaptiveDepthLayer(nn.Module):
    """Слой с адаптивной глубиной"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # Halting unit - решает, продолжать ли обработку
        self.halting_prob = nn.Linear(dim, 1)
        
    def forward(self, x, cumulative_halt_prob, threshold=0.99):
        B, L, D = x.shape
        
        # Compute halting probability
        halt_prob = torch.sigmoid(self.halting_prob(x.mean(dim=1)))
        
        # Update cumulative
        new_cumulative = cumulative_halt_prob + halt_prob
        
        # Compute this layer's weight
        weight = torch.clamp(1.0 - cumulative_halt_prob, min=0.0, max=1.0)
        
        # Process
        residual = x
        x = self.norm(x)
        x_attn, _ = self.attn(x, x, x)
        x = residual + x_attn
        
        x = self.ffn(x)
        
        # Weight output
        x = weight.unsqueeze(1) * x + (1 - weight.unsqueeze(1)) * residual
        
        return x, new_cumulative, halt_prob


class AdaptiveDepth(BaseArchitecture):
    """
    AdaptiveDepth - архитектура с динамической глубиной
    
    Идея: простые примеры проходят через меньше слоёв
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AdaptiveDepth")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.max_layers = config.get('num_layers', 8)
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        self.layers = nn.ModuleList([
            AdaptiveDepthLayer(self.dim)
            for _ in range(self.max_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        cumulative_halt = torch.zeros(B, 1, device=x.device)
        
        for layer in self.layers:
            x, cumulative_halt, halt_prob = layer(x, cumulative_halt)
            
            # Early exit если достигли threshold
            if (cumulative_halt > 0.99).all():
                break
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'AdaptiveDepth',
            'dim': self.dim,
            'max_layers': self.max_layers,
            'features': ['Adaptive computation time', 'Dynamic depth', 'Efficient processing']
        }

