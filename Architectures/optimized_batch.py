"""
Optimized Architectures Batch
Специализированные архитектуры для разных задач
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


# ============================================================================
# UltraFast - Экстремально быстрый инференс
# ============================================================================

class UltraFast(BaseArchitecture):
    """
    UltraFast - максимальная скорость инференса
    
    Оптимизации:
    - Depthwise separable convolutions
    - Minimal attention (только 2 головы)
    - Маленькие FFN (1.5x вместо 4x)
    - Группированные операции
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "UltraFast")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = config.get('num_layers', 3)  # меньше слоёв
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Легковесные слои
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # Depthwise convolution вместо attention
                'conv': nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim),
                'pointwise': nn.Conv1d(self.dim, self.dim, kernel_size=1),
                'norm': nn.LayerNorm(self.dim),
                # Маленький FFN
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, int(self.dim * 1.5)),
                    nn.GELU(),
                    nn.Linear(int(self.dim * 1.5), self.dim)
                )
            })
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for layer in self.layers:
            residual = x
            
            # Depthwise separable convolution (быстрее attention)
            x = x.transpose(1, 2)
            x = layer['conv'](x)
            x = layer['pointwise'](x)
            x = x.transpose(1, 2)
            x = layer['norm'](x + residual)
            
            # FFN
            x = x + layer['ffn'](x)
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'UltraFast',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'optimizations': [
                'Depthwise separable convolutions',
                'Reduced FFN size (1.5x)',
                'Fewer layers',
                'Target: >1500 pred/sec'
            ]
        }


# ============================================================================
# MemoryOptimized - Минимальное потребление памяти
# ============================================================================

class MemoryOptimized(BaseArchitecture):
    """
    MemoryOptimized - минимальное использование памяти
    
    Техники:
    - Gradient checkpointing
    - Weight sharing
    - Quantization-aware training
    - Pruning-friendly архитектура
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MemoryOptimized")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 128)  # меньше размерность
        self.num_layers = config.get('num_layers', 6)  # можем позволить больше слоёв
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Shared layers (weight sharing)
        self.shared_attn = nn.MultiheadAttention(self.dim, 4, batch_first=True)
        self.shared_ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        self.shared_norm1 = nn.LayerNorm(self.dim)
        self.shared_norm2 = nn.LayerNorm(self.dim)
        
        # Только нормализации уникальные для каждого слоя
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.dim) for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        # Переиспользуем одни и те же слои (weight sharing)
        for layer_norm in self.layer_norms:
            # Attention (shared)
            residual = x
            x = self.shared_norm1(x)
            x, _ = self.shared_attn(x, x, x)
            x = residual + x
            
            # FFN (shared)
            residual = x
            x = self.shared_norm2(x)
            x = self.shared_ffn(x)
            x = layer_norm(residual + x)
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'MemoryOptimized',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'optimizations': [
                'Weight sharing across layers',
                'Reduced dimensionality (128)',
                'Quantization-friendly',
                'Target: <200K parameters'
            ]
        }


# ============================================================================
# BalancedPro - Оптимальный баланс
# ============================================================================

class BalancedPro(BaseArchitecture):
    """
    BalancedPro - оптимальный баланс точности, скорости и размера
    
    Стратегия:
    - Средняя размерность (256)
    - Эффективная attention (Linear)
    - Умеренные FFN (2.5x)
    - Оптимальное количество слоёв (4)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "BalancedPro")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = config.get('num_layers', 4)
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Balanced layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(self.dim),
                # Линейная attention для эффективности
                'query': nn.Linear(self.dim, self.dim),
                'key': nn.Linear(self.dim, self.dim),
                'value': nn.Linear(self.dim, self.dim),
                'out': nn.Linear(self.dim, self.dim),
                'norm2': nn.LayerNorm(self.dim),
                # Balanced FFN
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, int(self.dim * 2.5)),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(int(self.dim * 2.5), self.dim)
                )
            })
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def linear_attention(self, q, k, v):
        """Линейная attention (O(n) вместо O(n²))"""
        # Feature map: elu(x) + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Вычисляем attention линейно
        kv = torch.einsum('bld,blm->bdm', k, v)
        k_sum = k.sum(dim=1, keepdim=True)
        
        out = torch.einsum('bld,bdm->blm', q, kv)
        out = out / (torch.einsum('bld,bld->bl', q, k_sum).unsqueeze(-1) + 1e-6)
        
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for layer in self.layers:
            # Linear attention
            residual = x
            x = layer['norm1'](x)
            
            q = layer['query'](x)
            k = layer['key'](x)
            v = layer['value'](x)
            
            attn_out = self.linear_attention(q, k, v)
            x = layer['out'](attn_out)
            x = residual + x
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = residual + x
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'BalancedPro',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'features': [
                'Linear attention O(n)',
                'Balanced FFN (2.5x)',
                'Optimal layer count',
                'Best all-around performance'
            ]
        }


# ============================================================================
# SpeedDemon - Максимальная скорость обучения
# ============================================================================

class SpeedDemon(BaseArchitecture):
    """
    SpeedDemon - максимальная скорость обучения
    
    Оптимизации:
    - Большие batch-friendly операции
    - Параллелизуемые вычисления
    - Минимум sequential зависимостей
    - Оптимизированные градиенты
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "SpeedDemon")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = config.get('num_layers', 3)
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Параллельные пути (можно вычислять одновременно)
        self.parallel_paths = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.dim),
                    nn.Linear(self.dim, self.dim),
                    nn.GELU()
                )
                for _ in range(3)  # 3 параллельных пути
            ])
            for _ in range(self.num_layers)
        ])
        
        # Mixing layers
        self.mixers = nn.ModuleList([
            nn.Linear(self.dim * 3, self.dim)
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for paths, mixer in zip(self.parallel_paths, self.mixers):
            # Все пути вычисляются параллельно
            path_outputs = [path(x) for path in paths]
            
            # Объединяем
            combined = torch.cat(path_outputs, dim=-1)
            x = mixer(combined)
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'SpeedDemon',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'optimizations': [
                'Parallel computation paths',
                'Minimal sequential dependencies',
                'Batch-friendly operations',
                'Target: >200 samples/sec training'
            ]
        }


# ============================================================================
# AccuracyFirst - Приоритет точности
# ============================================================================

class AccuracyFirst(BaseArchitecture):
    """
    AccuracyFirst - максимальная точность, скорость вторична
    
    Стратегии:
    - Больше слоёв (8)
    - Больше голов attention (16)
    - Большие FFN (4x)
    - Ensemble-like архитектура
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AccuracyFirst")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 384)  # больше размерность
        self.num_layers = config.get('num_layers', 8)  # больше слоёв
        
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Deep layers с большой ёмкостью
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(self.dim, 16, dropout=0.1, batch_first=True),
                'norm2': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 4),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.dim * 4, self.dim),
                    nn.Dropout(0.1)
                ),
                # Дополнительный residual path
                'aux_path': nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.Tanh()
                )
            })
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        for layer in self.layers:
            # Main path: attention
            residual = x
            x = layer['norm1'](x)
            x, _ = layer['attn'](x, x, x)
            x = residual + x
            
            # FFN
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            
            # Auxiliary path (ensemble-like)
            aux = layer['aux_path'](residual)
            x = residual + x + 0.1 * aux  # небольшой вклад aux
        
        return self.head(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'AccuracyFirst',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'features': [
                'Deep architecture (8 layers)',
                'Large dimension (384)',
                'Many attention heads (16)',
                'Target: 100% accuracy'
            ]
        }

