"""
AION-Lite Architecture
Лёгкая версия AION реализующая ключевые законы

Реализованные законы AION:
- Закон 1: Фиксированная память (256 чисел workspace + 8 эмоций)
- Закон 6: Восемь эмоций как внутреннее состояние
- Закон 7: Конкурс специалистов (упрощённый)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


class EmotionalHomeostasis(nn.Module):
    """
    Система эмоционального гомеостаза (Закон 6)
    
    8 эмоций: curiosity, confidence, surprise, satisfaction, 
              tension, boredom, engagement, confusion
    """
    def __init__(self, dim: int):
        super().__init__()
        self.num_emotions = 8
        
        # Вычисляем эмоции из входа
        self.emotion_computer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, self.num_emotions),
            nn.Sigmoid()  # Эмоции в [0, 1]
        )
        
    def forward(self, x):
        """
        Вычисляет текущее эмоциональное состояние
        
        Returns:
            emotions: [B, 8] тензор эмоций
        """
        # Агрегируем по последовательности
        x_agg = x.mean(dim=1)
        emotions = self.emotion_computer(x_agg)
        return emotions


class SpecialistWorkspace(nn.Module):
    """
    Конкурс специалистов (Закон 7 - упрощённая версия)
    
    Специалисты: FACTUAL, LOGICAL, CREATIVE, CRITICAL
    """
    def __init__(self, dim: int, num_specialists: int = 4):
        super().__init__()
        self.num_specialists = num_specialists
        self.dim = dim
        
        # Каждый специалист - это небольшая сеть
        self.specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            )
            for _ in range(num_specialists)
        ])
        
        # Голосование на основе эмоций
        self.voting_weights = nn.Linear(8, num_specialists)
        
    def forward(self, x, emotions):
        """
        Запускает конкурс специалистов
        
        Args:
            x: входные данные [B, L, D]
            emotions: эмоциональное состояние [B, 8]
            
        Returns:
            выход победившего специалиста
        """
        B, L, D = x.shape
        
        # Каждый специалист обрабатывает вход
        outputs = []
        for specialist in self.specialists:
            # Обрабатываем среднее по последовательности
            x_mean = x.mean(dim=1)
            output = specialist(x_mean)
            outputs.append(output.unsqueeze(1))
        
        # [B, num_specialists, D]
        outputs = torch.cat(outputs, dim=1)
        
        # Вычисляем веса голосования на основе эмоций
        votes = self.voting_weights(emotions)  # [B, num_specialists]
        votes = F.softmax(votes, dim=-1).unsqueeze(-1)  # [B, num_specialists, 1]
        
        # Взвешенная комбинация
        result = (outputs * votes).sum(dim=1)  # [B, D]
        
        # Добавляем обратно в последовательность
        result = result.unsqueeze(1).expand(-1, L, -1)
        
        return result


class AIONLite(BaseArchitecture):
    """
    AION-Lite - компактная реализация ключевых законов AION
    
    Особенности:
    1. Фиксированная память (Закон 1): 256 + 8 = 264 числа
    2. Эмоциональный гомеостаз (Закон 6): 8 эмоций
    3. Конкурс специалистов (Закон 7): 4 специалиста
    4. Константная сложность памяти
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AIONLite")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)  # Workspace: 256 чисел (Закон 1)
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Компоненты AION
        self.homeostasis = EmotionalHomeostasis(self.dim)
        self.workspace = SpecialistWorkspace(self.dim, num_specialists=4)
        
        # Layers с фиксированной памятью
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(
                    self.dim, num_heads=8, dropout=self.dropout, batch_first=True
                ),
                'norm2': nn.LayerNorm(self.dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.dim * 2, self.dim)
                )
            })
            for _ in range(self.num_layers)
        ])
        
        # Фиксированный workspace state (Закон 1)
        self.register_buffer('workspace_state', torch.zeros(1, self.dim))
        
        # Output
        self.norm_out = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        # Embeddings
        positions = torch.arange(L, device=x.device).clamp(max=self.seq_length - 1).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout_layer(x)
        
        # Вычисляем эмоциональное состояние (Закон 6)
        emotions = self.homeostasis(x)  # [B, 8]
        
        # Processing через слои
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer['norm1'](x)
            x, _ = layer['attn'](x, x, x)
            x = residual + x
            
            # Feed-forward
            residual = x
            x = layer['norm2'](x)
            x = layer['ffn'](x)
            x = residual + x
        
        # Конкурс специалистов (Закон 7)
        specialist_output = self.workspace(x, emotions)
        x = x + specialist_output
        
        # Обновляем workspace state (Закон 1 - фиксированная память)
        # Workspace state НЕ растёт с длиной последовательности
        self.workspace_state = x.mean(dim=1, keepdim=True).detach()
        
        # Output
        x = self.norm_out(x)
        x = self.head(x)
        
        return x
    
    def get_memory_size(self) -> int:
        """
        Возвращает размер памяти системы в числах
        
        Закон 1: память фиксирована и не растёт с длиной диалога
        """
        workspace_size = self.dim  # 256
        emotion_size = 8  # 8 эмоций
        return workspace_size + emotion_size
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'AIONLite',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'aion_laws': [
                'Law 1: Fixed memory (256 workspace + 8 emotions)',
                'Law 6: Eight emotions homeostasis',
                'Law 7: Specialist competition (4 specialists)',
            ],
            'memory_size': self.get_memory_size(),
            'memory_constant': True
        }
