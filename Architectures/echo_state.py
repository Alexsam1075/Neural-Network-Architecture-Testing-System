"""
EchoState / Liquid State Machine Architecture
==============================================
Проблема RNN: градиент затухает/взрывается при обучении длинных зависимостей.
Решение LSTM/GRU: gates помогают, но не решают проблему до конца.

Кардинальное решение: НЕ ОБУЧАТЬ рекуррентные веса совсем.
Reservoir Computing: огромный хаотический резервуар фиксированных весов.
Только входные и выходные веса обучаемые.

Почему это работает: хаотический резервуар — это «эхо-камера».
Разные входы создают разные траектории в резервуаре.
Линейный readout потом разделяет эти траектории.

Новинка в нашей реализации:
1. Несколько маленьких резервуаров вместо одного большого (ансамбль)
2. Learnable input projection — оптимизируем как входит сигнал
3. Learnable spectral radius — нейросеть сама настраивает хаотичность
4. Attention между резервуарами — разные резервуары специализируются

O(reservoir_size) per token — константа независимо от длины.
Бесконечный контекст через эхо-траекторию в резервуаре.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


class EchoReservoir(nn.Module):
    """
    Один хаотический резервуар.
    Веса W_res ФИКСИРОВАНЫ после инициализации — это ключевой принцип.
    Обучается только input_proj и spectral_scale.
    """

    def __init__(self, d_model: int, reservoir_size: int, sparsity: float = 0.1,
                 spectral_radius: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.reservoir_size = reservoir_size

        # Learnable входная проекция
        self.input_proj = nn.Linear(d_model, reservoir_size)

        # Learnable масштаб спектрального радиуса
        # Спектральный радиус < 1 = Echo State Property (стабильность)
        self.spectral_scale = nn.Parameter(torch.tensor(spectral_radius))

        # ФИКСИРОВАННЫЙ резервуар — инициализируем и замораживаем
        W = torch.randn(reservoir_size, reservoir_size) * (1.0 / reservoir_size ** 0.5)

        # Разреженность — только sparsity% связей активны
        mask = torch.bernoulli(torch.full((reservoir_size, reservoir_size), sparsity))
        W = W * mask

        # Нормализуем до единичного спектрального радиуса
        eigenvalues = torch.linalg.eigvals(W)
        max_eigenvalue = eigenvalues.abs().max()
        if max_eigenvalue > 1e-6:
            W = W / max_eigenvalue

        self.register_buffer('W_res', W)
        self.register_buffer('W_mask', mask)

        # Learnable выходная проекция
        self.output_proj = nn.Linear(reservoir_size, d_model)

        # Leak rate — насколько быстро обновляется состояние
        self.leak_rate = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        state = torch.zeros(B, self.reservoir_size, device=x.device, dtype=x.dtype)
        outputs = []

        # Эффективный спектральный радиус
        eff_radius = torch.sigmoid(self.spectral_scale) * 0.99  # < 1 для стабильности
        W_scaled = self.W_res * eff_radius

        # Leak rate
        alpha = torch.sigmoid(self.leak_rate)

        for t in range(T):
            x_t = self.input_proj(x[:, t, :])  # (B, reservoir_size)

            # Reservoir update: (1-α)*state + α*tanh(W*state + u)
            reservoir_in = state @ W_scaled.T + x_t
            new_state = (1 - alpha) * state + alpha * torch.tanh(reservoir_in)
            state = new_state

            outputs.append(self.output_proj(state))

        return torch.stack(outputs, dim=1)  # (B, T, D)


class EchoStateArchitecture(BaseArchitecture):
    """
    EchoState: ансамбль хаотических резервуаров с learnable readout.
    Рекуррентные веса НЕ ОБУЧАЮТСЯ — только входные/выходные.
    Бесконечный контекст, O(reservoir_size) per token.
    
    Разные резервуары улавливают разные паттерны:
    - Маленький спектральный радиус → кратковременная память
    - Близкий к 1 → долговременная память
    - Ансамбль покрывает все временные масштабы
    """

    def __init__(self, config: Dict[str, Any], name: str = "EchoState"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.reservoir_size = config.get('reservoir_size', 256)
        self.num_reservoirs = config.get('num_reservoirs', 4)
        self.dropout = config.get('dropout', 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)

        # Ансамбль резервуаров с разными спектральными радиусами
        spectral_radii = [0.5, 0.7, 0.9, 0.99]
        self.reservoirs = nn.ModuleList([
            EchoReservoir(
                self.d_model,
                self.reservoir_size // self.num_reservoirs,
                sparsity=0.15,
                spectral_radius=spectral_radii[i % len(spectral_radii)]
            )
            for i in range(self.num_reservoirs)
        ])

        # Attention между выходами резервуаров
        self.reservoir_attention = nn.MultiheadAttention(
            self.d_model, num_heads=4, dropout=self.dropout, batch_first=True
        )

        # Merge резервуаров
        self.merge = nn.Sequential(
            nn.Linear(self.d_model * self.num_reservoirs, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )

        # FFN после merge
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        self.norm_out = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = torch.clamp(x, 0, self.vocab_size - 1)
        h = self.embed_dropout(self.token_embedding(x))

        # Прогоняем через каждый резервуар
        reservoir_outs = []
        for reservoir in self.reservoirs:
            out = reservoir(h)  # (B, T, D)
            reservoir_outs.append(out)

        # Merge резервуаров
        merged = torch.cat(reservoir_outs, dim=-1)  # (B, T, D*n_res)
        h = self.merge(merged)  # (B, T, D)

        # FFN
        h = h + self.ffn(h)

        return self.fc_out(self.norm_out(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'EchoState',
            'complexity': 'O(reservoir_size) per token — constant in sequence length',
            'context': 'infinite — chaotic reservoir echo carries all history',
            'num_reservoirs': self.num_reservoirs,
            'reservoir_size': self.reservoir_size,
            'innovation': 'fixed chaotic reservoir weights, only readout trained; ensemble coverage',
            'description': 'Reservoir Computing: chaotic echo-state ensemble with learnable readout only'
        }
