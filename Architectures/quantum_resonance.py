"""
QuantumResonance Architecture
==============================
Проблема трансформеров: softmax attention коллапсирует распределение — теряет неопределённость.
Проблема всех архитектур: линейное мышление — взвешенные суммы.

Решение: Интерференция признаков как в квантовой механике.
Представляем фичи как волны (амплитуда + фаза).
Интерференция: конструктивная (похожие фичи усиливаются) и деструктивная (противоположные гасятся).
Это нелинейнее softmax — фичи взаимодействуют через фазовые соотношения.

Ключевой инсайт: язык полон «фазовых переходов» — слово меняет смысл в зависимости от окружения.
QuantumResonance кодирует это явно через фазу.
O(n) — полностью рекуррентная реализация через накопленную фазу.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math
from .base_architecture import BaseArchitecture


class WaveEmbedding(nn.Module):
    """
    Кодирует токены как волны (амплитуда + фаза).
    Каждый токен → вектор в комплексном пространстве.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for wave encoding"
        self.d_half = d_model // 2

        self.amp_embed = nn.Embedding(vocab_size, self.d_half)
        self.phase_embed = nn.Embedding(vocab_size, self.d_half)

        nn.init.normal_(self.amp_embed.weight, 0, 0.02)
        nn.init.uniform_(self.phase_embed.weight, -math.pi, math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает (B, T, d_model) где первая половина — real, вторая — imag"""
        amp = torch.abs(self.amp_embed(x)) + 1e-6  # амплитуда > 0
        phase = self.phase_embed(x)                 # фаза
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        return torch.cat([real, imag], dim=-1)


class ResonanceLayer(nn.Module):
    """
    Квантовая интерференция: токены «резонируют» друг с другом.
    
    Механизм:
    1. Вычисляем фазовую разницу между токенами
    2. Конструктивная интерференция (|Δφ| < π/2) усиливает признак
    3. Деструктивная (|Δφ| > π/2) ослабляет
    4. Рекуррентно накапливаем «суперпозицию»
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_half = d_model // 2

        # Проекции в волновое пространство
        self.real_proj = nn.Linear(d_model, self.d_half)
        self.imag_proj = nn.Linear(d_model, self.d_half)

        # Частотные фильтры — какие частоты усиливаем
        self.freq_filter = nn.Parameter(torch.randn(self.d_half) * 0.1)

        # Интерференционный gate
        self.interference_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

        # Рекуррентное состояние суперпозиции
        self.state_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.state_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def interference(self, x_real: torch.Tensor, x_imag: torch.Tensor,
                     state_real: torch.Tensor, state_imag: torch.Tensor):
        """Вычисляет интерференцию текущего токена с накопленным состоянием."""
        # Фазовая разница: cos(φ_new - φ_state)
        # = cos(φ_new)cos(φ_state) + sin(φ_new)sin(φ_state)
        constructive = x_real * state_real + x_imag * state_imag  # (B, d_half)

        # Деструктивная интерференция
        destructive = x_real * state_imag - x_imag * state_real   # (B, d_half)

        # Частотная фильтрация
        freq = torch.sigmoid(self.freq_filter)
        out_real = freq * constructive
        out_imag = (1 - freq) * destructive

        return torch.cat([out_real, out_imag], dim=-1)

    def step(self, x_t: torch.Tensor, sup_real: torch.Tensor, sup_imag: torch.Tensor):
        """Один шаг интерференции."""
        x_real = self.real_proj(x_t)
        x_imag = self.imag_proj(x_t)

        # Интерференция с суперпозицией
        inter = self.interference(x_real, x_imag, sup_real, sup_imag)

        # Gate — насколько обновляем суперпозицию
        combined = torch.cat([x_t, inter], dim=-1)
        g = self.state_gate(combined)
        update = self.state_update(combined)

        new_state = g * update + (1 - g) * x_t

        # Новая суперпозиция
        new_real = torch.cos(self.freq_filter) * x_real + new_state[:, :self.d_half]
        new_imag = torch.sin(self.freq_filter) * x_imag + new_state[:, self.d_half:]
        new_real = F.normalize(new_real, dim=-1) * math.sqrt(self.d_half)
        new_imag = F.normalize(new_imag, dim=-1) * math.sqrt(self.d_half)

        out = self.out_proj(new_state)
        return out, new_real, new_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        sup_real = torch.zeros(B, self.d_half, device=x.device)
        sup_imag = torch.zeros(B, self.d_half, device=x.device)

        outputs = []
        for t in range(T):
            out, sup_real, sup_imag = self.step(x[:, t, :], sup_real, sup_imag)
            outputs.append(out)

        out_seq = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(out_seq))


class QuantumResonanceArchitecture(BaseArchitecture):
    """
    QuantumResonance: интерференция признаков как квантовые волны.
    Конструктивная/деструктивная интерференция заменяет softmax attention.
    O(1) per token, бесконечный контекст через суперпозицию состояний.
    """

    def __init__(self, config: Dict[str, Any], name: str = "QuantumResonance"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.1)

        # Убеждаемся что d_model чётный
        if self.d_model % 2 != 0:
            self.d_model += 1

        self.wave_embedding = WaveEmbedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)

        self.resonance_layers = nn.ModuleList([
            ResonanceLayer(self.d_model, self.dropout)
            for _ in range(self.num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.d_model),
            )
            for _ in range(self.num_layers)
        ])

        self.norm_out = nn.LayerNorm(self.d_model)
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.wave_embedding.amp_embed \
               and m is not self.wave_embedding.phase_embed:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = torch.clamp(x, 0, self.vocab_size - 1)
        h = self.embed_dropout(self.wave_embedding(x))

        for res_layer, ffn in zip(self.resonance_layers, self.ffns):
            h = res_layer(h)
            h = h + ffn(h)

        return self.fc_out(self.norm_out(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'QuantumResonance',
            'complexity': 'O(1) per token — recurrent wave superposition',
            'context': 'infinite — superposition accumulates quantum state',
            'innovation': 'constructive/destructive interference replaces softmax attention',
            'description': 'Wave-based feature interference: amplitude+phase token encoding'
        }
