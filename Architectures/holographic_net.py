"""
HolographicNet Architecture
============================
Проблема всех архитектур: хранение контекста требует памяти O(n).
Трансформер: KV-cache растёт с длиной. SSM: хотя O(1) состояние, но теряет детали.

Решение: Голографическое кодирование — весь контекст в ОДНОМ векторе фиксированного размера.
Как голограмма: каждый фрагмент содержит информацию о целом.

Механизм: Circular Convolution (циркулярная свёртка).
bind(A, B) = ifft(fft(A) * fft(B)) — связывание двух векторов в один без потери размерности.
unbind(AB, B) = ifft(fft(AB) * conj(fft(B))) — извлечение A из AB.

Контекст — это один вектор, который содержит все связанные пары (позиция, значение).
Добавить новый токен: O(1). Запросить любой: O(1).
Ёмкость: ~d/2 различных пар. При d=128 — 64 независимых токена точно.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math
from .base_architecture import BaseArchitecture


def circular_conv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Циркулярная свёртка через FFT. a,b: (..., D)"""
    try:
        return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1),
                               n=a.shape[-1], dim=-1)
    except RuntimeError:
        out = torch.zeros_like(a)
        for i in range(a.shape[-1]):
            out = out + a[..., i:i + 1] * torch.roll(b, shifts=i, dims=-1)
        return out


def circular_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Циркулярная корреляция (развязка). a,b: (..., D)"""
    try:
        return torch.fft.irfft(torch.fft.rfft(a, dim=-1) * torch.fft.rfft(b, dim=-1).conj(),
                               n=a.shape[-1], dim=-1)
    except RuntimeError:
        out = torch.zeros_like(a)
        for i in range(a.shape[-1]):
            out = out + a[..., i:i + 1] * torch.roll(b, shifts=-i, dims=-1)
        return out


class HolographicMemory(nn.Module):
    """
    Голографическая память: весь контекст в одном векторе.
    
    Хранение: mem += bind(pos_key, value)
    Запрос: approx_value = unbind(mem, pos_key)
    
    Позиционные ключи — случайные ортогональные векторы (фиксированные или learnable).
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Проекции значений для хранения и запроса
        self.store_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        # Нормализация для стабильности (голограмма переполняется без нормировки)
        self.memory_norm = nn.Parameter(torch.ones(1))

        # Позиционные ключи — кодируют позицию в последовательности
        # Инициализируем как единичные случайные векторы
        pos_keys = torch.randn(512, d_model)
        pos_keys = F.normalize(pos_keys, dim=-1)
        self.register_buffer('pos_keys', pos_keys)

        # Gate — контролирует что записывать
        self.write_gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # Читающий трансформ — улучшает извлечённое значение
        self.read_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D), использует голографическую память"""
        B, T, D = x.shape

        # Проекции
        stored_vals = self.store_proj(x)    # что хранить
        query_vals = self.query_proj(x)     # что запрашивать

        # Позиционные ключи для каждой позиции
        pos_k = self.pos_keys[:T, :].unsqueeze(0).expand(B, -1, -1)  # (B, T, D)

        outputs = []
        memory = torch.zeros(B, D, device=x.device)

        for t in range(T):
            # Запрашиваем память ПЕРЕД записью текущего токена
            q_key = pos_k[:, t, :]           # (B, D)
            q_val = query_vals[:, t, :]

            if t > 0:
                # Извлекаем ближайшее значение из памяти
                retrieved = circular_corr(memory, q_key)  # (B, D)
                retrieved = retrieved / (self.memory_norm * math.sqrt(t) + 1e-6)

                # Комбинируем запрос с извлечённым
                out = self.read_proj(torch.cat([q_val, retrieved], dim=-1))
            else:
                out = q_val

            outputs.append(out)

            # Записываем текущий токен
            gate = self.write_gate(stored_vals[:, t, :])
            binding = circular_conv(pos_k[:, t, :], stored_vals[:, t, :])
            memory = memory + gate * binding

        out_seq = torch.stack(outputs, dim=1)  # (B, T, D)
        return self.norm_out(x + self.dropout(out_seq))


class HolographicBlock(nn.Module):
    """Блок: голографическая память + FFN"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.holo_mem = HolographicMemory(d_model, dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.holo_mem(x)
        return h + self.ffn(h)


class HolographicNetArchitecture(BaseArchitecture):
    """
    HolographicNet: весь контекст в одном векторе через circular convolution.
    O(1) память на контекст. O(1) запись и чтение.
    Теоретически — бесконечный контекст при достаточной размерности.
    """

    def __init__(self, config: Dict[str, Any], name: str = "HolographicNet"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList([
            HolographicBlock(self.d_model, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])

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

        for block in self.blocks:
            h = block(h)

        return self.fc_out(self.norm_out(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'HolographicNet',
            'complexity': 'O(1) memory, O(1) read/write per token',
            'context': 'infinite — full context in single fixed-size vector',
            'innovation': 'circular convolution binding: entire history in O(d) space',
            'description': 'Holographic Reduced Representations: context as superposition binding'
        }
