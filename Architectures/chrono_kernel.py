"""
ChronoKernel Architecture
=========================
Проблема трансформеров: O(n^2) attention — квадратичная сложность.
Проблема свёрточных сетей (CNN): фиксированное поле зрения — короткий контекст.
Проблема S4/Hyena: сложная математика дискретизации, нестабильное обучение.

Решение: Learnable kernel в спектральном домене.
FFT превращает свёртку O(n^2) в умножение O(n log n).
Ядро — learnable параметры в частотном пространстве.
Разные частоты = разные масштабы контекста одновременно.
Нет ограничения на длину — ядро применяется глобально через FFT.

Ключевой инсайт: язык имеет многочастотную структуру.
Высокие частоты = локальный синтаксис. Низкие = глобальная семантика.
ChronoKernel обучается работать на всех частотах параллельно.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


class SpectralKernelLayer(nn.Module):
    """
    Глобальная свёртка через FFT.
    Ядро — learnable комплексные веса в частотном пространстве.
    O(n log n) для любой длины последовательности.
    """

    def __init__(self, d_model: int, num_kernels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_kernels = num_kernels

        # Комплексные веса ядра в частотном домене
        # Размер 512 — максимальная длина; для коротких последовательностей обрезаем
        self.kernel_real = nn.Parameter(torch.randn(num_kernels, d_model, 512) * 0.02)
        self.kernel_imag = nn.Parameter(torch.randn(num_kernels, d_model, 512) * 0.02)

        # Смешивание результатов разных ядер
        self.kernel_mix = nn.Linear(num_kernels * d_model, d_model)

        # Частотный dropout — случайно зануляет частоты (регуляризация)
        self.freq_dropout = nn.Dropout(dropout * 0.5)

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape

        # FFT по временной оси
        x_freq = torch.fft.rfft(x, dim=1)  # (B, T//2+1, D) комплексный

        freq_len = x_freq.shape[1]
        k_len = min(freq_len, 512)

        # Применяем learnable ядра
        outputs = []
        for i in range(self.num_kernels):
            kr = self.kernel_real[i, :, :k_len].T  # (freq_len_capped, D)
            ki = self.kernel_imag[i, :, :k_len].T

            if k_len < freq_len:
                # Паддим ядро нулями если последовательность длиннее 512
                pad = freq_len - k_len
                kr = F.pad(kr, (0, 0, 0, pad))
                ki = F.pad(ki, (0, 0, 0, pad))

            # Комплексное ядро
            kernel_complex = torch.complex(kr, ki)  # (freq_len, D)

            # Умножение в частотном домене = свёртка в временном
            out_freq = x_freq * kernel_complex.unsqueeze(0)  # (B, freq_len, D)
            out_freq = self.freq_dropout(out_freq.real) + 1j * self.freq_dropout(out_freq.imag)

            # Обратное FFT
            out = torch.fft.irfft(out_freq, n=T, dim=1)  # (B, T, D)
            outputs.append(out)

        # Конкатенация и смешивание
        mixed = torch.cat(outputs, dim=-1)   # (B, T, D*num_kernels)
        out = self.kernel_mix(mixed)          # (B, T, D)
        out = self.act(out)

        return self.norm(x + out)


class ChronoKernelBlock(nn.Module):
    """Блок: SpectralKernel + локальный conv + FFN"""

    def __init__(self, d_model: int, d_ff: int, num_kernels: int = 4, dropout: float = 0.1):
        super().__init__()

        # Глобальный спектральный путь
        self.spectral = SpectralKernelLayer(d_model, num_kernels, dropout)

        # Локальный свёрточный путь (дополняет глобальный)
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.local_norm = nn.LayerNorm(d_model)

        # Merge
        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Глобальный спектральный путь
        global_out = self.spectral(x)

        # Локальный свёрточный путь
        local_out = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        local_out = self.local_norm(x + local_out)

        # Слияние глобального и локального
        h = self.merge(torch.cat([global_out, local_out], dim=-1))

        # FFN
        h = self.norm_ffn(h + self.dropout(self.ffn(h)))
        return h


class ChronoKernelArchitecture(BaseArchitecture):
    """
    ChronoKernel: глобальная свёртка через FFT с learnable ядром.
    Видит весь контекст любой длины за O(n log n).
    Разные частоты = разные масштабы восприятия.
    """

    def __init__(self, config: Dict[str, Any], name: str = "ChronoKernel"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.d_ff = config.get('d_ff', 512)
        self.num_kernels = config.get('num_kernels', 4)
        self.dropout = config.get('dropout', 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList([
            ChronoKernelBlock(self.d_model, self.d_ff, self.num_kernels, self.dropout)
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
            'type': 'ChronoKernel',
            'complexity': 'O(n log n) — global convolution via FFT',
            'context': 'infinite — FFT kernel covers full sequence',
            'num_kernels': self.num_kernels,
            'innovation': 'learnable spectral kernel captures multi-frequency patterns',
            'description': 'Spectral domain learnable convolution: local+global via FFT'
        }
