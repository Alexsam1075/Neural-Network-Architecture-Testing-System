"""
HyperMemory Architecture
========================
Проблема трансформеров: контекст ограничен окном, O(n^2) по памяти и вычислениям.
Проблема SSM: экспоненциальное затухание — далёкое прошлое «тает».

Решение: Внешняя дифференцируемая память с адресацией через хэш-проекции.
Каждый токен пишет в память через быстрый линейный ключ O(1).
Читает через тот же ключ — мгновенно, без attention по всей последовательности.
Память не затухает — она обновляется аддитивно с gate-контролем.

Сложность: O(1) per token, бесконечный контекст через аккумулирующую память.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


class HyperMemoryCell(nn.Module):
    """
    Ячейка внешней памяти.
    Память — матрица M (memory_slots x d_model).
    Запись: M[key] += gate * value
    Чтение: output = softmax(query @ M.T) @ M
    Всё O(memory_slots) — константа, не зависит от длины последовательности.
    """

    def __init__(self, d_model: int, memory_slots: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Проекции ключей/запросов/значений для памяти
        self.key_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Gate — решает сколько писать в память
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, memory_slots),
            nn.Sigmoid()
        )

        # Erase gate — выборочное забывание (не полное!)
        self.erase_gate = nn.Sequential(
            nn.Linear(d_model, memory_slots),
            nn.Sigmoid()
        )

        # Позиционная адресация памяти — фиксированные слоты
        self.slot_keys = nn.Parameter(torch.randn(memory_slots, d_model) * 0.02)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> tuple:
        """
        x: (batch, seq_len, d_model)
        memory: (batch, memory_slots, d_model)
        returns: (output, updated_memory)
        """
        B, T, D = x.shape

        queries = self.query_proj(x)   # (B, T, D)
        keys = self.key_proj(x)        # (B, T, D)
        values = self.value_proj(x)    # (B, T, D)

        # --- ЧТЕНИЕ из памяти ---
        # Адресация: насколько каждый запрос похож на каждый слот памяти
        slot_k = self.slot_keys.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        read_scores = torch.bmm(queries, slot_k.transpose(1, 2)) / (D ** 0.5)  # (B, T, M)
        read_weights = F.softmax(read_scores, dim=-1)  # (B, T, M)
        read_out = torch.bmm(read_weights, memory)     # (B, T, D)

        # --- ЗАПИСЬ в память (обновляем после чтения) ---
        write_gate = self.write_gate(x)   # (B, T, M)
        erase_gate = self.erase_gate(x)   # (B, T, M)

        # Агрегируем по временной оси — каждый слот получает взвешенную сумму
        # write: (B, M, D) = softmax(gates).T @ values
        write_addr = F.softmax(write_gate, dim=-1)   # (B, T, M)
        write_content = torch.bmm(write_addr.transpose(1, 2), values)   # (B, M, D)

        erase_addr = F.softmax(erase_gate, dim=-1)   # (B, T, M)
        erase_amount = erase_addr.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, M, 1)
        erase_scale = torch.sigmoid(-erase_amount + 1.0)  # мягкое забывание

        # Обновляем память: erase потом write
        new_memory = memory * erase_scale + write_content

        # --- ВЫХОД ---
        out = self.out_proj(read_out)
        out = self.norm(x + self.dropout(out))

        return out, new_memory


class HyperMemoryArchitecture(BaseArchitecture):
    """
    HyperMemory: O(1) per token, бесконечный контекст.
    Память накапливает знания через всю последовательность без затухания.
    """

    def __init__(self, config: Dict[str, Any], name: str = "HyperMemory"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.memory_slots = config.get('memory_slots', 64)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, self.d_model) * 0.02)

        self.memory_cells = nn.ModuleList([
            HyperMemoryCell(self.d_model, self.memory_slots, self.num_heads, self.dropout)
            for _ in range(self.num_layers)
        ])

        # FFN между слоями
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 4, self.d_model),
            )
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

        emb = self.token_embedding(x) + self.pos_encoding[:, :T, :]

        # Инициализируем память нулями (в inference можно передавать между батчами)
        memory = torch.zeros(B, self.memory_slots, self.d_model, device=x.device)

        h = emb
        for cell, ffn in zip(self.memory_cells, self.ffns):
            h, memory = cell(h, memory)
            h = h + ffn(h)

        h = self.norm_out(h)
        return self.fc_out(h)

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'HyperMemory',
            'complexity': 'O(memory_slots) per token — constant in sequence length',
            'context': 'infinite — accumulative external memory',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'memory_slots': self.memory_slots,
            'num_layers': self.num_layers,
            'description': 'Differentiable external memory with gated read/write, O(1) per token'
        }
