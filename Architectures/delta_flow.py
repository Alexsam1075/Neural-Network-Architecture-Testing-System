"""
DeltaFlow Architecture
======================
Проблема трансформеров: каждый слой видит токены независимо — нет «инерции» знания.
Проблема RNN/SSM: один скрытый вектор сжимает весь контекст — бутылочное горлышко.

Решение: Многоуровневый поток дельт (изменений).
Вместо хранения состояния храним СКОРОСТЬ изменения состояния (momentum).
Как физика: позиция + скорость + ускорение = точное предсказание.
Каждый токен обновляет не состояние, а градиент состояния — это принципиально другое.

Сложность: O(1) per token (рекуррентно), O(k) если разворачивать блоки.
Бесконечный контекст через накопленный momentum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


class MomentumCell(nn.Module):
    """
    Ячейка с физическим momentum.
    state_{t+1} = state_t + velocity_t
    velocity_{t+1} = α * velocity_t + β * acceleration(x_t)
    
    acceleration вычисляется из входного токена.
    Это позволяет «помнить» тренд изменений, а не только последнее значение.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Входная проекция → ускорение
        self.accel_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Адаптивные коэффициенты трения и ускорения
        self.alpha_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())  # трение velocity
        self.beta_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())   # масштаб ускорения

        # Gate для обновления state
        self.state_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

        self.norm_state = nn.LayerNorm(d_model)
        self.norm_vel = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Проекция выхода
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def step(self, x_t: torch.Tensor, state: torch.Tensor, velocity: torch.Tensor):
        """Один шаг momentum динамики"""
        # Ускорение из входа и текущего состояния
        combined = torch.cat([x_t, state], dim=-1)
        acceleration = self.accel_proj(combined)

        # Адаптивные коэффициенты
        alpha = self.alpha_gate(state)   # насколько сохраняем velocity
        beta = self.beta_gate(x_t)       # насколько применяем acceleration

        # Обновление velocity (momentum)
        new_velocity = alpha * velocity + beta * acceleration
        new_velocity = self.norm_vel(new_velocity)

        # Gate для обновления state
        sg = self.state_gate(torch.cat([state, new_velocity], dim=-1))
        new_state = self.norm_state(state + sg * new_velocity)

        # Выход — комбинация state и velocity
        out = self.out_proj(torch.cat([new_state, new_velocity], dim=-1))
        return out, new_state, new_velocity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape
        state = torch.zeros(B, D, device=x.device)
        velocity = torch.zeros(B, D, device=x.device)

        outputs = []
        for t in range(T):
            out, state, velocity = self.step(x[:, t, :], state, velocity)
            outputs.append(out)

        out_seq = torch.stack(outputs, dim=1)
        return self.norm_out(x + self.dropout(out_seq))


class DeltaFlowBlock(nn.Module):
    """Блок с двунаправленным momentum и FFN"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.forward_cell = MomentumCell(d_model, dropout)
        # Обратный проход — понимает будущий контекст через реверс
        self.backward_cell = MomentumCell(d_model, dropout)

        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_cell(x)
        bwd = self.backward_cell(x.flip(1)).flip(1)
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        return merged + self.ffn(merged)


class DeltaFlowArchitecture(BaseArchitecture):
    """
    DeltaFlow: физический momentum в нейросети.
    Хранит скорость и ускорение изменений — понимает тренды, не только значения.
    O(1) per token, бесконечный контекст.
    """

    def __init__(self, config: Dict[str, Any], name: str = "DeltaFlow"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.d_ff = config.get('d_ff', 512)
        self.dropout = config.get('dropout', 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList([
            DeltaFlowBlock(self.d_model, self.d_ff, self.dropout)
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
            'type': 'DeltaFlow',
            'complexity': 'O(1) per token — recurrent momentum dynamics',
            'context': 'infinite — momentum accumulates indefinitely',
            'innovation': 'stores velocity+acceleration of state, not just state',
            'description': 'Physics-inspired momentum architecture: position + velocity + acceleration'
        }
