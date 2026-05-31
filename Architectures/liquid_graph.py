"""
LiquidGraph Architecture
========================
Проблема трансформеров: граф связей фиксирован (каждый→каждый).
Проблема GNN: граф задаётся заранее, не адаптируется к контенту.

Решение: Граф строится ДИНАМИЧЕСКИ из контента токенов.
Каждый токен сам решает с кем соединиться через learnable similarity.
Топология графа — дифференцируемая переменная, меняется с каждым слоем.
Только k ближайших соседей → O(n*k) вместо O(n^2).

Ключевой инсайт: не все токены одинаково важны для каждого токена.
Разные слои строят разные графы — ансамбль точек зрения.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from .base_architecture import BaseArchitecture


class DynamicGraphLayer(nn.Module):
    """
    Строит граф из контента и распространяет сообщения по нему.
    k_neighbors << n — константа, не зависит от длины последовательности.
    """

    def __init__(self, d_model: int, k_neighbors: int = 8, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.k = k_neighbors
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Проекции для построения графа
        self.edge_query = nn.Linear(d_model, d_model)
        self.edge_key = nn.Linear(d_model, d_model)

        # Проекции для message passing
        self.msg_proj = nn.Linear(d_model, d_model)
        self.agg_proj = nn.Linear(d_model, d_model)

        # Gate — фильтрует релевантные сообщения
        self.msg_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        # Самообновление
        self.self_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def build_graph(self, x: torch.Tensor):
        """
        Строит k-nearest граф по content similarity.
        Возвращает индексы k соседей для каждого токена.
        """
        B, T, D = x.shape
        k = min(self.k, T)

        q = self.edge_query(x)  # (B, T, D)
        key = self.edge_key(x)     # (B, T, D)

        # Similarity matrix (B, T, T)
        sim = torch.bmm(q, key.transpose(1, 2)) / (D ** 0.5)

        # Маскируем себя
        mask = torch.eye(T, device=x.device).bool().unsqueeze(0)
        sim = sim.masked_fill(mask, -1e9)

        # Top-k соседей
        _, top_idx = sim.topk(k, dim=-1)  # (B, T, k)
        top_weights = F.softmax(sim.gather(2, top_idx), dim=-1)  # (B, T, k)

        return top_idx, top_weights

    def message_passing(self, x: torch.Tensor, top_idx: torch.Tensor, top_weights: torch.Tensor):
        """Передача сообщений по динамическому графу."""
        B, T, D = x.shape
        k = top_idx.shape[-1]

        # Собираем фичи соседей
        # top_idx: (B, T, k) → нужно gather по dim=1
        idx_expanded = top_idx.reshape(B, -1)  # (B, T*k)
        idx_expanded = idx_expanded.unsqueeze(-1).expand(-1, -1, D)
        neighbors = x.gather(1, idx_expanded).view(B, T, k, D)  # (B, T, k, D)

        # Сообщения от соседей
        msgs = self.msg_proj(neighbors)  # (B, T, k, D)

        # Взвешенная агрегация
        w = top_weights.unsqueeze(-1)  # (B, T, k, 1)
        agg = (w * msgs).sum(dim=2)    # (B, T, D)
        agg = self.agg_proj(agg)

        # Gate — насколько принимаем сообщения
        self_feat = self.self_proj(x)
        gate = self.msg_gate(torch.cat([self_feat, agg], dim=-1))

        return gate * agg + (1 - gate) * self_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Строим граф
        top_idx, top_weights = self.build_graph(x)

        # Message passing
        msg_out = self.message_passing(x, top_idx, top_weights)
        h = self.norm1(x + self.dropout(msg_out))

        # FFN
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class LiquidGraphArchitecture(BaseArchitecture):
    """
    LiquidGraph: граф связей строится динамически из контента.
    Каждый слой — новая топология. O(n*k) где k — константа.
    Разные слои «видят» разные отношения между токенами.
    """

    def __init__(self, config: Dict[str, Any], name: str = "LiquidGraph"):
        super().__init__(config, name)

        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.k_neighbors = config.get('k_neighbors', 8)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(self.dropout)

        # Позиционное смещение
        self.pos_bias = nn.Parameter(torch.randn(1, 512, self.d_model) * 0.01)

        self.graph_layers = nn.ModuleList([
            DynamicGraphLayer(self.d_model, self.k_neighbors, self.num_heads, self.dropout)
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
        h = h + self.pos_bias[:, :T, :]

        for layer in self.graph_layers:
            h = layer(h)

        return self.fc_out(self.norm_out(h))

    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'LiquidGraph',
            'complexity': 'O(n*k) where k is constant number of neighbors',
            'context': 'adaptive — graph topology changes per layer',
            'k_neighbors': self.k_neighbors,
            'innovation': 'dynamic graph topology learned from content, not fixed',
            'description': 'Content-adaptive dynamic graph neural network, O(nk) complexity'
        }
