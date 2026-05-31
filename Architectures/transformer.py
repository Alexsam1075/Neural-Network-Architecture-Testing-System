"""
Transformer Architecture
Классическая архитектура на основе self-attention механизма
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import math
from .base_architecture import BaseArchitecture


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
        batch_size = Q.shape[0]
        
        # Linear transformations
        Q = self.W_q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        
        # Final linear layer
        out = self.W_o(out)
        
        return out


class FeedForward(nn.Module):
    """Feed Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer Block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Self-attention with residual connection
        attention_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        # Feed forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerArchitecture(BaseArchitecture):
    """
    Transformer Architecture
    
    Уникальная архитектура на основе self-attention.
    Параметры можно полностью кастомизировать.
    """
    
    def __init__(self, config: Dict[str, any], name: str = "Transformer"):
        super().__init__(config, name)
        
        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.num_layers = config.get('num_layers', 2)
        self.num_heads = config.get('num_heads', 4)
        self.d_ff = config.get('d_ff', 512)
        self.max_seq_len = config.get('max_seq_len', 128)
        self.dropout = config.get('dropout', 0.1)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) - token indices
            
        Returns:
            (batch_size, seq_len, vocab_size) - logits
        """
        batch_size, seq_len = x.shape
        
        # Обрезаем индексы до vocab_size
        x = torch.clamp(x, min=0, max=self.vocab_size - 1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.positional_embedding(pos_ids)
        
        x = self.dropout_layer(token_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Output layer
        logits = self.fc_out(x)
        
        return logits
    
    def get_architecture_info(self) -> Dict[str, any]:
        return {
            'type': 'Transformer',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'description': 'Classic Transformer with Multi-Head Self-Attention'
        }
