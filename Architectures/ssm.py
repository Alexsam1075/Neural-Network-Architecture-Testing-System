"""
State Space Model (SSM) Architecture
Эффективная архитектура на основе линейной системы динамики состояний
Вдохновлено Mamba/S4
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import math
from .base_architecture import BaseArchitecture
from .positional import DynamicSinusoidalPositionEncoding


class SSMLayer(nn.Module):
    """State Space Model Layer"""
    
    def __init__(self, d_model: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Input projection to state_dim
        self.input_proj = nn.Linear(d_model, state_dim)
        self.output_proj = nn.Linear(state_dim, d_model)
        
        # A matrix - state transition (state_dim x state_dim)
        # Initialized as random, typically kept as learnable
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        
        # B matrix - input-to-state (state_dim x d_model)
        self.B = nn.Parameter(torch.randn(state_dim, d_model) * 0.1)
        
        # C matrix - state-to-output (d_model x state_dim)
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        
        # D matrix - direct feedthrough (d_model x d_model)
        self.D = nn.Parameter(torch.zeros(d_model, d_model))
        
        # Delta parameter - discretization
        self.delta = nn.Parameter(torch.ones(state_dim) * 0.1)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # Sequential computation through sequence
        for t in range(seq_len):
            # Current input
            x_t = x_norm[:, t, :]  # (batch_size, d_model)
            
            # State update: h_{t+1} = A @ h_t + B @ x_t
            A_discrete = torch.eye(self.state_dim, device=x.device, dtype=x.dtype) + self.A * self.delta.unsqueeze(0)
            
            h = torch.matmul(h, A_discrete.transpose(-2, -1)) + torch.matmul(x_t.unsqueeze(1), self.B.transpose(-2, -1)).squeeze(1)
            
            # Output: y_t = C @ h_t + D @ x_t
            y_t = torch.matmul(h.unsqueeze(1), self.C.transpose(-2, -1)).squeeze(1) + torch.matmul(x_t.unsqueeze(1), self.D.transpose(-2, -1)).squeeze(1)
            
            outputs.append(y_t)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        
        return output


class SSMBlock(nn.Module):
    """SSM Block with residual connection and feed-forward"""
    
    def __init__(self, d_model: int, state_dim: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.ssm = SSMLayer(d_model, state_dim, dropout)
        
        # Feed-forward network
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM with residual
        ssm_out = self.ssm(x)
        x = self.norm1(x + self.dropout(ssm_out))
        
        # Feed-forward with residual
        ff_out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SSMArchitecture(BaseArchitecture):
    """
    State Space Model Architecture
    
    Уникальная архитектура на основе линейной динамики состояний.
    Более эффективная чем attention, особенно для длинных последовательностей.
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "SSM"):
        super().__init__(config, name)
        
        self.vocab_size = config.get('vocab_size', 256)
        self.d_model = config.get('d_model', 128)
        self.state_dim = config.get('state_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.d_ff = config.get('d_ff', 512)
        self.max_seq_len = config.get('max_seq_len', config.get('max_context_len', 128))
        self.max_context_len = config.get('max_context_len', self.max_seq_len)
        self.dropout = config.get('dropout', 0.1)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = DynamicSinusoidalPositionEncoding(self.d_model)
        
        # SSM layers
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(self.d_model, self.state_dim, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
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
        pos_emb = self.positional_encoding(batch_size, seq_len, x.device, token_emb.dtype)
        
        x = self.dropout_layer(token_emb + pos_emb)
        
        # SSM blocks
        for block in self.ssm_blocks:
            x = block(x)
        
        # Output layer
        logits = self.fc_out(x)
        
        return logits
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'SSM',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'state_dim': self.state_dim,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'max_context_len': self.max_context_len,
            'position_encoding': 'dynamic_sinusoidal',
            'dropout': self.dropout,
            'description': 'State Space Model - linear state transition architecture'
        }
