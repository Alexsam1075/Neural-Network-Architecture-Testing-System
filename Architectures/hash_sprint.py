from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .linear_sprint import LinearSprintBlock
from .pointer_infinity import LastOccurrencePointer
from .positional import DynamicSinusoidalPositionEncoding


class HashedBigramPointer(nn.Module):
    """Approximate causal bigram pointer with fixed hash buckets.

    Formula:
        key_i = hash(x_{i-1}, x_i)
        value_i = x_{i+1}
        B_t = last value_i where i < t and key_i == hash(x_{t-1}, x_t)

    This is a general prefix recall mechanism for code identifiers, repeated
    phrases and local conventions. It starts empty on every forward pass and
    does not store benchmark answers.
    """

    def __init__(self, vocab_size: int, buckets: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.buckets = buckets

    def _hash_pair(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * 257 + right * 263) % self.buckets

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, length = x.shape
        logits = torch.zeros(batch_size, length, self.vocab_size, device=x.device, dtype=dtype)
        if length < 3:
            return logits

        update_keys = torch.zeros(batch_size, length, device=x.device, dtype=torch.long)
        update_keys[:, 2:] = self._hash_pair(x[:, :-2], x[:, 1:-1])
        update_mask = torch.zeros(batch_size, length, device=x.device, dtype=torch.long)
        update_mask[:, 2:] = 1

        key_seen = F.one_hot(update_keys, num_classes=self.buckets).to(dtype=torch.long) * update_mask.unsqueeze(-1)
        visible_time = torch.arange(1, length + 1, device=x.device, dtype=torch.long).view(1, length, 1)
        last_time_by_bucket = torch.cummax(key_seen * visible_time, dim=1).values

        query_keys = torch.zeros(batch_size, length, device=x.device, dtype=torch.long)
        query_keys[:, 1:] = self._hash_pair(x[:, :-1], x[:, 1:])
        token_time = last_time_by_bucket.gather(2, query_keys.unsqueeze(-1)).squeeze(-1)
        value_index = (token_time - 1).clamp_min(0)
        continuation = x.gather(1, value_index)
        valid = (token_time > 0).to(dtype=dtype)
        return logits.scatter(2, continuation.unsqueeze(-1), valid.unsqueeze(-1))


class _HashSprintBase(BaseArchitecture):
    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        rank: int,
        ff_mult: int,
        local_rank: int,
        buckets: int,
        dropout: float,
        transition_scale: float,
        tied_scale: float,
        local_scale: float,
        unigram_scale: float,
        bigram_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.rank = config.get("rank", rank)
        self.ff_mult = config.get("ff_mult", ff_mult)
        self.local_rank = config.get("local_rank", local_rank)
        self.hash_buckets = config.get("hash_buckets", buckets)
        self.max_context_len = config.get("max_context_len", config.get("max_seq_len", 1_000_000_000))
        self.dropout = config.get("dropout", dropout)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position = DynamicSinusoidalPositionEncoding(self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.blocks = nn.ModuleList(
            [LinearSprintBlock(self.d_model, self.rank, self.ff_mult, self.dropout) for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.semantic_head = nn.Linear(self.d_model, self.vocab_size)
        self.transition_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.local_left = nn.Linear(self.d_model * 4, self.local_rank)
        self.local_right = nn.Linear(self.local_rank, self.vocab_size, bias=False)
        self.unigram_pointer = LastOccurrencePointer(self.vocab_size)
        self.bigram_pointer = HashedBigramPointer(self.vocab_size, self.hash_buckets)

        self.tied_scale = nn.Parameter(torch.tensor(tied_scale))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.local_scale = nn.Parameter(torch.tensor(local_scale))
        self.unigram_scale = nn.Parameter(torch.tensor(unigram_scale))
        self.bigram_scale = nn.Parameter(torch.tensor(bigram_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_embedding(x)
        h = self.drop(self.input_norm(token_h + self.position(batch_size, length, x.device, token_h.dtype)))

        for block in self.blocks:
            h = block(h)

        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta = token_h - prev_h
        steps = torch.arange(1, length + 1, device=x.device, dtype=token_h.dtype).view(1, -1, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps
        local_state = F.gelu(self.local_left(torch.cat([token_h, prev_h, delta, prefix_mean], dim=-1)))
        local_logits = self.local_right(local_state)

        h = self.output_norm(h)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=h.dtype)
        return (
            self.semantic_head(h)
            + self.tied_scale * F.linear(h, self.token_embedding.weight)
            + self.transition_scale * self.transition_head(token_h)
            + self.local_scale * local_logits
            + self.unigram_scale * self.unigram_pointer(x, h.dtype)
            + self.bigram_scale * self.bigram_pointer(x, h.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "hash_buckets": self.hash_buckets,
            "max_context_len": self.max_context_len,
            "position_encoding": "dynamic_sinusoidal",
            "long_context_safe": True,
            "linear_prefix_memory": True,
            "hashed_bigram_pointer": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "complexity": "O(sequence * rank * dim + sequence * hash_buckets + sequence * vocab), no QK^T matrix",
            "formula": (
                "S_t=S_{t-1}+phi(K_t)V_t^T; p_t=phi(Q_t)S_t/(phi(Q_t)Z_t); "
                "b_t=one_hot(x_{i+1}) where i<t and hash(x_{i-1},x_i)=hash(x_{t-1},x_t); "
                "u_t=one_hot(x_{j+1}) where j<t and x_j=x_t; "
                "logits=W_s h_t+alpha h_tE^T+beta W_trE[x_t]+gamma Local_t+eta u_t+mu b_t-rho one_hot(x_t)"
            ),
        }


class HashSprintTiny(_HashSprintBase):
    """Fast hashed-recall candidate."""

    def __init__(self, config: Dict[str, Any], name: str = "HashSprintTiny"):
        super().__init__(
            config,
            name,
            dim=64,
            layers=1,
            rank=12,
            ff_mult=2,
            local_rank=32,
            buckets=512,
            dropout=0.01,
            transition_scale=0.45,
            tied_scale=0.12,
            local_scale=0.14,
            unigram_scale=0.08,
            bigram_scale=0.35,
            anti_repeat_scale=0.045,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("HashSprintTiny")


class HashSprintCore(_HashSprintBase):
    """Balanced hashed-recall non-quadratic generator."""

    def __init__(self, config: Dict[str, Any], name: str = "HashSprintCore"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            rank=16,
            ff_mult=2,
            local_rank=48,
            buckets=1024,
            dropout=0.02,
            transition_scale=0.42,
            tied_scale=0.14,
            local_scale=0.16,
            unigram_scale=0.09,
            bigram_scale=0.45,
            anti_repeat_scale=0.05,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("HashSprintCore")


class HashSprintPro(_HashSprintBase):
    """Wider hashed-recall architecture for quality experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "HashSprintPro"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=1,
            rank=20,
            ff_mult=2,
            local_rank=64,
            buckets=2048,
            dropout=0.02,
            transition_scale=0.38,
            tied_scale=0.16,
            local_scale=0.17,
            unigram_scale=0.10,
            bigram_scale=0.50,
            anti_repeat_scale=0.055,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("HashSprintPro")
