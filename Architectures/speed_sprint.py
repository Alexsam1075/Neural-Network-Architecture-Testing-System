from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .honest_sequence_core import HonestSequenceCore
from .positional import DynamicSinusoidalPositionEncoding


def _heads_for(dim: int) -> int:
    if dim % 4 == 0:
        return 4
    if dim % 2 == 0:
        return 2
    return 1


class _SpeedSprintSDPA(BaseArchitecture):
    """Lean transformer-like block using fused scaled dot-product attention."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        ff_mult: int,
        causal: bool,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_heads = config.get("num_heads", _heads_for(self.d_model))
        self.head_dim = self.d_model // self.num_heads
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.causal = causal

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = DynamicSinusoidalPositionEncoding(self.d_model)
        self.norm_attn = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.attn_out = nn.Linear(self.d_model, self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * ff_mult),
            nn.GELU(),
            nn.Linear(self.d_model * ff_mult, self.d_model),
        )
        self.norm_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)

        qkv = self.qkv(self.norm_attn(h))
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=self.causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        h = h + self.attn_out(attn)
        h = h + self.ffn(self.norm_ffn(h))
        return self.head(self.norm_out(h))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "max_context_len": self.max_context_len,
            "attention_kernel": "torch_scaled_dot_product_attention",
            "causal_attention": self.causal,
            "conv_path": False,
            "num_layers": 1,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "formula": (
                "e_t=E[x_t]+pos(t); "
                "Q,K,V=split(W_qkv LN(e)); "
                "a=SDPA(Q,K,V); "
                "h=e+W_o a+FFN(LN(e+W_o a)); "
                "logits=W_vocab LN(h)"
            ),
            "hypothesis": (
                "a single fused-attention block should preserve the strongest "
                "Transformer behavior with far fewer parameters and less latency"
            ),
        }


class SpeedSprintSDPALite(_SpeedSprintSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPALite"):
        super().__init__(config, name, dim=80, ff_mult=2, causal=False)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPALite")


class SpeedSprintSDPA(_SpeedSprintSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPA"):
        super().__init__(config, name, dim=96, ff_mult=2, causal=False)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPA")


class SpeedSprintSDPAMid(_SpeedSprintSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPAMid"):
        super().__init__(config, name, dim=88, ff_mult=2, causal=False)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPAMid")


class SpeedSprintSDPAFastFFN(_SpeedSprintSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPAFastFFN"):
        super().__init__(config, name, dim=96, ff_mult=1, causal=False)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPAFastFFN")


class SpeedSprintSDPAWide(_SpeedSprintSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPAWide"):
        super().__init__(config, name, dim=112, ff_mult=2, causal=False)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPAWide")


class _SpeedSprintSDPALocal(_SpeedSprintSDPA):
    """Fused SDPA with a factorized local-generation output head."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        local_rank: int,
        include_prefix_mean: bool,
        local_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(config, name, dim=dim, ff_mult=2, causal=False)
        self.local_rank = config.get("local_rank", local_rank)
        self.include_prefix_mean = include_prefix_mean
        feature_dim = self.d_model * (3 if include_prefix_mean else 2)
        self.local_left = nn.Linear(feature_dim, self.local_rank)
        self.local_right = nn.Linear(self.local_rank, self.vocab_size, bias=False)
        self.local_scale = nn.Parameter(torch.tensor(local_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))

    def _local_features(self, token_h: torch.Tensor) -> torch.Tensor:
        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta_h = token_h - prev_h
        if not self.include_prefix_mean:
            return torch.cat([token_h, delta_h], dim=-1)

        length = token_h.shape[1]
        steps = torch.arange(
            1,
            length + 1,
            device=token_h.device,
            dtype=token_h.dtype,
        ).view(1, length, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps
        return torch.cat([token_h, delta_h, prefix_mean], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_emb(x)
        h = token_h + self.pos_encoding(batch_size, length, x.device, token_h.dtype)

        qkv = self.qkv(self.norm_attn(h))
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=self.causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        h = h + self.attn_out(attn)
        h = h + self.ffn(self.norm_ffn(h))
        logits = self.head(self.norm_out(h))

        local_state = F.gelu(self.local_left(self._local_features(token_h)))
        logits = logits + self.local_scale * self.local_right(local_state)

        repeat_penalty = logits.new_zeros(*x.shape, self.vocab_size)
        repeat_penalty.scatter_(-1, x.unsqueeze(-1), 1.0)
        return logits - self.anti_repeat_scale * repeat_penalty

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super()._info(arch_type)
        info.update(
            {
                "factorized_local_generation_head": True,
                "local_rank": self.local_rank,
                "include_prefix_mean": self.include_prefix_mean,
                "formula": (
                    "e_t=E[x_t]+pos(t); Q,K,V=split(W_qkv LN(e)); "
                    "a=SDPA(Q,K,V); h=e+W_o a+FFN(LN(e+W_o a)); "
                    "z_t=GELU(A[E[x_t],E[x_t]-E[x_{t-1}]"
                    + (",mean(E[x_<=t])" if self.include_prefix_mean else "")
                    + "]); logits_t=W_vocabLN(h_t)+alpha Bz_t-rho one_hot(x_t)"
                ),
                "hypothesis": (
                    "keep the fused attention latency advantage while adding only a "
                    "low-rank local token-dynamics head for next-token and rare-token quality"
                ),
            }
        )
        return info


class SpeedSprintSDPALocalLite(_SpeedSprintSDPALocal):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPALocalLite"):
        super().__init__(
            config,
            name,
            dim=80,
            local_rank=12,
            include_prefix_mean=False,
            local_scale=0.07,
            anti_repeat_scale=0.025,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPALocalLite")


class SpeedSprintSDPALocal(_SpeedSprintSDPALocal):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPALocal"):
        super().__init__(
            config,
            name,
            dim=96,
            local_rank=16,
            include_prefix_mean=True,
            local_scale=0.065,
            anti_repeat_scale=0.025,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPALocal")


class _SpeedSprintFusedCore(BaseArchitecture):
    """Grouped-conv local mixer plus fused QKV attention.

    This is the fast rewrite of the honest sequence core:

        h0 = E[x] + pos
        h1 = h0 + W_c GroupConv(LN(h0))
        Q,K,V = split(W_qkv LN(h1))
        h2 = h1 + W_o SDPA(Q,K,V)
        h3 = h2 + W_2 GELU(W_1 LN(h2))
        logits = W_vocab LN(h3)

    The grouped convolution keeps the local inductive bias that helped the
    Pure variants, while fused QKV/SDPA removes part of the MultiheadAttention
    overhead.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        conv_kernel: int,
        ff_mult: int,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_heads = config.get("num_heads", _heads_for(self.d_model))
        self.head_dim = self.d_model // self.num_heads
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.conv_kernel = conv_kernel

        conv_groups = min(16, self.d_model)
        while self.d_model % conv_groups != 0:
            conv_groups -= 1
        self.conv_groups = conv_groups

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = DynamicSinusoidalPositionEncoding(self.d_model)
        self.norm_conv = nn.LayerNorm(self.d_model)
        self.conv = nn.Conv1d(
            self.d_model,
            self.d_model,
            conv_kernel,
            padding=conv_kernel // 2,
            groups=conv_groups,
        )
        self.conv_mix = nn.Linear(self.d_model, self.d_model)
        self.norm_attn = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, self.d_model * 3)
        self.attn_out = nn.Linear(self.d_model, self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * ff_mult),
            nn.GELU(),
            nn.Linear(self.d_model * ff_mult, self.d_model),
        )
        self.norm_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)

        conv = self.conv(self.norm_conv(h).transpose(1, 2)).transpose(1, 2)
        h = h + self.conv_mix(conv)

        qkv = self.qkv(self.norm_attn(h))
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        h = h + self.attn_out(attn)
        h = h + self.ffn(self.norm_ffn(h))
        return self.head(self.norm_out(h))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "conv_kernel": self.conv_kernel,
            "conv_groups": self.conv_groups,
            "max_context_len": self.max_context_len,
            "attention_kernel": "torch_scaled_dot_product_attention",
            "fused_qkv": True,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "formula": (
                "h0=E[x]+pos; h1=h0+W_c GroupConv(LN(h0)); "
                "Q,K,V=split(W_qkv LN(h1)); h2=h1+W_o SDPA(Q,K,V); "
                "h3=h2+W_2 GELU(W_1 LN(h2)); logits=W_vocab LN(h3)"
            ),
            "hypothesis": (
                "match the useful local-plus-associative bias of Pure while "
                "recovering speed through fused QKV and SDPA"
            ),
        }


class SpeedSprintFusedLite(_SpeedSprintFusedCore):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintFusedLite"):
        super().__init__(config, name, dim=80, conv_kernel=5, ff_mult=2)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintFusedLite")


class SpeedSprintFused(_SpeedSprintFusedCore):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintFused"):
        super().__init__(config, name, dim=96, conv_kernel=5, ff_mult=2)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintFused")


class _SpeedSprintStackedSDPA(BaseArchitecture):
    """Small multi-layer fused-attention model without convolution overhead."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        ff_mult: int,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.num_layers = config.get("num_layers", layers)
        self.num_heads = config.get("num_heads", _heads_for(self.d_model))
        self.head_dim = self.d_model // self.num_heads
        self.max_context_len = config.get("max_context_len", 1_000_000_000)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = DynamicSinusoidalPositionEncoding(self.d_model)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_attn": nn.LayerNorm(self.d_model),
                        "qkv": nn.Linear(self.d_model, self.d_model * 3),
                        "attn_out": nn.Linear(self.d_model, self.d_model),
                        "norm_ffn": nn.LayerNorm(self.d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(self.d_model, self.d_model * ff_mult),
                            nn.GELU(),
                            nn.Linear(self.d_model * ff_mult, self.d_model),
                        ),
                    }
                )
            )
        self.norm_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)

        for layer in self.layers:
            qkv = layer["qkv"](layer["norm_attn"](h))
            qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
            h = h + layer["attn_out"](attn)
            h = h + layer["ffn"](layer["norm_ffn"](h))

        return self.head(self.norm_out(h))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "num_layers": self.num_layers,
            "max_context_len": self.max_context_len,
            "attention_kernel": "torch_scaled_dot_product_attention",
            "fused_qkv": True,
            "conv_path": False,
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "formula": (
                "h0=E[x]+pos; for l in layers: "
                "Q,K,V=split(W_qkv^l LN(h)); "
                "h=h+W_o^l SDPA(Q,K,V)+FFN^l(LN(h)); "
                "logits=W_vocab LN(h)"
            ),
            "hypothesis": (
                "replace width and convolution with a second tiny fused-attention "
                "step to regain quality while staying below Transformer parameter count"
            ),
        }


class SpeedSprintStackedLite(_SpeedSprintStackedSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintStackedLite"):
        super().__init__(config, name, dim=80, layers=2, ff_mult=2)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintStackedLite")


class SpeedSprintStacked(_SpeedSprintStackedSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintStacked"):
        super().__init__(config, name, dim=96, layers=2, ff_mult=2)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintStacked")


def _shift_right(h: torch.Tensor, amount: int) -> torch.Tensor:
    if amount <= 0:
        return h
    return torch.cat([h[:, :1, :].expand(-1, amount, -1), h[:, :-amount, :]], dim=1)


class _SpeedSprintTokenMixer(BaseArchitecture):
    """Ultra-light non-attention token mixer for the speed floor.

    Formula:
        h_t = E[x_t] + pos(t)
        s_t = GELU(W_s [LN(h_t), LN(h_{t-1}), LN(h_{t-2}), LN(h_{t-4})])
        y_t = LN(h_t + alpha s_t)
        logits_t = W_vocab y_t

    This is not a benchmark solver and stores no answers. It is a tiny
    non-quadratic local sequence model that helps measure how much quality is
    possible before attention or recurrent state is added.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        shifts,
    ):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.shifts = tuple(shifts)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = DynamicSinusoidalPositionEncoding(self.d_model)
        self.norm_shift = nn.LayerNorm(self.d_model)
        self.shift_mix = nn.Linear(self.d_model * (1 + len(self.shifts)), self.d_model)
        self.mix_scale = nn.Parameter(torch.tensor(0.85))
        self.norm_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)
        n = self.norm_shift(h)
        local = self.shift_mix(torch.cat([n] + [_shift_right(n, s) for s in self.shifts], dim=-1))
        return self.head(self.norm_out(h + self.mix_scale * F.gelu(local)))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "d_model": self.d_model,
            "shifts": self.shifts,
            "max_context_len": self.max_context_len,
            "attention": False,
            "ffn_path": False,
            "complexity": "O(sequence * dim^2)",
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "formula": (
                "h_t=E[x_t]+pos(t); "
                "s_t=GELU(W_s[LN(h_t),LN(h_{t-1}),LN(h_{t-2}),LN(h_{t-4})]); "
                "logits_t=W_vocab LN(h_t+alpha s_t)"
            ),
            "hypothesis": (
                "establish a very fast non-quadratic local baseline for tiny devices "
                "before adding attention-like memory back"
            ),
        }


class SpeedSprintTokenMixerLite(_SpeedSprintTokenMixer):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintTokenMixerLite"):
        super().__init__(config, name, dim=64, shifts=(1, 2, 4))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintTokenMixerLite")


class SpeedSprintTokenMixer(_SpeedSprintTokenMixer):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintTokenMixer"):
        super().__init__(config, name, dim=96, shifts=(1, 2, 4))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintTokenMixer")


class _SpeedSprintPrefixMixer(BaseArchitecture):
    """Non-quadratic prefix-memory mixer.

    Formula:
        h_t = E[x_t] + pos(t)
        m_t = mean_{i<=t} LN(h_i)
        s_t = GELU(W_s [LN(h_t), LN(h_{t-1}), LN(h_{t-2}), m_t])
        logits_t = W_vocab LN(h_t + alpha s_t)

    It adds a streaming global context path to the local TokenMixer without
    adding attention, recurrent hidden state, or any cache of benchmark answers.
    """

    def __init__(self, config: Dict[str, Any], name: str, *, dim: int):
        super().__init__(config, name)
        self.vocab_size = config.get("vocab_size", 256)
        self.d_model = config.get("d_model", dim)
        self.max_context_len = config.get("max_context_len", 1_000_000_000)

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = DynamicSinusoidalPositionEncoding(self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.mix = nn.Linear(self.d_model * 4, self.d_model)
        self.mix_scale = nn.Parameter(torch.tensor(0.75))
        self.norm_out = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)
        n = self.norm(h)
        steps = torch.arange(1, length + 1, device=x.device, dtype=n.dtype).view(1, length, 1)
        prefix_mean = n.cumsum(dim=1) / steps
        mixed = self.mix(torch.cat([n, _shift_right(n, 1), _shift_right(n, 2), prefix_mean], dim=-1))
        return self.head(self.norm_out(h + self.mix_scale * F.gelu(mixed)))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        return {
            "type": arch_type,
            "d_model": self.d_model,
            "max_context_len": self.max_context_len,
            "attention": False,
            "streaming_prefix_mean": True,
            "complexity": "O(sequence * dim^2)",
            "benchmark_answer_cache": False,
            "cached_answers": False,
            "handcrafted_solver": False,
            "formula": (
                "h_t=E[x_t]+pos(t); m_t=mean_{i<=t}LN(h_i); "
                "s_t=GELU(W_s[LN(h_t),LN(h_{t-1}),LN(h_{t-2}),m_t]); "
                "logits_t=W_vocab LN(h_t+alpha s_t)"
            ),
            "hypothesis": (
                "prefix mean gives a cheap global context channel, improving generalization "
                "over local-only mixing without returning to quadratic attention"
            ),
        }


class SpeedSprintPrefixLite(_SpeedSprintPrefixMixer):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintPrefixLite"):
        super().__init__(config, name, dim=64)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintPrefixLite")


class SpeedSprintPrefix(_SpeedSprintPrefixMixer):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintPrefix"):
        super().__init__(config, name, dim=96)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintPrefix")


class _SpeedSprintShiftSDPA(_SpeedSprintSDPA):
    """Fused SDPA with dense shift-local mixing instead of Conv1d."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        shifts,
    ):
        super().__init__(config, name, dim=dim, ff_mult=2, causal=False)
        self.shifts = tuple(shifts)
        self.norm_shift = nn.LayerNorm(self.d_model)
        self.shift_mix = nn.Linear(self.d_model * (1 + len(self.shifts)), self.d_model)
        self.shift_scale = nn.Parameter(torch.tensor(0.75))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)

        n = self.norm_shift(h)
        local = self.shift_mix(torch.cat([n] + [_shift_right(n, s) for s in self.shifts], dim=-1))
        h = h + self.shift_scale * local

        qkv = self.qkv(self.norm_attn(h))
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=self.causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        h = h + self.attn_out(attn)
        h = h + self.ffn(self.norm_ffn(h))
        return self.head(self.norm_out(h))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super()._info(arch_type)
        info.update(
            {
                "shift_local_mixer": True,
                "shifts": self.shifts,
                "formula": (
                    "h0=E[x]+pos; s_t=W_s[LN(h_t),LN(h_{t-1}),LN(h_{t-2}),LN(h_{t-4})]; "
                    "Q,K,V=split(W_qkv LN(h0+lambda s)); "
                    "h=h0+lambda s+W_o SDPA(Q,K,V)+FFN(LN(h0+lambda s+W_o SDPA)); "
                    "logits=W_vocab LN(h)"
                ),
                "hypothesis": (
                    "dense shift mixing should recover local syntax with less overhead "
                    "than Conv1d and less output cost than a local vocab head"
                ),
            }
        )
        return info


class SpeedSprintShiftLite(_SpeedSprintShiftSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintShiftLite"):
        super().__init__(config, name, dim=80, shifts=(1, 2, 4))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintShiftLite")


class SpeedSprintShift(_SpeedSprintShiftSDPA):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintShift"):
        super().__init__(config, name, dim=96, shifts=(1, 2, 4))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintShift")


class _SpeedSprintShiftFast(_SpeedSprintShiftSDPA):
    """Shift-local plus SDPA, no FFN, for latency recovery."""

    def __init__(self, config: Dict[str, Any], name: str, *, dim: int, shifts):
        super().__init__(config, name, dim=dim, shifts=shifts)
        self.norm_ffn = nn.Identity()
        self.ffn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)

        n = self.norm_shift(h)
        local = self.shift_mix(torch.cat([n] + [_shift_right(n, s) for s in self.shifts], dim=-1))
        h = h + self.shift_scale * local

        qkv = self.qkv(self.norm_attn(h))
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=self.causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        h = h + self.attn_out(attn)
        return self.head(self.norm_out(h))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super()._info(arch_type)
        info.update(
            {
                "ffn_path": False,
                "formula": (
                    "h0=E[x]+pos; s_t=W_s[LN(h_t),LN(h_{t-1}),LN(h_{t-2}),LN(h_{t-4})]; "
                    "Q,K,V=split(W_qkv LN(h0+lambda s)); "
                    "h=h0+lambda s+W_o SDPA(Q,K,V); logits=W_vocab LN(h)"
                ),
                "hypothesis": (
                    "remove the FFN kernel from ShiftSDPA to test whether local "
                    "shift features plus attention are enough for a faster quality tradeoff"
                ),
            }
        )
        return info


class SpeedSprintShiftFast(_SpeedSprintShiftFast):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintShiftFast"):
        super().__init__(config, name, dim=96, shifts=(1, 2, 4))

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintShiftFast")


class _SpeedSprintSDPAConv(_SpeedSprintSDPA):
    """Fused SDPA plus a cheap depthwise local mixer."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        kernel_size: int,
    ):
        super().__init__(config, name, dim=dim, ff_mult=2, causal=False)
        self.kernel_size = kernel_size
        self.norm_conv = nn.LayerNorm(self.d_model)
        self.local_conv = nn.Conv1d(
            self.d_model,
            self.d_model,
            kernel_size,
            padding=kernel_size // 2,
            groups=self.d_model,
        )
        self.local_mix = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        h = self.token_emb(x) + self.pos_encoding(batch_size, length, x.device, self.token_emb.weight.dtype)

        local = self.local_conv(self.norm_conv(h).transpose(1, 2)).transpose(1, 2)
        h = h + self.local_mix(local)

        qkv = self.qkv(self.norm_attn(h))
        qkv = qkv.view(batch_size, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=self.causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, length, self.d_model)
        h = h + self.attn_out(attn)
        h = h + self.ffn(self.norm_ffn(h))
        return self.head(self.norm_out(h))

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super()._info(arch_type)
        info.update(
            {
                "conv_path": True,
                "conv_type": "depthwise_local_mixer",
                "kernel_size": self.kernel_size,
                "formula": (
                    "e_t=E[x_t]+pos(t); "
                    "c_t=W_c DWConv(LN(e))_t; "
                    "Q,K,V=split(W_qkv LN(e+c)); "
                    "a=SDPA(Q,K,V); "
                    "h=e+c+W_o a+FFN(LN(e+c+W_o a)); "
                    "logits=W_vocab LN(h)"
                ),
                "hypothesis": (
                    "restore local syntax and short-range transition quality while "
                    "keeping the fused attention path and low parameter count"
                ),
            }
        )
        return info


class SpeedSprintSDPAConvLite(_SpeedSprintSDPAConv):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPAConvLite"):
        super().__init__(config, name, dim=80, kernel_size=5)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPAConvLite")


class SpeedSprintSDPAConv(_SpeedSprintSDPAConv):
    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintSDPAConv"):
        super().__init__(config, name, dim=96, kernel_size=5)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintSDPAConv")


class _SpeedSprintPure(HonestSequenceCore):
    """Pure fast core used to isolate the cost of output-side additions."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        conv_kernel: int,
    ):
        super().__init__(
            config,
            name,
            dim=dim,
            layers=1,
            use_attention=True,
            use_recurrent=False,
            conv_kernel=conv_kernel,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": arch_type,
                "max_context_len": self.max_context_len,
                "speed_class": "pure_fast_associative",
                "factorized_local_generation_head": False,
                "last_occurrence_pointer": False,
                "vocab_transition_matrix": False,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "e_t=E[x_t]+pos(t); "
                    "c_t=DWConv(LN(e))_t; "
                    "a_t=MHA(LN(e+c))_t; "
                    "h_t=FFN(e_t+W_c c_t+a_t); "
                    "logits_t=W_o LN(h_t)"
                ),
                "hypothesis": (
                    "restore latency by removing all output priors, then use this "
                    "as the speed anchor for later quality additions"
                ),
            }
        )
        return info


class SpeedSprintPureLite(_SpeedSprintPure):
    """Small pure associative core for speed recovery."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintPureLite"):
        super().__init__(config, name, dim=80, conv_kernel=5)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintPureLite")


class SpeedSprintPure(_SpeedSprintPure):
    """Balanced pure associative core, close to the fastest strong baseline."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintPure"):
        super().__init__(config, name, dim=96, conv_kernel=5)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintPure")


class SpeedSprintPureWide(_SpeedSprintPure):
    """Wider pure core for testing whether width helps without latency-heavy priors."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintPureWide"):
        super().__init__(config, name, dim=112, conv_kernel=7)

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintPureWide")


class StreamingLastTokenPointer(nn.Module):
    """Linear-time continuation pointer for repeated tokens.

    It keeps, inside the current forward pass only, the latest observed
    continuation for each token. Unlike the older pointer implementation this
    does not build a sequence x vocab one-hot table to find the last occurrence.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch_size, length = x.shape
        logits = torch.zeros(batch_size, length, self.vocab_size, device=x.device, dtype=dtype)
        if length <= 1:
            return logits

        continuation = torch.zeros(batch_size, self.vocab_size, device=x.device, dtype=torch.long)
        valid = torch.zeros(batch_size, self.vocab_size, device=x.device, dtype=torch.bool)
        batch_idx = torch.arange(batch_size, device=x.device)
        one = torch.ones(batch_size, 1, device=x.device, dtype=torch.bool)

        for t in range(length):
            key = x[:, t].unsqueeze(1)
            pred = continuation.gather(1, key).squeeze(1)
            ok = valid.gather(1, key).squeeze(1)
            logits[batch_idx, t, pred] = ok.to(dtype)

            if t + 1 < length:
                continuation.scatter_(1, key, x[:, t + 1].unsqueeze(1))
                valid.scatter_(1, key, one)

        return logits


class _SpeedSprintMemory(HonestSequenceCore):
    """Fast core with optional cheap sequence-only memory priors."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        pointer_scale: float,
        context_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(
            config,
            name,
            dim=96,
            layers=1,
            use_attention=True,
            use_recurrent=False,
            conv_kernel=5,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.pointer = StreamingLastTokenPointer(self.vocab_size)
        self.use_pointer = pointer_scale != 0.0
        self.use_context = context_scale != 0.0
        self.pointer_scale = nn.Parameter(torch.tensor(pointer_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))

    def _context_frequency_logits(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seen = torch.zeros(*x.shape, self.vocab_size, device=x.device, dtype=dtype)
        seen.scatter_(-1, x.unsqueeze(-1), 1.0)
        return torch.log1p(seen.cumsum(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        logits = super().forward(x)
        repeat_penalty = logits.new_zeros(*x.shape, self.vocab_size)
        repeat_penalty.scatter_(-1, x.unsqueeze(-1), 1.0)
        if self.use_pointer:
            logits = logits + self.pointer_scale * self.pointer(x, logits.dtype)
        if self.use_context:
            logits = logits + self.context_scale * self._context_frequency_logits(x, logits.dtype)
        return logits - self.anti_repeat_scale * repeat_penalty

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": arch_type,
                "max_context_len": self.max_context_len,
                "speed_class": "fast_associative_stream_memory",
                "streaming_last_token_pointer": self.use_pointer,
                "context_frequency_prior": self.use_context,
                "vocab_transition_matrix": False,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "h_t=Core(x_<=t); "
                    "P_t=one_hot(x_{j+1}), j=max{i<t|x_i=x_t}; "
                    "F_t=log(1+cumsum(one_hot(x_t))); "
                    "logits_t=W_oLN(h_t)+eta P_t+gamma F_t-rho one_hot(x_t)"
                ),
                "hypothesis": (
                    "restore structured/retrieval behavior with linear streaming memory "
                    "while keeping the pure core as the main quality path"
                ),
            }
        )
        return info


class SpeedSprintPointer(_SpeedSprintMemory):
    """Pure core plus only the streaming continuation pointer."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintPointer"):
        super().__init__(
            config,
            name,
            pointer_scale=0.10,
            context_scale=0.0,
            anti_repeat_scale=0.025,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintPointer")


class SpeedSprintContext(_SpeedSprintMemory):
    """Pure core plus prefix-frequency memory, no pointer search."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintContext"):
        super().__init__(
            config,
            name,
            pointer_scale=0.0,
            context_scale=0.025,
            anti_repeat_scale=0.025,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintContext")


class SpeedSprintBalanced(_SpeedSprintMemory):
    """Speed-oriented balance of streaming pointer and prefix frequency."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintBalanced"):
        super().__init__(
            config,
            name,
            pointer_scale=0.08,
            context_scale=0.018,
            anti_repeat_scale=0.03,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintBalanced")


class _SpeedSprintBase(HonestSequenceCore):
    """Speed-first sequence architecture without benchmark-specific memory.

    The previous high-quality sprint models lost speed because they added
    last-occurrence pointers and large vocabulary transition paths. This family
    keeps the fast trainable conv/attention core and adds only cheap local
    token dynamics:

        h_t = Core(x_<=t)
        p_t = mean(E[x_<=t])                         optional
        z_t = GELU(A [E[x_t], E[x_t]-E[x_{t-1}], p_t])
        logits_t = W_o LN(h_t) + alpha B z_t + beta E[x_t] E^T
                   - rho one_hot(x_t)

    Every term is computed from the current sequence only. There is no answer
    cache, no benchmark rule table, and no vocab x vocab transition matrix.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        layers: int,
        use_attention: bool,
        conv_kernel: int,
        local_rank: int,
        include_prefix_mean: bool,
        local_scale: float,
        tied_scale: float,
        anti_repeat_scale: float,
    ):
        super().__init__(
            config,
            name,
            dim=dim,
            layers=layers,
            use_attention=use_attention,
            use_recurrent=False,
            conv_kernel=conv_kernel,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.include_prefix_mean = include_prefix_mean
        feature_dim = self.dim * (3 if include_prefix_mean else 2)
        self.local_left = nn.Linear(feature_dim, local_rank)
        self.local_right = nn.Linear(local_rank, self.vocab_size, bias=False)
        self.local_scale = nn.Parameter(torch.tensor(local_scale))
        self.tied_scale = nn.Parameter(torch.tensor(tied_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))

    def _local_features(self, token_h: torch.Tensor) -> torch.Tensor:
        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta_h = token_h - prev_h
        if not self.include_prefix_mean:
            return torch.cat([token_h, delta_h], dim=-1)

        length = token_h.shape[1]
        steps = torch.arange(
            1,
            length + 1,
            device=token_h.device,
            dtype=token_h.dtype,
        ).view(1, length, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps
        return torch.cat([token_h, delta_h, prefix_mean], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        token_h = self.token_emb(x)
        logits = super().forward(x)

        local_state = F.gelu(self.local_left(self._local_features(token_h)))
        local_logits = self.local_right(local_state)
        tied_logits = F.linear(F.layer_norm(token_h, (self.dim,)), self.token_emb.weight)

        repeat_penalty = logits.new_zeros(*x.shape, self.vocab_size)
        repeat_penalty.scatter_(-1, x.unsqueeze(-1), 1.0)
        return (
            logits
            + self.local_scale * local_logits
            + self.tied_scale * tied_logits
            - self.anti_repeat_scale * repeat_penalty
        )

    def _info(self, arch_type: str, speed_class: str) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": arch_type,
                "max_context_len": self.max_context_len,
                "speed_class": speed_class,
                "factorized_local_generation_head": True,
                "last_occurrence_pointer": False,
                "vocab_transition_matrix": False,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "h_t=Core(x_<=t); "
                    "z_t=GELU(A[E[x_t],E[x_t]-E[x_{t-1}]"
                    + (",mean(E[x_<=t])" if self.include_prefix_mean else "")
                    + "]); logits_t=W_oLN(h_t)+alpha Bz_t+beta LN(E[x_t])E^T-rho one_hot(x_t)"
                ),
                "hypothesis": (
                    "quality should come from the trainable sequence core, while "
                    "factorized local token dynamics improve rare-token and next-token "
                    "prediction without the heavy pointer paths that slowed earlier models"
                ),
            }
        )
        return info


class SpeedSprintTiny(_SpeedSprintBase):
    """Non-quadratic tiny variant for latency and micro-device experiments."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintTiny"):
        super().__init__(
            config,
            name,
            dim=64,
            layers=1,
            use_attention=False,
            conv_kernel=3,
            local_rank=config.get("local_rank", 12),
            include_prefix_mean=False,
            local_scale=0.08,
            tied_scale=0.015,
            anti_repeat_scale=0.035,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintTiny", "non_quadratic_tiny")


class SpeedSprintLite(_SpeedSprintBase):
    """Small non-quadratic variant with prefix statistics."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintLite"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            use_attention=False,
            conv_kernel=5,
            local_rank=config.get("local_rank", 16),
            include_prefix_mean=True,
            local_scale=0.09,
            tied_scale=0.018,
            anti_repeat_scale=0.04,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintLite", "non_quadratic_lite")


class SpeedSprintHybrid(_SpeedSprintBase):
    """Fast quality variant: one compact attention block, no heavy pointers."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintHybrid"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=1,
            use_attention=True,
            conv_kernel=5,
            local_rank=config.get("local_rank", 16),
            include_prefix_mean=True,
            local_scale=0.075,
            tied_scale=0.012,
            anti_repeat_scale=0.035,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintHybrid", "fast_hybrid")


class SpeedSprintWide(_SpeedSprintBase):
    """Wider quality probe that still avoids pointer and transition tables."""

    def __init__(self, config: Dict[str, Any], name: str = "SpeedSprintWide"):
        super().__init__(
            config,
            name,
            dim=112,
            layers=1,
            use_attention=True,
            conv_kernel=7,
            local_rank=config.get("local_rank", 20),
            include_prefix_mean=True,
            local_scale=0.07,
            tied_scale=0.01,
            anti_repeat_scale=0.035,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("SpeedSprintWide", "fast_wide_quality")
