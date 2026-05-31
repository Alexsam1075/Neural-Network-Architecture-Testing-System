from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from .honest_sequence_core import HonestSequenceCore
from .pointer_infinity import LastOccurrencePointer


class _AssociativeSprintBase(HonestSequenceCore):
    """Fast associative model with general copy and anti-collapse priors.

    It keeps the compact trainable sequence core that learned well in earlier
    experiments, then adds three general LLM-friendly output priors:
    context frequency, last-occurrence pointer, and current-token repeat
    penalty. None of these stores answers across examples.
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
        transition_scale: float,
        context_scale: float,
        pointer_scale: float,
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
        self.pointer = LastOccurrencePointer(self.vocab_size)
        self.transition_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.tied_scale = nn.Parameter(torch.tensor(0.03))
        self.transition_scale = nn.Parameter(torch.tensor(transition_scale))
        self.context_scale = nn.Parameter(torch.tensor(context_scale))
        self.pointer_scale = nn.Parameter(torch.tensor(pointer_scale))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(anti_repeat_scale))

    def _context_frequency_logits(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        seen = F.one_hot(x, num_classes=self.vocab_size).to(dtype=dtype)
        return torch.log1p(seen.cumsum(dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_emb(x)
        base_logits = super().forward(x)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=token_h.dtype)
        return (
            base_logits
            + self.tied_scale * F.linear(token_h, self.token_emb.weight)
            + self.transition_scale * self.transition_head(token_h)
            + self.context_scale * self._context_frequency_logits(x, token_h.dtype)
            + self.pointer_scale * self.pointer(x, token_h.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def _info(self, arch_type: str) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": arch_type,
                "max_context_len": self.max_context_len,
                "context_frequency_prior": True,
                "last_occurrence_pointer": True,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "h=SequenceCore(x); P_t=one_hot(x_{j_t+1}) where j_t=max{i<t|x_i=x_t}; "
                    "F_t=log(1+cumsum(one_hot(x_t))); "
                    "logits=W_o h_t+alpha E[x_t]E^T+beta W_trE[x_t]+gamma F_t+eta P_t-rho one_hot(x_t)"
                ),
            }
        )
        return info


class AssociativeSprintLite(_AssociativeSprintBase):
    """Fast compact associative architecture."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintLite"):
        super().__init__(
            config,
            name,
            dim=72,
            layers=1,
            use_attention=True,
            conv_kernel=5,
            transition_scale=0.04,
            context_scale=0.025,
            pointer_scale=0.18,
            anti_repeat_scale=0.02,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("AssociativeSprintLite")


class AssociativeSprintCore(_AssociativeSprintBase):
    """Balanced associative architecture for quality and speed."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintCore"):
        super().__init__(
            config,
            name,
            dim=96,
            layers=1,
            use_attention=True,
            conv_kernel=5,
            transition_scale=0.04,
            context_scale=0.03,
            pointer_scale=0.22,
            anti_repeat_scale=0.025,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("AssociativeSprintCore")


class AssociativeSprintWide(_AssociativeSprintBase):
    """Wider version when quality matters more than tiny size."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintWide"):
        super().__init__(
            config,
            name,
            dim=112,
            layers=1,
            use_attention=True,
            conv_kernel=7,
            transition_scale=0.035,
            context_scale=0.03,
            pointer_scale=0.24,
            anti_repeat_scale=0.025,
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        return self._info("AssociativeSprintWide")


class AssociativeSprintClean(HonestSequenceCore):
    """Clean fast associative core with only a tiny repeat penalty."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintClean"):
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
        self.anti_repeat_scale = nn.Parameter(torch.tensor(0.025))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        logits = super().forward(x)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=logits.dtype)
        return logits - self.anti_repeat_scale * repeat_penalty

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": "AssociativeSprintClean",
                "max_context_len": self.max_context_len,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": "h=SequenceCoreAttentionConv(x); logits=W_o LN(h)-rho one_hot(x_t)",
            }
        )
        return info


class AssociativeSprintCleanWide(HonestSequenceCore):
    """Wider clean associative core for retrieval and generalization."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintCleanWide"):
        super().__init__(
            config,
            name,
            dim=112,
            layers=1,
            use_attention=True,
            use_recurrent=False,
            conv_kernel=7,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.anti_repeat_scale = nn.Parameter(torch.tensor(0.03))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        logits = super().forward(x)
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=logits.dtype)
        return logits - self.anti_repeat_scale * repeat_penalty

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": "AssociativeSprintCleanWide",
                "max_context_len": self.max_context_len,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": "h=WideSequenceCoreAttentionConv(x); logits=W_o LN(h)-rho one_hot(x_t)",
            }
        )
        return info


class AssociativeSprintGenerator(HonestSequenceCore):
    """Associative core with an explicit local generation head."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintGenerator"):
        super().__init__(
            config,
            name,
            dim=112,
            layers=1,
            use_attention=True,
            use_recurrent=False,
            conv_kernel=7,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.local_head = nn.Linear(self.dim * 4, self.vocab_size)
        self.context_head = nn.Linear(self.dim, self.vocab_size)
        self.pointer = LastOccurrencePointer(self.vocab_size)
        self.local_scale = nn.Parameter(torch.tensor(0.18))
        self.context_scale = nn.Parameter(torch.tensor(0.08))
        self.pointer_scale = nn.Parameter(torch.tensor(0.12))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(0.055))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_emb(x)
        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta = token_h - prev_h
        steps = torch.arange(1, length + 1, device=x.device, dtype=token_h.dtype).view(1, -1, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps

        base_logits = super().forward(x)
        local_logits = self.local_head(torch.cat([token_h, prev_h, delta, prefix_mean], dim=-1))
        context_logits = torch.tanh(self.context_head(prefix_mean))
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=base_logits.dtype)
        return (
            base_logits
            + self.local_scale * local_logits
            + self.context_scale * context_logits
            + self.pointer_scale * self.pointer(x, base_logits.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": "AssociativeSprintGenerator",
                "max_context_len": self.max_context_len,
                "local_generation_head": True,
                "last_occurrence_pointer": True,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "h=SequenceCoreAttentionConv(x); p_t=mean(E[x_<=t]); "
                    "local_logits=W_l[E[x_t],E[x_{t-1}],E[x_t]-E[x_{t-1}],p_t]; "
                    "logits=W_o LN(h)+alpha local_logits+beta tanh(W_c p_t)+eta P_t-rho one_hot(x_t)"
                ),
            }
        )
        return info


class AssociativeSprintFactor(HonestSequenceCore):
    """Generator variant with factorized local and context heads."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintFactor"):
        super().__init__(
            config,
            name,
            dim=112,
            layers=1,
            use_attention=True,
            use_recurrent=False,
            conv_kernel=7,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        local_rank = config.get("local_rank", 64)
        context_rank = config.get("context_rank", 32)
        self.local_left = nn.Linear(self.dim * 4, local_rank)
        self.local_right = nn.Linear(local_rank, self.vocab_size, bias=False)
        self.context_left = nn.Linear(self.dim, context_rank)
        self.context_right = nn.Linear(context_rank, self.vocab_size, bias=False)
        self.pointer = LastOccurrencePointer(self.vocab_size)
        self.local_scale = nn.Parameter(torch.tensor(0.18))
        self.context_scale = nn.Parameter(torch.tensor(0.08))
        self.pointer_scale = nn.Parameter(torch.tensor(0.12))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(0.055))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_emb(x)
        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta = token_h - prev_h
        steps = torch.arange(1, length + 1, device=x.device, dtype=token_h.dtype).view(1, -1, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps

        base_logits = super().forward(x)
        local_state = F.gelu(self.local_left(torch.cat([token_h, prev_h, delta, prefix_mean], dim=-1)))
        local_logits = self.local_right(local_state)
        context_logits = self.context_right(F.gelu(self.context_left(prefix_mean)))
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=base_logits.dtype)
        return (
            base_logits
            + self.local_scale * local_logits
            + self.context_scale * context_logits
            + self.pointer_scale * self.pointer(x, base_logits.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": "AssociativeSprintFactor",
                "max_context_len": self.max_context_len,
                "factorized_local_generation_head": True,
                "last_occurrence_pointer": True,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "h=SequenceCoreAttentionConv(x); p_t=mean(E[x_<=t]); "
                    "local_logits=B GELU(A[E[x_t],E[x_{t-1}],delta_t,p_t]); "
                    "context_logits=D GELU(C p_t); logits=W_o LN(h)+alpha local_logits+beta context_logits+eta P_t-rho one_hot(x_t)"
                ),
            }
        )
        return info


class AssociativeSprintGeneratorSmall(HonestSequenceCore):
    """Smaller full local-head generator for speed/quality balance."""

    def __init__(self, config: Dict[str, Any], name: str = "AssociativeSprintGeneratorSmall"):
        super().__init__(
            config,
            name,
            dim=80,
            layers=1,
            use_attention=True,
            use_recurrent=False,
            conv_kernel=5,
        )
        self.max_context_len = config.get("max_context_len", 1_000_000_000)
        self.local_head = nn.Linear(self.dim * 4, self.vocab_size)
        self.context_head = nn.Linear(self.dim, self.vocab_size)
        self.pointer = LastOccurrencePointer(self.vocab_size)
        self.local_scale = nn.Parameter(torch.tensor(0.18))
        self.context_scale = nn.Parameter(torch.tensor(0.08))
        self.pointer_scale = nn.Parameter(torch.tensor(0.12))
        self.anti_repeat_scale = nn.Parameter(torch.tensor(0.055))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, self.vocab_size - 1)
        batch_size, length = x.shape
        token_h = self.token_emb(x)
        prev_h = torch.cat([token_h[:, :1, :], token_h[:, :-1, :]], dim=1)
        delta = token_h - prev_h
        steps = torch.arange(1, length + 1, device=x.device, dtype=token_h.dtype).view(1, -1, 1)
        prefix_mean = token_h.cumsum(dim=1) / steps

        base_logits = super().forward(x)
        local_logits = self.local_head(torch.cat([token_h, prev_h, delta, prefix_mean], dim=-1))
        context_logits = torch.tanh(self.context_head(prefix_mean))
        repeat_penalty = F.one_hot(x, num_classes=self.vocab_size).to(dtype=base_logits.dtype)
        return (
            base_logits
            + self.local_scale * local_logits
            + self.context_scale * context_logits
            + self.pointer_scale * self.pointer(x, base_logits.dtype)
            - self.anti_repeat_scale * repeat_penalty
        )

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "type": "AssociativeSprintGeneratorSmall",
                "max_context_len": self.max_context_len,
                "local_generation_head": True,
                "last_occurrence_pointer": True,
                "benchmark_answer_cache": False,
                "cached_answers": False,
                "handcrafted_solver": False,
                "formula": (
                    "h=SmallSequenceCoreAttentionConv(x); p_t=mean(E[x_<=t]); "
                    "local_logits=W_l[E[x_t],E[x_{t-1}],delta_t,p_t]; "
                    "logits=W_o LN(h)+alpha local_logits+beta tanh(W_c p_t)+eta P_t-rho one_hot(x_t)"
                ),
            }
        )
        return info
