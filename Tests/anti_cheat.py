import hashlib
import inspect
import re
from typing import Iterable, Tuple

import torch


def _stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{name}:{base_seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def token_permutation(
    vocab_size: int,
    name: str,
    base_seed: int = 1337,
    protected_tokens: Iterable[int] = (),
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(name, base_seed))
    perm = torch.randperm(vocab_size, generator=generator)

    for token in protected_tokens:
        token = int(token)
        if token < 0 or token >= vocab_size:
            continue
        idx = int((perm == token).nonzero(as_tuple=False)[0].item())
        old = int(perm[token].item())
        perm[token] = token
        perm[idx] = old

    return perm


def remap_tokens(
    tensors: Tuple[torch.Tensor, ...],
    vocab_size: int,
    name: str,
    base_seed: int = 1337,
    protected_tokens: Iterable[int] = (),
) -> Tuple[torch.Tensor, ...]:
    perm = token_permutation(vocab_size, name, base_seed, protected_tokens)
    return tuple(perm.to(t.device)[t] for t in tensors)


def audit_architecture_class(architecture_class, allowlisted: bool = False) -> dict:
    """Static heuristic audit for explicit benchmark-solving logic."""
    module = inspect.getmodule(architecture_class)
    source = ""
    if module is not None and getattr(module, "__file__", None):
        try:
            with open(module.__file__, "r", encoding="utf-8") as handle:
                source = handle.read()
        except OSError:
            source = inspect.getsource(architecture_class)
    else:
        source = inspect.getsource(architecture_class)

    checks = {
        "suffix_pattern_matching": r"hist\s*==\s*pattern|candidate\s*=\s*x\[:,\s*i\s*\+\s*1\]",
        "sentinel_repair": r"sentinel_tokens\s*=|sentinel_strength\s*=|def\s+_sentinel_repair",
        "hardcoded_rule_strength": r"scatter_add_|def\s+_numeric_solver_logits|geometric_mask|arithmetic_context",
        "seed_program_solver": r"self\.seed_strength|self\.seed_program|self\.factor_seed_head|self\.seed_memory",
        "benchmark_token_bands": r"x\s*>?=\s*240|x\s*<=\s*200|rare_tokens|common_tokens",
    }
    findings = [
        {"name": name, "pattern": pattern}
        for name, pattern in checks.items()
        if re.search(pattern, source)
    ]

    return {
        "passed": allowlisted or not findings,
        "allowlisted": allowlisted,
        "findings": findings,
        "source_file": getattr(module, "__file__", None) if module is not None else None,
    }
