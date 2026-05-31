from typing import Any, Dict

from .honest_sequence_core import HonestSequenceCore


class _AssociativeInductionCore(HonestSequenceCore):
    """Backward-compatible honest replacement for associative induction solvers."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int = 96,
        memory_dim: int = 64,
        periods=(),
        use_attention: bool = False,
        use_conv: bool = True,
        **_removed_kwargs: Any,
    ):
        super().__init__(
            config,
            name,
            dim=dim,
            layers=max(1, config.get("num_layers", 1)),
            use_attention=use_attention,
            use_recurrent=not use_conv,
            conv_kernel=5,
        )
        self.memory_dim = memory_dim
        self.periods = tuple(periods)
        self.compat_removed_biases = {
            "copy": 0.0,
            "pair": 0.0,
            "seed": 0.0,
            "recall": 0.0,
        }

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "memory_dim": self.memory_dim,
                "periods": self.periods,
                "copy_induction_logits": False,
                "pair_memory_logits": False,
                "seed_program_logits": False,
                "numeric_solver_logits": False,
                "hypothesis": "honest trainable sequence model; handcrafted induction solvers removed",
            }
        )
        return info


class InductionTiny(_AssociativeInductionCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "InductionTiny", dim=48, memory_dim=32)


class PairTransitionNet(_AssociativeInductionCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "PairTransitionNet", dim=64, memory_dim=64)


class SeedProgrammer(_AssociativeInductionCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "SeedProgrammer", dim=72, memory_dim=96)


class CopyInductionHybrid(_AssociativeInductionCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "CopyInductionHybrid", dim=96, memory_dim=64, use_attention=True)


class MetaRouteKernel(_AssociativeInductionCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MetaRouteKernel", dim=112, memory_dim=96, use_attention=True)


class _NumericalInductionSolver(_AssociativeInductionCore):
    """Compatibility shim with no explicit numerical solver."""


class ApexInductionSolver(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexInductionSolver", dim=96, memory_dim=96)


class ApexInductionSolverFast(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexInductionSolverFast", dim=64, memory_dim=64)


class ApexSeedSolver(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolver", dim=72, memory_dim=96)


class ApexSeedSolverTurbo(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolverTurbo", dim=56, memory_dim=72, use_conv=False)


class ApexSeedSolverFlash(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolverFlash", dim=72, memory_dim=96, use_conv=False)


class ApexSeedSolverSwift(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolverSwift", dim=64, memory_dim=88, use_conv=False)


class ApexSeedSolverBlade(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolverBlade", dim=64, memory_dim=80)


class ApexSeedSolverCompact(_NumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolverCompact", dim=48, memory_dim=64)


class _GatedNumericalInductionSolver(_NumericalInductionSolver):
    """Compatibility shim with no input-content gate."""


class ApexSeedSolverGated(_GatedNumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexSeedSolverGated", dim=72, memory_dim=96)


class ApexFactorSeedSolver(_GatedNumericalInductionSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexFactorSeedSolver", dim=72, memory_dim=64)
        self.factor_dim = 64

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info["factor_dim"] = self.factor_dim
        info["long_context_safe"] = False
        info["long_context_note"] = "Uses bounded learned sequence processing; no estimated ultra-long-context credit."
        return info


class ApexFactorSeedSolverPlus(ApexFactorSeedSolver):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "ApexFactorSeedSolverPlus"
        self.factor_dim = 80
