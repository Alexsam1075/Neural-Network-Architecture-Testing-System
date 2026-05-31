from typing import Any, Dict

from .honest_sequence_core import HonestSequenceCore


class _VectorPatternCore(HonestSequenceCore):
    """Backward-compatible honest replacement for former vector pattern solvers."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int,
        periods=(),
        neural: bool = True,
        strength: float = 0.0,
        recurrence_strength: float = 0.0,
        procedural_strength: float = 0.0,
    ):
        super().__init__(
            config,
            name,
            dim=dim,
            layers=max(1, config.get("num_layers", 1)),
            use_attention=neural and dim >= 40,
            use_recurrent=neural,
            conv_kernel=5,
        )
        self.periods = tuple(periods)
        self.neural = neural

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "periods": self.periods,
                "neural_path": self.neural,
                "handcrafted_pattern_logits": False,
                "hypothesis": "honest trainable sequence model; vectorized pattern logits removed",
            }
        )
        return info


class VectorPatternSprint(_VectorPatternCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "VectorPatternSprint", dim=24, periods=(1, 2, 3, 4, 5, 8), neural=True)


class SentinelDeltaNet(_VectorPatternCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "SentinelDeltaNet", dim=32, periods=(1, 2, 4, 8), neural=True)


class PeriodicMemoryEngine(_VectorPatternCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "PeriodicMemoryEngine", dim=32, periods=(1, 2, 3, 4, 5, 6, 7, 8, 11), neural=True)


class ProceduralRecallNet(_VectorPatternCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ProceduralRecallNet", dim=48, periods=(1, 2, 3, 5, 8, 11), neural=True)


class OmniPatternKernel(_VectorPatternCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "OmniPatternKernel", dim=40, periods=(1, 2, 3, 4, 5, 6, 7, 8, 11), neural=True)
