from typing import Any, Dict

from .honest_sequence_core import HonestSequenceCore


class _RuleEnhancedCore(HonestSequenceCore):
    """Backward-compatible honest replacement for the former rule-logit core."""

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        *,
        dim: int = 96,
        layers: int = 2,
        rule_strength: float = 0.0,
        use_attention: bool = False,
        use_scan: bool = False,
    ):
        super().__init__(
            config,
            name,
            dim=dim,
            layers=layers,
            use_attention=use_attention,
            use_recurrent=use_scan,
            conv_kernel=5,
        )
        self.rule_strength = 0.0

    def get_architecture_info(self) -> Dict[str, Any]:
        info = super().get_architecture_info()
        info.update(
            {
                "rule_strength": self.rule_strength,
                "explicit_rules": False,
                "hypothesis": "honest neural mixer; former rule logits removed",
            }
        )
        return info


class NeuroSymbolicSprint(_RuleEnhancedCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "NeuroSymbolicSprint", dim=64, layers=1)


class AlgorithmicCortex(_RuleEnhancedCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AlgorithmicCortex", dim=96, layers=2, use_scan=True)


class PatternSavant(_RuleEnhancedCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "PatternSavant", dim=96, layers=2)


class TurboRuleMixer(_RuleEnhancedCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "TurboRuleMixer", dim=80, layers=1)


class ApexContextEngine(_RuleEnhancedCore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexContextEngine", dim=128, layers=2, use_attention=True, use_scan=True)
