from typing import Any, Dict

from .new_architectures_batch6 import _RuleEnhancedCore


class _ApexRulePlus(_RuleEnhancedCore):
    """Backward-compatible honest replacement for former Apex rule variants."""


class ApexRulePlus(_ApexRulePlus):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexRulePlus", dim=128, layers=2, use_attention=True, use_scan=True)


class ApexRulePlusFast(_ApexRulePlus):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexRulePlusFast", dim=96, layers=2, use_attention=True, use_scan=True)


class MaskProofSprint(_ApexRulePlus):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "MaskProofSprint", dim=80, layers=1)


class GeneralistRuleEngine(_ApexRulePlus):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "GeneralistRuleEngine", dim=128, layers=2, use_attention=True)


class ApexUltraLite(_ApexRulePlus):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "ApexUltraLite", dim=64, layers=1)
