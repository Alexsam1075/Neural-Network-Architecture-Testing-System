# Architectures Package v3.0
# Модульная система для добавления новых архитектур без изменения других компонентов

from .base_architecture import BaseArchitecture

# Оригинальные архитектуры v1.0
from .transformer import TransformerArchitecture as TransformerArch
from .ssm import SSMArchitecture as SSMArch
from .hyper_memory import HyperMemoryArchitecture as HyperMemory
from .delta_flow import DeltaFlowArchitecture as DeltaFlow
from .fractal_net import FractalNetArchitecture as FractalNet
from .liquid_graph import LiquidGraphArchitecture as LiquidGraph
from .chrono_kernel import ChronoKernelArchitecture as ChronoKernel
from .quantum_resonance import QuantumResonanceArchitecture as QuantumResonance
from .holographic_net import HolographicNetArchitecture as HolographicNet
from .causal_conv_net import CausalConvNetArchitecture as CausalConvNet
from .neural_ode import NeuralODEArchitecture as NeuralODE
from .tensor_train import TensorTrainArchitecture as TensorTrain
from .echo_state import EchoStateArchitecture as EchoState

# Новые архитектуры v3.0 - Улучшенные версии
from .hyper_memory_plus import HyperMemoryPlusPlus
from .fractal_net_v2 import FractalNetV2

# Новые архитектуры v3.0 - Batch 1
from .new_architectures_batch1 import EchoStateV2, HybridMemory, AdaptiveDepth

# Новые архитектуры v3.0 - AION-inspired
from .aion_lite import AIONLite
from .aion_batch import AIONOntology, AIONHonest, AIONModular

# Новые архитектуры v3.0 - Optimized
from .optimized_batch import UltraFast, MemoryOptimized, BalancedPro, SpeedDemon, AccuracyFirst

# Новые архитектуры v4.0 - Batch 2
from .new_architectures_batch2 import (
    HyperSpeed, FlashConv, GatedMemoryNet, ParallelFractal,
    MambaLite, TurboTransformer, CausalMixer, DeepResidual,
    WaveNet2, HybridSSMAttn
)
from .new_architectures_batch3 import RecallMixerPro, FractalMemoryPro, StableSSMTransformer
from .new_architectures_batch4 import CausalRecallHybrid, HoloCausalMemory, AttentiveRecallSSM
from .new_architectures_batch5 import (
    PrimeRecallNet, MosaicMixer, CausalDeltaMemory, RareTokenSentinel,
    AntiCollapseTransformer, SwiftRecallConv, HoloFractalLite,
    StateSpaceRecall, OmniMemoryMixer, RolloutResonator,
)
from .new_architectures_batch6 import (
    NeuroSymbolicSprint, AlgorithmicCortex, PatternSavant,
    TurboRuleMixer, ApexContextEngine,
)
from .new_architectures_batch7 import (
    ApexRulePlus, ApexRulePlusFast, MaskProofSprint,
    GeneralistRuleEngine, ApexUltraLite,
)
from .new_architectures_batch8 import (
    VectorPatternSprint, SentinelDeltaNet, PeriodicMemoryEngine,
    ProceduralRecallNet, OmniPatternKernel,
)
from .new_architectures_batch9 import (
    LocalFormulaMixer, GatedDeltaMixer, LinearAssociativeMemory,
    FactorizedTransitionMixer, RecurrentFormulaCore,
)
from .aion_long_memory_core import (
    AIONLongMemoryCore,
    AIONLongMemoryCoreV2,
    AIONLongMemoryCoreV3,
    AIONLongMemoryCoreV4,
)

__all__ = [
    'BaseArchitecture',
    # Original v1.0
    'TransformerArch',
    'SSMArch', 
    'HyperMemory',
    'DeltaFlow',
    'FractalNet',
    'LiquidGraph',
    'ChronoKernel',
    'QuantumResonance',
    'HolographicNet',
    'CausalConvNet',
    'NeuralODE',
    'TensorTrain',
    'EchoState',
    # New v3.0 - Improved
    'HyperMemoryPlusPlus',
    'FractalNetV2',
    # New v3.0 - Batch 1
    'EchoStateV2',
    'HybridMemory',
    'AdaptiveDepth',
    # New v3.0 - AION
    'AIONLite',
    'AIONOntology',
    'AIONHonest',
    'AIONModular',
    # New v3.0 - Optimized
    'UltraFast',
    'MemoryOptimized',
    'BalancedPro',
    'SpeedDemon',
    'AccuracyFirst',
    # New v4.0 - Batch 2
    'HyperSpeed',
    'FlashConv',
    'GatedMemoryNet',
    'ParallelFractal',
    'MambaLite',
    'TurboTransformer',
    'CausalMixer',
    'DeepResidual',
    'WaveNet2',
    'HybridSSMAttn',
    'RecallMixerPro',
    'FractalMemoryPro',
    'StableSSMTransformer',
    'CausalRecallHybrid',
    'HoloCausalMemory',
    'AttentiveRecallSSM',
    'PrimeRecallNet',
    'MosaicMixer',
    'CausalDeltaMemory',
    'RareTokenSentinel',
    'AntiCollapseTransformer',
    'SwiftRecallConv',
    'HoloFractalLite',
    'StateSpaceRecall',
    'OmniMemoryMixer',
    'RolloutResonator',
    'NeuroSymbolicSprint',
    'AlgorithmicCortex',
    'PatternSavant',
    'TurboRuleMixer',
    'ApexContextEngine',
    'ApexRulePlus',
    'ApexRulePlusFast',
    'MaskProofSprint',
    'GeneralistRuleEngine',
    'ApexUltraLite',
    'VectorPatternSprint',
    'SentinelDeltaNet',
    'PeriodicMemoryEngine',
    'ProceduralRecallNet',
    'OmniPatternKernel',
    'LocalFormulaMixer',
    'GatedDeltaMixer',
    'LinearAssociativeMemory',
    'FactorizedTransitionMixer',
    'RecurrentFormulaCore',
    'AIONLongMemoryCore',
    'AIONLongMemoryCoreV2',
    'AIONLongMemoryCoreV3',
    'AIONLongMemoryCoreV4',
]

# New v4.0 - Batch 2 (already imported above, adding to __all__ via tail)
