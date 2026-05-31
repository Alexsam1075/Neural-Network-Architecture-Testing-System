"""
Architecture Tester v3.0
С автоматической регистрацией архитектур и тестов
"""

import warnings
warnings.filterwarnings("ignore", message="Failed to find CUDA.", category=UserWarning)

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import importlib
import inspect
from pathlib import Path

# Импортируем все архитектуры
from Architectures import *
from Architectures.base_architecture import BaseArchitecture

# Импортируем тесты
from Tests.number_sequence_test import NumberSequenceTest
from Tests.next_token_test import NextTokenPredictionTest as NextTokenTest
from Tests.new_tests_batch2 import LongRangeDependencyTest, NoiseRobustnessTest, GeneralizationTest
from Tests.new_tests_batch3 import LengthExtrapolationTest, DistractorRetrievalTest, CausalMaskStabilityTest
from Tests.generation_quality_tests import (
    StructuredGenerationTest,
    RareTokenGenerationTest,
    AlgorithmicRolloutTest,
    RepetitionCollapseTest,
)
from Tests.ultra_long_context_test import UltraLongContextTest
from Tests.base_test import BaseTest
from Tests.anti_cheat import audit_architecture_class


class ArchitectureTesterV3:
    """
    Улучшенный тестер v3.0 с автоматическим обнаружением
    
    Новые возможности:
    - Автоматическая регистрация всех архитектур из Architectures/
    - Автоматическая регистрация всех тестов из Tests/
    - Поддержка кэширования через ResultsCache
    - Запуск отдельных тестов
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cpu')
        
        # Автоматически находим все архитектуры
        self.architectures = self._discover_architectures()
        
        # Автоматически находим все тесты
        self.tests = self._discover_tests()
        
        print(f"✓ Initialized {len(self.architectures)} architectures:")
        for name, arch in self.architectures.items():
            try:
                arch_config = self.config.copy()
                if 'architectures' in self.config and name in self.config['architectures']:
                    arch_config.update(self.config['architectures'][name])
                if 'dim' not in arch_config and 'd_model' in arch_config:
                    arch_config['dim'] = arch_config['d_model']
                instance = arch(arch_config)
                params = instance.count_parameters()
                print(f"  - {name:25s} ({params:,} params)")
            except Exception as e:
                print(f"  - {name:25s} (initialization error: {e})")
        
        print(f"\n✓ Initialized {len(self.tests)} tests:")
        for name in self.tests.keys():
            print(f"  - {name}")
        print()
        
    def _discover_architectures(self) -> Dict[str, type]:
        """
        Автоматически обнаруживает все классы архитектур
        
        Returns:
            Dict: {имя_архитектуры: класс_архитектуры}
        """
        architectures = {}
        
        # Список всех архитектур из __init__.py
        arch_classes = [
            # Original v1.0
            ('Transformer', TransformerArch),
            ('SSM', SSMArch),
            ('HyperMemory', HyperMemory),
            ('DeltaFlow', DeltaFlow),
            ('FractalNet', FractalNet),
            ('LiquidGraph', LiquidGraph),
            ('ChronoKernel', ChronoKernel),
            ('QuantumResonance', QuantumResonance),
            ('HolographicNet', HolographicNet),
            ('CausalConvNet', CausalConvNet),
            ('NeuralODE', NeuralODE),
            ('TensorTrain', TensorTrain),
            ('EchoState', EchoState),
            # v3.0 Improved
            ('HyperMemoryPlusPlus', HyperMemoryPlusPlus),
            ('FractalNetV2', FractalNetV2),
            # v3.0 Batch1
            ('EchoStateV2', EchoStateV2),
            ('HybridMemory', HybridMemory),
            ('AdaptiveDepth', AdaptiveDepth),
            # v3.0 AION
            ('AIONLite', AIONLite),
            ('AIONOntology', AIONOntology),
            ('AIONHonest', AIONHonest),
            ('AIONModular', AIONModular),
            # v3.0 Optimized
            ('UltraFast', UltraFast),
            ('MemoryOptimized', MemoryOptimized),
            ('BalancedPro', BalancedPro),
            ('SpeedDemon', SpeedDemon),
            ('AccuracyFirst', AccuracyFirst),
            # v4.0 Batch 2
            ('HyperSpeed', HyperSpeed),
            ('FlashConv', FlashConv),
            ('GatedMemoryNet', GatedMemoryNet),
            ('ParallelFractal', ParallelFractal),
            ('MambaLite', MambaLite),
            ('TurboTransformer', TurboTransformer),
            ('CausalMixer', CausalMixer),
            ('DeepResidual', DeepResidual),
            ('WaveNet2', WaveNet2),
            ('HybridSSMAttn', HybridSSMAttn),
            # v5.0 Batch 3
            ('RecallMixerPro', RecallMixerPro),
            ('FractalMemoryPro', FractalMemoryPro),
            ('StableSSMTransformer', StableSSMTransformer),
            # v6.0 Batch 4
            ('CausalRecallHybrid', CausalRecallHybrid),
            ('HoloCausalMemory', HoloCausalMemory),
            ('AttentiveRecallSSM', AttentiveRecallSSM),
            # v7.0 Batch 5
            ('PrimeRecallNet', PrimeRecallNet),
            ('MosaicMixer', MosaicMixer),
            ('CausalDeltaMemory', CausalDeltaMemory),
            ('RareTokenSentinel', RareTokenSentinel),
            ('AntiCollapseTransformer', AntiCollapseTransformer),
            ('SwiftRecallConv', SwiftRecallConv),
            ('HoloFractalLite', HoloFractalLite),
            ('StateSpaceRecall', StateSpaceRecall),
            ('OmniMemoryMixer', OmniMemoryMixer),
            ('RolloutResonator', RolloutResonator),
            # v8.0 Batch 6
            ('NeuroSymbolicSprint', NeuroSymbolicSprint),
            ('AlgorithmicCortex', AlgorithmicCortex),
            ('PatternSavant', PatternSavant),
            ('TurboRuleMixer', TurboRuleMixer),
            ('ApexContextEngine', ApexContextEngine),
            # v9.0 Batch 7
            ('ApexRulePlus', ApexRulePlus),
            ('ApexRulePlusFast', ApexRulePlusFast),
            ('MaskProofSprint', MaskProofSprint),
            ('GeneralistRuleEngine', GeneralistRuleEngine),
            ('ApexUltraLite', ApexUltraLite),
            # v10.0 Batch 8
            ('VectorPatternSprint', VectorPatternSprint),
            ('SentinelDeltaNet', SentinelDeltaNet),
            ('PeriodicMemoryEngine', PeriodicMemoryEngine),
            ('ProceduralRecallNet', ProceduralRecallNet),
            ('OmniPatternKernel', OmniPatternKernel),
            # v11.0 Batch 9
            ('LocalFormulaMixer', LocalFormulaMixer),
            ('GatedDeltaMixer', GatedDeltaMixer),
            ('LinearAssociativeMemory', LinearAssociativeMemory),
            ('FactorizedTransitionMixer', FactorizedTransitionMixer),
            ('RecurrentFormulaCore', RecurrentFormulaCore),
            ('AIONLongMemoryCore', AIONLongMemoryCore),
            ('AIONLongMemoryCoreV2', AIONLongMemoryCoreV2),
            ('AIONLongMemoryCoreV3', AIONLongMemoryCoreV3),
            ('AIONLongMemoryCoreV4', AIONLongMemoryCoreV4),
        ]
        
        for name, cls in arch_classes:
            try:
                # Проверяем что это подкласс BaseArchitecture
                if inspect.isclass(cls) and issubclass(cls, BaseArchitecture) and cls != BaseArchitecture:
                    architectures[name] = cls
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        if self.config.get('trusted_architectures_only', True):
            trusted = set(self.config.get('trusted_architectures', []))
            if trusted:
                architectures = {name: cls for name, cls in architectures.items() if name in trusted}

        return architectures
    
    def _discover_tests(self) -> Dict[str, type]:
        """
        Автоматически обнаруживает все тесты
        
        Returns:
            Dict: {имя_теста: класс_теста}
        """
        tests = {
            'NumberSequence': NumberSequenceTest,
            'NextToken': NextTokenTest,
            'LongRange': LongRangeDependencyTest,
            'NoiseRobustness': NoiseRobustnessTest,
            'Generalization': GeneralizationTest,
            'LengthExtrapolation': LengthExtrapolationTest,
            'DistractorRetrieval': DistractorRetrievalTest,
            'CausalMaskStability': CausalMaskStabilityTest,
            'StructuredGeneration': StructuredGenerationTest,
            'RareTokenGeneration': RareTokenGenerationTest,
            'AlgorithmicRollout': AlgorithmicRolloutTest,
            'RepetitionCollapse': RepetitionCollapseTest,
            'UltraLongContext': UltraLongContextTest,
        }
        
        return tests
    
    def create_architecture(self, arch_name: str) -> Optional[BaseArchitecture]:
        """
        Создаёт экземпляр архитектуры
        
        Args:
            arch_name: Имя архитектуры
            
        Returns:
            Экземпляр архитектуры или None если не найдена
        """
        if arch_name not in self.architectures:
            print(f"Architecture {arch_name} not found!")
            print(f"Available: {list(self.architectures.keys())}")
            return None
        
        try:
            arch_class = self.architectures[arch_name]
            # Подготавливаем конфиг - объединяем глобальный и архитектурный конфиги
            arch_config = self.config.copy()
            if 'architectures' in self.config and arch_name in self.config['architectures']:
                arch_config.update(self.config['architectures'][arch_name])
            if 'dim' not in arch_config and 'd_model' in arch_config:
                arch_config['dim'] = arch_config['d_model']
            architecture = arch_class(arch_config)
            architecture.to(self.device)
            return architecture
        except Exception as e:
            print(f"Error creating {arch_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_test(self, test_name: str) -> Optional[BaseTest]:
        """
        Создаёт экземпляр теста
        
        Args:
            test_name: Имя теста
            
        Returns:
            Экземпляр теста или None если не найден
        """
        if test_name not in self.tests:
            print(f"Test {test_name} not found!")
            print(f"Available: {list(self.tests.keys())}")
            return None
        
        try:
            test_class = self.tests[test_name]
            test_config = self.config.copy()
            if 'tests' in self.config and test_name in self.config['tests']:
                test_config.update(self.config['tests'][test_name])
            test = test_class(test_config)
            return test
        except Exception as e:
            print(f"Error creating test {test_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_single_test(self, arch_name: str, test_name: str) -> Dict[str, Any]:
        """
        Запускает один тест на одной архитектуре
        
        Args:
            arch_name: Имя архитектуры
            test_name: Имя теста
            
        Returns:
            Результаты теста
        """
        # Создаём архитектуру
        if self.config.get('run_architecture_anti_cheat', True):
            arch_class = self.architectures.get(arch_name)
            if arch_class is not None:
                allowlisted = arch_name in set(self.config.get('anti_cheat_allowlist', []))
                audit = audit_architecture_class(arch_class, allowlisted=allowlisted)
                if not audit['passed']:
                    return {'error': 'Architecture failed anti-cheat audit', 'anti_cheat': audit}

        architecture = self.create_architecture(arch_name)
        if architecture is None:
            return {'error': f'Architecture {arch_name} not found'}
        
        # Создаём тест
        test = self.create_test(test_name)
        if test is None:
            return {'error': f'Test {test_name} not found'}
        
        # Reset архитектуры
        if hasattr(architecture, 'reset_parameters'):
            architecture.reset_parameters()
        
        # Запускаем тест
        try:
            results = test.run(architecture)
            return results
        except Exception as e:
            print(f"Error running test: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def get_architecture_summary(self, arch_name: str) -> Dict[str, Any]:
        """Возвращает информацию об архитектуре"""
        architecture = self.create_architecture(arch_name)
        if architecture is None:
            return {}
        
        return architecture.get_summary()
    
    def get_all_architectures(self) -> List[str]:
        """Возвращает список всех доступных архитектур"""
        return list(self.architectures.keys())
    
    def get_all_tests(self) -> List[str]:
        """Возвращает список всех доступных тестов"""
        return list(self.tests.keys())
