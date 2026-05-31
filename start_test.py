"""
Neural Architecture Testing System v3.0
С умным кэшированием результатов и инкрементальным тестированием
"""

import warnings
warnings.filterwarnings("ignore", message="Failed to find CUDA.", category=UserWarning)

import torch
import json
import os
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import time

from helper.architecture_tester_v3 import ArchitectureTesterV3
from helper.results_cache import ResultsCache


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


class NeuralArchitectureTestingSystemV3:
    """
    Улучшенная система тестирования v3.0
    
    Новые возможности:
    - Кэширование результатов с версионированием тестов
    - Инкрементальное тестирование (только новое/изменённое)
    - Автоматическое обнаружение новых архитектур
    - Умная загрузка предыдущих результатов
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Инициализация системы"""
        print("="*80)
        print("NEURAL ARCHITECTURE TESTING SYSTEM v3.0")
        print("="*80)
        
        # Загрузка конфигурации
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = self.device
        
        if self.device.type == "cuda":
            print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Device: {self.device}")
        
        # Результаты
        self.results_dir = Path(self.config.get('results_dir', 'results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results directory: {self.results_dir}")
        
        # Инициализируем кэш
        self.cache = ResultsCache(cache_dir=str(self.results_dir / "cache"))
        
        # Инициализируем тестер
        self.tester = ArchitectureTesterV3(self.config)
        
        # Регистрируем версии тестов
        self._register_test_versions()
        self._register_architecture_versions()
        
        print("="*80)
        print()
        
    def _register_test_versions(self):
        """Регистрирует версии всех тестов"""
        print("📦 Registering test versions...")
        
        tests_changed = []
        for test_name, test_class in self.tester.tests.items():
            changed = self.cache.register_test_version(test_name, test_class)
            if changed:
                tests_changed.append(test_name)
        
        if tests_changed:
            print(f"   ⚠️  Tests updated (will re-run affected architectures): {', '.join(tests_changed)}")
            for test_name in tests_changed:
                self.cache.invalidate_test(test_name)
        else:
            print("   ✓ All tests unchanged")

        removed_errors = self.cache.prune_error_results()
        if removed_errors:
            print(f"   Removed {removed_errors} cached failed runs")
        
        print()

    def _register_architecture_versions(self):
        """Registers architecture source versions so changed models are retested."""
        print("Registering architecture versions...")

        architectures_changed = []
        for architecture_name, architecture_class in self.tester.architectures.items():
            changed = self.cache.register_architecture_version(architecture_name, architecture_class)
            if changed:
                architectures_changed.append(architecture_name)

        if architectures_changed:
            print(f"   Architectures updated: {', '.join(architectures_changed)}")
            for architecture_name in architectures_changed:
                self.cache.invalidate_architecture(architecture_name)
        else:
            print("   All architectures unchanged")

        removed_stale = self.cache.prune_stale_results()
        if removed_stale:
            print(f"   Removed {removed_stale} stale cached results")

        print()
    
    def discover_architectures(self) -> List[str]:
        """Автоматически обнаруживает все доступные архитектуры"""
        available = list(self.tester.architectures.keys())
        print(f"✓ Discovered {len(available)} architectures:")
        for arch in available:
            print(f"  - {arch}")
        print()
        return available
    
    def discover_tests(self) -> List[str]:
        """Автоматически обнаруживает все доступные тесты"""
        available = list(self.tester.tests.keys())
        print(f"✓ Discovered {len(available)} tests:")
        for test in available:
            print(f"  - {test}")
        print()
        return available
    
    def run_tests_incremental(
        self, 
        architectures: List[str] = None,
        tests: List[str] = None,
        force_retest: bool = False,
        time_limit_seconds: float = 60.0
    ):
        """
        Запускает тесты инкрементально (только новые/изменённые)
        
        Args:
            architectures: список архитектур (None = все)
            tests: список тестов (None = все)
            force_retest: принудительно перетестировать всё
            time_limit_seconds: общий лимит перед запуском следующего теста
        """
        
        # Определяем что тестировать
        if architectures is None:
            architectures = self.discover_architectures()
        if tests is None:
            tests = self.discover_tests()
        
        # Проверяем что нужно протестировать
        if force_retest:
            print("🔄 Force retest enabled - will run all tests")
            missing = [(arch, test) for arch in architectures for test in tests]
        else:
            missing = self.cache.get_missing_tests(architectures, tests)
        
        total_tests = len(architectures) * len(tests)
        cached_tests = total_tests - len(missing)
        
        print("="*80)
        print("TESTING STRATEGY")
        print("="*80)
        print(f"Total tests needed: {total_tests}")
        print(f"Cached results: {cached_tests} ({cached_tests/total_tests*100:.1f}%)")
        print(f"Tests to run: {len(missing)} ({len(missing)/total_tests*100:.1f}%)")
        print("="*80)
        print()
        
        if len(missing) == 0:
            print("✓ All results are cached! No testing needed.")
            print("  Use force_retest=True to re-run all tests.")
            return self.load_all_results(architectures, tests)
        
        # Запускаем только недостающие тесты
        print("="*80)
        print(f"RUNNING {len(missing)} TESTS")
        print("="*80)
        print()
        
        run_start = time.time()
        for i, (arch_name, test_name) in enumerate(missing, 1):
            elapsed = time.time() - run_start
            if time_limit_seconds and elapsed >= time_limit_seconds:
                print(f"Time limit reached ({time_limit_seconds:.1f}s). Stopping before next test.")
                break

            print("="*60)
            print(f"Test {i}/{len(missing)}: {arch_name} on {test_name}")
            print("="*60)
            
            try:
                # Запускаем тест
                result = self.tester.run_single_test(arch_name, test_name)
                
                # Сохраняем в кэш
                self.cache.save_result(arch_name, test_name, result)
                
                # Показываем результаты
                print(f"✓ Completed")
                accuracy = result.get('accuracy', 'N/A')
                training_time = result.get('training_time_seconds', result.get('training_time', 'N/A'))
                inference_speed = result.get('inference_speed', 'N/A')
                
                print(f"  Accuracy: {accuracy}")
                
                if isinstance(training_time, (int, float)):
                    print(f"  Training time: {training_time:.2f}s")
                else:
                    print(f"  Training time: {training_time}")
                    
                if isinstance(inference_speed, (int, float)):
                    print(f"  Inference speed: {inference_speed:.1f} samples/sec")
                else:
                    print(f"  Inference speed: {inference_speed}")
                print()
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                print()
        
        # Загружаем все результаты (включая кэшированные)
        return self.load_all_results(architectures, tests)
    
    def load_all_results(self, architectures: List[str], tests: List[str]) -> Dict:
        """Загружает все результаты (из кэша + новые)"""
        all_results = {}
        
        for arch in architectures:
            all_results[arch] = {}
            for test in tests:
                cached = self.cache.get_cached_result(arch, test)
                if cached:
                    all_results[arch][test] = cached
        
        return all_results
    
    def generate_report(self, results: Dict):
        """Генерирует отчёт по результатам"""
        print("="*80)
        print("GENERATING REPORT")
        print("="*80)
        print()
        
        # Сохраняем результаты
        results_file = self.results_dir / "test_results_v3.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {results_file}")
        
        # Генерируем сравнительный отчёт
        report = self._generate_comparison_report(results)
        report_file = self.results_dir / "architecture_comparison_v3.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved to: {report_file}")
        
        # Экспортируем в старый формат для совместимости
        legacy_file = self.results_dir / "test_results.json"
        self.cache.export_to_legacy_format(str(legacy_file))
        print(f"✓ Legacy format exported to: {legacy_file}")
        
        # Статистика кэша
        cache_stats = self.cache.get_cache_stats()
        print()
        print("CACHE STATISTICS:")
        print(f"  Cached results: {cache_stats['total_cached_results']}")
        print(f"  Unique architectures: {cache_stats['unique_architectures']}")
        print(f"  Unique tests: {cache_stats['unique_tests']}")
        print(f"  Cache size: {cache_stats['cache_size_kb']} KB")
        
        print()
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("="*80)
        
    def _generate_comparison_report(self, results: Dict) -> str:
        """Генерирует сравнительный отчёт"""
        report = []
        report.append("="*80)
        report.append("ARCHITECTURE COMPARISON REPORT v3.0")
        report.append("="*80)
        report.append("")
        
        # Группируем по тестам
        tests = set()
        for arch_results in results.values():
            tests.update(arch_results.keys())
        
        for test_name in sorted(tests):
            report.append("")
            report.append("─"*80)
            report.append(f"TEST: {test_name}")
            report.append("─"*80)
            report.append("")
            
            # Собираем метрики
            metrics = []
            for arch_name, arch_results in results.items():
                if test_name in arch_results:
                    test_result = arch_results[test_name]
                    metrics.append({
                        'name': arch_name,
                        'accuracy': test_result.get('accuracy', 0),
                        'training_time': test_result.get('training_time_seconds', test_result.get('training_time', float('inf'))),
                        'inference_speed': test_result.get('inference_speed', 0),
                        'top5_accuracy': test_result.get('top5_accuracy', None)
                    })
            
            # Сортируем по точности
            report.append("ACCURACY:")
            sorted_by_acc = sorted(metrics, key=lambda x: x['accuracy'], reverse=True)
            for i, m in enumerate(sorted_by_acc[:15], 1):  # топ-15
                acc_str = f"{m['accuracy']:.4f}"
                if m['top5_accuracy'] is not None:
                    acc_str += f" (top-5: {m['top5_accuracy']:.4f})"
                report.append(f"  {i:2d}. {m['name']:25s} {acc_str}")
            
            report.append("")
            
            # Сортируем по скорости обучения
            report.append("TRAINING SPEED (faster is better):")
            sorted_by_train = sorted(metrics, key=lambda x: x['training_time'])
            for i, m in enumerate(sorted_by_train[:15], 1):
                if m['training_time'] < float('inf'):
                    samples_per_sec = 1000 / m['training_time']  # примерная оценка
                    report.append(f"  {i:2d}. {m['name']:25s} {m['training_time']:.2f}s ({samples_per_sec:.1f} samples/sec)")
            
            report.append("")
            
            # Сортируем по скорости инференса
            report.append("INFERENCE SPEED (predictions/sec):")
            sorted_by_inf = sorted(metrics, key=lambda x: x['inference_speed'], reverse=True)
            for i, m in enumerate(sorted_by_inf[:15], 1):
                report.append(f"  {i:2d}. {m['name']:25s} {m['inference_speed']:.1f}")
            
            report.append("")
        
        return "\n".join(report)


def _parse_csv(value: str) -> List[str]:
    if not value:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Neural Architecture Testing System v3.0")
    parser.add_argument("--architectures", "--arch", default=None,
                        help="Comma-separated architecture names to test")
    parser.add_argument("--tests", default=None,
                        help="Comma-separated test names to run")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache and retest selected pairs")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Overall run time limit in seconds before starting another test")
    parser.add_argument("--list", action="store_true",
                        help="List available architectures and tests, then exit")
    args = parser.parse_args()

    # Инициализация
    system = NeuralArchitectureTestingSystemV3()

    if args.list:
        system.discover_architectures()
        system.discover_tests()
        return
    
    # Запускаем инкрементальные тесты
    results = system.run_tests_incremental(
        architectures=_parse_csv(args.architectures),  # None = все архитектуры
        tests=_parse_csv(args.tests),                  # None = все тесты
        force_retest=args.force,
        time_limit_seconds=args.timeout
    )
    
    # Генерируем отчёт
    system.generate_report(results)


if __name__ == "__main__":
    main()
