"""
Results Cache System v3.0
Кэширование результатов тестирования архитектур с версионированием
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import hashlib


class ResultsCache:
    """
    Умный кэш результатов тестирования.
    
    Особенности:
    - Версионирование тестов (при изменении теста старые результаты не используются)
    - Быстрая проверка необходимости повторного тестирования
    - Инкрементальное обновление (тестируем только новое)
    """
    
    def __init__(self, cache_dir: str = "results/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "results_cache.json"
        self.test_versions_file = self.cache_dir / "test_versions.json"
        self.architecture_versions_file = self.cache_dir / "architecture_versions.json"
        
        self.cache = self._load_cache()
        self.test_versions = self._load_test_versions()
        self.architecture_versions = self._load_architecture_versions()
        
    def _load_cache(self) -> Dict:
        """Загружает кэш результатов"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _load_test_versions(self) -> Dict:
        """Загружает версии тестов"""
        if self.test_versions_file.exists():
            try:
                with open(self.test_versions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _load_architecture_versions(self) -> Dict:
        if self.architecture_versions_file.exists():
            try:
                with open(self.architecture_versions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Сохраняет кэш на диск"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)
    
    def _save_test_versions(self):
        """Сохраняет версии тестов"""
        with open(self.test_versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_versions, f, indent=2)

    def _save_architecture_versions(self):
        with open(self.architecture_versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.architecture_versions, f, indent=2)
    
    def get_test_version(self, test_class) -> str:
        """
        Вычисляет версию теста на основе его исходного кода
        
        Если код теста изменился, версия тоже изменится
        """
        import inspect
        source = inspect.getsource(test_class)
        hash_obj = hashlib.md5(source.encode())
        return hash_obj.hexdigest()[:8]
    
    def register_test_version(self, test_name: str, test_class) -> bool:
        """
        Регистрирует версию теста.
        
        Returns:
            True если версия изменилась (нужно перетестировать архитектуры)
            False если версия та же самая
        """
        new_version = self.get_test_version(test_class)
        old_version = self.test_versions.get(test_name)
        
        if old_version != new_version:
            self.test_versions[test_name] = new_version
            self._save_test_versions()
            return True
        return False

    def get_architecture_version(self, architecture_class) -> str:
        import inspect
        sources = []
        for cls in inspect.getmro(architecture_class):
            if cls is object:
                continue
            module = inspect.getmodule(cls)
            if module is not None and getattr(module, "__file__", None):
                try:
                    with open(module.__file__, "r", encoding="utf-8") as f:
                        sources.append(f.read())
                    continue
                except OSError:
                    pass
            try:
                sources.append(inspect.getsource(cls))
            except OSError:
                pass
        source = "\n".join(sources)
        hash_obj = hashlib.md5(source.encode())
        return hash_obj.hexdigest()[:8]

    def register_architecture_version(self, architecture_name: str, architecture_class) -> bool:
        new_version = self.get_architecture_version(architecture_class)
        old_version = self.architecture_versions.get(architecture_name)

        if old_version != new_version:
            self.architecture_versions[architecture_name] = new_version
            self._save_architecture_versions()
            return True
        return False
    
    def get_cached_result(self, architecture_name: str, test_name: str) -> Optional[Dict]:
        """
        Получает закэшированный результат если он актуален
        
        Returns:
            Dict с результатами или None если кэш устарел/отсутствует
        """
        key = f"{architecture_name}::{test_name}"
        
        if key not in self.cache:
            return None
        
        cached_entry = self.cache[key]
        cached_results = cached_entry.get('results', {})
        if isinstance(cached_results, dict) and 'error' in cached_results:
            return None

        cached_test_version = cached_entry.get('test_version')
        current_test_version = self.test_versions.get(test_name)
        cached_architecture_version = cached_entry.get('architecture_version')
        current_architecture_version = self.architecture_versions.get(architecture_name)
        
        # Проверяем актуальность
        if (
            cached_test_version == current_test_version
            and cached_architecture_version == current_architecture_version
        ):
            return cached_results
        else:
            # Кэш устарел
            return None
    
    def save_result(self, architecture_name: str, test_name: str, results: Dict):
        """Сохраняет результат в кэш"""
        key = f"{architecture_name}::{test_name}"
        
        self.cache[key] = {
            'results': results,
            'test_version': self.test_versions.get(test_name),
            'architecture_version': self.architecture_versions.get(architecture_name),
            'timestamp': datetime.now().isoformat(),
            'architecture': architecture_name,
            'test': test_name
        }
        
        self._save_cache()
    
    def get_missing_tests(self, architectures: List[str], tests: List[str]) -> List[tuple]:
        """
        Возвращает список (architecture, test) пар которые нужно протестировать
        
        Returns:
            List of tuples (architecture_name, test_name)
        """
        missing = []
        
        for arch in architectures:
            for test in tests:
                if self.get_cached_result(arch, test) is None:
                    missing.append((arch, test))
        
        return missing
    
    def get_cache_stats(self) -> Dict:
        """Возвращает статистику кэша"""
        total_entries = len(self.cache)
        
        architectures = set()
        tests = set()
        
        for key in self.cache.keys():
            arch, test = key.split('::', 1)
            architectures.add(arch)
            tests.add(test)
        
        return {
            'total_cached_results': total_entries,
            'unique_architectures': len(architectures),
            'unique_tests': len(tests),
            'test_versions': len(self.test_versions),
            'architecture_versions': len(self.architecture_versions),
            'cache_file': str(self.cache_file),
            'cache_size_kb': round(self.cache_file.stat().st_size / 1024, 2) if self.cache_file.exists() else 0
        }

    def prune_error_results(self) -> int:
        """Removes cached failed runs so fixed architectures/tests can be retried."""
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if isinstance(entry.get('results'), dict) and 'error' in entry.get('results', {})
        ]
        for key in keys_to_remove:
            del self.cache[key]
        if keys_to_remove:
            self._save_cache()
        return len(keys_to_remove)

    def prune_stale_results(self) -> int:
        """Removes results whose test or architecture version is no longer current."""
        keys_to_remove = []
        for key, entry in self.cache.items():
            arch, test = key.split('::', 1)
            current_test_version = self.test_versions.get(test)
            current_architecture_version = self.architecture_versions.get(arch)
            if not current_test_version or not current_architecture_version:
                keys_to_remove.append(key)
                continue
            if entry.get('test_version') != current_test_version:
                keys_to_remove.append(key)
                continue
            if entry.get('architecture_version') != current_architecture_version:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]
        if keys_to_remove:
            self._save_cache()
        return len(keys_to_remove)
    
    def invalidate_test(self, test_name: str):
        """Инвалидирует все результаты для конкретного теста"""
        keys_to_remove = [k for k in self.cache.keys() if k.endswith(f"::{test_name}")]
        for key in keys_to_remove:
            del self.cache[key]
        self._save_cache()
    
    def invalidate_architecture(self, architecture_name: str):
        """Инвалидирует все результаты для конкретной архитектуры"""
        keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{architecture_name}::")]
        for key in keys_to_remove:
            del self.cache[key]
        self._save_cache()
    
    def clear_all(self):
        """Полностью очищает кэш"""
        self.cache = {}
        self._save_cache()
    
    def export_to_legacy_format(self, output_file: str):
        """
        Экспортирует результаты в старый формат для совместимости
        """
        legacy_results = {}
        
        for key, entry in self.cache.items():
            if isinstance(entry.get('results'), dict) and 'error' in entry.get('results', {}):
                continue

            arch, test = key.split('::', 1)
            current_test_version = self.test_versions.get(test)
            current_architecture_version = self.architecture_versions.get(arch)
            if not current_test_version or not current_architecture_version:
                continue
            if entry.get('test_version') != current_test_version:
                continue
            if entry.get('architecture_version') != current_architecture_version:
                continue
            
            if arch not in legacy_results:
                legacy_results[arch] = {}
            
            legacy_results[arch][test] = entry['results']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(legacy_results, f, indent=2, ensure_ascii=False)
