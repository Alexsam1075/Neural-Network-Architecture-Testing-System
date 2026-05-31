"""
Base Test Interface
Все тесты должны наследоваться от этого класса
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import json
import os
from datetime import datetime


class BaseTest(ABC):
    """
    Базовый класс для всех тестов архитектур.
    
    Гарантирует единообразный интерфейс:
    - Подготовка данных
    - Запуск теста
    - Сбор метрик
    - Сохранение результатов
    """
    
    def __init__(self, config: Dict[str, Any], name: str):
        self.config = config
        self.name = name
        self.device = config.get('device', 'cpu')
        self.results = {}
        self.metrics = {}
        
    @abstractmethod
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Подготавливает данные для теста
        
        Returns:
            (X_train, y_train) - обучающие данные
        """
        pass
    
    @abstractmethod
    def run(self, model: nn.Module) -> Dict[str, Any]:
        """
        Запускает тест на модели
        
        Args:
            model: Архитектура для тестирования
            
        Returns:
            Dict с метриками теста
        """
        pass
    
    def get_test_info(self) -> Dict[str, Any]:
        """Возвращает информацию о тесте"""
        return {
            'name': self.name,
            'config': self.config,
            'description': self.__doc__
        }
    
    def save_results(self, path: str, results: Dict[str, Any]) -> None:
        """Сохраняет результаты теста"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        results_with_meta = {
            'test_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': results
        }
        
        with open(path, 'w') as f:
            json.dump(results_with_meta, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает итоговую информацию о тесте"""
        return {
            'name': self.name,
            'config': self.config,
            'results': self.results,
            'metrics': self.metrics
        }
