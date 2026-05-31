"""
Base Architecture Interface
Все архитектуры должны наследоваться от этого класса
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import json
import os


class BaseArchitecture(ABC, nn.Module):
    """
    Базовый класс для всех архитектур нейросетей.
    
    Гарантирует единообразный интерфейс:
    - Инициализация с конфигом
    - Forward pass
    - Сохранение/загрузка весов
    - Сбор статистики
    """
    
    def __init__(self, config: Dict[str, Any], name: str):
        super().__init__()
        self.config = config
        self.name = name
        self.device = config.get('device', 'cpu')
        self.to(self.device)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass архитектуры
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def get_architecture_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию об архитектуре
        
        Returns:
            Dict с параметрами и описанием архитектуры
        """
        pass
    
    def count_parameters(self) -> int:
        """Подсчитывает количество параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str) -> None:
        """Сохраняет веса и конфиг"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'name': self.name,
            'architecture_info': self.get_architecture_info()
        }
        torch.save(checkpoint, path)
        
    def load(self, path: str) -> None:
        """Загружает веса и конфиг"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
    def reset_parameters(self) -> None:
        """Переинициализирует параметры модели"""
        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module is not self:
                try:
                    module.reset_parameters()
                except:
                    pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Возвращает итоговую информацию о модели"""
        return {
            'name': self.name,
            'total_parameters': self.count_parameters(),
            'architecture': self.get_architecture_info(),
            'device': str(self.device),
            'config': self.config
        }
