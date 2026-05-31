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
        self.stateless_forward = config.get('stateless_forward', True)
        self._resetting_runtime_state = False
        self.to(self.device)

    def __call__(self, *args, **kwargs):
        if self.stateless_forward and not self._resetting_runtime_state:
            self.reset_runtime_state()
        return super().__call__(*args, **kwargs)

    def reset_runtime_state(self) -> None:
        """Clear inference caches/state carried between independent test calls."""
        self._resetting_runtime_state = True
        try:
            for module in self.modules():
                if module is not self and hasattr(module, 'disable_cache'):
                    try:
                        module.disable_cache()
                    except Exception:
                        pass
            state_markers = ('state', 'cache', 'history', 'previous')
            for name, buffer in self.named_buffers():
                leaf_name = name.rsplit('.', 1)[-1].lower()
                if any(marker in leaf_name for marker in state_markers):
                    buffer.detach().zero_()
        finally:
            self._resetting_runtime_state = False
        
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
