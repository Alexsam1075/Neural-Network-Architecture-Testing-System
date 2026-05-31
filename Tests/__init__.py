# Tests Package
# Модульная система для добавления новых тестов

from .base_test import BaseTest
from .number_sequence_test import NumberSequenceTest
from .next_token_test import NextTokenPredictionTest

__all__ = [
    'BaseTest',
    'NumberSequenceTest',
    'NextTokenPredictionTest'
]
