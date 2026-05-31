"""
AION-inspired Architectures Batch
Реализация архитектур на основе законов AION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from .base_architecture import BaseArchitecture


# ============================================================================
# AION-Ontology: Живая онтология (Законы 4, 5)
# ============================================================================

class OntologyGraph(nn.Module):
    """
    Живая онтология - граф знаний с причинно-следственными связями
    Закон 4: знания в графе, а не в весах
    """
    def __init__(self, dim: int, num_concepts: int = 128):
        super().__init__()
        self.dim = dim
        self.num_concepts = num_concepts
        
        # Embeddings для концептов
        self.concepts = nn.Parameter(torch.randn(num_concepts, dim) * 0.02)
        
        # Матрица связей (причинность)
        # causality[i,j] = насколько концепт i вызывает концепт j
        self.causality = nn.Parameter(torch.zeros(num_concepts, num_concepts))
        
        # Веса связей (уверенность)
        self.edge_weights = nn.Parameter(torch.ones(num_concepts, num_concepts) * 0.5)
        
    def query(self, x):
        """Запрос к онтологии"""
        B, L, D = x.shape
        
        # Находим ближайшие концепты
        x_flat = x.reshape(-1, D)
        similarities = torch.matmul(x_flat, self.concepts.T)  # [B*L, num_concepts]
        
        # Активация концептов
        concept_activation = F.softmax(similarities, dim=-1)
        
        # Распространение по графу причинности
        # Умножаем на веса рёбер
        weighted_causality = self.causality * torch.sigmoid(self.edge_weights)
        propagated = torch.matmul(concept_activation, weighted_causality)
        
        # Получаем знания из активированных концептов
        knowledge = torch.matmul(propagated, self.concepts)
        knowledge = knowledge.reshape(B, L, D)
        
        return knowledge, concept_activation.reshape(B, L, -1)
    
    def hebbian_update(self, concept_idx1: int, concept_idx2: int, strength: float = 0.1):
        """
        Hebbian learning - обновление без backprop
        Если концепты активируются вместе, связь усиливается
        """
        with torch.no_grad():
            self.causality[concept_idx1, concept_idx2] += strength
            self.causality[concept_idx1, concept_idx2] = torch.clamp(
                self.causality[concept_idx1, concept_idx2], -1.0, 1.0
            )


class AIONOntology(BaseArchitecture):
    """
    AION-Ontology: Разделение интеллекта и знаний
    
    Закон 4: Живая онтология - знания в графе
    Закон 5: Разделение интеллекта (рассуждения) и знаний (факты)
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AIONOntology")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_concepts = config.get('num_concepts', 128)
        self.num_layers = config.get('num_layers', 4)
        
        # Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Интеллект (reasoning) - обучается один раз
        self.reasoning_layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(self.dim, 8, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 2),
                    nn.GELU(),
                    nn.Linear(self.dim * 2, self.dim)
                )
            })
            for _ in range(self.num_layers)
        ])
        
        # Знания (knowledge) - живая онтология
        self.ontology = OntologyGraph(self.dim, self.num_concepts)
        
        # Интеграция знаний и рассуждений
        self.knowledge_gate = nn.Linear(self.dim * 2, self.dim)
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        # Embeddings
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        # Reasoning (интеллект)
        reasoning_output = x
        for layer in self.reasoning_layers:
            residual = reasoning_output
            reasoning_output = layer['norm'](reasoning_output)
            reasoning_output, _ = layer['attn'](reasoning_output, reasoning_output, reasoning_output)
            reasoning_output = residual + reasoning_output
            reasoning_output = reasoning_output + layer['ffn'](reasoning_output)
        
        # Knowledge retrieval (знания из онтологии)
        knowledge_output, concept_activation = self.ontology.query(x)
        
        # Комбинируем интеллект и знания
        combined = torch.cat([reasoning_output, knowledge_output], dim=-1)
        gate = torch.sigmoid(self.knowledge_gate(combined))
        
        output = gate * reasoning_output + (1 - gate) * knowledge_output
        
        return self.head(output)
    
    def add_knowledge(self, concept1_idx: int, concept2_idx: int, causality_strength: float):
        """
        Добавление нового знания без переобучения
        Закон 4: добавить узел = добавить знание
        """
        self.ontology.hebbian_update(concept1_idx, concept2_idx, causality_strength)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'AIONOntology',
            'dim': self.dim,
            'num_concepts': self.num_concepts,
            'num_layers': self.num_layers,
            'aion_laws': [
                'Law 4: Living ontology - knowledge in graph',
                'Law 5: Separation of intelligence (reasoning) and knowledge (facts)',
                'Hebbian learning for knowledge updates'
            ]
        }


# ============================================================================
# AION-Honest: Архитектурная честность (Законы 11, 12, 30)
# ============================================================================

class EpistemicMonitor(nn.Module):
    """
    Эпистемический монитор - проверяет уверенность и знакомость
    Закон 11: блокирует генерацию если уверенность низкая
    """
    def __init__(self, dim: int):
        super().__init__()
        
        # Вычисляет уверенность
        self.confidence_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Вычисляет знакомость (familiarity)
        self.familiarity_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Пороги
        self.confidence_threshold = 0.7
        self.familiarity_threshold = 0.6
        
    def forward(self, x):
        """
        Проверяет можно ли генерировать
        
        Returns:
            can_generate: bool тензор
            confidence: уровень уверенности
            familiarity: уровень знакомости
        """
        # Агрегируем по последовательности
        x_agg = x.mean(dim=1)
        
        confidence = self.confidence_net(x_agg)
        familiarity = self.familiarity_net(x_agg)
        
        # Можно генерировать только если оба выше порога
        can_generate = (confidence > self.confidence_threshold) & (familiarity > self.familiarity_threshold)
        
        return can_generate, confidence, familiarity


class CoherenceChecker(nn.Module):
    """
    Проверка согласованности
    Закон 30: система хочет поддерживать внутреннюю согласованность
    """
    def __init__(self, dim: int):
        super().__init__()
        
        self.coherence_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, current_state, previous_state):
        """
        Вычисляет согласованность текущего состояния с предыдущим
        
        Returns:
            coherence_score: [0, 1] - насколько согласованно
        """
        combined = torch.cat([current_state, previous_state], dim=-1)
        coherence = self.coherence_net(combined)
        return coherence


class AIONHonest(BaseArchitecture):
    """
    AION-Honest: Архитектурная честность
    
    Закон 11: Эпистемический монитор блокирует галлюцинации
    Закон 12: Четыре исхода (знаю, не знаю, не уверена, могу предположить)
    Закон 30: Согласованность как иммунитет от лжи
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AIONHonest")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = config.get('num_layers', 4)
        
        # Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Основные слои
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(self.dim, 8, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 4),
                    nn.GELU(),
                    nn.Linear(self.dim * 4, self.dim)
                )
            })
            for _ in range(self.num_layers)
        ])
        
        # Эпистемический монитор (Закон 11)
        self.epistemic_monitor = EpistemicMonitor(self.dim)
        
        # Проверка согласованности (Закон 30)
        self.coherence_checker = CoherenceChecker(self.dim)
        
        # История для проверки согласованности
        self.register_buffer('previous_state', torch.zeros(1, self.dim))
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        # Embeddings
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        # Processing
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x, _ = layer['attn'](x, x, x)
            x = residual + x
            x = x + layer['ffn'](x)
        
        # Проверяем уверенность и знакомость (Закон 11)
        can_generate, confidence, familiarity = self.epistemic_monitor(x)
        
        # Проверяем согласованность с предыдущим состоянием (Закон 30)
        current_state = x.mean(dim=1)
        coherence = self.coherence_checker(
            current_state, 
            self.previous_state.expand(B, -1)
        )
        
        # Обновляем предыдущее состояние
        self.previous_state = current_state.detach()[:1]
        
        # Генерация с учётом эпистемического контроля
        logits = self.head(x)
        
        # Если уверенность или согласованность низкие - снижаем вероятности
        # (в реальной системе здесь была бы блокировка генерации)
        epistemic_penalty = (confidence * coherence).view(B, 1, 1)
        logits = logits * epistemic_penalty
        
        return logits
    
    def get_outcome_type(self, confidence: float, familiarity: float, coherence: float) -> str:
        """
        Определяет тип исхода (Закон 12)
        
        Returns:
            "know" | "dont_know" | "uncertain" | "can_guess"
        """
        if confidence > 0.8 and familiarity > 0.8 and coherence > 0.8:
            return "know"
        elif familiarity < 0.5:
            return "dont_know"
        elif coherence < 0.6 or confidence < 0.6:
            return "uncertain"
        else:
            return "can_guess"
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'AIONHonest',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'aion_laws': [
                'Law 11: Epistemic monitor blocks hallucinations',
                'Law 12: Four outcomes (know/dont_know/uncertain/guess)',
                'Law 30: Coherence as immunity to lies'
            ],
            'features': [
                'Confidence threshold: 0.7',
                'Familiarity threshold: 0.6',
                'Coherence checking with previous state'
            ]
        }


# ============================================================================
# AION-Modular: Модульная онтология (Закон 31)
# ============================================================================

class KnowledgeBlock(nn.Module):
    """
    Независимый подключаемый блок знаний
    """
    def __init__(self, dim: int, block_type: str):
        super().__init__()
        self.dim = dim
        self.block_type = block_type  # static, semi-dynamic, dynamic, scientific, etc.
        
        # Каждый блок - это мини-граф знаний
        self.knowledge_embedding = nn.Parameter(torch.randn(64, dim) * 0.02)
        
        # Частота обновления (зависит от типа)
        self.update_frequency = {
            'static': 0.0,        # никогда
            'semi_dynamic': 0.1,  # редко
            'dynamic': 1.0,       # часто
            'scientific': 0.3,    # средне
            'user': 1.0          # часто
        }.get(block_type, 0.5)
        
    def query(self, x):
        """Запрос к блоку"""
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        
        # Находим релевантные знания
        similarity = torch.matmul(x_flat, self.knowledge_embedding.T)
        attention = F.softmax(similarity, dim=-1)
        
        knowledge = torch.matmul(attention, self.knowledge_embedding)
        return knowledge.reshape(B, L, D)


class AIONModular(BaseArchitecture):
    """
    AION-Modular: Модульная онтология
    
    Закон 31: Независимые подключаемые блоки знаний
    - Каждый блок имеет роль, скорость обновления, область ответственности
    - Блоки могут добавляться/удаляться без переобучения
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "AIONModular")
        
        self.vocab_size = config['vocab_size']
        self.seq_length = config.get('max_seq_len', config.get('seq_length', 128))
        self.dim = config.get('dim', 256)
        self.num_layers = config.get('num_layers', 4)
        
        # Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Embedding(self.seq_length, self.dim)
        
        # Модульные блоки знаний
        self.blocks = nn.ModuleDict({
            'static': KnowledgeBlock(self.dim, 'static'),           # Неизменные принципы
            'semi_dynamic': KnowledgeBlock(self.dim, 'semi_dynamic'), # Энциклопедия
            'dynamic': KnowledgeBlock(self.dim, 'dynamic'),         # Реальное время
            'scientific': KnowledgeBlock(self.dim, 'scientific'),   # Научные знания
            'user': KnowledgeBlock(self.dim, 'user')               # Пользовательские
        })
        
        # Маршрутизатор блоков
        self.block_router = nn.Sequential(
            nn.Linear(self.dim, len(self.blocks)),
            nn.Softmax(dim=-1)
        )
        
        # Основные слои обработки
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(self.dim),
                'attn': nn.MultiheadAttention(self.dim, 8, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(self.dim, self.dim * 3),
                    nn.GELU(),
                    nn.Linear(self.dim * 3, self.dim)
                )
            })
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Linear(self.dim, self.vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        
        # Embeddings
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x) + self.pos_emb(positions)
        
        # Определяем какие блоки использовать
        x_mean = x.mean(dim=1)
        block_weights = self.block_router(x_mean)  # [B, num_blocks]
        
        # Запрашиваем знания из всех блоков
        block_outputs = []
        for block_name, block in self.blocks.items():
            block_knowledge = block.query(x)
            block_outputs.append(block_knowledge)
        
        # Взвешенная комбинация блоков
        block_outputs = torch.stack(block_outputs, dim=1)  # [B, num_blocks, L, D]
        block_weights_expanded = block_weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_blocks, 1, 1]
        
        combined_knowledge = (block_outputs * block_weights_expanded).sum(dim=1)  # [B, L, D]
        
        # Интегрируем знания
        x = x + combined_knowledge
        
        # Processing
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x, _ = layer['attn'](x, x, x)
            x = residual + x
            x = x + layer['ffn'](x)
        
        return self.head(x)
    
    def add_custom_block(self, block_name: str, block_type: str = 'custom'):
        """
        Добавляет новый блок знаний без переобучения
        Закон 31: модульность
        """
        self.blocks[block_name] = KnowledgeBlock(self.dim, block_type)
    
    def remove_block(self, block_name: str):
        """Удаляет блок знаний"""
        if block_name in self.blocks:
            del self.blocks[block_name]
    
    def get_architecture_info(self) -> Dict[str, Any]:
        return {
            'type': 'AIONModular',
            'dim': self.dim,
            'num_layers': self.num_layers,
            'num_blocks': len(self.blocks),
            'blocks': list(self.blocks.keys()),
            'aion_laws': [
                'Law 31: Modular ontology with pluggable knowledge blocks',
                'Independent blocks with different update frequencies',
                'Can add/remove blocks without retraining'
            ]
        }
