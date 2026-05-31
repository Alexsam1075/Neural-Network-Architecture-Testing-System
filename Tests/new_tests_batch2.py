"""
New Tests Batch 2
Три новых теста для выявления будущих проблем нейросети:
1. LongRangeTest - длинные зависимости (выявляет gradient vanishing, забывание)
2. NoiseRobustnessTest - устойчивость к шуму (выявляет overfit на train distribution)
3. GeneralizationTest - обобщение на unseen паттерны (выявляет memorization vs learning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List
import time
import numpy as np
import random
from .base_test import BaseTest
from .anti_cheat import remap_tokens


# ============================================================================
# 1. LongRange Dependency Test
# Тест на длинные зависимости - выявляет gradient vanishing и проблемы памяти
# Модель должна помнить токен из начала последовательности и повторить его в конце
# ============================================================================

class LongRangeDependencyTest(BaseTest):
    """
    Тест на длинные зависимости.
    
    Генерирует последовательности где ответ зависит от токена в начале:
    [key_token, ...noise..., query_token] -> key_token
    
    Выявляет:
    - Gradient vanishing/exploding
    - Неспособность запоминать долгосрочные зависимости  
    - Проблемы с positional encoding на длинных последовательностях
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "LongRangeTest"):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_sequences', config.get('num_samples', 1000))
        self.num_eval_samples = config.get('num_eval_samples', max(128, self.num_samples // 4))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 10)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.vocab_size = config.get('vocab_size', 256)
        
        # Специальные токены
        self.NUM_KEYS = 10       # 10 различных "ключей" которые надо запомнить
        self.NOISE_TOKEN = 100   # токен шума
        self.QUERY_TOKEN = 250   # токен запроса; не должен пересекаться с шумом
        self.anti_cheat_token_permutation = config.get('anti_cheat_token_permutation', True)
        self.anti_cheat_seed = config.get('anti_cheat_seed', 1337)
        self.random_seed = config.get('random_seed', self.anti_cheat_seed)
        
    def generate_sample(self, key_id: int, noise_len: int) -> Tuple[torch.Tensor, int]:
        """
        Генерирует: [key_id, noise..., QUERY_TOKEN] -> key_id
        noise_len контролирует расстояние зависимости
        """
        seq = [key_id]
        # Шум в середине
        for _ in range(noise_len):
            seq.append(self.NOISE_TOKEN + (hash(len(seq)) % 5))
        seq.append(self.QUERY_TOKEN)
        
        # Паддинг до seq_length
        while len(seq) < self.seq_length:
            seq.append(self.NOISE_TOKEN)
        seq = seq[:self.seq_length]
        
        return torch.tensor(seq, dtype=torch.long), key_id
    
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        
        offset = int(getattr(self, "_data_offset", self.random_seed))
        count = int(getattr(self, "_data_count", self.num_samples))
        for i in range(count):
            sample_id = i + offset
            key_id = sample_id % self.NUM_KEYS + 1  # ключи 1..10
            # Варьируем расстояние зависимости
            noise_len = min(sample_id % (self.seq_length - 5) + 1, self.seq_length - 3)
            seq, target = self.generate_sample(key_id, noise_len)
            sequences.append(seq)
            targets.append(target)
        
        x = torch.stack(sequences)
        y = torch.tensor(targets, dtype=torch.long)
        if self.anti_cheat_token_permutation:
            x, y = remap_tokens(
                (x, y),
                self.vocab_size,
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.QUERY_TOKEN,),
            )
        return x, y
    
    def run(self, model: nn.Module) -> Dict[str, Any]:
        model.train()
        device = next(model.parameters()).device
        
        X, y = self.prepare_data()
        X, y = X.to(device), y.to(device)
        self._data_offset = int(self.random_seed) + 10_000
        self._data_count = self.num_eval_samples
        X_eval, y_eval = self.prepare_data()
        del self._data_offset
        del self._data_count
        X_eval, y_eval = X_eval.to(device), y_eval.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_start = time.time()
        initial_loss = None
        final_loss = None
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_eval, y_eval),
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)  # B, L, vocab
                # Берём предсказание на последнем токене (позиция query)
                query_positions = (xb == self.QUERY_TOKEN).long().argmax(dim=1)
                # Безопасно получаем выход на позиции query
                batch_idx = torch.arange(len(xb), device=device)
                pred = out[batch_idx, query_positions, :]  # B, vocab
                loss = criterion(pred, yb)
                loss.backward()
                # Мониторим gradient norm для выявления vanishing/exploding
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss
        
        training_time = time.time() - train_start
        
        # Оценка
        model.eval()
        eval_start = time.time()
        
        with torch.no_grad():
            correct = 0
            total = 0
            # Тест на разных дистанциях
            short_correct = 0
            long_correct = 0
            short_total = 0
            long_total = 0
            
            for xb, yb in eval_loader:
                out = model(xb)
                query_positions = (xb == self.QUERY_TOKEN).long().argmax(dim=1)
                batch_idx = torch.arange(len(xb), device=device)
                pred = out[batch_idx, query_positions, :].argmax(dim=-1)
                
                correct += (pred == yb).sum().item()
                total += len(yb)
                
                # Разделяем по дистанции
                for j in range(len(xb)):
                    dist = query_positions[j].item()
                    is_correct = (pred[j] == yb[j]).item()
                    if dist <= self.seq_length // 3:
                        short_correct += is_correct
                        short_total += 1
                    else:
                        long_correct += is_correct
                        long_total += 1
        
        eval_time = time.time() - eval_start
        accuracy = correct / total if total > 0 else 0
        
        # Gradient health check
        grad_norms = []
        xb_sample, yb_sample = X[:self.batch_size], y[:self.batch_size]
        model.train()
        optimizer.zero_grad()
        out = model(xb_sample)
        qpos = (xb_sample == self.QUERY_TOKEN).long().argmax(dim=1)
        bidx = torch.arange(len(xb_sample), device=device)
        pred_sample = out[bidx, qpos, :]
        loss_sample = criterion(pred_sample, yb_sample)
        loss_sample.backward()
        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())
        
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
        max_grad_norm = np.max(grad_norms) if grad_norms else 0
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'short_range_accuracy': short_correct / short_total if short_total > 0 else 0,
            'long_range_accuracy': long_correct / long_total if long_total > 0 else 0,
            'long_range_gap': (short_correct / short_total if short_total > 0 else 0) - 
                             (long_correct / long_total if long_total > 0 else 0),
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': total / eval_time if eval_time > 0 else 0,
            'training_speed': (self.num_samples * self.epochs) / training_time if training_time > 0 else 0,
            'train_loss_final': final_loss,
            'train_loss_initial': initial_loss,
            'avg_gradient_norm': avg_grad_norm,
            'max_gradient_norm': max_grad_norm,
            'gradient_health': 'ok' if 0.001 < avg_grad_norm < 10 else ('vanishing' if avg_grad_norm < 0.001 else 'exploding'),
            'num_samples': self.num_samples,
            'num_eval_samples': self.num_eval_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


# ============================================================================
# 2. Noise Robustness Test
# Тест на устойчивость к шуму - выявляет overfit и хрупкость модели
# ============================================================================

class NoiseRobustnessTest(BaseTest):
    """
    Тест на устойчивость к шуму.
    
    Обучает модель на чистых последовательностях,
    тестирует на зашумлённых (замена случайных токенов).
    
    Выявляет:
    - Overfit на тренировочное распределение
    - Хрупкость к входным пертурбациям
    - Способность к robust generalization
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "NoiseRobustnessTest"):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_sequences', config.get('num_samples', 1000))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 10)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.vocab_size = config.get('vocab_size', 256)
        self.noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]  # доля замененных токенов
        
    def generate_clean_sequence(self, pattern_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Генерирует чистую последовательность и её таргет"""
        pattern_type = pattern_id % 4
        if pattern_type == 0:
            # Арифметическая: предскажи следующий
            start = (pattern_id * 3) % 100
            step = (pattern_id % 5) + 1
            seq = [(start + i * step) % 200 + 1 for i in range(self.seq_length)]
            target = [(start + (i + 1) * step) % 200 + 1 for i in range(self.seq_length)]
        elif pattern_type == 1:
            # Повтор паттерна
            p_len = (pattern_id % 4) + 2
            base = [(pattern_id * 7 + i) % 50 + 1 for i in range(p_len)]
            seq = [base[i % p_len] for i in range(self.seq_length)]
            target = [base[(i + 1) % p_len] for i in range(self.seq_length)]
        elif pattern_type == 2:
            # XOR паттерн
            a = (pattern_id % 10) + 1
            b = ((pattern_id // 10) % 10) + 1
            seq = [a if i % 2 == 0 else b for i in range(self.seq_length)]
            target = [b if i % 2 == 0 else a for i in range(self.seq_length)]
        else:
            # Счётчик с wrap
            mod = (pattern_id % 8) + 3
            seq = [i % mod + 1 for i in range(self.seq_length)]
            target = [(i + 1) % mod + 1 for i in range(self.seq_length)]
        
        x = torch.tensor(seq[:self.seq_length], dtype=torch.long)
        y = torch.tensor(target[:self.seq_length], dtype=torch.long)
        return x, y
    
    def add_noise(self, seq: torch.Tensor, noise_ratio: float) -> torch.Tensor:
        """Добавляет шум: случайно заменяет noise_ratio долю токенов"""
        if noise_ratio == 0:
            return seq.clone()
        noisy = seq.clone()
        mask = torch.rand(len(seq), device=seq.device) < noise_ratio
        random_tokens = torch.randint(
            1,
            min(200, self.vocab_size),
            (mask.sum().item(),),
            device=seq.device,
        )
        noisy[mask] = random_tokens
        return noisy
    
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        for i in range(self.num_samples):
            x, y = self.generate_clean_sequence(i)
            sequences.append(x)
            targets.append(y)
        return torch.stack(sequences), torch.stack(targets)
    
    def run(self, model: nn.Module) -> Dict[str, Any]:
        model.train()
        device = next(model.parameters()).device
        
        X_clean, Y_clean = self.prepare_data()
        X_clean, Y_clean = X_clean.to(device), Y_clean.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(X_clean, Y_clean)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        train_start = time.time()
        initial_loss = None
        final_loss = None
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out.reshape(-1, out.shape[-1]), yb.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(loader)
            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss
        
        training_time = time.time() - train_start
        
        # Тест при разных уровнях шума
        model.eval()
        noise_accuracies = {}
        
        eval_start = time.time()
        with torch.no_grad():
            for noise_level in self.noise_levels:
                correct = 0
                total = 0
                for xb, yb in loader:
                    # Добавляем шум к входу
                    xb_noisy = torch.stack([self.add_noise(x, noise_level) for x in xb]).to(device)
                    out = model(xb_noisy)
                    pred = out.argmax(dim=-1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
                noise_accuracies[f'noise_{int(noise_level*100)}pct'] = correct / total if total > 0 else 0
        
        eval_time = time.time() - eval_start
        
        # Базовая точность (без шума)
        base_acc = noise_accuracies['noise_0pct']
        # Деградация точности при максимальном шуме
        max_noise_acc = noise_accuracies[f'noise_{int(self.noise_levels[-1]*100)}pct']
        robustness_score = max_noise_acc / base_acc if base_acc > 0 else 0
        
        result = {
            'accuracy': base_acc,
            'correct_predictions': int(base_acc * self.num_samples * self.seq_length),
            'total_predictions': self.num_samples * self.seq_length,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': (self.num_samples * len(self.noise_levels)) / eval_time if eval_time > 0 else 0,
            'training_speed': (self.num_samples * self.epochs) / training_time if training_time > 0 else 0,
            'train_loss_final': final_loss,
            'train_loss_initial': initial_loss,
            'robustness_score': robustness_score,  # 1.0 = идеальная робастность
            'accuracy_degradation_at_30pct_noise': base_acc - max_noise_acc,
            'num_samples': self.num_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }
        result.update(noise_accuracies)
        return result


# ============================================================================
# 3. Generalization Test
# Тест на обобщение - выявляет memorization vs real learning
# ============================================================================

class GeneralizationTest(BaseTest):
    """
    Тест на обобщение на unseen паттерны.
    
    Обучает на одних параметрах паттернов, тестирует на других.
    
    Выявляет:
    - Memorization вместо реального обучения
    - Способность к compositional generalization  
    - Устойчивость к distribution shift
    - Настоящий OOD (out-of-distribution) performance
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "GeneralizationTest"):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_sequences', config.get('num_samples', 1000))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 10)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.vocab_size = config.get('vocab_size', 256)
        
    def generate_arithmetic(self, start: int, step: int, length: int) -> List[int]:
        return [(start + i * step) % 200 + 1 for i in range(length)]
    
    def generate_geometric_mod(self, start: int, ratio: int, length: int) -> List[int]:
        seq = []
        val = start
        for _ in range(length):
            seq.append(val % 200 + 1)
            val = (val * ratio) % 500 + 1
        return seq
    
    def make_sample(self, start: int, step: int, use_geo: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_geo:
            seq = self.generate_geometric_mod(start, step, self.seq_length + 1)
        else:
            seq = self.generate_arithmetic(start, step, self.seq_length + 1)
        x = torch.tensor(seq[:self.seq_length], dtype=torch.long)
        y = torch.tensor(seq[1:self.seq_length + 1], dtype=torch.long)
        return x, y
    
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Тренировочные данные: starts 0-49, steps 1-9"""
        sequences, targets = [], []
        for i in range(self.num_samples):
            start = (i * 7) % 50          # starts: 0-49
            step = (i % 9) + 1            # steps: 1-9
            use_geo = i % 5 == 0
            x, y = self.make_sample(start, step, use_geo)
            sequences.append(x)
            targets.append(y)
        return torch.stack(sequences), torch.stack(targets)
    
    def prepare_ood_data(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """OOD данные: starts 100-149, steps 11-19 (unseen range)"""
        sequences, targets = [], []
        for i in range(self.num_samples // 4):
            start = (i * 7) % 50 + 100    # starts: 100-149 (OOD)
            step = (i % 9) + 11           # steps: 11-19 (OOD)
            use_geo = i % 5 == 0
            x, y = self.make_sample(start, step, use_geo)
            sequences.append(x)
            targets.append(y)
        return torch.stack(sequences).to(device), torch.stack(targets).to(device)
    
    def run(self, model: nn.Module) -> Dict[str, Any]:
        model.train()
        device = next(model.parameters()).device
        
        X, Y = self.prepare_data()
        X, Y = X.to(device), Y.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        n_train = int(0.8 * len(X))
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_val, Y_val = X[n_train:], Y[n_train:]
        
        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        train_start = time.time()
        initial_loss = None
        final_loss = None
        
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out.reshape(-1, out.shape[-1]), yb.reshape(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            if epoch == 0:
                initial_loss = avg_loss
            final_loss = avg_loss
        
        training_time = time.time() - train_start
        
        model.eval()
        eval_start = time.time()
        
        def eval_acc(Xd, Yd):
            correct = total = 0
            with torch.no_grad():
                for i in range(0, len(Xd), self.batch_size):
                    xb = Xd[i:i+self.batch_size]
                    yb = Yd[i:i+self.batch_size]
                    out = model(xb)
                    pred = out.argmax(dim=-1)
                    correct += (pred == yb).sum().item()
                    total += yb.numel()
            return correct / total if total > 0 else 0
        
        # Train accuracy
        train_acc = eval_acc(X_train, Y_train)
        # Validation accuracy (same distribution)
        val_acc = eval_acc(X_val, Y_val)
        # OOD accuracy (distribution shift)
        X_ood, Y_ood = self.prepare_ood_data(device)
        ood_acc = eval_acc(X_ood, Y_ood)
        
        eval_time = time.time() - eval_start
        
        # Generalization gap: разница train vs val
        gen_gap = train_acc - val_acc
        # OOD gap: разница val vs ood
        ood_gap = val_acc - ood_acc
        # Generalization score: насколько хорошо модель обобщает (0=memorize, 1=perfect generalize)
        gen_score = ood_acc / train_acc if train_acc > 0 else 0
        
        return {
            'accuracy': val_acc,
            'correct_predictions': int(val_acc * len(X_val) * self.seq_length),
            'total_predictions': len(X_val) * self.seq_length,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'ood_accuracy': ood_acc,
            'generalization_gap': gen_gap,      # маленький = хорошо (нет overfit)
            'ood_gap': ood_gap,                  # маленький = хорошо (robust OOD)
            'generalization_score': gen_score,   # близко к 1.0 = хорошо
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': (len(X_val) + len(X_ood)) / eval_time if eval_time > 0 else 0,
            'training_speed': (n_train * self.epochs) / training_time if training_time > 0 else 0,
            'train_loss_final': final_loss,
            'train_loss_initial': initial_loss,
            'num_samples': self.num_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }
