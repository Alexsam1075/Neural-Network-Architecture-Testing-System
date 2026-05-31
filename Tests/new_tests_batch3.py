import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .base_test import BaseTest
from .anti_cheat import remap_tokens


class LengthExtrapolationTest(BaseTest):
    """Train on short delays, evaluate on longer unseen delays."""

    def __init__(self, config: Dict[str, Any], name: str = "LengthExtrapolationTest"):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_samples', config.get('num_sequences', 512))
        self.num_eval_samples = config.get('num_eval_samples', max(128, self.num_samples // 4))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 6)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.num_keys = 12
        self.noise_base = 120
        self.query_token = 251
        self.anti_cheat_token_permutation = config.get('anti_cheat_token_permutation', True)
        self.anti_cheat_seed = config.get('anti_cheat_seed', 1337)
        self.random_seed = config.get('random_seed', self.anti_cheat_seed)

    def _make_data(self, n: int, min_delay: int, max_delay: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        span = max_delay - min_delay + 1
        for i in range(n):
            key = i % self.num_keys + 1
            delay = min_delay + (i % span)
            seq = [key]
            seq.extend(self.noise_base + ((i + j) % 20) for j in range(delay))
            seq.append(self.query_token)
            seq.extend([self.noise_base] * max(0, self.seq_length - len(seq)))
            xs.append(torch.tensor(seq[:self.seq_length], dtype=torch.long))
            ys.append(key)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        if self.anti_cheat_token_permutation:
            x, y = remap_tokens(
                (x, y),
                self.config.get('vocab_size', 256),
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.query_token,),
            )
        return x, y

    def prepare_data(self):
        return self._make_data(self.num_samples, 1, max(2, self.seq_length // 3), self.device)

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        train_x, train_y = self._make_data(self.num_samples, 1, max(2, self.seq_length // 3), device)
        test_x, test_y = self._make_data(self.num_eval_samples, self.seq_length // 2, self.seq_length - 3, device)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        start = time.time()
        losses = []
        model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                qpos = (xb == self.query_token).long().argmax(dim=1)
                pred = out[torch.arange(len(xb), device=device), qpos, :]
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / max(1, len(loader)))
        training_time = time.time() - start

        def accuracy(x, y):
            correct = total = 0
            with torch.no_grad():
                for i in range(0, len(x), self.batch_size):
                    xb, yb = x[i:i + self.batch_size], y[i:i + self.batch_size]
                    out = model(xb)
                    qpos = (xb == self.query_token).long().argmax(dim=1)
                    pred = out[torch.arange(len(xb), device=device), qpos, :].argmax(dim=-1)
                    correct += (pred == yb).sum().item()
                    total += len(yb)
            return correct, total

        model.eval()
        eval_start = time.time()
        train_correct, train_total = accuracy(train_x, train_y)
        test_correct, test_total = accuracy(test_x, test_y)
        eval_time = time.time() - eval_start
        train_acc = train_correct / train_total if train_total else 0
        test_acc = test_correct / test_total if test_total else 0

        return {
            'accuracy': test_acc,
            'train_accuracy': train_acc,
            'extrapolation_gap': train_acc - test_acc,
            'correct_predictions': test_correct,
            'total_predictions': test_total,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': (train_total + test_total) / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'num_eval_samples': self.num_eval_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


class DistractorRetrievalTest(BaseTest):
    """Copy the cued key while ignoring many distractor key-value pairs."""

    def __init__(self, config: Dict[str, Any], name: str = "DistractorRetrievalTest"):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_samples', config.get('num_sequences', 512))
        self.num_eval_samples = config.get('num_eval_samples', max(128, self.num_samples // 4))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 6)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.query_token = 252
        self.pad_token = 0
        self.anti_cheat_token_permutation = config.get('anti_cheat_token_permutation', True)
        self.anti_cheat_seed = config.get('anti_cheat_seed', 1337)
        self.random_seed = config.get('random_seed', self.anti_cheat_seed)

    def _make_sample(self, i: int):
        gen = torch.Generator()
        gen.manual_seed(int(self.random_seed) + int(i) * 1_000_003)
        num_pairs = max(2, (self.seq_length - 2) // 2)
        target_key = int(torch.randint(1, 21, (1,), generator=gen).item())
        values = (80 + torch.randperm(40, generator=gen)).tolist()
        target_value = int(values[0])
        target_slot = int(torch.randint(0, num_pairs, (1,), generator=gen).item())
        value_idx = 1

        seq = []
        for j in range(num_pairs):
            if j == target_slot:
                seq.extend([target_key, target_value])
                continue
            key = int(torch.randint(21, 41, (1,), generator=gen).item())
            seq.extend([key, int(values[value_idx % len(values)])])
            value_idx += 1
        seq.append(target_value)
        seq.append(self.query_token)
        seq.extend([self.pad_token] * max(0, self.seq_length - len(seq)))
        return torch.tensor(seq[:self.seq_length], dtype=torch.long), target_key

    def prepare_data(self, device, count=None, offset=0):
        xs, ys = [], []
        count = self.num_samples if count is None else count
        for i in range(count):
            x, y = self._make_sample(i + offset)
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        if self.anti_cheat_token_permutation:
            x, y = remap_tokens(
                (x, y),
                self.config.get('vocab_size', 256),
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.pad_token, self.query_token),
            )
        return x, y

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        x, y = self.prepare_data(device, self.num_samples, int(self.random_seed))
        x_eval, y_eval = self.prepare_data(device, self.num_eval_samples, int(self.random_seed) + 10_000)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_eval, y_eval),
            batch_size=self.batch_size,
            shuffle=False,
        )

        start = time.time()
        losses = []
        model.train()
        for _ in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                qpos = (xb == self.query_token).long().argmax(dim=1)
                pred = out[torch.arange(len(xb), device=device), qpos, :]
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / max(1, len(loader)))
        training_time = time.time() - start

        model.eval()
        eval_start = time.time()
        correct = total = near_miss = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                out = model(xb)
                qpos = (xb == self.query_token).long().argmax(dim=1)
                pred = out[torch.arange(len(xb), device=device), qpos, :].argmax(dim=-1)
                correct += (pred == yb).sum().item()
                near_miss += ((pred >= 21) & (pred <= 40)).sum().item()
                total += len(yb)
        eval_time = time.time() - eval_start

        return {
            'accuracy': correct / total if total else 0,
            'correct_predictions': correct,
            'total_predictions': total,
            'distractor_pick_rate': near_miss / total if total else 0,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': total / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'num_eval_samples': self.num_eval_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }


class CausalMaskStabilityTest(BaseTest):
    """Detect models that rely on future leakage or brittle token positions."""

    def __init__(self, config: Dict[str, Any], name: str = "CausalMaskStabilityTest"):
        super().__init__(config, name)
        self.seq_length = config.get('seq_length', 32)
        self.num_samples = config.get('num_samples', config.get('num_sequences', 512))
        self.num_eval_samples = config.get('num_eval_samples', max(128, self.num_samples // 4))
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 6)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.mask_token = 253
        self.anti_cheat_token_permutation = config.get('anti_cheat_token_permutation', True)
        self.anti_cheat_seed = config.get('anti_cheat_seed', 1337)
        self.random_seed = config.get('random_seed', self.anti_cheat_seed)

    def _make_data(self, device, count=None, offset=0):
        xs, ys = [], []
        count = self.num_samples if count is None else count
        for i in range(count):
            sample_id = i + offset
            a = (sample_id % 30) + 1
            b = ((sample_id * 5) % 30) + 31
            seq = [a if j % 2 == 0 else b for j in range(self.seq_length)]
            target = [b if j % 2 == 0 else a for j in range(self.seq_length)]
            xs.append(torch.tensor(seq, dtype=torch.long))
            ys.append(torch.tensor(target, dtype=torch.long))
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        if self.anti_cheat_token_permutation:
            x, y = remap_tokens(
                (x, y),
                self.config.get('vocab_size', 256),
                self.name,
                self.anti_cheat_seed,
                protected_tokens=(self.mask_token,),
            )
        return x, y

    def prepare_data(self):
        return self._make_data(self.device)

    def _masked(self, xb: torch.Tensor) -> torch.Tensor:
        masked = xb.clone()
        mask = (torch.arange(xb.shape[1], device=xb.device).unsqueeze(0) % 7 == 3)
        masked[mask.expand_as(masked)] = self.mask_token
        return masked

    def run(self, model: nn.Module) -> Dict[str, Any]:
        device = next(model.parameters()).device
        x, y = self._make_data(device, self.num_samples, int(self.random_seed))
        x_eval, y_eval = self._make_data(device, self.num_eval_samples, int(self.random_seed) + 10_000)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=True,
        )
        eval_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_eval, y_eval),
            batch_size=self.batch_size,
            shuffle=False,
        )

        start = time.time()
        losses = []
        model.train()
        for _ in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out.reshape(-1, out.shape[-1]), yb.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / max(1, len(loader)))
        training_time = time.time() - start

        model.eval()
        eval_start = time.time()
        clean_correct = masked_correct = flips = total = 0
        with torch.no_grad():
            for xb, yb in eval_loader:
                clean_pred = model(xb).argmax(dim=-1)
                masked_pred = model(self._masked(xb)).argmax(dim=-1)
                clean_correct += (clean_pred == yb).sum().item()
                masked_correct += (masked_pred == yb).sum().item()
                flips += (clean_pred != masked_pred).sum().item()
                total += yb.numel()
        eval_time = time.time() - eval_start
        clean_acc = clean_correct / total if total else 0
        masked_acc = masked_correct / total if total else 0

        return {
            'accuracy': masked_acc,
            'clean_accuracy': clean_acc,
            'masked_accuracy': masked_acc,
            'mask_degradation': clean_acc - masked_acc,
            'prediction_flip_rate': flips / total if total else 0,
            'correct_predictions': masked_correct,
            'total_predictions': total,
            'training_time_seconds': training_time,
            'evaluation_time_seconds': eval_time,
            'inference_speed': (2 * self.num_eval_samples) / eval_time if eval_time > 0 else 0,
            'train_loss_initial': losses[0] if losses else 0,
            'train_loss_final': losses[-1] if losses else 0,
            'num_samples': self.num_samples,
            'num_eval_samples': self.num_eval_samples,
            'sequence_length': self.seq_length,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
        }
