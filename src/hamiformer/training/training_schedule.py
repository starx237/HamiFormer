from __future__ import annotations

import math
from typing import Any

import torch


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    maximum: float,
    minimum: float,
) -> float:
    if not 1 <= step <= total_steps:
        raise ValueError('learning-rate step lies outside its update budget')
    if not 0 <= warmup_steps < total_steps:
        raise ValueError('warmup must lie in [0,total_steps)')
    if not (math.isfinite(maximum) and math.isfinite(minimum) and maximum >= minimum > 0.0):
        raise ValueError('learning-rate endpoints must be finite positive')
    if warmup_steps and step <= warmup_steps:
        return maximum * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError('optimizer learning rate must be finite positive')
    for group in optimizer.param_groups:
        group['lr'] = float(value)


class ExplicitEpochBatchStream:
    def __init__(self, size: int, batch_size: int, *, seed: int) -> None:
        if min(int(size), int(batch_size)) < 1:
            raise ValueError('stream size and batch size must be positive')
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.epoch = 0
        self.offset = 0
        self.permutation = torch.randperm(self.size, generator=self.generator)

    def _next_epoch(self) -> None:
        self.epoch += 1
        self.offset = 0
        self.permutation = torch.randperm(self.size, generator=self.generator)

    def next_indices(self) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        needed = self.batch_size
        while needed:
            available = self.size - self.offset
            take = min(needed, available)
            chunks.append(self.permutation[self.offset:self.offset + take])
            self.offset += take
            needed -= take
            if self.offset == self.size:
                self._next_epoch()
        return torch.cat(chunks, dim=0)

    def state_dict(self) -> dict[str, Any]:
        return {
            'size': self.size,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'offset': self.offset,
            'permutation': self.permutation.clone(),
            'generator_state': self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get('size', -1)) != self.size or int(state.get('batch_size', -1)) != self.batch_size:
            raise ValueError('batch-stream shape contract differs from checkpoint')
        permutation = state.get('permutation')
        generator_state = state.get('generator_state')
        if not isinstance(permutation, torch.Tensor) or permutation.shape != (self.size,):
            raise ValueError('checkpoint has an invalid epoch permutation')
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError('checkpoint lacks a batch-stream RNG state')
        epoch, offset = int(state.get('epoch', -1)), int(state.get('offset', -1))
        if epoch < 0 or not 0 <= offset < self.size:
            raise ValueError('checkpoint has an invalid epoch/offset ledger')
        self.epoch = epoch
        self.offset = offset
        self.permutation = permutation.clone().to(device='cpu', dtype=torch.long)
        self.generator.set_state(generator_state.clone().to(device='cpu'))


def generator_state_dict(generators: dict[str, torch.Generator]) -> dict[str, torch.Tensor]:
    return {name: generator.get_state().cpu() for name, generator in generators.items()}


def load_generator_state_dict(
    generators: dict[str, torch.Generator],
    state: dict[str, torch.Tensor],
) -> None:
    if set(state) != set(generators):
        raise ValueError('explicit RNG ledger names differ from checkpoint')
    for name, generator in generators.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'RNG state {name} is not a tensor')
        generator.set_state(value.cpu())


__all__ = [
    'ExplicitEpochBatchStream',
    'cosine_learning_rate',
    'generator_state_dict',
    'load_generator_state_dict',
    'set_optimizer_learning_rate',
]
