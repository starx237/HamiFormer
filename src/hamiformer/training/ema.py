from __future__ import annotations
from collections.abc import Iterator
import torch
from torch import nn

class ExponentialMovingAverage:

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError('EMA decay 必须位于 (0,1)')
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError('EMA state keys 与模型不一致')
        for name, value in current.items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError('待恢复 EMA 的 state keys 与模型不一致')
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def tensors(self) -> Iterator[torch.Tensor]:
        yield from self.shadow.values()
