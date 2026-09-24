from __future__ import annotations
import torch
from torch import nn
from hamiformer.models import DirectVectorExpert, HamiltonianExpert, PhaseDiT
from hamiformer.types import FieldOutput
from .rectified_flow import clean_to_velocity

class DField(nn.Module):

    def __init__(self, model: PhaseDiT, *, mixed_precision: str='fp32', t_eps: float=0.05) -> None:
        super().__init__()
        if mixed_precision not in {'fp32', 'bf16'}:
            raise ValueError('DField mixed_precision 只能是 fp32 或 bf16')
        self.model = model
        self.mixed_precision = mixed_precision
        self.t_eps = float(t_eps)

    def forward(self, noisy: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> FieldOutput:
        enabled = noisy.device.type == 'cuda' and self.mixed_precision == 'bf16'
        with torch.autocast(device_type=noisy.device.type, dtype=torch.bfloat16, enabled=enabled):
            clean = self.model(noisy, tau, x0=x0, attrs=attrs, physical_time=physical_time)
        return FieldOutput(clean=clean, velocity=clean_to_velocity(clean, noisy, tau, t_eps=self.t_eps))

class HField(nn.Module):

    def __init__(self, model: HamiltonianExpert, *, q_scale: torch.Tensor, p_scale: torch.Tensor, t_eps: float=0.05) -> None:
        super().__init__()
        self.model = model
        self.register_buffer('q_scale', q_scale.float().clone())
        self.register_buffer('p_scale', p_scale.float().clone())
        self.t_eps = float(t_eps)

    def forward(self, noisy: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> FieldOutput:
        output = self.model(noisy, tau, x0=x0, attrs=attrs, physical_time=physical_time, q_scale=self.q_scale, p_scale=self.p_scale, create_graph=False)
        velocity = clean_to_velocity(output.clean, noisy, tau, t_eps=self.t_eps)
        return FieldOutput(clean=output.clean, velocity=velocity, disagreement=output.disagreement, diagnostics={'occurrences': output.occurrences})

class SField(nn.Module):

    def __init__(self, model: DirectVectorExpert, *, q_scale: torch.Tensor, p_scale: torch.Tensor, t_eps: float=0.05) -> None:
        super().__init__()
        self.model = model
        self.register_buffer('q_scale', q_scale.float().clone())
        self.register_buffer('p_scale', p_scale.float().clone())
        self.t_eps = float(t_eps)

    def forward(self, noisy: torch.Tensor, tau: torch.Tensor, **conditions: torch.Tensor) -> FieldOutput:
        output = self.model(noisy, tau, q_scale=self.q_scale, p_scale=self.p_scale, **conditions)
        return FieldOutput(clean=output.clean, velocity=clean_to_velocity(output.clean, noisy, tau, t_eps=self.t_eps), disagreement=output.disagreement, diagnostics={'occurrences': output.occurrences})
