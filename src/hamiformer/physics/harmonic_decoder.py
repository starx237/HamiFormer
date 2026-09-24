from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class DecodedPhaseMass:
    phase: torch.Tensor
    mass: torch.Tensor

class PhaseMassNormalizer(nn.Module):

    def __init__(self, *, num_objects: int, state_scale: torch.Tensor, log_mass_center: float, log_mass_scale: float, mass_min: float, mass_max: float) -> None:
        super().__init__()
        if num_objects <= 0:
            raise ValueError('num_objects 必须为正')
        if state_scale.ndim != 1 or not bool((state_scale > 0).all().item()):
            raise ValueError('state_scale 必须是一维有限正数')
        if log_mass_scale <= 0.0:
            raise ValueError('log_mass_scale 必须为正')
        if not 0.0 < mass_min < mass_max:
            raise ValueError('质量支持必须满足 0 < mass_min < mass_max')
        self.num_objects = int(num_objects)
        self.state_dim = int(state_scale.numel())
        self.object_latent_dim = self.state_dim + 1
        self.register_buffer('state_scale', state_scale.detach().float().clone())
        self.register_buffer('log_mass_center', torch.tensor(float(log_mass_center), dtype=torch.float32))
        self.register_buffer('log_mass_scale', torch.tensor(float(log_mass_scale), dtype=torch.float32))
        self.register_buffer('log_mass_min', torch.tensor(math.log(float(mass_min)), dtype=torch.float32))
        self.register_buffer('log_mass_max', torch.tensor(math.log(float(mass_max)), dtype=torch.float32))

    @property
    def latent_dim(self) -> int:
        return self.num_objects * self.object_latent_dim

    def pack(self, phase: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        if phase.ndim != 3 or phase.shape[1:] != (self.num_objects, self.state_dim):
            raise ValueError('phase 必须为 [B,K,state_dim]')
        if mass.shape != phase.shape[:2]:
            raise ValueError('mass 必须为 [B,K]')
        if not bool((mass > 0).all().item()):
            raise ValueError('mass 必须为正')
        state = phase / self.state_scale.to(phase).view(1, 1, -1)
        log_mass = torch.log(mass)
        mass_latent = (log_mass - self.log_mass_center.to(log_mass)) / self.log_mass_scale.to(log_mass)
        objects = torch.cat([state, mass_latent.unsqueeze(-1)], dim=-1)
        return objects.reshape(phase.shape[0], self.latent_dim)

    def unpack(self, latent: torch.Tensor) -> DecodedPhaseMass:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError('latent 必须为 [B,latent_dim]')
        objects = latent.reshape(latent.shape[0], self.num_objects, self.object_latent_dim)
        phase = objects[..., :self.state_dim] * self.state_scale.to(latent).view(1, 1, -1)
        log_mass = objects[..., self.state_dim] * self.log_mass_scale.to(latent) + self.log_mass_center.to(latent)
        log_mass = log_mass.clamp(min=float(self.log_mass_min.item()), max=float(self.log_mass_max.item()))
        return DecodedPhaseMass(phase=phase, mass=torch.exp(log_mass))

class HarmonicSymplecticDecoder(nn.Module):

    def __init__(self, *, q_dim: int, substeps: int, harmonic_k: float) -> None:
        super().__init__()
        if q_dim <= 0 or substeps <= 0:
            raise ValueError('q_dim/substeps 必须为正')
        if harmonic_k < 0.0:
            raise ValueError('harmonic_k 不能为负')
        self.q_dim = int(q_dim)
        self.substeps = int(substeps)
        self.register_buffer('default_harmonic_k', torch.tensor(float(harmonic_k), dtype=torch.float32))

    @staticmethod
    def _expand_k(harmonic_k: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
        batch = reference.shape[0]
        value = torch.as_tensor(harmonic_k, device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand(batch)
        if value.shape != (batch,):
            raise ValueError('harmonic_k 必须为 scalar 或 [B]')
        if bool((value < 0).any().item()):
            raise ValueError('harmonic_k 不能为负')
        return value.view(batch, 1, 1)

    def _forward_interval(self, q: torch.Tensor, p: torch.Tensor, mass: torch.Tensor, total_dt: torch.Tensor, harmonic_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dt = total_dt.view(-1, 1, 1) / self.substeps
        mass_view = mass.unsqueeze(-1)
        for _ in range(self.substeps):
            p = p - dt * harmonic_k * q
            q = q + dt * p / mass_view
        return (q, p)

    def _backward_interval(self, q: torch.Tensor, p: torch.Tensor, mass: torch.Tensor, total_dt: torch.Tensor, harmonic_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dt = total_dt.view(-1, 1, 1) / self.substeps
        mass_view = mass.unsqueeze(-1)
        for _ in range(self.substeps):
            q = q - dt * p / mass_view
            p = p + dt * harmonic_k * q
        return (q, p)

    def forward(self, reference_phase: torch.Tensor, mass: torch.Tensor, physical_time: torch.Tensor, *, reference_index: int=0, harmonic_k: torch.Tensor | float | None=None) -> torch.Tensor:
        if reference_phase.ndim != 3 or reference_phase.shape[-1] != 2 * self.q_dim:
            raise ValueError('reference_phase 必须为 [B,K,2*q_dim]')
        batch, objects, _ = reference_phase.shape
        if mass.shape != (batch, objects):
            raise ValueError('mass 必须为 [B,K]')
        if not bool((mass > 0).all().item()):
            raise ValueError('mass 必须为正')
        if physical_time.ndim != 2 or physical_time.shape[0] != batch:
            raise ValueError('physical_time 必须为 [B,T]')
        frames = physical_time.shape[1]
        if frames < 1 or not 0 <= reference_index < frames:
            raise ValueError('reference_index 超出物理时间范围')
        if frames > 1 and (not bool((physical_time[:, 1:] > physical_time[:, :-1]).all().item())):
            raise ValueError('physical_time 必须严格递增')
        k_value: torch.Tensor | float
        if harmonic_k is None:
            k_value = self.default_harmonic_k
        else:
            k_value = harmonic_k
        k_view = self._expand_k(k_value, reference_phase)
        states: list[torch.Tensor | None] = [None] * frames
        states[reference_index] = reference_phase
        q = reference_phase[..., :self.q_dim]
        p = reference_phase[..., self.q_dim:]
        for index in range(reference_index + 1, frames):
            delta = physical_time[:, index] - physical_time[:, index - 1]
            q, p = self._forward_interval(q, p, mass, delta, k_view)
            states[index] = torch.cat([q, p], dim=-1)
        q = reference_phase[..., :self.q_dim]
        p = reference_phase[..., self.q_dim:]
        for index in range(reference_index - 1, -1, -1):
            delta = physical_time[:, index + 1] - physical_time[:, index]
            q, p = self._backward_interval(q, p, mass, delta, k_view)
            states[index] = torch.cat([q, p], dim=-1)
        if any((state is None for state in states)):
            raise RuntimeError('内部错误：trajectory decoder 未填满全部物理时刻')
        return torch.stack([state for state in states if state is not None], dim=1)

    def energy(self, phase: torch.Tensor, mass: torch.Tensor, *, harmonic_k: torch.Tensor | float | None=None) -> torch.Tensor:
        if phase.ndim != 4 or phase.shape[-1] != 2 * self.q_dim:
            raise ValueError('phase 必须为 [B,T,K,2*q_dim]')
        if mass.shape != (phase.shape[0], phase.shape[2]):
            raise ValueError('mass 必须为 [B,K]')
        k_value: torch.Tensor | float
        if harmonic_k is None:
            k_value = self.default_harmonic_k
        else:
            k_value = harmonic_k
        k_view = self._expand_k(k_value, phase[:, 0]).unsqueeze(1)
        q = phase[..., :self.q_dim]
        p = phase[..., self.q_dim:]
        kinetic = p.square().sum(dim=-1) / (2.0 * mass[:, None, :])
        potential = 0.5 * k_view.squeeze(-1) * q.square().sum(dim=-1)
        return (kinetic + potential).sum(dim=-1)
