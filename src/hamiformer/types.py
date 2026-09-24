from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class PhaseBatch:
    x0: torch.Tensor
    future: torch.Tensor
    attrs: torch.Tensor
    time: torch.Tensor
    sample_id: list[str] | None = None
    scene_id: list[str] | None = None

    def to(self, device: torch.device | str, *, dtype: torch.dtype | None=None) -> 'PhaseBatch':

        def move(value: torch.Tensor) -> torch.Tensor:
            target_dtype = dtype if dtype is not None and value.is_floating_point() else value.dtype
            return value.to(device=device, dtype=target_dtype)
        return PhaseBatch(x0=move(self.x0), future=move(self.future), attrs=move(self.attrs), time=move(self.time), sample_id=self.sample_id, scene_id=self.scene_id)

@dataclass
class PhaseObservation:
    phase: torch.Tensor
    phase_mask: torch.Tensor
    attrs: torch.Tensor
    attr_mask: torch.Tensor
    time: torch.Tensor

    def to(self, device: torch.device | str, *, dtype: torch.dtype | None=None) -> 'PhaseObservation':
        target_dtype = dtype if dtype is not None else self.phase.dtype
        return PhaseObservation(phase=self.phase.to(device=device, dtype=target_dtype), phase_mask=self.phase_mask.to(device=device, dtype=torch.bool), attrs=self.attrs.to(device=device, dtype=target_dtype), attr_mask=self.attr_mask.to(device=device, dtype=torch.bool), time=self.time.to(device=device, dtype=target_dtype))

@dataclass
class RFPair:
    clean: torch.Tensor
    noise: torch.Tensor
    noisy: torch.Tensor
    tau: torch.Tensor
    target_velocity: torch.Tensor

@dataclass
class PosteriorRFPair:
    clean: torch.Tensor
    source: torch.Tensor
    noisy: torch.Tensor
    tau: torch.Tensor
    sigma: torch.Tensor
    target_velocity: torch.Tensor

@dataclass
class VectorRFPair:
    clean: torch.Tensor
    noise: torch.Tensor
    noisy: torch.Tensor
    signal: torch.Tensor

@dataclass
class FieldOutput:
    clean: torch.Tensor
    velocity: torch.Tensor
    disagreement: torch.Tensor | None = None
    diagnostics: dict[str, Any] | None = None

@dataclass
class HamiltonianOccurrences:
    q_plus: torch.Tensor
    p_plus: torch.Tensor
    q_minus: torch.Tensor
    p_minus: torch.Tensor

@dataclass
class HamiltonianOutput:
    clean: torch.Tensor
    disagreement: torch.Tensor
    occurrences: HamiltonianOccurrences

@dataclass
class SamplerTrace:
    d_only_intervals: int = 0
    h_predictor_calls: int = 0
    h_corrector_calls: int = 0
    d_predictor_calls: int = 0
    d_corrector_calls: int = 0
    h_committed_samples: int = 0
    d_fallback_samples: int = 0
