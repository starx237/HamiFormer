from __future__ import annotations
import torch
from torch import nn
from hamiformer.types import FieldOutput, SamplerTrace

def _finite_per_sample(value: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1)

class HMSRFHOnlySampler:

    def __init__(self, *, field: nn.Module, num_intervals: int, tau_max: float) -> None:
        if num_intervals < 2:
            raise ValueError('num_intervals 至少为 2')
        if not 0.0 < tau_max < 1.0:
            raise ValueError('tau_max 必须严格位于 (0,1)')
        self.field = field
        self.num_intervals = int(num_intervals)
        self.tau_max = float(tau_max)

    @staticmethod
    def _tau(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return value.expand(reference.shape[0]).to(device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _clamp_known_path(state: torch.Tensor, tau: torch.Tensor, *, known_values: torch.Tensor, known_mask: torch.Tensor, known_source_noise: torch.Tensor) -> torch.Tensor:
        if known_values.shape != state.shape or known_source_noise.shape != state.shape:
            raise ValueError('known_values/source_noise 必须与 state 同形')
        if known_mask.shape != state.shape or known_mask.dtype != torch.bool:
            raise ValueError('known_mask 必须是与 state 同形的 bool tensor')
        tau_view = tau.reshape(tau.shape[0], 1, 1, 1)
        path = tau_view * known_values + (1.0 - tau_view) * known_source_noise
        return torch.where(known_mask, path, state)

    def _evaluate(self, state: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, known_values: torch.Tensor, known_mask: torch.Tensor) -> FieldOutput:
        with torch.no_grad():
            output = self.field(state, tau, x0=x0, attrs=attrs, physical_time=physical_time, known_values=known_values, known_mask=known_mask)
        if not bool((_finite_per_sample(output.clean) & _finite_per_sample(output.velocity)).all().item()):
            raise FloatingPointError('HMS-RF H field 出现 NaN/Inf')
        return output

    def sample(self, initial_noise: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, known_values: torch.Tensor | None=None, known_mask: torch.Tensor | None=None, known_source_noise: torch.Tensor | None=None) -> tuple[torch.Tensor, SamplerTrace]:
        state = initial_noise
        if known_values is None:
            known_values = torch.zeros_like(state)
        if known_mask is None:
            known_mask = torch.zeros_like(state, dtype=torch.bool)
        if known_source_noise is None:
            known_source_noise = initial_noise
        trace = SamplerTrace()
        grid = torch.linspace(0.0, self.tau_max, self.num_intervals, device=state.device, dtype=state.dtype)
        tau_zero = self._tau(state, grid[0])
        state = self._clamp_known_path(state, tau_zero, known_values=known_values, known_mask=known_mask, known_source_noise=known_source_noise)
        for index in range(self.num_intervals - 1):
            tau = self._tau(state, grid[index])
            tau_next = self._tau(state, grid[index + 1])
            step = (tau_next - tau).reshape(-1, 1, 1, 1)
            output0 = self._evaluate(state, tau, x0=x0, attrs=attrs, physical_time=physical_time, known_values=known_values, known_mask=known_mask)
            trace.h_predictor_calls += 1
            predictor = state + step * output0.velocity
            predictor = self._clamp_known_path(predictor, tau_next, known_values=known_values, known_mask=known_mask, known_source_noise=known_source_noise)
            output1 = self._evaluate(predictor, tau_next, x0=x0, attrs=attrs, physical_time=physical_time, known_values=known_values, known_mask=known_mask)
            trace.h_corrector_calls += 1
            state = state + 0.5 * step * (output0.velocity + output1.velocity)
            state = self._clamp_known_path(state, tau_next, known_values=known_values, known_mask=known_mask, known_source_noise=known_source_noise)
        tau = self._tau(state, grid[-1])
        tau_one = self._tau(state, grid.new_tensor(1.0))
        final_output = self._evaluate(state, tau, x0=x0, attrs=attrs, physical_time=physical_time, known_values=known_values, known_mask=known_mask)
        trace.h_predictor_calls += 1
        final_step = (tau_one - tau).reshape(-1, 1, 1, 1)
        state = state + final_step * final_output.velocity
        state = self._clamp_known_path(state, tau_one, known_values=known_values, known_mask=known_mask, known_source_noise=known_source_noise)
        return (state, trace)
