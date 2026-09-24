from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from hamiformer.routing import HardRoutingPolicy, ThreeZoneSoftPolicy
from hamiformer.types import FieldOutput, SamplerTrace

def _finite_per_sample(*tensors: torch.Tensor) -> torch.Tensor:
    masks = [torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1) for value in tensors]
    output = masks[0]
    for mask in masks[1:]:
        output = output & mask
    return output

@dataclass(frozen=True)
class _Conditions:
    x0: torch.Tensor
    attrs: torch.Tensor
    physical_time: torch.Tensor

    def subset(self, mask: torch.Tensor) -> '_Conditions':
        return _Conditions(self.x0[mask], self.attrs[mask], self.physical_time[mask])

    def kwargs(self) -> dict[str, torch.Tensor]:
        return {'x0': self.x0, 'attrs': self.attrs, 'physical_time': self.physical_time}

class RoutedSampler:

    def __init__(self, *, d_field: nn.Module, h_field: nn.Module, policy: HardRoutingPolicy | ThreeZoneSoftPolicy, num_intervals: int, tau_max: float, mode: str='hard') -> None:
        if mode not in {'hard', 'soft_three_zone'}:
            raise ValueError('sampler mode 只能是 hard 或 soft_three_zone')
        if mode == 'soft_three_zone' and (not isinstance(policy, ThreeZoneSoftPolicy)):
            raise ValueError('soft mode 必须搭配 ThreeZoneSoftPolicy')
        self.d_field = d_field
        self.h_field = h_field
        self.policy = policy
        self.num_intervals = int(num_intervals)
        self.tau_max = float(tau_max)
        if self.num_intervals < 2:
            raise ValueError('num_intervals 至少为 2')
        if not 0.0 < self.tau_max < 1.0:
            raise ValueError('tau_max 必须严格位于 (0,1)')
        self.mode = mode

    @staticmethod
    def _tau(batch: int, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.expand(batch).to(device=reference.device, dtype=reference.dtype)

    def _d_eval(self, z: torch.Tensor, tau: torch.Tensor, conditions: _Conditions) -> FieldOutput:
        with torch.no_grad():
            output = self.d_field(z, tau, **conditions.kwargs())
        if not _finite_per_sample(output.clean, output.velocity).all():
            raise FloatingPointError('D field 出现 NaN/Inf，不能静默继续采样')
        return output

    def _structured_eval(self, z: torch.Tensor, tau: torch.Tensor, conditions: _Conditions) -> FieldOutput:
        with torch.no_grad():
            return self.h_field(z, tau, **conditions.kwargs())

    def _d_heun(self, z: torch.Tensor, tau: torch.Tensor, tau_next: torch.Tensor, conditions: _Conditions, trace: SamplerTrace) -> torch.Tensor:
        step = (tau_next - tau).reshape(-1, 1, 1, 1)
        out0 = self._d_eval(z, tau, conditions)
        trace.d_predictor_calls += 1
        predictor = z + step * out0.velocity
        out1 = self._d_eval(predictor, tau_next, conditions)
        trace.d_corrector_calls += 1
        return z + 0.5 * step * (out0.velocity + out1.velocity)

    def _hard_heun(self, z: torch.Tensor, tau: torch.Tensor, tau_next: torch.Tensor, conditions: _Conditions, trace: SamplerTrace) -> torch.Tensor:
        threshold, d_only = self.policy.parameters(tau)
        if bool(d_only.all().item()):
            trace.d_only_intervals += 1
            return self._d_heun(z, tau, tau_next, conditions, trace)
        step = (tau_next - tau).reshape(-1, 1, 1, 1)
        h0 = self._structured_eval(z, tau, conditions)
        trace.h_predictor_calls += 1
        if h0.disagreement is None:
            raise RuntimeError('H field 必须返回 disagreement')
        predictor = z + step * h0.velocity
        accept0 = ~d_only & (h0.disagreement <= threshold) & _finite_per_sample(h0.clean, h0.velocity, h0.disagreement, predictor)
        result = torch.empty_like(z)
        reject0 = ~accept0
        if bool(reject0.any().item()):
            result[reject0] = self._d_heun(z[reject0], tau[reject0], tau_next[reject0], conditions.subset(reject0), trace)
            trace.d_fallback_samples += int(reject0.sum().item())
        if bool(accept0.any().item()):
            h1 = self._structured_eval(predictor[accept0], tau_next[accept0], conditions.subset(accept0))
            trace.h_corrector_calls += 1
            if h1.disagreement is None:
                raise RuntimeError('H corrector 必须返回 disagreement')
            endpoint = z[accept0] + 0.5 * step[accept0] * (h0.velocity[accept0] + h1.velocity)
            accept1_local = (h1.disagreement <= threshold[accept0]) & _finite_per_sample(h1.clean, h1.velocity, h1.disagreement, endpoint)
            accepted_indices = torch.nonzero(accept0, as_tuple=False).flatten()
            if bool(accept1_local.any().item()):
                result[accepted_indices[accept1_local]] = endpoint[accept1_local]
                trace.h_committed_samples += int(accept1_local.sum().item())
            reject1_local = ~accept1_local
            if bool(reject1_local.any().item()):
                reject_indices = accepted_indices[reject1_local]
                result[reject_indices] = self._d_heun(z[reject_indices], tau[reject_indices], tau_next[reject_indices], _Conditions(conditions.x0[reject_indices], conditions.attrs[reject_indices], conditions.physical_time[reject_indices]), trace)
                trace.d_fallback_samples += int(reject1_local.sum().item())
        return result

    def _mixed_eval(self, z: torch.Tensor, tau: torch.Tensor, conditions: _Conditions, trace: SamplerTrace, *, corrector: bool) -> FieldOutput:
        assert isinstance(self.policy, ThreeZoneSoftPolicy)
        _, d_only = self.policy.parameters(tau)
        result_clean = torch.empty_like(z)
        result_velocity = torch.empty_like(z)
        if bool(d_only.any().item()):
            d_output = self._d_eval(z[d_only], tau[d_only], conditions.subset(d_only))
            result_clean[d_only], result_velocity[d_only] = (d_output.clean, d_output.velocity)
            if corrector:
                trace.d_corrector_calls += 1
            else:
                trace.d_predictor_calls += 1
        routable = ~d_only
        if bool(routable.any().item()):
            h_output = self._structured_eval(z[routable], tau[routable], conditions.subset(routable))
            if corrector:
                trace.h_corrector_calls += 1
            else:
                trace.h_predictor_calls += 1
            if h_output.disagreement is None:
                raise RuntimeError('H field 必须返回 disagreement')
            weight = self.policy.weight(tau[routable], h_output.disagreement)
            h_finite = _finite_per_sample(h_output.clean, h_output.velocity, h_output.disagreement)
            weight = torch.where(h_finite, weight, torch.zeros_like(weight))
            finite_view = h_finite.reshape(-1, 1, 1, 1)
            clean = torch.where(finite_view, h_output.clean, torch.zeros_like(h_output.clean))
            velocity = torch.where(finite_view, h_output.velocity, torch.zeros_like(h_output.velocity))
            need_d = weight < 1.0
            if bool(need_d.any().item()):
                local_conditions = conditions.subset(routable)
                d_output = self._d_eval(z[routable][need_d], tau[routable][need_d], local_conditions.subset(need_d))
                if corrector:
                    trace.d_corrector_calls += 1
                else:
                    trace.d_predictor_calls += 1
                w = weight[need_d].reshape(-1, 1, 1, 1)
                clean[need_d] = w * clean[need_d] + (1.0 - w) * d_output.clean
                velocity[need_d] = w * velocity[need_d] + (1.0 - w) * d_output.velocity
            result_clean[routable], result_velocity[routable] = (clean, velocity)
        return FieldOutput(clean=result_clean, velocity=result_velocity)

    def _soft_heun(self, z: torch.Tensor, tau: torch.Tensor, tau_next: torch.Tensor, conditions: _Conditions, trace: SamplerTrace) -> torch.Tensor:
        step = (tau_next - tau).reshape(-1, 1, 1, 1)
        out0 = self._mixed_eval(z, tau, conditions, trace, corrector=False)
        predictor = z + step * out0.velocity
        out1 = self._mixed_eval(predictor, tau_next, conditions, trace, corrector=True)
        return z + 0.5 * step * (out0.velocity + out1.velocity)

    def sample(self, initial_noise: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> tuple[torch.Tensor, SamplerTrace]:
        z = initial_noise
        batch = z.shape[0]
        conditions = _Conditions(x0, attrs, physical_time)
        trace = SamplerTrace()
        grid = torch.linspace(0.0, self.tau_max, self.num_intervals, device=z.device, dtype=z.dtype)
        for index in range(self.num_intervals - 1):
            tau = self._tau(batch, grid[index], z)
            tau_next = self._tau(batch, grid[index + 1], z)
            if self.mode == 'hard':
                z = self._hard_heun(z, tau, tau_next, conditions, trace)
            else:
                z = self._soft_heun(z, tau, tau_next, conditions, trace)
        tau = self._tau(batch, grid[-1], z)
        tau_next = self._tau(batch, grid.new_tensor(1.0), z)
        step = (tau_next - tau).reshape(-1, 1, 1, 1)
        d_final = self._d_eval(z, tau, conditions)
        trace.d_predictor_calls += 1
        z = z + step * d_final.velocity
        return (z, trace)

class FixedExpertSampler:

    def __init__(self, *, field: nn.Module, expert: str, num_intervals: int, tau_max: float, solver: str='heun') -> None:
        if expert not in {'d', 'h'}:
            raise ValueError('fixed expert 只能是 d 或 h')
        if num_intervals < 2:
            raise ValueError('num_intervals 至少为 2')
        if not 0.0 < tau_max < 1.0:
            raise ValueError('tau_max 必须严格位于 (0,1)')
        if solver not in {'heun', 'clean_euler'}:
            raise ValueError('fixed expert solver 只能是 heun 或 clean_euler')
        self.field = field
        self.expert = expert
        self.num_intervals = int(num_intervals)
        self.tau_max = float(tau_max)
        self.solver = solver

    @staticmethod
    def _tau(batch: int, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.expand(batch).to(device=reference.device, dtype=reference.dtype)

    def _evaluate(self, z: torch.Tensor, tau: torch.Tensor, conditions: _Conditions) -> FieldOutput:
        with torch.no_grad():
            output = self.field(z, tau, **conditions.kwargs())
        values = [output.clean, output.velocity]
        if output.disagreement is not None:
            values.append(output.disagreement)
        if not bool(_finite_per_sample(*values).all().item()):
            raise FloatingPointError(f'{self.expert.upper()}-only field 出现 NaN/Inf')
        return output

    def _record_call(self, trace: SamplerTrace, *, corrector: bool) -> None:
        if self.expert == 'd':
            if corrector:
                trace.d_corrector_calls += 1
            else:
                trace.d_predictor_calls += 1
        elif corrector:
            trace.h_corrector_calls += 1
        else:
            trace.h_predictor_calls += 1

    def sample(self, initial_noise: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> tuple[torch.Tensor, SamplerTrace]:
        z = initial_noise
        batch = z.shape[0]
        conditions = _Conditions(x0, attrs, physical_time)
        trace = SamplerTrace()
        grid = torch.linspace(0.0, self.tau_max, self.num_intervals, device=z.device, dtype=z.dtype)
        for index in range(self.num_intervals - 1):
            tau = self._tau(batch, grid[index], z)
            tau_next = self._tau(batch, grid[index + 1], z)
            out0 = self._evaluate(z, tau, conditions)
            self._record_call(trace, corrector=False)
            if self.solver == 'heun':
                step = (tau_next - tau).reshape(-1, 1, 1, 1)
                predictor = z + step * out0.velocity
                out1 = self._evaluate(predictor, tau_next, conditions)
                self._record_call(trace, corrector=True)
                z = z + 0.5 * step * (out0.velocity + out1.velocity)
            else:
                tau_view = tau.reshape(-1, 1, 1, 1)
                tau_next_view = tau_next.reshape(-1, 1, 1, 1)
                keep = (1.0 - tau_next_view) / (1.0 - tau_view)
                z = keep * z + (1.0 - keep) * out0.clean
            if self.expert == 'h':
                trace.h_committed_samples += batch
            else:
                trace.d_only_intervals += 1
        tau = self._tau(batch, grid[-1], z)
        final = self._evaluate(z, tau, conditions)
        self._record_call(trace, corrector=False)
        if self.solver == 'heun':
            step = (grid.new_tensor(1.0) - grid[-1]).reshape(1, 1, 1, 1)
            z = z + step * final.velocity
        else:
            z = final.clean
        return (z, trace)
