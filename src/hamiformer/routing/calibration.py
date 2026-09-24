from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn
from hamiformer.types import PhaseBatch
from .artifact import RoutingArtifact

@dataclass
class CalibrationTable:
    bin_index: torch.Tensor
    d_predictor: torch.Tensor
    d_corrector: torch.Tensor
    error_h: torch.Tensor
    error_d: torch.Tensor

    def subset(self, mask: torch.Tensor) -> 'CalibrationTable':
        return CalibrationTable(self.bin_index[mask], self.d_predictor[mask], self.d_corrector[mask], self.error_h[mask], self.error_d[mask])

    @staticmethod
    def concatenate(tables: list['CalibrationTable']) -> 'CalibrationTable':
        if not tables:
            raise ValueError('没有 calibration records')
        return CalibrationTable(*[torch.cat([getattr(table, name).cpu() for table in tables], dim=0) for name in ('bin_index', 'd_predictor', 'd_corrector', 'error_h', 'error_d')])

@dataclass(frozen=True)
class RiskSummary:
    threshold: float
    count: int
    accepted: int
    coverage: float
    far: float
    cvar: float
    predictor_pass_rate: float
    corrector_reject_rate: float
    expected_cost_over_d: float
    valid: bool

@dataclass(frozen=True)
class CostCurve:
    batch_size: int
    fractions: tuple[float, ...]
    d_seconds: tuple[float, ...]
    h_seconds: tuple[float, ...]
    device_type: str = 'unknown'
    device_name: str = 'unknown'
    torch_version: str = 'unknown'
    structured_expert: str = 'h'

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError('cost curve 的 batch_size 必须为正')
        if not self.fractions or not len(self.fractions) == len(self.d_seconds) == len(self.h_seconds):
            raise ValueError('cost curve 三个序列必须同长且非空')
        if tuple(sorted(self.fractions)) != self.fractions:
            raise ValueError('cost curve fractions 必须严格按升序保存')
        if self.fractions[-1] != 1.0 or any((left <= 0.0 or left > 1.0 or left >= right for left, right in zip(self.fractions, self.fractions[1:], strict=False))):
            raise ValueError('cost curve fractions 必须位于 (0,1]、互异且以 1.0 结束')
        if any((value <= 0.0 or not math.isfinite(value) for value in (*self.d_seconds, *self.h_seconds))):
            raise ValueError('cost curve 延迟必须为有限正数')

        def cumulative_max(values: tuple[float, ...]) -> tuple[float, ...]:
            result: list[float] = []
            running = 0.0
            for value in values:
                running = max(running, value)
                result.append(running)
            return tuple(result)
        object.__setattr__(self, 'd_seconds', cumulative_max(self.d_seconds))
        object.__setattr__(self, 'h_seconds', cumulative_max(self.h_seconds))

    def latency(self, expert: str, fraction: float) -> float:
        if fraction <= 0.0:
            return 0.0
        values = self.d_seconds if expert == 'd' else self.h_seconds if expert == 'h' else None
        if values is None:
            raise ValueError('expert 只能是 d 或 h')
        clipped = min(float(fraction), 1.0)
        for measured_fraction, value in zip(self.fractions, values, strict=True):
            if clipped <= measured_fraction:
                return value
        return values[-1]

    def expected_ratio(self, predictor_pass_rate: float, corrector_reject_rate: float) -> float:
        a = min(max(float(predictor_pass_rate), 0.0), 1.0)
        b = min(max(float(corrector_reject_rate), 0.0), 1.0)
        routed = self.latency('h', 1.0) + self.latency('h', a)
        routed += 2.0 * self.latency('d', 1.0 - a)
        routed += 2.0 * self.latency('d', a * b)
        return routed / (2.0 * self.latency('d', 1.0))

def _per_sample_error(prediction: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    scale = state_scale.to(prediction).view(1, 1, 1, -1)
    return ((prediction - target) / scale).pow(2).flatten(1).mean(dim=1)

def _field_heun(field: nn.Module, z: torch.Tensor, tau: torch.Tensor, tau_next: torch.Tensor, batch: PhaseBatch) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    step = (tau_next - tau).reshape(-1, 1, 1, 1)
    with torch.no_grad():
        out0 = field(z, tau, x0=batch.x0, attrs=batch.attrs, physical_time=batch.time)
    predictor = z + step * out0.velocity
    out1 = field(predictor, tau_next, x0=batch.x0, attrs=batch.attrs, physical_time=batch.time)
    endpoint = z + 0.5 * step * (out0.velocity + out1.velocity)
    return (endpoint, out0.disagreement, out1.disagreement)

def _finite_per_sample(*values: torch.Tensor) -> torch.Tensor:
    masks = [torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1) for value in values]
    result = masks[0]
    for mask in masks[1:]:
        result = result & mask
    return result

def _h_field_heun_safe(field: nn.Module, z: torch.Tensor, tau: torch.Tensor, tau_next: torch.Tensor, batch: PhaseBatch, predictor_threshold: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step = (tau_next - tau).reshape(-1, 1, 1, 1)
    out0 = field(z, tau, x0=batch.x0, attrs=batch.attrs, physical_time=batch.time)
    if out0.disagreement is None:
        raise RuntimeError('H calibration 需要 predictor disagreement')
    predictor = z + step * out0.velocity
    predictor_finite = _finite_per_sample(out0.clean, out0.velocity, out0.disagreement, predictor)
    infinity = torch.full_like(out0.disagreement, float('inf'))
    d0 = torch.where(predictor_finite, out0.disagreement, infinity)
    d1 = infinity.clone()
    endpoint = torch.full_like(z, float('inf'))
    corrector_mask = predictor_finite
    if predictor_threshold is not None:
        if predictor_threshold.shape != d0.shape:
            raise ValueError('predictor_threshold 必须为 [B]')
        corrector_mask = corrector_mask & (d0 <= predictor_threshold)
    if bool(corrector_mask.any().item()):
        with torch.no_grad():
            out1 = field(predictor[corrector_mask], tau_next[corrector_mask], x0=batch.x0[corrector_mask], attrs=batch.attrs[corrector_mask], physical_time=batch.time[corrector_mask])
        if out1.disagreement is None:
            raise RuntimeError('H calibration 需要 corrector disagreement')
        local_endpoint = z[corrector_mask] + 0.5 * step[corrector_mask] * (out0.velocity[corrector_mask] + out1.velocity)
        corrector_finite = _finite_per_sample(out1.clean, out1.velocity, out1.disagreement, local_endpoint)
        indices = torch.nonzero(corrector_mask, as_tuple=False).flatten()
        good_indices = indices[corrector_finite]
        endpoint[good_indices] = local_endpoint[corrector_finite]
        d1[good_indices] = out1.disagreement[corrector_finite]
    return (endpoint, d0, d1)

def collect_analytic_records(*, d_field: nn.Module, h_field: nn.Module, batch: PhaseBatch, noise: torch.Tensor, state_scale: torch.Tensor, num_intervals: int, tau_max: float, num_bins: int, interval_indices: list[int] | None=None) -> CalibrationTable:
    if noise.shape != batch.future.shape:
        raise ValueError('noise 与 clean future shape 必须一致')
    if not 0.0 < tau_max < 1.0:
        raise ValueError('tau_max 必须严格位于 (0,1)')
    grid = torch.linspace(0.0, tau_max, num_intervals, device=batch.future.device, dtype=batch.future.dtype)
    if interval_indices is None:
        interval_indices = list(range(num_intervals - 1))
    tables: list[CalibrationTable] = []
    for index in interval_indices:
        if not 0 <= index < num_intervals - 1:
            raise ValueError('校准 interval 不能包含最终 D-Euler')
        tau = grid[index].expand(batch.future.shape[0])
        tau_next = grid[index + 1].expand(batch.future.shape[0])
        tau_view = tau.reshape(-1, 1, 1, 1)
        next_view = tau_next.reshape(-1, 1, 1, 1)
        z = tau_view * batch.future + (1.0 - tau_view) * noise
        true_next = next_view * batch.future + (1.0 - next_view) * noise
        with torch.no_grad():
            endpoint_d, _, _ = _field_heun(d_field, z, tau, tau_next, batch)
        if not bool(_finite_per_sample(endpoint_d).all().item()):
            raise FloatingPointError('calibration 的 D-Heun endpoint 出现 NaN/Inf；不能生成 fallback artifact')
        endpoint_h, d0, d1 = _h_field_heun_safe(h_field, z, tau, tau_next, batch)
        error_d = _per_sample_error(endpoint_d, true_next, state_scale)
        error_h = _per_sample_error(endpoint_h, true_next, state_scale)
        finite_endpoint = _finite_per_sample(endpoint_h, d1)
        error_h = torch.where(finite_endpoint, error_h, torch.full_like(error_h, float('inf')))
        bin_value = min(int(float(grid[index]) * num_bins), num_bins - 1)
        tables.append(CalibrationTable(bin_index=torch.full_like(d0, bin_value, dtype=torch.long), d_predictor=d0.detach(), d_corrector=d1.detach(), error_h=error_h.detach(), error_d=error_d.detach()))
    return CalibrationTable.concatenate(tables)

def collect_on_policy_records(*, d_field: nn.Module, h_field: nn.Module, batch: PhaseBatch, noise: torch.Tensor, state_scale: torch.Tensor, num_intervals: int, tau_max: float, num_bins: int, thresholds: list[float], d_only: list[bool]) -> CalibrationTable:
    if len(thresholds) != num_bins or len(d_only) != num_bins:
        raise ValueError('冻结阈值长度与 num_bins 不一致')
    if noise.shape != batch.future.shape:
        raise ValueError('noise 与 clean future shape 必须一致')
    grid = torch.linspace(0.0, tau_max, num_intervals, device=batch.future.device, dtype=batch.future.dtype)
    z = noise.clone()
    tables: list[CalibrationTable] = []
    batch_size = batch.future.shape[0]
    for index in range(num_intervals - 1):
        tau = grid[index].expand(batch_size)
        tau_next = grid[index + 1].expand(batch_size)
        step_next = tau_next.reshape(-1, 1, 1, 1)
        true_next = step_next * batch.future + (1.0 - step_next) * noise
        with torch.no_grad():
            endpoint_d, _, _ = _field_heun(d_field, z, tau, tau_next, batch)
        if not bool(_finite_per_sample(endpoint_d).all().item()):
            raise FloatingPointError('on-policy audit 的 D-Heun endpoint 出现 NaN/Inf')
        bin_value = min(int(float(grid[index]) * num_bins), num_bins - 1)
        threshold_value = thresholds[bin_value]
        if d_only[bin_value]:
            d0 = torch.full((batch_size,), float('inf'), device=z.device, dtype=z.dtype)
            d1 = d0.clone()
            endpoint_h = torch.full_like(z, float('inf'))
        else:
            threshold = torch.full((batch_size,), threshold_value, device=z.device, dtype=z.dtype)
            endpoint_h, d0, d1 = _h_field_heun_safe(h_field, z, tau, tau_next, batch, predictor_threshold=threshold)
        error_d = _per_sample_error(endpoint_d, true_next, state_scale)
        error_h = _per_sample_error(endpoint_h, true_next, state_scale)
        accepted = ~torch.full((batch_size,), d_only[bin_value], device=z.device, dtype=torch.bool) & torch.isfinite(d0) & torch.isfinite(d1) & _finite_per_sample(endpoint_h) & (d0 <= threshold_value) & (d1 <= threshold_value)
        selector = accepted.reshape(-1, 1, 1, 1)
        z = torch.where(selector, endpoint_h, endpoint_d).detach()
        tables.append(CalibrationTable(bin_index=torch.full((batch_size,), bin_value, device=z.device, dtype=torch.long), d_predictor=d0.detach(), d_corrector=d1.detach(), error_h=error_h.detach(), error_d=error_d.detach()))
    return CalibrationTable.concatenate(tables)

def summarize_threshold(table: CalibrationTable, threshold: float, *, far_limit: float, cvar_limit: float, tail_fraction: float, minimum_accepted: int, cost_curve: CostCurve) -> RiskSummary:
    predictor_finite = torch.isfinite(table.d_predictor) & torch.isfinite(table.error_d)
    predictor_pass = predictor_finite & (table.d_predictor <= threshold)
    accepted_finite = torch.isfinite(table.d_corrector) & torch.isfinite(table.error_h) & torch.isfinite(table.error_d)
    accepted_mask = predictor_pass & accepted_finite & (table.d_corrector <= threshold)
    count = int(table.error_d.numel())
    accepted = int(accepted_mask.sum().item())
    coverage = accepted / max(count, 1)
    if accepted == 0:
        far, cvar = (float('inf'), float('inf'))
    else:
        errors_h = table.error_h[accepted_mask]
        errors_d = table.error_d[accepted_mask]
        far = float((errors_h > errors_d).float().mean().item())
        stable_errors = table.error_d[torch.isfinite(table.error_d)]
        stable = max(float(torch.median(stable_errors).item()), 1e-12)
        regret = torch.clamp(errors_h - errors_d, min=0.0) / (errors_d + stable)
        tail_count = max(1, int(math.ceil(accepted * tail_fraction)))
        cvar = float(torch.topk(regret, k=tail_count).values.mean().item())
    a = float(predictor_pass.float().mean().item()) if count else 0.0
    passed_count = int(predictor_pass.sum().item())
    b = float((predictor_pass & ~accepted_mask).float().sum().item()) / passed_count if passed_count else 1.0
    expected = cost_curve.expected_ratio(a, b)
    valid = accepted >= minimum_accepted and far <= far_limit and (cvar <= cvar_limit) and (expected < 1.0)
    return RiskSummary(threshold=float(threshold), count=count, accepted=accepted, coverage=coverage, far=far, cvar=cvar, predictor_pass_rate=a, corrector_reject_rate=b, expected_cost_over_d=expected, valid=valid)

def fit_threshold(table: CalibrationTable, *, far_limit: float, cvar_limit: float, tail_fraction: float, minimum_accepted: int, cost_curve: CostCurve, max_candidates: int=256) -> RiskSummary | None:
    route_score = torch.maximum(table.d_predictor, table.d_corrector)
    finite = route_score[torch.isfinite(route_score)]
    if finite.numel() == 0:
        return None
    sorted_values = torch.unique(finite.sort().values)
    if sorted_values.numel() > max_candidates:
        indices = torch.linspace(0, sorted_values.numel() - 1, max_candidates).long()
        sorted_values = sorted_values[indices]
    candidates = [summarize_threshold(table, float(value), far_limit=far_limit, cvar_limit=cvar_limit, tail_fraction=tail_fraction, minimum_accepted=minimum_accepted, cost_curve=cost_curve) for value in sorted_values]
    valid = [item for item in candidates if item.valid]
    return max(valid, key=lambda item: item.coverage) if valid else None

def fit_provisional_thresholds(fit_table: CalibrationTable, *, num_bins: int, far_limit: float, cvar_limit: float, tail_fraction: float, minimum_accepted: int, cost_curve: CostCurve) -> tuple[list[float], list[bool]]:
    thresholds: list[float] = []
    d_only: list[bool] = []
    for bin_index in range(num_bins):
        fit_bin = fit_table.subset(fit_table.bin_index == bin_index)
        fitted = fit_threshold(fit_bin, far_limit=far_limit, cvar_limit=cvar_limit, tail_fraction=tail_fraction, minimum_accepted=minimum_accepted, cost_curve=cost_curve)
        if fitted is None:
            thresholds.append(0.0)
            d_only.append(True)
        else:
            thresholds.append(fitted.threshold)
            d_only.append(False)
    return (thresholds, d_only)

def audit_fitted_artifact(audit_table: CalibrationTable, *, thresholds: list[float], provisional_d_only: list[bool], far_limit: float, cvar_limit: float, tail_fraction: float, minimum_accepted: int, cost_curve: CostCurve, hashes: dict[str, str]) -> RoutingArtifact:
    num_bins = len(thresholds)
    if len(provisional_d_only) != num_bins:
        raise ValueError('provisional_d_only 与 thresholds 长度不一致')
    d_only: list[bool] = []
    audit_rows: list[dict[str, float | int | bool]] = []
    for bin_index, threshold in enumerate(thresholds):
        audit_bin = audit_table.subset(audit_table.bin_index == bin_index)
        if provisional_d_only[bin_index] or audit_bin.error_d.numel() == 0:
            d_only.append(True)
            audit_rows.append({'bin': bin_index, 'valid': False, 'reason_code': 1})
            continue
        audited = summarize_threshold(audit_bin, threshold, far_limit=far_limit, cvar_limit=cvar_limit, tail_fraction=tail_fraction, minimum_accepted=minimum_accepted, cost_curve=cost_curve)
        d_only.append(not audited.valid)
        audit_rows.append({'bin': bin_index, 'valid': audited.valid, 'threshold': audited.threshold, 'count': audited.count, 'accepted': audited.accepted, 'coverage': audited.coverage, 'far': audited.far, 'cvar': audited.cvar, 'expected_cost_over_d': audited.expected_cost_over_d})
    first_failure = next((index for index, disabled in enumerate(d_only) if disabled and (not provisional_d_only[index])), None)
    if first_failure is not None:
        for index in range(first_failure + 1, num_bins):
            d_only[index] = True
            audit_rows[index]['valid'] = False
            audit_rows[index]['reason_code'] = 2
    return RoutingArtifact(num_bins=num_bins, thresholds=thresholds, d_only=d_only, model_hash=hashes['model_hash'], stats_hash=hashes['stats_hash'], solver_hash=hashes['solver_hash'], config_hash=hashes['config_hash'], audit=audit_rows, cost_profile_hash=hashes.get('cost_profile_hash', ''), cost_device=f'{cost_curve.device_type}:{cost_curve.device_name}', cost_torch_version=cost_curve.torch_version, structured_expert=cost_curve.structured_expert, calibration_data_hash=hashes.get('calibration_data_hash', ''))

def fit_and_audit_artifact(fit_table: CalibrationTable, audit_table: CalibrationTable, *, num_bins: int, far_limit: float, cvar_limit: float, tail_fraction: float, minimum_accepted: int, cost_curve: CostCurve, hashes: dict[str, str]) -> RoutingArtifact:
    thresholds, provisional = fit_provisional_thresholds(fit_table, num_bins=num_bins, far_limit=far_limit, cvar_limit=cvar_limit, tail_fraction=tail_fraction, minimum_accepted=minimum_accepted, cost_curve=cost_curve)
    return audit_fitted_artifact(audit_table, thresholds=thresholds, provisional_d_only=provisional, far_limit=far_limit, cvar_limit=cvar_limit, tail_fraction=tail_fraction, minimum_accepted=minimum_accepted, cost_curve=cost_curve, hashes=hashes)
