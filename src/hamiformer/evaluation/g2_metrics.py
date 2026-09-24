from __future__ import annotations
import statistics
import time
from collections.abc import Callable
from typing import Any
import torch
from hamiformer.physics.integrators import symplectic_euler_step

def _selected_rmse(prediction: torch.Tensor, target: torch.Tensor, selector: torch.Tensor) -> float:
    if not bool(selector.any().item()):
        return float('nan')
    return float((prediction[selector] - target[selector]).square().mean().sqrt().cpu())

def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    return {'median': float(torch.quantile(values, 0.5)), 'q90': float(torch.quantile(values, 0.9)), 'q99': float(torch.quantile(values, 0.99)), 'max': float(values.max())}

def pendulum_conservative_energy_metrics(prediction: torch.Tensor, target: torch.Tensor, *, initial: torch.Tensor, theta_sys: torch.Tensor, conservative_selector: torch.Tensor) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('prediction/target 必须为 [B,T,2]')
    if initial.shape != (prediction.shape[0], 2):
        raise ValueError('initial 必须为 [B,2]')
    if theta_sys.shape != (prediction.shape[0],):
        raise ValueError('theta_sys 必须为 [B]')
    if conservative_selector.shape != (prediction.shape[0],):
        raise ValueError('conservative_selector 必须为 [B]')

    def energy(phase: torch.Tensor) -> torch.Tensor:
        return 0.5 * phase[..., 1].square() + theta_sys[:, None] * (1.0 - torch.cos(phase[..., 0]))
    pred_energy = energy(torch.cat([initial[:, None], prediction], dim=1))
    target_energy = energy(torch.cat([initial[:, None], target], dim=1))
    pred_delta = pred_energy - pred_energy[:, :1]
    target_delta = target_energy - target_energy[:, :1]
    target_referenced_error = pred_delta - target_delta
    result = {'target_referenced_energy_delta_mae': float(target_referenced_error.abs().mean().cpu()), 'target_referenced_energy_delta_rmse': float(target_referenced_error.square().mean().sqrt().cpu()), 'target_referenced_energy_delta_max_abs_mean': float(target_referenced_error.abs().amax(dim=1).mean().cpu()), 'target_referenced_energy_delta_final_abs_mean': float(target_referenced_error[:, -1].abs().mean().cpu())}
    if not bool(conservative_selector.any().item()):
        return {**result, 'conservative_pred_energy_drift_mean': float('nan'), 'conservative_target_energy_drift_mean': float('nan'), 'conservative_energy_mae': float('nan')}
    pred_drift = (pred_energy - pred_energy[:, :1]).abs().amax(dim=1)
    target_drift = (target_energy - target_energy[:, :1]).abs().amax(dim=1)
    energy_mae = (pred_energy - target_energy).abs().mean(dim=1)
    selector = conservative_selector
    return {**result, 'conservative_pred_energy_drift_mean': float(pred_drift[selector].mean().cpu()), 'conservative_target_energy_drift_mean': float(target_drift[selector].mean().cpu()), 'conservative_energy_mae': float(energy_mae[selector].mean().cpu())}

def trajectory_quality_metrics(prediction: torch.Tensor, target: torch.Tensor, *, initial: torch.Tensor, true_collision_edge: torch.Tensor, step_size: float, wall_limit: torch.Tensor | float, theta_sys: torch.Tensor | float, phase_scale: torch.Tensor | tuple[float, float] | list[float], collision_metric_contract: str='pendulum_symplectic_euler_pre_event.v1') -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('prediction/target 必须同形状 [B,T,2]')
    if true_collision_edge.shape != prediction.shape[:2]:
        raise ValueError('true_collision_edge 必须为 [B,T]')
    scale = torch.as_tensor(phase_scale, device=prediction.device, dtype=prediction.dtype)
    if scale.shape != (prediction.shape[-1],) or not bool(torch.isfinite(scale).all()):
        raise ValueError('phase_scale 必须是与 phase 维度一致的有限正向量')
    if not bool((scale > 0).all()):
        raise ValueError('phase_scale 的每一维必须为正')
    theta = torch.as_tensor(theta_sys, device=prediction.device, dtype=prediction.dtype)
    if theta.ndim == 0:
        theta = theta.expand(prediction.shape[0])
    if theta.shape != (prediction.shape[0],) or not bool(torch.isfinite(theta).all()):
        raise ValueError('theta_sys 必须是有限标量或每条轨迹一个 [B] 常量')
    collision_sample = true_collision_edge.any(dim=1)
    noncollision_sample = ~collision_sample
    sample_error = (prediction - target).square().mean(dim=(1, 2)).sqrt()
    normalized_error = (prediction - target) / scale.view(1, 1, -1)
    normalized_sample_error = normalized_error.square().mean(dim=(1, 2)).sqrt()
    if collision_metric_contract == 'pendulum_symplectic_euler_pre_event.v1':
        predicted_sources = torch.cat([initial[:, None], prediction[:, :-1]], dim=1)
        predicted_core = symplectic_euler_step(predicted_sources, step_size=step_size, theta_sys=theta[:, None])
        wall = torch.as_tensor(wall_limit, device=prediction.device, dtype=prediction.dtype)
        if wall.ndim == 1:
            wall = wall[:, None]
        predicted_collision_edge = predicted_core[..., 0].abs() > wall
        true_positive = (predicted_collision_edge & true_collision_edge).sum()
        total_true = true_collision_edge.sum()
        collision_recall = float((true_positive.float() / total_true.clamp_min(1)).cpu()) if int(total_true.item()) > 0 else float('nan')
    elif collision_metric_contract == 'rk4_event_marker.v1':
        predicted_collision_edge = torch.zeros_like(true_collision_edge)
        collision_recall = float('nan')
    else:
        raise ValueError('未知 collision_metric_contract')
    timing_errors: list[torch.Tensor] = []
    multi_collision_hits: list[float] = []
    horizon = prediction.shape[1]
    for batch_index in range(prediction.shape[0]):
        if collision_metric_contract == 'rk4_event_marker.v1':
            break
        true_indices = torch.nonzero(true_collision_edge[batch_index], as_tuple=False).flatten()
        if true_indices.numel() == 0:
            continue
        predicted_indices = torch.nonzero(predicted_collision_edge[batch_index], as_tuple=False).flatten()
        if predicted_indices.numel() == 0:
            timing_errors.append(torch.full_like(true_indices, horizon, dtype=torch.float32))
        else:
            distance = (true_indices[:, None] - predicted_indices[None, :]).abs().float()
            timing_errors.append(distance.min(dim=1).values)
        if true_indices.numel() > 1:
            multi_collision_hits.append(float(predicted_indices.numel() >= true_indices.numel()))
    if timing_errors:
        timing_error = float(torch.cat(timing_errors).mean().cpu())
    else:
        timing_error = float('nan')
    multi_recall = float(sum(multi_collision_hits) / len(multi_collision_hits)) if multi_collision_hits else float('nan')
    return {'phase_rmse': float(sample_error.square().mean().sqrt().cpu()), 'phase_quantiles': _quantiles(sample_error), 'normalized_phase_rmse': float(normalized_sample_error.square().mean().sqrt().cpu()), 'normalized_phase_quantiles': _quantiles(normalized_sample_error), 'phase_scale': [float(value) for value in scale.detach().cpu()], 'collision_metric_contract': collision_metric_contract, 'q_rmse': float((prediction[..., 0] - target[..., 0]).square().mean().sqrt().cpu()), 'p_rmse': float((prediction[..., 1] - target[..., 1]).square().mean().sqrt().cpu()), 'phase_collision_rmse': _selected_rmse(prediction, target, collision_sample), 'phase_noncollision_rmse': _selected_rmse(prediction, target, noncollision_sample), 'collision_edge_recall': collision_recall, 'collision_timing_mae_edges': timing_error, 'multi_collision_count_recall': multi_recall, 'predicted_collision_edge_rate': float(predicted_collision_edge.float().mean().cpu()) if collision_metric_contract == 'pendulum_symplectic_euler_pre_event.v1' else float('nan')}

def benchmark_sampler(sample_fn: Callable[[], Any], *, batch_size: int, device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    if batch_size < 1 or warmup < 0 or repeats < 1:
        raise ValueError('batch_size/repeats 必须为正，warmup 不得为负')
    with torch.no_grad():
        for _ in range(warmup):
            sample_fn()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations: list[float] = []
        for _ in range(repeats):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            sample_fn()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
        peak_mb = float(torch.cuda.max_memory_allocated(device) / 1024.0 ** 2) if device.type == 'cuda' else 0.0
    mean_seconds = statistics.fmean(durations)
    std_seconds = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    return {'sampling_wall_seconds_mean': mean_seconds, 'sampling_wall_seconds_std': std_seconds, 'sampling_latency_ms_per_sample': 1000.0 * mean_seconds / batch_size, 'sampling_peak_allocated_mb': peak_mb, 'timing_warmup': float(warmup), 'timing_repeats': float(repeats)}
__all__ = ['benchmark_sampler', 'pendulum_conservative_energy_metrics', 'trajectory_quality_metrics']
