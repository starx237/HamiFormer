from __future__ import annotations
import torch
from hamiformer.types import VectorRFPair

def _validate_state_and_signal(state: torch.Tensor, signal: torch.Tensor) -> None:
    if state.ndim != 4:
        raise ValueError('phase future 必须为 [B,F,K,D]')
    if signal.shape != state.shape[:2]:
        raise ValueError('signal 必须为 [B,F]')
    if not bool(torch.isfinite(signal).all().item()):
        raise ValueError('signal 含非有限值')
    if bool(((signal < 0) | (signal > 1)).any().item()):
        raise ValueError('signal 必须位于 [0,1]')

def make_vector_rf_pair(clean: torch.Tensor, signal: torch.Tensor, state_scale: torch.Tensor, *, noise_scale: float, generator: torch.Generator | None=None) -> VectorRFPair:
    _validate_state_and_signal(clean, signal)
    if state_scale.shape != (clean.shape[-1],):
        raise ValueError('state_scale 必须为 [D]')
    if bool((state_scale <= 0).any().item()) or noise_scale <= 0:
        raise ValueError('state_scale 与 noise_scale 必须严格为正')
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noise = noise * state_scale.to(clean).view(1, 1, 1, -1) * noise_scale
    weight = signal[..., None, None]
    noisy = weight * clean + (1.0 - weight) * noise
    return VectorRFPair(clean=clean, noise=noise, noisy=noisy, signal=signal)

def vector_clean_euler(noisy: torch.Tensor, clean_estimate: torch.Tensor, signal: torch.Tensor, next_signal: torch.Tensor, *, eps: float=1e-08) -> torch.Tensor:
    _validate_state_and_signal(noisy, signal)
    _validate_state_and_signal(clean_estimate, signal)
    if clean_estimate.shape != noisy.shape or next_signal.shape != signal.shape:
        raise ValueError('clean_estimate/next_signal shape 与 noisy/signal 不一致')
    if eps <= 0:
        raise ValueError('eps 必须为正')
    if bool(((next_signal < signal) | (next_signal > 1)).any().item()):
        raise ValueError('next_signal 必须逐元素不小于 signal 且不超过 1')
    active = next_signal > signal
    denominator = (1.0 - signal).clamp_min(eps)
    keep = (1.0 - next_signal) / denominator
    keep = keep[..., None, None]
    proposal = keep * noisy + (1.0 - keep) * clean_estimate
    return torch.where(active[..., None, None], proposal, noisy)

def build_wavefront_signal_grid(owner_stage: torch.Tensor, *, denoise_intervals: int, stage_lag: int=1, dtype: torch.dtype=torch.float32, device: torch.device | str | None=None) -> torch.Tensor:
    if owner_stage.ndim != 1 or owner_stage.numel() == 0:
        raise ValueError('owner_stage 必须为非空 [F]')
    if owner_stage.dtype == torch.bool or owner_stage.is_floating_point():
        raise ValueError('owner_stage 必须为整数张量')
    if denoise_intervals <= 0 or stage_lag <= 0:
        raise ValueError('denoise_intervals 与 stage_lag 必须为正')
    stages = owner_stage.to(device=device, dtype=torch.long)
    if bool((stages < 0).any().item()):
        raise ValueError('owner_stage 不能为负')
    num_groups = int(stages.max().item()) + 1
    expected = torch.arange(num_groups, device=stages.device)
    if not torch.equal(torch.unique(stages, sorted=True), expected):
        raise ValueError('owner_stage 必须从 0 连续编号且每组至少包含一帧')
    num_intervals = denoise_intervals + stage_lag * (num_groups - 1)
    nodes = torch.arange(num_intervals + 1, device=stages.device, dtype=dtype)
    start = (stage_lag * stages).to(dtype=dtype)
    future_signal = ((nodes[:, None] - start[None]) / float(denoise_intervals)).clamp(0.0, 1.0)
    known = torch.ones(num_intervals + 1, 1, device=stages.device, dtype=dtype)
    grid = torch.cat([known, future_signal], dim=1)
    if not bool((grid[1:] >= grid[:-1]).all().item()):
        raise RuntimeError('内部错误：wavefront signal 非单调')
    if not bool((grid[-1] == 1).all().item()):
        raise RuntimeError('内部错误：wavefront 终点没有全部 clean')
    return grid
