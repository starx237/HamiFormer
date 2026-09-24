from __future__ import annotations
from dataclasses import dataclass
import torch
from hamiformer.models.wavefront_hamiltonian_expert import WavefrontHamiltonianExpert
from .vector_signal import build_wavefront_signal_grid, vector_clean_euler

@dataclass(frozen=True)
class WavefrontSamplerTrace:
    global_intervals: int
    h_expert_forwards: int
    scalar_network_forwards: int
    factor_evaluations_per_sample: int
    owner_factor_evaluations_per_sample: int
    num_groups: int

@torch.no_grad()
def advance_wavefront_h(model: WavefrontHamiltonianExpert, state: torch.Tensor, signal_grid: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, start_interval: int=0, stop_interval: int | None=None) -> torch.Tensor:
    if state.ndim != 4 or state.shape[1] != model.future_steps:
        raise ValueError('state 必须为 [B,F,K,D] 且 F 与模型一致')
    if signal_grid.ndim != 2 or signal_grid.shape[1] != model.future_steps + 1:
        raise ValueError('signal_grid 必须为 [N+1,F+1]')
    total_intervals = int(signal_grid.shape[0] - 1)
    stop = total_intervals if stop_interval is None else int(stop_interval)
    if not 0 <= start_interval <= stop <= total_intervals:
        raise ValueError('必须满足 0 <= start_interval <= stop_interval <= N')
    z = state
    batch = state.shape[0]
    for interval in range(start_interval, stop):
        signal = signal_grid[interval].unsqueeze(0).expand(batch, -1)
        next_signal = signal_grid[interval + 1].unsqueeze(0).expand(batch, -1)
        output = model(z, signal, x0=x0, attrs=attrs, physical_time=physical_time, create_graph=False)
        z = vector_clean_euler(z, output.clean, signal[:, 1:], next_signal[:, 1:])
        if not bool(torch.isfinite(z).all().item()):
            raise FloatingPointError(f'Wavefront H 在 global interval {interval} 产生非有限状态')
    return z

@torch.no_grad()
def sample_wavefront_h(model: WavefrontHamiltonianExpert, noise: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, denoise_intervals: int, stage_lag: int=1) -> tuple[torch.Tensor, WavefrontSamplerTrace]:
    if noise.ndim != 4:
        raise ValueError('noise 必须为 [B,F,K,D]')
    if noise.shape[1] != model.future_steps:
        raise ValueError('noise 的 future 维与 Wavefront H expert 不一致')
    if denoise_intervals <= 0 or stage_lag <= 0:
        raise ValueError('denoise_intervals 与 stage_lag 必须为正')
    grid = build_wavefront_signal_grid(model.owner_stage, denoise_intervals=denoise_intervals, stage_lag=stage_lag, dtype=noise.dtype, device=noise.device)
    z = advance_wavefront_h(model, noise, grid, x0=x0, attrs=attrs, physical_time=physical_time)
    global_intervals = int(grid.shape[0] - 1)
    selected_windows = int(model.selected_window_indices.numel())
    all_windows = int(model.core.num_windows)
    return (z, WavefrontSamplerTrace(global_intervals=global_intervals, h_expert_forwards=global_intervals, scalar_network_forwards=2 * global_intervals, factor_evaluations_per_sample=2 * global_intervals * all_windows, owner_factor_evaluations_per_sample=2 * global_intervals * selected_windows, num_groups=model.num_groups))
