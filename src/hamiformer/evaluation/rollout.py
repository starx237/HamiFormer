from __future__ import annotations
import torch
from hamiformer.flow import RoutedSampler
from hamiformer.types import SamplerTrace

def rollout_chunks(sampler: RoutedSampler, *, x0: torch.Tensor, attrs: torch.Tensor, rollout_time: torch.Tensor, state_scale: torch.Tensor, noise_scale: float, num_chunks: int, generator: torch.Generator | None=None) -> tuple[torch.Tensor, list[SamplerTrace]]:
    states = [x0[:, None]]
    traces: list[SamplerTrace] = []
    current = x0
    total_steps = rollout_time.shape[1] - 1
    if total_steps <= 0 or total_steps % num_chunks != 0:
        raise ValueError('rollout_time 长度必须满足 1 + num_chunks*future_steps')
    future_steps = total_steps // num_chunks
    for chunk_index in range(num_chunks):
        start = chunk_index * future_steps
        chunk_time = rollout_time[:, start:start + future_steps + 1]
        noise = torch.randn(current.shape[0], future_steps, current.shape[1], current.shape[2], device=current.device, dtype=current.dtype, generator=generator)
        noise = noise * noise_scale * state_scale.to(noise).view(1, 1, 1, -1)
        prediction, trace = sampler.sample(noise, x0=current, attrs=attrs, physical_time=chunk_time)
        states.append(prediction)
        current = prediction[:, -1]
        traces.append(trace)
    return (torch.cat(states, dim=1), traces)
