from __future__ import annotations
import torch
from hamiformer.config import RectifiedFlowConfig
from hamiformer.types import RFPair

def _broadcast_tau(tau: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tau.ndim != 1 or tau.shape[0] != target.shape[0]:
        raise ValueError('tau 必须为 [B] 并与 target batch 一致')
    return tau.reshape(tau.shape[0], *[1] * (target.ndim - 1))

def prefix_sum_delta_clean(delta_clean: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    if delta_clean.ndim != 3:
        raise ValueError('delta_clean must have shape [B,T,D]')
    if initial.ndim == 3 and initial.shape[1] == 1:
        initial = initial[:, 0]
    if initial.ndim != 2:
        raise ValueError('initial must have shape [B,D] or [B,1,D]')
    if initial.shape[0] != delta_clean.shape[0] or initial.shape[1] != delta_clean.shape[2]:
        raise ValueError('initial and delta_clean batch/state dimensions must agree')
    return initial[:, None, :] + torch.cumsum(delta_clean, dim=1)

def sample_tau(batch_size: int, config: RectifiedFlowConfig, *, device: torch.device | str, generator: torch.Generator | None=None) -> torch.Tensor:
    choose_logit = torch.rand(batch_size, device=device, generator=generator)
    gaussian = torch.randn(batch_size, device=device, generator=generator)
    logit_tau = torch.sigmoid(config.logit_mean + config.logit_std * gaussian)
    uniform_tau = torch.rand(batch_size, device=device, generator=generator) * config.tau_max
    tau = torch.where(choose_logit < config.logit_normal_probability, logit_tau, uniform_tau)
    return tau.clamp(min=0.0, max=config.tau_max)

def make_rf_pair(clean: torch.Tensor, tau: torch.Tensor, state_scale: torch.Tensor, *, noise_scale: float, generator: torch.Generator | None=None) -> RFPair:
    if state_scale.shape != (clean.shape[-1],):
        raise ValueError('state_scale 必须与最后一个 phase channel 维一致')
    tau_view = _broadcast_tau(tau, clean)
    base_noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    noise = noise_scale * state_scale.to(clean).view(*[1] * (clean.ndim - 1), -1) * base_noise
    noisy = tau_view * clean + (1.0 - tau_view) * noise
    target_velocity = clean - noise
    return RFPair(clean=clean, noise=noise, noisy=noisy, tau=tau, target_velocity=target_velocity)

def clean_to_velocity(clean: torch.Tensor, noisy: torch.Tensor, tau: torch.Tensor, *, t_eps: float=0.05) -> torch.Tensor:
    if clean.shape != noisy.shape:
        raise ValueError('clean/noisy shape 必须一致')
    if torch.any(tau >= 1.0) or torch.any(tau < 0.0):
        raise ValueError('RF field 只允许 0 <= tau < 1')
    if not 0.0 < t_eps <= 1.0:
        raise ValueError('t_eps 必须位于 (0,1]')
    tau_view = _broadcast_tau(tau, clean)
    return (clean - noisy) / (1.0 - tau_view).clamp_min(t_eps)

def target_velocity_from_pair(clean: torch.Tensor, noise: torch.Tensor, noisy: torch.Tensor, tau: torch.Tensor, *, prediction_type: str, t_eps: float=0.05) -> torch.Tensor:
    kind = prediction_type.lower()
    if kind == 'velocity':
        if clean.shape != noise.shape:
            raise ValueError('clean/noise shape 必须一致')
        return clean - noise
    if kind == 'clean':
        return clean_to_velocity(clean, noisy, tau, t_eps=t_eps)
    raise ValueError('prediction_type 必须为 clean 或 velocity')
