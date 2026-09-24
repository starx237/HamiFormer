from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import torch
from hamiformer.types import PosteriorRFPair

def _tau_view(tau: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tau.ndim != 1 or tau.shape[0] != target.shape[0]:
        raise ValueError('tau 必须为 [B] 并与 latent batch 一致')
    return tau.reshape(tau.shape[0], *[1] * (target.ndim - 1))

def sample_posterior_tau(batch_size: int, *, device: torch.device | str, generator: torch.Generator | None=None) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError('batch_size 必须为正')
    return torch.rand(batch_size, device=device, generator=generator)

def make_posterior_rf_pair(clean: torch.Tensor, tau: torch.Tensor, *, sigma_min: float, generator: torch.Generator | None=None) -> PosteriorRFPair:
    if clean.ndim < 2:
        raise ValueError('clean latent 至少需要 batch 与 feature 两维')
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError('sigma_min 必须位于 [0,1)')
    tau_view = _tau_view(tau, clean)
    source = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    attenuation = 1.0 - sigma_min
    sigma = 1.0 - attenuation * tau
    sigma_view = _tau_view(sigma, clean)
    noisy = tau_view * clean + sigma_view * source
    target_velocity = clean - attenuation * source
    return PosteriorRFPair(clean=clean, source=source, noisy=noisy, tau=tau, sigma=sigma, target_velocity=target_velocity)

def posterior_clean_from_velocity(noisy: torch.Tensor, velocity: torch.Tensor, tau: torch.Tensor, *, sigma_min: float) -> torch.Tensor:
    if noisy.shape != velocity.shape:
        raise ValueError('noisy/velocity shape 必须一致')
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError('sigma_min 必须位于 [0,1)')
    tau_view = _tau_view(tau, noisy)
    attenuation = 1.0 - sigma_min
    sigma_view = 1.0 - attenuation * tau_view
    return attenuation * noisy + sigma_view * velocity

def posterior_velocity_mse(prediction: torch.Tensor, pair: PosteriorRFPair) -> torch.Tensor:
    if prediction.shape != pair.target_velocity.shape:
        raise ValueError('prediction 与 posterior target shape 不一致')
    return (prediction.float() - pair.target_velocity.float()).pow(2).mean()

class PosteriorVelocityField(Protocol):

    def __call__(self, latent: torch.Tensor, tau: torch.Tensor, *, condition: Any) -> torch.Tensor:
        ...

@dataclass
class PosteriorSample:
    terminal: torch.Tensor
    clean: torch.Tensor
    source: torch.Tensor
    nfe: int

def sample_posterior_heun(field: PosteriorVelocityField, condition: Any, *, batch_size: int, latent_dim: int, num_intervals: int, sigma_min: float, device: torch.device | str, dtype: torch.dtype, source: torch.Tensor | None=None) -> PosteriorSample:
    if batch_size <= 0 or latent_dim <= 0 or num_intervals <= 0:
        raise ValueError('batch_size/latent_dim/num_intervals 必须为正')
    if not 0.0 <= sigma_min < 1.0:
        raise ValueError('sigma_min 必须位于 [0,1)')
    if source is None:
        source = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
    elif source.shape != (batch_size, latent_dim):
        raise ValueError('source 必须为 [B,latent_dim]')
    else:
        source = source.to(device=device, dtype=dtype)
    latent = source
    grid = torch.linspace(0.0, 1.0, num_intervals + 1, device=device, dtype=dtype)
    nfe = 0
    for index in range(num_intervals):
        left = grid[index]
        right = grid[index + 1]
        step = right - left
        tau_left = left.expand(batch_size)
        velocity_left = field(latent, tau_left, condition=condition)
        proposal = latent + step * velocity_left
        tau_right = right.expand(batch_size)
        velocity_right = field(proposal, tau_right, condition=condition)
        latent = latent + 0.5 * step * (velocity_left + velocity_right)
        nfe += 2
    tau_one = torch.ones(batch_size, device=device, dtype=dtype)
    endpoint_velocity = field(latent, tau_one, condition=condition)
    nfe += 1
    clean = posterior_clean_from_velocity(latent, endpoint_velocity, tau_one, sigma_min=sigma_min)
    return PosteriorSample(terminal=latent, clean=clean, source=source, nfe=nfe)
