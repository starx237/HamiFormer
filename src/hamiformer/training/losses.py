from __future__ import annotations
import torch
from hamiformer.types import HamiltonianOccurrences, RFPair

def _tau_view(tau: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return tau.reshape(tau.shape[0], *[1] * (target.ndim - 1))

def weighted_velocity_mse(prediction: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError('prediction/target shape 必须一致')
    scale = state_scale.to(prediction).view(*[1] * (prediction.ndim - 1), -1)
    return ((prediction - target) / scale).pow(2).mean()

def d_rf_loss(clean_prediction: torch.Tensor, pair: RFPair, state_scale: torch.Tensor, *, t_eps: float=0.05) -> torch.Tensor:
    denominator = (1.0 - _tau_view(pair.tau, pair.noisy)).clamp_min(t_eps)
    prediction_velocity = (clean_prediction - pair.noisy) / denominator
    target_velocity = (pair.clean - pair.noisy) / denominator
    return weighted_velocity_mse(prediction_velocity, target_velocity, state_scale)

def occurrence_rf_loss(occurrences: HamiltonianOccurrences, pair: RFPair, *, q_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor, t_eps: float=0.05) -> torch.Tensor:
    noisy_q, noisy_p = (pair.noisy[..., :q_dim], pair.noisy[..., q_dim:])
    target_q = pair.target_velocity[..., :q_dim]
    target_p = pair.target_velocity[..., q_dim:]
    inv = 1.0 / (1.0 - _tau_view(pair.tau, pair.noisy)).clamp_min(t_eps)
    inv_q = inv[..., :1]
    groups = [((occurrences.q_plus - noisy_q) * inv_q, target_q, q_scale), ((occurrences.p_minus - noisy_p) * inv_q, target_p, p_scale), ((occurrences.q_minus[:, 1:] - noisy_q[:, :-1]) * inv_q, target_q[:, :-1], q_scale), ((occurrences.p_plus[:, 1:] - noisy_p[:, :-1]) * inv_q, target_p[:, :-1], p_scale)]
    total = pair.noisy.new_zeros((), dtype=torch.float32)
    count = 0
    for prediction, target, scale in groups:
        normalized = (prediction - target) / scale.to(prediction).view(1, 1, 1, -1)
        total = total + normalized.float().pow(2).sum()
        count += normalized.numel()
    if count == 0:
        raise RuntimeError('occurrence loss 没有有效元素')
    return total / count

def occurrence_clean_loss(occurrences: HamiltonianOccurrences, clean: torch.Tensor, *, q_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor) -> torch.Tensor:
    clean_q, clean_p = (clean[..., :q_dim], clean[..., q_dim:])
    groups = [(occurrences.q_plus, clean_q, q_scale), (occurrences.p_minus, clean_p, p_scale), (occurrences.q_minus[:, 1:], clean_q[:, :-1], q_scale), (occurrences.p_plus[:, 1:], clean_p[:, :-1], p_scale)]
    total = clean.new_zeros((), dtype=torch.float32)
    count = 0
    for prediction, target, scale in groups:
        normalized = (prediction - target) / scale.to(prediction).view(1, 1, 1, -1)
        total = total + normalized.float().pow(2).sum()
        count += normalized.numel()
    if count == 0:
        raise RuntimeError('occurrence clean loss 没有有效元素')
    return total / count
