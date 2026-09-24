from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class PhaseMetricAccumulator:
    q_sse: float = 0.0
    p_sse: float = 0.0
    q_count: int = 0
    p_count: int = 0
    normalized_sse: float = 0.0
    normalized_count: int = 0
    final_normalized_sse: float = 0.0
    final_normalized_count: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor, *, q_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError('prediction/target shape 不一致')
        q_error = prediction[..., :q_dim] - target[..., :q_dim]
        p_error = prediction[..., q_dim:] - target[..., q_dim:]
        q_norm = q_error / q_scale.to(q_error).view(1, 1, 1, -1)
        p_norm = p_error / p_scale.to(p_error).view(1, 1, 1, -1)
        normalized = torch.cat([q_norm, p_norm], dim=-1)
        self.q_sse += float(q_error.double().pow(2).sum().cpu())
        self.p_sse += float(p_error.double().pow(2).sum().cpu())
        self.q_count += q_error.numel()
        self.p_count += p_error.numel()
        self.normalized_sse += float(normalized.double().pow(2).sum().cpu())
        self.normalized_count += normalized.numel()
        final = normalized[:, -1]
        self.final_normalized_sse += float(final.double().pow(2).sum().cpu())
        self.final_normalized_count += final.numel()

    def result(self) -> dict[str, float]:
        if min(self.q_count, self.p_count, self.normalized_count, self.final_normalized_count) <= 0:
            raise RuntimeError('PhaseMetricAccumulator 没有样本')
        return {'q_mse': self.q_sse / self.q_count, 'p_mse': self.p_sse / self.p_count, 'phase_mse_normalized': self.normalized_sse / self.normalized_count, 'final_phase_rmse_normalized': (self.final_normalized_sse / self.final_normalized_count) ** 0.5}

def phase_metrics(prediction: torch.Tensor, target: torch.Tensor, *, q_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError('prediction/target shape 不一致')
    q_error = prediction[..., :q_dim] - target[..., :q_dim]
    p_error = prediction[..., q_dim:] - target[..., q_dim:]
    q_norm = q_error / q_scale.to(q_error).view(1, 1, 1, -1)
    p_norm = p_error / p_scale.to(p_error).view(1, 1, 1, -1)
    return {'q_mse': q_error.pow(2).mean(), 'p_mse': p_error.pow(2).mean(), 'phase_mse_normalized': torch.cat([q_norm, p_norm], dim=-1).pow(2).mean(), 'final_phase_rmse_normalized': torch.cat([q_norm[:, -1], p_norm[:, -1]], dim=-1).pow(2).mean().sqrt()}

def kinetic_energy(phase: torch.Tensor, attrs: torch.Tensor, q_dim: int) -> torch.Tensor:
    momentum = phase[..., q_dim:]
    mass = attrs[..., 0]
    return (momentum.pow(2).sum(dim=-1) / (2.0 * mass[:, None])).sum(dim=-1)
