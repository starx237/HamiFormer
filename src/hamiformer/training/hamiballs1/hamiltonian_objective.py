from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
from typing import Any
import torch
from hamiformer.physics.continuous_hamiltonian import canonical_vector_field
ROOT = project_root()
_equation_scale: torch.Tensor | None = None
_latest: dict[str, float] = {}

def _prepare_relation_objective(train: Any, state_scale: torch.Tensor, *, q_dim: int) -> None:
    global _equation_scale
    phase = torch.cat((train.raw_x0[:, None], train.raw_future), dim=1)
    if phase.shape[-1] != 2 * q_dim:
        raise ValueError('Training phase has the wrong q/p dimension')
    normalized_transition = (phase[:, 1:] - phase[:, :-1]) / state_scale.reshape(1, 1, 1, -1)
    scale = torch.quantile(normalized_transition.detach().abs().reshape(-1, 2 * q_dim), 0.5, dim=0)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 1e-08).any()):
        raise ValueError('Fixed anisotropic transition scale is degenerate')
    _equation_scale = scale.detach()

def _relation_loss(hamiltonian, phase: torch.Tensor, context: torch.Tensor, *, state_scale: torch.Tensor, frame_dt: float, dof: float) -> torch.Tensor:
    global _latest
    if _equation_scale is None:
        raise RuntimeError('Hamiltonian relation scale was not prepared')
    left, right = (phase[:, :-1], phase[:, 1:])
    midpoint = 0.5 * (left + right)
    expanded_context = context[:, None].expand(-1, midpoint.shape[1], -1, -1, -1)
    field = canonical_vector_field(hamiltonian, midpoint, expanded_context, create_graph=True)
    residual = (frame_dt * field - (right - left)) / state_scale.reshape(1, 1, 1, -1)
    standardized_squared_mean = (residual / _equation_scale.reshape(1, 1, 1, -1)).square().mean(dim=(-2, -1))
    retention = (dof / (dof + standardized_squared_mean)).detach()
    canonical_half_mse = 0.5 * residual.square().mean(dim=(-2, -1))
    location = (dof + 1.0) / dof * (retention * canonical_half_mse).mean()
    detached = retention.detach()
    _latest = {'retention_mean': float(detached.mean().cpu()), 'retention_p10': float(torch.quantile(detached, 0.1).cpu()), 'retention_p50': float(torch.quantile(detached, 0.5).cpu()), 'retention_p90': float(torch.quantile(detached, 0.9).cpu())}
    return location

def _relation_diagnostics() -> dict[str, float]:
    return dict(_latest)
