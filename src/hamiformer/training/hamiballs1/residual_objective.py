from __future__ import annotations
from hamiformer.utils.paths import project_root
import hashlib
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
FEATURE_DIM = 63
HIDDEN_DIM = 32
EDGES = 48
FRAME_DT = 1.0 / 30.0
CHECKPOINT_SCHEMA = 'hamiformer.hamiballs.residual_objective.r_confidence_head.v1'

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

class ConfidenceHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.input(features))).squeeze(-1)

def _nearest_distance(state: torch.Tensor) -> torch.Tensor:
    q = state[..., :2]
    delta = q[..., :, None, :] - q[..., None, :, :]
    diagonal = torch.eye(q.shape[-2], device=q.device, dtype=torch.bool)
    squared = delta.square().sum(dim=-1).masked_fill(diagonal, 1.0)
    distance = squared.sqrt()
    distance = distance.masked_fill(diagonal, torch.inf)
    return distance.min(dim=-1, keepdim=True).values

def observable_features_step(h: torch.Tensor, d: torch.Tensor, residual: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, *, frame_dt: float=FRAME_DT, edges: int=EDGES) -> torch.Tensor:
    if h.shape != d.shape or h.shape != residual.shape or h.ndim != 3:
        raise ValueError('abstention states must align as [B,K,4]')
    if h.shape[-1] != 4 or attrs.shape[:2] != h.shape[:2]:
        raise ValueError('abstention attrs/state shape changed')
    if physical_time.shape != (h.shape[0],):
        raise ValueError('abstention physical_time must be [B]')
    if edges < 2 or frame_dt <= 0.0:
        raise ValueError('abstention edge/time contract is invalid')
    disagreement = d - h
    hr = h + residual
    eps = h.new_tensor(1e-08)
    geometry: list[torch.Tensor] = []
    for component in (slice(0, 2), slice(2, 4)):
        r_component = residual[..., component]
        d_component = disagreement[..., component]
        r_norm = (r_component.square().sum(dim=-1, keepdim=True) + eps).sqrt()
        d_norm = (d_component.square().sum(dim=-1, keepdim=True) + eps).sqrt()
        dot = (r_component * d_component).sum(dim=-1, keepdim=True)
        cosine = dot / (r_norm * d_norm + eps)
        geometry.extend((r_norm, d_norm, dot, cosine))
    global_features: list[torch.Tensor] = []
    for value in (h, d, residual, disagreement):
        mean = value.mean(dim=1, keepdim=True)
        std = (value - mean).square().mean(dim=1, keepdim=True).sqrt()
        global_features.extend((mean.expand_as(value), std.expand_as(value)))
    edge_fraction = ((physical_time / float(frame_dt) - 1.0) / float(edges - 1)).clamp(0.0, 1.0)[:, None, None].expand(-1, h.shape[1], 1)
    time_feature = physical_time[:, None, None].expand(-1, h.shape[1], 1)
    features = torch.cat((h, d, residual, disagreement, attrs, edge_fraction, time_feature, *geometry, *global_features, _nearest_distance(h), _nearest_distance(d)), dim=-1)
    if features.shape != h.shape[:-1] + (FEATURE_DIM,):
        raise AssertionError(f'unexpected abstention feature shape {features.shape}')
    if not bool(torch.isfinite(features).all()):
        raise FloatingPointError('abstention features are non-finite')
    return features

class ObservableAbstainingResidual(nn.Module):

    def __init__(self, residual: nn.Module, checkpoint: Path, *, expected_sha256: str) -> None:
        super().__init__()
        if _sha256(checkpoint) != expected_sha256:
            raise ValueError('abstention checkpoint SHA changed')
        payload: dict[str, Any] = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload.get('schema') != CHECKPOINT_SCHEMA or payload.get('status') != 'PASS' or payload.get('seed') != 42 or (payload.get('feature_dim') != FEATURE_DIM) or (payload.get('hidden_dim') != HIDDEN_DIM):
            raise ValueError('abstention checkpoint contract changed')
        self.residual = residual
        self.head = ConfidenceHead()
        self.head.load_state_dict(payload['model_state_dict'], strict=True)
        self.register_buffer('feature_mean', payload['feature_mean'].float())
        self.register_buffer('feature_std', payload['feature_std'].float())
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.head.eval()
        self.state_dim = int(getattr(residual, 'state_dim'))
        self.per_object_previous_g = bool(getattr(residual, 'per_object_previous_g', False))

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool=True):
        return self.residual.load_state_dict(state_dict, strict=strict)

    def forward_step_with_diagnostics(self, d_tokens: torch.Tensor, noisy: torch.Tensor, x0: torch.Tensor, previous: torch.Tensor, h_candidate: torch.Tensor, d_candidate: torch.Tensor, attrs: torch.Tensor, tau: torch.Tensor, physical_time: torch.Tensor, previous_g: torch.Tensor):
        values = self.residual.forward_step_with_diagnostics(d_tokens, noisy, x0, previous, h_candidate, d_candidate, attrs, tau, physical_time, previous_g)
        innovation, hidden, gain, direction_rms, _innovation_rms = values
        features = observable_features_step(h_candidate, d_candidate, innovation, attrs, physical_time)
        normalised = (features - self.feature_mean.to(features)) / self.feature_std.to(features)
        alpha = torch.sigmoid(self.head(normalised))
        scaled = alpha[..., None] * innovation
        scaled_gain = alpha * gain
        scaled_rms = scaled.square().mean(dim=-1).sqrt()
        return (scaled, hidden, scaled_gain, direction_rms, scaled_rms)
__all__ = ['ConfidenceHead', 'ObservableAbstainingResidual', 'observable_features_step']
