from __future__ import annotations
from hamiformer.utils.paths import project_root
from typing import Any
import torch
from hamiformer.training.hamiballs_formal import previous_gate_sequence

@torch.no_grad()
def _fit_observable_statistics(residual, carrier: Any, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, state_scale: torch.Tensor) -> int:
    if int(residual.statistics_fitted.item()) != 0:
        raise RuntimeError('FeedbackCarrier observable statistics may be fitted only once')
    residual.state_scale.copy_(state_scale.to(residual.state_scale))
    sums = {'feature': torch.zeros(44, dtype=torch.float64, device=x0.device), 'feature2': torch.zeros(44, dtype=torch.float64, device=x0.device), 'q': torch.zeros(13, dtype=torch.float64, device=x0.device), 'q2': torch.zeros(13, dtype=torch.float64, device=x0.device), 'p': torch.zeros(13, dtype=torch.float64, device=x0.device), 'p2': torch.zeros(13, dtype=torch.float64, device=x0.device)}
    count = 0
    for field in carrier.trace.traces:
        rollout = field.rollout
        if rollout is None or field.d_tokens is None:
            continue
        incoming = previous_gate_sequence(rollout.gate.detach(), initial=field.anchor.previous_g)
        context = None
        frames = int(rollout.mixed.shape[1])
        for edge in range(frames):
            raw, context = residual.raw_step_features_with_context(field.d_tokens[:, edge].detach(), field.state[:, edge].detach(), x0.detach(), rollout.previous_mixed[:, edge].detach(), rollout.h_candidate[:, edge].detach(), rollout.d_candidate[:, edge].detach(), attrs.detach(), field.tau.detach(), physical_time[:, 1 + edge].detach(), incoming[:, edge].detach(), context=context)
            q_raw, p_raw = residual.raw_quality_features(raw)
            feature_flat = raw.reshape(-1, 44).double()
            q_flat = q_raw.reshape(-1, 13).double()
            p_flat = p_raw.reshape(-1, 13).double()
            rows = int(feature_flat.shape[0])
            count += rows
            sums['feature'] += feature_flat.sum(dim=0)
            sums['feature2'] += feature_flat.square().sum(dim=0)
            sums['q'] += q_flat.sum(dim=0)
            sums['q2'] += q_flat.square().sum(dim=0)
            sums['p'] += p_flat.sum(dim=0)
            sums['p2'] += p_flat.square().sum(dim=0)
    if count < 1:
        raise RuntimeError('FeedbackCarrier first carrier exposed no observable rows')

    def mean_scale(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        mean64 = sums[name] / float(count)
        variance = (sums[f'{name}2'] / float(count) - mean64.square()).clamp_min(0.0)
        scale64 = variance.sqrt().clamp_min(1e-05)
        return (mean64.float(), scale64.float())
    feature_mean, feature_scale = mean_scale('feature')
    q_mean, q_scale = mean_scale('q')
    p_mean, p_scale = mean_scale('p')
    residual.set_observable_statistics(state_scale=state_scale, feature_mean=feature_mean, feature_scale=feature_scale, quality_q_mean=q_mean, quality_q_scale=q_scale, quality_p_mean=p_mean, quality_p_scale=p_scale)
    return count
