from __future__ import annotations
from dataclasses import dataclass
import math
import torch

@dataclass(frozen=True)
class ConvexRoutingTarget:
    responsibility: torch.Tensor
    disagreement_sq: torch.Tensor
    identifiable: torch.Tensor
    relative_gain: torch.Tensor

def _convex_routing_target_core(d_clean: torch.Tensor, h_clean: torch.Tensor, clean_target: torch.Tensor, phase_scale: torch.Tensor, *, disagreement_floor: float) -> ConvexRoutingTarget:
    scale = phase_scale.to(clean_target)
    direction = (h_clean - d_clean) / scale
    target_direction = (clean_target - d_clean) / scale
    disagreement_sq = direction.square().sum(dim=-1, keepdim=True)
    floor = torch.as_tensor(disagreement_floor, device=disagreement_sq.device, dtype=disagreement_sq.dtype)
    identifiable = disagreement_sq > floor
    numerator = (target_direction * direction).sum(dim=-1, keepdim=True)
    projected = (numerator / disagreement_sq.clamp_min(floor)).clamp(0.0, 1.0)
    neutral = torch.full_like(projected, 0.5)
    responsibility = torch.where(identifiable, projected, neutral)
    oracle_clean = d_clean + responsibility * (h_clean - d_clean)
    d_error = ((d_clean - clean_target) / scale).square().sum(dim=-1, keepdim=True)
    oracle_error = ((oracle_clean - clean_target) / scale).square().sum(dim=-1, keepdim=True)
    relative_gain = ((d_error - oracle_error) / d_error.clamp_min(floor)).clamp(0.0, 1.0)
    relative_gain = torch.where(identifiable, relative_gain, torch.zeros_like(relative_gain))
    return ConvexRoutingTarget(responsibility=responsibility.detach(), disagreement_sq=disagreement_sq.detach(), identifiable=identifiable.detach(), relative_gain=relative_gain.detach())

def convex_routing_target(d_clean: torch.Tensor, h_clean: torch.Tensor, clean_target: torch.Tensor, phase_scale: torch.Tensor, *, disagreement_floor: float=1e-12) -> ConvexRoutingTarget:
    if d_clean.shape != h_clean.shape or d_clean.shape != clean_target.shape:
        raise ValueError('D/H/clean proposal 必须同形')
    if d_clean.ndim < 2:
        raise ValueError('D/H/clean proposal 至少需要 batch 与 state 两个维度')
    if phase_scale.shape != (clean_target.shape[-1],):
        raise ValueError('phase_scale shape 错误')
    if not bool(torch.isfinite(phase_scale).all().item()) or bool((phase_scale <= 0.0).any().item()):
        raise ValueError('phase_scale 必须全部为 finite 正数')
    if not math.isfinite(disagreement_floor) or disagreement_floor <= 0.0:
        raise ValueError('disagreement_floor 必须为正')
    return _convex_routing_target_core(d_clean, h_clean, clean_target, phase_scale, disagreement_floor=disagreement_floor)

def convex_routing_target_prevalidated(d_clean: torch.Tensor, h_clean: torch.Tensor, clean_target: torch.Tensor, phase_scale: torch.Tensor, *, disagreement_floor: float=1e-12) -> ConvexRoutingTarget:
    if d_clean.shape != h_clean.shape or d_clean.shape != clean_target.shape:
        raise ValueError('D/H/clean proposal 必须同形')
    if phase_scale.shape != (clean_target.shape[-1],):
        raise ValueError('phase_scale shape 错误')
    if not math.isfinite(disagreement_floor) or disagreement_floor <= 0.0:
        raise ValueError('disagreement_floor 必须为正')
    return _convex_routing_target_core(d_clean, h_clean, clean_target, phase_scale, disagreement_floor=disagreement_floor)

def disagreement_weighted_routing_loss(responsibility: torch.Tensor, target: ConvexRoutingTarget, *, extra_weight: torch.Tensor | None=None, denominator_floor: float=1e-12) -> torch.Tensor:
    if responsibility.shape != target.responsibility.shape:
        raise ValueError('responsibility 与 target shape 不一致')
    if not bool(torch.isfinite(responsibility).all().item()) or bool(((responsibility < 0.0) | (responsibility > 1.0)).any().item()):
        raise ValueError('responsibility 必须为 [0,1] 内的 finite 数')
    if not math.isfinite(denominator_floor) or denominator_floor <= 0.0:
        raise ValueError('denominator_floor 必须为正')
    weight = target.disagreement_sq
    if extra_weight is not None:
        if extra_weight.shape != weight.shape:
            raise ValueError('extra_weight 必须与逐边 disagreement 同形')
        if not bool(torch.isfinite(extra_weight).all().item()) or bool((extra_weight < 0.0).any().item()):
            raise ValueError('extra_weight 必须为 finite 非负数')
        weight = weight * extra_weight.detach()
    numerator = (weight * (responsibility - target.responsibility).square()).sum()
    denominator = weight.sum().clamp_min(torch.as_tensor(denominator_floor, device=weight.device, dtype=weight.dtype))
    return numerator / denominator
__all__ = ['ConvexRoutingTarget', 'convex_routing_target', 'convex_routing_target_prevalidated', 'disagreement_weighted_routing_loss']
