from __future__ import annotations
import torch

def wavefront_active_clean_loss(clean_estimate: torch.Tensor, clean_target: torch.Tensor, active: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    if clean_estimate.shape != clean_target.shape or clean_estimate.ndim != 4:
        raise ValueError('clean estimate/target 必须是相同shape的 [B,F,K,D]')
    if active.shape != clean_estimate.shape[:2] or active.dtype != torch.bool:
        raise ValueError('active 必须是 [B,F] bool mask')
    if state_scale.shape != (clean_estimate.shape[-1],):
        raise ValueError('state_scale 必须为 [D]')
    if bool((state_scale <= 0).any().item()):
        raise ValueError('state_scale 必须严格为正')
    if not bool(active.any().item()):
        raise ValueError('当前batch没有active wavefront state')
    normalized = (clean_estimate - clean_target) / state_scale.to(clean_estimate).view(1, 1, 1, -1)
    per_state = normalized.square().mean(dim=(2, 3))
    return per_state[active].mean()
