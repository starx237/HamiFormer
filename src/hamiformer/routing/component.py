from __future__ import annotations
from dataclasses import dataclass
import torch

def future_normalized_mse(prediction: torch.Tensor, target: torch.Tensor, state_scale: torch.Tensor, *, reference_index: int) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('prediction/target 必须是同形 [B,T,K,D]')
    if not 0 <= reference_index < prediction.shape[1] - 1:
        raise ValueError('reference_index 必须至少留下一个 future frame')
    if state_scale.shape != (prediction.shape[-1],):
        raise ValueError('state_scale shape 与 phase channel 不一致')
    scale = state_scale.to(prediction).view(1, 1, 1, -1)
    error = (prediction[:, reference_index + 1:] - target[:, reference_index + 1:]) / scale
    return error.float().square().mean(dim=(1, 2, 3))

def clipped_risk_difference_target(h_loss: torch.Tensor, d_loss: torch.Tensor, *, clip_value: float) -> torch.Tensor:
    if h_loss.shape != d_loss.shape or h_loss.ndim != 1:
        raise ValueError('h_loss/d_loss 必须是同形 [B]')
    if clip_value <= 0.0:
        raise ValueError('clip_value 必须为正')
    return (h_loss - d_loss).clamp(min=-clip_value, max=clip_value)

@dataclass(frozen=True)
class ComponentThreshold:
    risk_threshold: float
    h_coverage: float
    routed_mean_loss: float
    d_only_mean_loss: float
    relative_tolerance: float
    safe: bool

def calibrate_component_threshold(predicted_difference: torch.Tensor, h_loss: torch.Tensor, d_loss: torch.Tensor, *, relative_tolerance: float) -> ComponentThreshold:
    if not (predicted_difference.ndim == h_loss.ndim == d_loss.ndim == 1 and predicted_difference.shape == h_loss.shape == d_loss.shape and (predicted_difference.numel() > 0)):
        raise ValueError('score/h_loss/d_loss 必须是非空同形 [N]')
    if relative_tolerance < 0.0:
        raise ValueError('relative_tolerance 不能为负')
    if not (torch.isfinite(predicted_difference).all() and torch.isfinite(h_loss).all() and torch.isfinite(d_loss).all()):
        raise ValueError('score/loss 含 NaN 或 Inf')
    d_mean = float(d_loss.double().mean().item())
    safe_limit = d_mean * (1.0 + relative_tolerance)
    candidates = [float('-inf'), *sorted(set((float(value) for value in predicted_difference.tolist())))]
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        choose_h = predicted_difference <= threshold
        routed = torch.where(choose_h, h_loss, d_loss)
        routed_mean = float(routed.double().mean().item())
        coverage = float(choose_h.double().mean().item())
        if routed_mean <= safe_limit:
            candidate = (coverage, -routed_mean, threshold)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return ComponentThreshold(risk_threshold=float('-inf'), h_coverage=0.0, routed_mean_loss=d_mean, d_only_mean_loss=d_mean, relative_tolerance=relative_tolerance, safe=False)
    coverage, negative_loss, threshold = best
    return ComponentThreshold(risk_threshold=threshold, h_coverage=coverage, routed_mean_loss=-negative_loss, d_only_mean_loss=d_mean, relative_tolerance=relative_tolerance, safe=True)
