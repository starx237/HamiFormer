from __future__ import annotations
import torch
from hamiformer.models.block_hamiltonian_expert import BlockHamiltonianOccurrences

def block_occurrence_clean_loss(occurrences: BlockHamiltonianOccurrences, clean_full: torch.Tensor, *, q_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor) -> torch.Tensor:
    if clean_full.ndim != 4 or clean_full.shape[-1] != 2 * q_dim:
        raise ValueError('clean_full 必须为 [B,F+1,K,2*q_dim]')
    if q_scale.shape != (q_dim,) or p_scale.shape != (q_dim,):
        raise ValueError('q_scale/p_scale 必须为 [q_dim]')
    clean_q = clean_full[..., :q_dim]
    clean_p = clean_full[..., q_dim:]
    source = occurrences.source_indices
    target = occurrences.target_indices
    groups = ((occurrences.q_plus, clean_q[:, target], q_scale), (occurrences.p_plus, clean_p[:, source], p_scale), (occurrences.q_minus, clean_q[:, source], q_scale), (occurrences.p_minus, clean_p[:, target], p_scale))
    total = clean_full.new_zeros((), dtype=torch.float32)
    count = 0
    for prediction, truth, scale in groups:
        if prediction.shape != truth.shape:
            raise ValueError('occurrence proposal 与 clean target shape 不一致')
        denominator = scale.to(prediction).view(1, 1, 1, 1, -1)
        error = ((prediction - truth) / denominator).float()
        total = total + error.square().sum()
        count += error.numel()
    if count == 0:
        raise RuntimeError('block occurrence loss 没有有效元素')
    return total / count

def block_occurrence_rmse_by_chart(occurrences: BlockHamiltonianOccurrences, clean_full: torch.Tensor, *, q_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor) -> dict[str, torch.Tensor]:
    clean_q = clean_full[..., :q_dim]
    clean_p = clean_full[..., q_dim:]
    source = occurrences.source_indices
    target = occurrences.target_indices
    groups = {'q_plus': (occurrences.q_plus, clean_q[:, target], q_scale), 'p_plus': (occurrences.p_plus, clean_p[:, source], p_scale), 'q_minus': (occurrences.q_minus, clean_q[:, source], q_scale), 'p_minus': (occurrences.p_minus, clean_p[:, target], p_scale)}
    result: dict[str, torch.Tensor] = {}
    for name, (prediction, truth, scale) in groups.items():
        denominator = scale.to(prediction).view(1, 1, 1, 1, -1)
        result[name] = ((prediction - truth) / denominator).float().square().mean().sqrt()
    return result
