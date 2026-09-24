from __future__ import annotations
import hashlib
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from hamiformer.data.hamiballs2 import HamiBalls2CropPackDataset
from hamiformer.models.hamiballs2_hamiltonian import HamiBalls2ContinuousHamiltonian, graph_context, parameter_count
from hamiformer.physics.continuous_hamiltonian import canonical_vector_field
from hamiformer.physics.generic_type2 import type2_vector_eom
from hamiformer.physics.generic_type2 import type2_vector_health_barrier
from hamiformer.training import ExplicitEpochBatchStream

def derived_seed(master: int, label: str) -> int:
    digest = hashlib.sha256(f'hamiballs2-h-capacity-v1|{master}|{label}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 63 - 1)

def learning_rate(step: int, total: int) -> float:
    warmup = min(1000, total - 1)
    if step <= warmup:
        return 0.0001 * step / warmup
    progress = (step - warmup) / (total - warmup)
    return 1e-06 + 0.5 * (0.0001 - 1e-06) * (1.0 + math.cos(math.pi * progress))

def fetch(dataset: HamiBalls2CropPackDataset, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    rows = indices.cpu().numpy()
    keys = ('phase', 'attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length', 'contact', 'physical_seed')
    return {key: torch.from_numpy(np.array(dataset.arrays[key][rows], copy=True)).to(device=device, non_blocking=True) for key in keys if key in dataset.arrays}

def relation_terms(model: HamiBalls2ContinuousHamiltonian, batch: dict[str, torch.Tensor], phase_scale: torch.Tensor, equation_scale: torch.Tensor, *, frame_dt: float, dof: float, create_graph: bool, robust_granularity: str='whole_edge', objective_mode: str='midpoint_field') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = batch['phase'].float()
    left, right = (phase[:, :-1], phase[:, 1:])
    context = graph_context(batch['attrs'].float(), batch['object_mask'].bool(), batch['spring_mask'], batch['spring_k'].float(), batch['spring_rest_length'].float())
    context = context[:, None].expand(-1, left.shape[1], -1, -1, -1)
    if objective_mode == 'midpoint_field':
        midpoint = 0.5 * (left + right)
        field = canonical_vector_field(model, midpoint, context, create_graph=create_graph)
        raw_residual = frame_dt * field - (right - left)
    elif objective_mode == 'type2_endpoint':
        q_left = left[..., :3].reshape(*left.shape[:2], -1)
        p_right = right[..., 3:].reshape(*right.shape[:2], -1)
        predicted_p_left, predicted_q_right = type2_vector_eom(model, q_left, p_right, context, step_size=frame_dt, create_graph=create_graph)
        q_residual = predicted_q_right.reshape(*left.shape[:3], 3) - right[..., :3]
        p_residual = predicted_p_left.reshape(*left.shape[:3], 3) - left[..., 3:]
        raw_residual = torch.cat((q_residual, p_residual), dim=-1)
    else:
        raise ValueError(f'unsupported H objective mode: {objective_mode}')
    residual = raw_residual / phase_scale.reshape(1, 1, 1, 6)
    valid = batch['object_mask'].bool()[:, None, :].expand(residual.shape[:3])
    weight = valid[..., None].to(residual)
    count = (valid.sum(dim=-1) * residual.shape[-1]).clamp_min(1).to(residual)
    standardized = (residual / equation_scale.reshape(1, 1, 1, 6)).square()
    if robust_granularity == 'whole_edge':
        standardized_edge = (standardized * weight).sum(dim=(-2, -1)) / count
        retention = (dof / (dof + standardized_edge)).detach()
        half_mse = 0.5 * (residual.square() * weight).sum(dim=(-2, -1)) / count
        loss = (dof + 1.0) / dof * (retention * half_mse).mean()
    elif robust_granularity == 'object_edge':
        standardized_object = standardized.mean(dim=-1)
        retention = (dof / (dof + standardized_object)).detach() * valid.to(residual)
        half_mse = 0.5 * residual.square().mean(dim=-1)
        loss = (dof + 1.0) / dof * ((retention * half_mse).sum() / valid.sum().clamp_min(1).to(residual))
    else:
        raise ValueError(f'unsupported robust granularity: {robust_granularity}')
    return (loss, residual, retention)

def compute_equation_scale(dataset: HamiBalls2CropPackDataset, phase_scale: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    phase = dataset.arrays['phase']
    mask = dataset.arrays['object_mask']
    for start in range(0, len(dataset), 512):
        stop = min(start + 512, len(dataset))
        local_phase = np.asarray(phase[start:stop], dtype=np.float32)
        delta = (local_phase[:, 1:] - local_phase[:, :-1]) / phase_scale.reshape(1, 1, 1, 6)
        valid = np.broadcast_to(np.asarray(mask[start:stop], dtype=bool)[:, None, :], delta.shape[:3])
        values.append(np.abs(delta[valid]))
    result = np.median(np.concatenate(values, axis=0), axis=0).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result <= 1e-08):
        raise ValueError('anisotropic transition scale is non-finite or degenerate')
    return result
