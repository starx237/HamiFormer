from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

@dataclass(frozen=True)
class PhaseScales:
    q: torch.Tensor
    p: torch.Tensor

    def state(self, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        return torch.cat([self.q, self.p]).to(device=device, dtype=dtype)

def load_phase_scales(path: str | Path, q_dim: int, *, expected_train_manifest_sha256: str) -> PhaseScales:
    stats_path = Path(path)
    with stats_path.open('r', encoding='utf-8') as handle:
        raw = json.load(handle)
    required = {'format_version', 'train_manifest_sha256', 'q_scale', 'p_scale'}
    if set(raw) != required or raw['format_version'] != 1:
        raise ValueError('phase_scales JSON 格式或版本错误')
    if raw['train_manifest_sha256'] != expected_train_manifest_sha256:
        raise ValueError('phase_scales 不是由当前 train manifest 计算得到')
    q = torch.as_tensor(raw['q_scale'], dtype=torch.float32)
    p = torch.as_tensor(raw['p_scale'], dtype=torch.float32)
    if q.shape != (q_dim,) or p.shape != (q_dim,):
        raise ValueError(f'q/p scale 必须都是 [{q_dim}]，实际为 {q.shape}/{p.shape}')
    if not torch.isfinite(q).all() or not torch.isfinite(p).all() or (q <= 0).any() or (p <= 0).any():
        raise ValueError('q/p scale 必须为有限正数')
    return PhaseScales(q=q, p=p)

def validate_phase_arrays(phase: np.ndarray, attrs: np.ndarray, time: np.ndarray, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int) -> None:
    expected_phase = (future_steps + 1, num_objects, 2 * q_dim)
    expected_attrs = (num_objects, attr_dim)
    expected_time = (future_steps + 1,)
    if phase.shape != expected_phase:
        raise ValueError(f'phase 期望 {expected_phase}，实际 {phase.shape}')
    if attrs.shape != expected_attrs:
        raise ValueError(f'attrs 期望 {expected_attrs}，实际 {attrs.shape}')
    if time.shape != expected_time:
        raise ValueError(f'time 期望 {expected_time}，实际 {time.shape}')
    if phase.dtype != np.float32 or attrs.dtype != np.float32 or time.dtype != np.float32:
        raise ValueError('phase/attrs/time 必须严格为 float32，loader 不做静默转换')
    if not np.isfinite(phase).all() or not np.isfinite(attrs).all() or (not np.isfinite(time).all()):
        raise ValueError('phase/attrs/time 含 NaN 或 Inf')
    if not np.all(np.diff(time) > 0):
        raise ValueError('time 必须严格递增')
    if np.any(attrs[:, 0] <= 0) or np.any(attrs[:, 1] <= 0):
        raise ValueError('mass 与 radius 必须为正')
    if np.any((attrs[:, 2] < 0) | (attrs[:, 2] > 1)):
        raise ValueError('restitution 必须位于 [0,1]')
