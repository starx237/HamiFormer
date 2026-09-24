from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from hamiformer.types import PhaseBatch
from hamiformer.utils import sha256_file
from .phase_array_pack import PHASE_ARRAY_PACK_SCHEMA, PhaseArrayPack
from .schema import validate_phase_arrays

class PhaseWindowDataset(Dataset[dict[str, Any]]):

    def __init__(self, manifest_path: str | Path, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, verify_content_hash: bool=False) -> None:
        self.manifest_path = Path(manifest_path)
        self.num_objects = int(num_objects)
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.attr_dim = int(attr_dim)
        self.verify_content_hash = bool(verify_content_hash)
        self.array_pack: PhaseArrayPack | None = self._maybe_load_array_pack()
        self.records = self.array_pack.records if self.array_pack is not None else self._read_manifest()

    def _maybe_load_array_pack(self) -> PhaseArrayPack | None:
        resolved_manifest = self.manifest_path.expanduser().resolve()
        if resolved_manifest.suffix.lower() != '.json':
            return None
        if not resolved_manifest.is_file():
            raise FileNotFoundError(f'manifest 不存在: {self.manifest_path}。本仓库不会自动下载或生成数据。')
        with resolved_manifest.open('r', encoding='utf-8') as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict) or raw.get('schema') != PHASE_ARRAY_PACK_SCHEMA:
            raise ValueError(f'.json 数据入口必须是 {PHASE_ARRAY_PACK_SCHEMA}: {self.manifest_path}')
        return PhaseArrayPack(resolved_manifest, num_objects=self.num_objects, future_steps=self.future_steps, q_dim=self.q_dim, attr_dim=self.attr_dim, verify_content_hash=self.verify_content_hash)

    def _read_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f'manifest 不存在: {self.manifest_path}。本仓库不会自动下载或生成数据。')
        records: list[dict[str, str]] = []
        with self.manifest_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                required = {'sample_id', 'scene_id', 'npz_path', 'npz_sha256'}
                if not required.issubset(raw):
                    raise ValueError(f'manifest 第 {line_number} 行缺少字段: {sorted(required - set(raw))}')
                records.append({key: str(raw[key]) for key in required})
        if not records:
            raise ValueError(f'manifest 为空: {self.manifest_path}')
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.array_pack is not None:
            record = self.records[index]
            phase = np.array(self.array_pack.phase[index], copy=True)
            attrs = np.array(self.array_pack.attrs[index], copy=True)
            time = np.array(self.array_pack.time[index], copy=True)
            return {'x0': torch.from_numpy(phase[0]), 'future': torch.from_numpy(phase[1:]), 'attrs': torch.from_numpy(attrs), 'time': torch.from_numpy(time), 'sample_id': str(record['sample_id']), 'scene_id': str(record['scene_id'])}
        record = self.records[index]
        npz_path = Path(record['npz_path'])
        if not npz_path.is_absolute():
            npz_path = (self.manifest_path.parent / npz_path).resolve()
        if self.verify_content_hash and sha256_file(npz_path) != record['npz_sha256']:
            raise ValueError(f'样本内容 hash 与 manifest 不一致: {npz_path}')
        with np.load(npz_path, allow_pickle=False) as sample:
            required = {'phase', 'attrs', 'time'}
            if not required.issubset(sample.files):
                raise ValueError(f'样本 {npz_path} 缺少字段: {sorted(required - set(sample.files))}')
            phase = np.asarray(sample['phase'])
            attrs = np.asarray(sample['attrs'])
            time = np.asarray(sample['time'])
        validate_phase_arrays(phase, attrs, time, num_objects=self.num_objects, future_steps=self.future_steps, q_dim=self.q_dim, attr_dim=self.attr_dim)
        return {'x0': torch.from_numpy(phase[0]), 'future': torch.from_numpy(phase[1:]), 'attrs': torch.from_numpy(attrs), 'time': torch.from_numpy(time), 'sample_id': record['sample_id'], 'scene_id': record['scene_id']}

def collate_phase_windows(samples: list[dict[str, Any]]) -> PhaseBatch:
    if not samples:
        raise ValueError('不能 collate 空 batch')
    return PhaseBatch(x0=torch.stack([item['x0'] for item in samples], dim=0), future=torch.stack([item['future'] for item in samples], dim=0), attrs=torch.stack([item['attrs'] for item in samples], dim=0), time=torch.stack([item['time'] for item in samples], dim=0), sample_id=[item['sample_id'] for item in samples], scene_id=[item['scene_id'] for item in samples])

class RawPhaseWindowDataset(Dataset[torch.Tensor]):

    def __init__(self, manifest_path: str | Path, *, num_objects: int, future_steps: int, q_dim: int, verify_content_hash: bool=False) -> None:
        if min(int(num_objects), int(future_steps), int(q_dim)) < 1:
            raise ValueError('raw phase dataset dimensions must be positive')
        self.manifest_path = Path(manifest_path)
        self.num_objects = int(num_objects)
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.verify_content_hash = bool(verify_content_hash)
        self.records = self._read_manifest()

    def _read_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f'raw phase manifest does not exist: {self.manifest_path}')
        required = {'npz_path', 'npz_sha256'}
        records: list[dict[str, str]] = []
        with self.manifest_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if set(raw) != required:
                    raise ValueError(f'raw phase manifest must contain only npz_path/npz_sha256; line {line_number} has {sorted(raw)}')
                records.append({key: str(raw[key]) for key in required})
        if not records:
            raise ValueError(f'raw phase manifest is empty: {self.manifest_path}')
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        npz_path = Path(record['npz_path'])
        if not npz_path.is_absolute():
            npz_path = (self.manifest_path.parent / npz_path).resolve()
        if self.verify_content_hash and sha256_file(npz_path) != record['npz_sha256']:
            raise ValueError(f'raw phase sample hash differs from manifest: {npz_path}')
        with np.load(npz_path, allow_pickle=False) as sample:
            if 'phase' not in sample.files:
                raise ValueError(f'raw phase sample lacks phase: {npz_path}')
            phase = np.asarray(sample['phase'])
        expected_shape = (self.future_steps + 1, self.num_objects, 2 * self.q_dim)
        if phase.shape != expected_shape:
            raise ValueError(f'raw phase shape must be {expected_shape}, got {phase.shape}')
        if phase.dtype.kind != 'f' or not np.isfinite(phase).all():
            raise ValueError('raw phase must be finite floating point')
        return torch.from_numpy(np.array(phase, copy=True))

class RawPhaseContextWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __init__(self, manifest_path: str | Path, *, num_objects: int, future_steps: int, q_dim: int, context_dim: int, verify_content_hash: bool=False) -> None:
        if min(int(num_objects), int(future_steps), int(q_dim), int(context_dim)) < 1:
            raise ValueError('raw phase/context dataset dimensions must be positive')
        self.manifest_path = Path(manifest_path)
        self.num_objects = int(num_objects)
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.context_dim = int(context_dim)
        self.verify_content_hash = bool(verify_content_hash)
        self.records = self._read_manifest()

    def _read_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f'raw phase/context manifest does not exist: {self.manifest_path}')
        required = {'npz_path', 'npz_sha256'}
        records: list[dict[str, str]] = []
        with self.manifest_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if set(raw) != required:
                    raise ValueError(f'raw phase/context manifest must contain only npz_path/npz_sha256; line {line_number} has {sorted(raw)}')
                records.append({key: str(raw[key]) for key in required})
        if not records:
            raise ValueError('raw phase/context manifest is empty')
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        npz_path = Path(record['npz_path'])
        if not npz_path.is_absolute():
            npz_path = (self.manifest_path.parent / npz_path).resolve()
        if self.verify_content_hash and sha256_file(npz_path) != record['npz_sha256']:
            raise ValueError(f'raw phase/context sample hash differs from manifest: {npz_path}')
        with np.load(npz_path, allow_pickle=False) as sample:
            if set(sample.files) != {'phase', 'system_context'}:
                raise ValueError(f'raw phase/context sample must expose only phase/system_context; got {sorted(sample.files)}')
            phase = np.asarray(sample['phase'])
            context = np.asarray(sample['system_context'])
        expected_phase = (self.future_steps + 1, self.num_objects, 2 * self.q_dim)
        if phase.shape != expected_phase:
            raise ValueError(f'raw phase shape must be {expected_phase}, got {phase.shape}')
        if context.shape != (self.context_dim,):
            raise ValueError(f'raw system_context shape must be {(self.context_dim,)}, got {context.shape}')
        if phase.dtype.kind != 'f' or context.dtype.kind != 'f' or (not np.isfinite(phase).all()) or (not np.isfinite(context).all()):
            raise ValueError('raw phase/system_context must be finite floating point')
        return (torch.from_numpy(np.array(phase, copy=True)), torch.from_numpy(np.array(context, copy=True)))

class RawPhaseObjectContextWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):

    def __init__(self, manifest_path: str | Path, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, verify_content_hash: bool=False) -> None:
        if min(int(num_objects), int(future_steps), int(q_dim), int(attr_dim)) < 1:
            raise ValueError('raw phase/object-context dimensions must be positive')
        self.manifest_path = Path(manifest_path)
        self.num_objects = int(num_objects)
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.attr_dim = int(attr_dim)
        self.verify_content_hash = bool(verify_content_hash)
        self.records = self._read_manifest()

    def _read_manifest(self) -> list[dict[str, str]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f'raw phase/object-context manifest does not exist: {self.manifest_path}')
        required = {'npz_path', 'npz_sha256'}
        records: list[dict[str, str]] = []
        with self.manifest_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if set(raw) != required:
                    raise ValueError(f'raw phase/object-context manifest must contain only npz_path/npz_sha256; line {line_number} has {sorted(raw)}')
                records.append({key: str(raw[key]) for key in required})
        if not records:
            raise ValueError('raw phase/object-context manifest is empty')
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        npz_path = Path(record['npz_path'])
        if not npz_path.is_absolute():
            npz_path = (self.manifest_path.parent / npz_path).resolve()
        if self.verify_content_hash and sha256_file(npz_path) != record['npz_sha256']:
            raise ValueError(f'raw phase/object-context sample hash differs from manifest: {npz_path}')
        with np.load(npz_path, allow_pickle=False) as sample:
            if set(sample.files) != {'phase', 'attrs', 'time'}:
                raise ValueError(f'raw HamiBalls sample must expose exactly phase/attrs/time; got {sorted(sample.files)}')
            phase = np.asarray(sample['phase'])
            attrs = np.asarray(sample['attrs'])
        expected_phase = (self.future_steps + 1, self.num_objects, 2 * self.q_dim)
        expected_attrs = (self.num_objects, self.attr_dim)
        if phase.shape != expected_phase:
            raise ValueError(f'raw phase shape must be {expected_phase}, got {phase.shape}')
        if attrs.shape != expected_attrs:
            raise ValueError(f'raw attrs shape must be {expected_attrs}, got {attrs.shape}')
        if phase.dtype.kind != 'f' or attrs.dtype.kind != 'f' or (not np.isfinite(phase).all()) or (not np.isfinite(attrs).all()):
            raise ValueError('raw phase/attrs must be finite floating point')
        return (torch.from_numpy(np.array(phase, copy=True)), torch.from_numpy(np.array(attrs, copy=True)))
