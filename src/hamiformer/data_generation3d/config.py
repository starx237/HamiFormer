from __future__ import annotations
import hashlib
import json
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class OutputConfig:
    root: str
    compressed: bool
    workers: int
    reserve_free_gib: float
    storage_format: str = 'sample_npz_v1'
    shard_size: int = 1

@dataclass(frozen=True)
class PhysicsConfig:
    engine: str
    required_pybullet_version: str
    frame_dt: float
    substeps: int
    solver_iterations: int
    restitution_velocity_threshold: float
    contact_processing_threshold: float
    contact_erp: float
    gravity: float
    box_half_extent_xy: float
    box_height: float
    wall_thickness: float
    boundary_restitution: float
    friction: float

@dataclass(frozen=True)
class SamplingConfig:
    episode_steps: int
    window_steps: int
    n_min: int
    n_max: int
    radius_min: float
    radius_max: float
    mass_min: float
    mass_max: float
    restitution_min: float
    restitution_max: float
    speed_min: float
    speed_max: float
    placement_margin: float
    max_placement_attempts: int
    spring_expected_degree: float
    spring_period_min: float
    spring_period_max: float
    spring_log_strain_abs: float

@dataclass(frozen=True)
class PerSeedCounts:
    train: int
    calibration: int
    validation: int
    expansion: int

@dataclass(frozen=True)
class HamiBalls2Config:
    format_version: int
    dataset_name: str
    physical_seeds: tuple[int, ...]
    output: OutputConfig
    physics: PhysicsConfig
    sampling: SamplingConfig
    per_seed_counts: PerSeedCounts

    def validate(self) -> None:
        p, s, c = (self.physics, self.sampling, self.per_seed_counts)
        if self.format_version not in (1, 2) or not self.dataset_name.strip():
            raise ValueError('format_version must be 1 or 2 and dataset_name must be nonempty')
        if self.physical_seeds != (40, 41, 42, 43):
            raise ValueError('HamiBalls-2 v1 freezes physical_seeds to [40,41,42,43]')
        if p.engine != 'pybullet' or not p.required_pybullet_version:
            raise ValueError('engine must be pybullet with a fixed version')
        if p.frame_dt <= 0 or p.substeps <= 0 or p.solver_iterations <= 0:
            raise ValueError('time step, substeps, and solver iterations must be positive')
        if p.gravity <= 0 or p.box_half_extent_xy <= 0 or p.box_height <= 0:
            raise ValueError('gravity and box dimensions must be positive')
        if p.wall_thickness <= 0 or not 0 <= p.boundary_restitution <= 1:
            raise ValueError('invalid boundary parameters')
        if p.contact_processing_threshold != 0 or p.contact_erp != 0:
            raise ValueError('v1 uses impulse-only contacts: processing threshold=0 and ERP=0')
        if p.friction != 0:
            raise ValueError('v1 has no rotation state, so friction must be zero')
        if s.episode_steps < s.window_steps or s.window_steps != 48:
            raise ValueError('episode must contain the frozen 48-edge window')
        if (s.n_min, s.n_max) != (5, 10):
            raise ValueError('v1 freezes N to 5..10')
        for lo, hi, name in ((s.radius_min, s.radius_max, 'radius'), (s.mass_min, s.mass_max, 'mass'), (s.restitution_min, s.restitution_max, 'restitution'), (s.speed_min, s.speed_max, 'speed'), (s.spring_period_min, s.spring_period_max, 'spring_period')):
            if lo < 0 or hi < lo or (name not in {'restitution', 'speed'} and lo <= 0):
                raise ValueError(f'invalid {name} range')
        if s.restitution_max > 1 or s.max_placement_attempts <= 0:
            raise ValueError('invalid restitution or placement attempts')
        if s.spring_expected_degree <= 0 or s.spring_log_strain_abs < 0:
            raise ValueError('invalid spring graph parameters')
        if self.output.workers <= 0 or self.output.reserve_free_gib < 0:
            raise ValueError('invalid runtime output parameters')
        if self.output.storage_format not in {'sample_npz_v1', 'npz_shard_v1'}:
            raise ValueError('unsupported output storage format')
        if self.output.shard_size <= 0:
            raise ValueError('shard_size must be positive')
        if self.format_version == 1 and (self.output.storage_format != 'sample_npz_v1' or self.output.shard_size != 1):
            raise ValueError('format v1 uses per-sample NPZ storage')
        if self.format_version == 2 and self.output.storage_format != 'npz_shard_v1':
            raise ValueError('format v2 uses sharded NPZ storage')
        if c.train != 7680 or c.calibration != 128 or c.validation != 128 or (c.expansion != 256):
            raise ValueError('v1 freezes per-seed counts to 7680/128/128/256')
        if c.train % (s.n_max - s.n_min + 1):
            raise ValueError('train count must balance N exactly')

    def semantic_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['physical_seeds'] = list(self.physical_seeds)
        payload['output'] = {'compressed': self.output.compressed}
        if self.format_version >= 2:
            payload['output'].update({'storage_format': self.output.storage_format, 'shard_size': self.output.shard_size})
        return payload

    @property
    def semantic_hash(self) -> str:
        raw = json.dumps(self.semantic_payload(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

def _section(raw: dict[str, Any], key: str, cls: type) -> Any:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{key} must be a mapping')
    fields = cls.__dataclass_fields__
    expected = set(fields)
    required = {name for name, field in fields.items() if field.default is MISSING and field.default_factory is MISSING}
    if not required.issubset(value) or not set(value).issubset(expected):
        raise ValueError(f'{key} fields differ: missing={sorted(required - set(value))}, extra={sorted(set(value) - expected)}')
    return cls(**value)

def load_config(path: str | Path) -> HamiBalls2Config:
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('config root must be a mapping')
    expected = {'format_version', 'dataset_name', 'physical_seeds', 'output', 'physics', 'sampling', 'per_seed_counts'}
    if set(raw) != expected:
        raise ValueError(f'config root fields differ: missing={sorted(expected - set(raw))}, extra={sorted(set(raw) - expected)}')
    cfg = HamiBalls2Config(format_version=raw['format_version'], dataset_name=raw['dataset_name'], physical_seeds=tuple(raw['physical_seeds']), output=_section(raw, 'output', OutputConfig), physics=_section(raw, 'physics', PhysicsConfig), sampling=_section(raw, 'sampling', SamplingConfig), per_seed_counts=_section(raw, 'per_seed_counts', PerSeedCounts))
    cfg.validate()
    return cfg
