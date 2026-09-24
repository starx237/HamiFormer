from __future__ import annotations
import math
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints
import yaml
from hamiformer.utils import hash_jsonable
T = TypeVar('T')

def _strict_dataclass(cls: type[T], raw: Any, section: str) -> T:
    if not isinstance(raw, dict):
        raise ValueError(f'生成配置段 {section!r} 必须是映射')
    known = {item.name for item in fields(cls)}
    unknown = set(raw) - known
    missing = {item.name for item in fields(cls) if item.default is MISSING and item.default_factory is MISSING} - set(raw)
    if unknown or missing:
        raise ValueError(f'生成配置段 {section!r} 字段错误；未知={sorted(unknown)}，缺失={sorted(missing)}')
    for name, expected in get_type_hints(cls).items():
        value = raw[name]
        if expected in {bool, int, float, str} and type(value) is not expected:
            raise ValueError(f'生成配置段 {section!r} 的 {name} 必须为 {expected.__name__}，实际为 {type(value).__name__}')
    return cls(**raw)

@dataclass(frozen=True)
class GeneratorOutputConfig:
    root: str
    compressed: bool
    reserve_free_gib: float

@dataclass(frozen=True)
class GeneratorRuntimeConfig:
    workers: int
    required_pymunk_version: str

@dataclass(frozen=True)
class PhysicsConfig:
    collisions_enabled: bool
    frame_dt: float
    substeps: int
    box_half_extent: float
    wall_radius: float
    wall_restitution: float
    damping: float
    gravity_x: float
    gravity_y: float
    harmonic_k: float
    iterations: int
    collision_persistence: int
    collision_slop: float
    friction: float
    sleep_enabled: bool
    threaded: bool
    smooth_integrator: str | None = None

    @property
    def resolved_smooth_integrator(self) -> str:
        return self.smooth_integrator or 'pymunk_explicit_euler_reference'

    @property
    def physics_dt(self) -> float:
        return self.frame_dt / self.substeps

@dataclass(frozen=True)
class SamplingConfig:
    collision_mode: str
    num_objects: int
    short_steps: int
    long_chunks: int
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
    max_scene_attempts: int
    collision_enriched_train_fraction: float
    minimum_collision_events: int
    closure_check_frames: int
    seam_quiet_substeps: int
    closure_float64_tolerance: float
    closure_float32_tolerance: float
    all_split_steps: int | None = None
    handoff_validation: str | None = None

    @property
    def short_frames(self) -> int:
        return self.short_steps + 1

    @property
    def long_frames(self) -> int:
        return 1 + self.short_steps * self.long_chunks

    @property
    def resolved_handoff_validation(self) -> str:
        return self.handoff_validation or 'required'

    def frames_for_split(self, split: str) -> int:
        if self.all_split_steps is not None:
            return 1 + self.all_split_steps
        return self.long_frames if split == 'long_test' else self.short_frames

@dataclass(frozen=True)
class SplitCounts:
    train: int
    dev: int
    calibration_fit: int
    calibration_audit: int
    test: int
    long_test: int

@dataclass(frozen=True)
class GeneratorConfig:
    format_version: int
    seed: int
    output: GeneratorOutputConfig
    runtime: GeneratorRuntimeConfig
    physics: PhysicsConfig
    sampling: SamplingConfig
    splits: SplitCounts

    def validate(self) -> None:
        if self.format_version != 1:
            raise ValueError('生成配置 format_version 当前只能为 1')
        for section_name, section in (('output', self.output), ('physics', self.physics), ('sampling', self.sampling)):
            for item in fields(section):
                value = getattr(section, item.name)
                if type(value) is float and (not math.isfinite(value)):
                    raise ValueError(f'{section_name}.{item.name} 必须是有限浮点数')
        if not self.output.root.strip():
            raise ValueError('output.root 不能为空')
        if self.output.reserve_free_gib < 0.0:
            raise ValueError('reserve_free_gib 不能为负')
        if self.runtime.workers <= 0 or self.runtime.workers > 64:
            raise ValueError('workers 必须位于 [1,64]')
        if not self.runtime.required_pymunk_version.strip():
            raise ValueError('必须显式固定 required_pymunk_version')
        physics = self.physics
        if physics.frame_dt <= 0.0 or physics.substeps <= 0:
            raise ValueError('frame_dt/substeps 必须为正')
        if physics.box_half_extent <= 0.0 or physics.wall_radius < 0.0:
            raise ValueError('box_half_extent 必须为正，wall_radius 不能为负')
        if type(physics.collisions_enabled) is not bool:
            raise ValueError('collisions_enabled 必须是布尔值')
        if not 0.0 <= physics.wall_restitution <= 1.0:
            raise ValueError('wall_restitution 必须位于 [0,1]；1 只用于保守弹性数据')
        if physics.damping != 1.0:
            raise ValueError('v1.0 固定 damping=1.0；连续 drag 不与碰撞耗散混用')
        if physics.harmonic_k < 0.0 or physics.iterations <= 0:
            raise ValueError('harmonic_k 不能为负，solver iterations 必须为正')
        if physics.smooth_integrator not in {None, 'pre_kick_symplectic_euler'}:
            raise ValueError('smooth_integrator 必须省略（reference pymunk_explicit_euler_reference）或为 pre_kick_symplectic_euler')
        if physics.collision_persistence < 1 or physics.collision_slop < 0.0:
            raise ValueError('collision_persistence 必须至少为 1，collision_slop 不能为负')
        if physics.friction != 0.0 or physics.sleep_enabled or physics.threaded:
            raise ValueError('v1.0 必须 friction=0、sleep_enabled=false、threaded=false')
        sampling = self.sampling
        if sampling.num_objects <= 1 or sampling.short_steps < 2 or sampling.long_chunks <= 0:
            raise ValueError('至少需要两个物体、两个 future steps 和一个 long chunk')
        if sampling.all_split_steps is not None:
            if sampling.all_split_steps < sampling.short_steps:
                raise ValueError('all_split_steps 不能短于 short_steps')
            if sampling.all_split_steps % sampling.short_steps != 0:
                raise ValueError('all_split_steps 必须是 short_steps 的整数倍')
        if sampling.handoff_validation not in {None, 'required', 'audit_only'}:
            raise ValueError('handoff_validation 必须省略、required 或 audit_only')
        if not 0.0 < sampling.radius_min <= sampling.radius_max:
            raise ValueError('radius 范围非法')
        if sampling.radius_max + physics.wall_radius + sampling.placement_margin >= physics.box_half_extent:
            raise ValueError('最大球半径、墙半径与 placement margin 之和必须小于盒子半边长')
        if not 0.0 < sampling.mass_min <= sampling.mass_max:
            raise ValueError('mass 范围非法')
        if sampling.collision_mode not in {'mixed', 'zero', 'natural', 'at_least'}:
            raise ValueError('collision_mode 只能是 mixed、zero、natural 或 at_least')
        if not 0.0 <= sampling.restitution_min <= sampling.restitution_max <= 1.0:
            raise ValueError('球 restitution 范围必须位于 [0,1]')
        if not physics.collisions_enabled and sampling.collision_mode != 'zero':
            raise ValueError('关闭碰撞后 collision_mode 必须为 zero')
        if not 0.0 <= sampling.speed_min <= sampling.speed_max:
            raise ValueError('speed 范围非法')
        if 2.0 * sampling.speed_max * physics.physics_dt > sampling.radius_min / 4.0:
            raise ValueError('physics_dt 过粗：必须满足 2*speed_max*dt <= radius_min/4')
        if sampling.placement_margin < 0.0:
            raise ValueError('placement_margin 不能为负')
        if sampling.max_placement_attempts <= 0 or sampling.max_scene_attempts <= 0:
            raise ValueError('placement/scene attempts 必须为正')
        if not 0.0 <= sampling.collision_enriched_train_fraction <= 1.0:
            raise ValueError('collision_enriched_train_fraction 必须位于 [0,1]')
        if sampling.minimum_collision_events <= 0:
            raise ValueError('minimum_collision_events 必须为正')
        if not 1 <= sampling.closure_check_frames <= sampling.short_steps:
            raise ValueError('closure_check_frames 必须位于 [1,short_steps]')
        if sampling.seam_quiet_substeps < physics.collision_persistence + 1:
            raise ValueError('seam_quiet_substeps 必须覆盖 collision persistence 后再多一步')
        if sampling.closure_float64_tolerance <= 0.0 or sampling.closure_float32_tolerance <= 0.0:
            raise ValueError('两种 fresh-restart closure tolerance 必须为正')
        if sampling.closure_float64_tolerance > sampling.closure_float32_tolerance:
            raise ValueError('float64 closure tolerance 不应宽于落盘 float32 tolerance')
        counts = asdict(self.splits)
        if any((type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError('生成配置的六个 split 样本数都必须为非负整数')
        if counts['train'] <= 0:
            raise ValueError('train split 必须为正，以生成train-only phase scales')

    def semantic_payload(self) -> dict[str, Any]:
        raw = asdict(self)
        raw['output'] = {'compressed': self.output.compressed}
        raw['runtime'] = {'required_pymunk_version': self.runtime.required_pymunk_version}
        if raw['physics']['smooth_integrator'] is None:
            del raw['physics']['smooth_integrator']
        if raw['sampling']['all_split_steps'] is None:
            del raw['sampling']['all_split_steps']
        if raw['sampling']['handoff_validation'] is None:
            del raw['sampling']['handoff_validation']
        return raw

    @property
    def semantic_hash(self) -> str:
        return hash_jsonable(self.semantic_payload())

def load_generator_config(path: str | Path) -> GeneratorConfig:
    config_path = Path(path)
    with config_path.open('r', encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f'生成配置根节点必须是映射: {config_path}')
    expected = {'format_version', 'seed', 'output', 'runtime', 'physics', 'sampling', 'splits'}
    unknown, missing = (set(raw) - expected, expected - set(raw))
    if unknown or missing:
        raise ValueError(f'生成配置根字段错误；未知={sorted(unknown)}，缺失={sorted(missing)}')
    if type(raw['format_version']) is not int or type(raw['seed']) is not int:
        raise ValueError('format_version/seed 必须是整数')
    physics_raw = raw['physics']
    if isinstance(physics_raw, dict) and 'smooth_integrator' not in physics_raw:
        physics_raw = {**physics_raw, 'smooth_integrator': None}
    sampling_raw = raw['sampling']
    if isinstance(sampling_raw, dict):
        sampling_raw = {'all_split_steps': None, 'handoff_validation': None, **sampling_raw}
    config = GeneratorConfig(format_version=raw['format_version'], seed=raw['seed'], output=_strict_dataclass(GeneratorOutputConfig, raw['output'], 'output'), runtime=_strict_dataclass(GeneratorRuntimeConfig, raw['runtime'], 'runtime'), physics=_strict_dataclass(PhysicsConfig, physics_raw, 'physics'), sampling=_strict_dataclass(SamplingConfig, sampling_raw, 'sampling'), splits=_strict_dataclass(SplitCounts, raw['splits'], 'splits'))
    config.validate()
    return config
