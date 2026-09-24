from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .config import GeneratorConfig
from .plan import SceneTask, stable_uint64
from .scene import InitialScene, sample_initial_scene, scene_from_phase
BALL_COLLISION_TYPE = 1
WALL_COLLISION_TYPE = 2

class SceneRejected(RuntimeError):
    pass

def _import_pymunk(required_version: str) -> Any:
    try:
        import pymunk
    except ImportError as error:
        raise RuntimeError('数据生成需要可选依赖 Pymunk；请在目标环境安装 `pip install -e ".[data]"`') from error
    if str(pymunk.version) != required_version:
        raise RuntimeError(f'Pymunk 版本必须为 {required_version}，实际为 {pymunk.version}；拒绝在未审计版本上生成正式数据')
    return pymunk

def pymunk_runtime_info(required_version: str) -> dict[str, str]:
    pymunk = _import_pymunk(required_version)
    return {'pymunk_version': str(pymunk.version), 'chipmunk_version': str(pymunk.chipmunk_version)}

@dataclass
class CollisionTracker:
    substep_index: int = 0
    last_contact_substep: int = -10 ** 12
    ball_ball_events: int = 0
    ball_wall_events: int = 0
    total_impulse: float = 0.0
    _contact_this_substep: bool = False
    _max_penetration_this_substep: float = 0.0
    shape_labels: dict[int, str] | None = None
    events: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.shape_labels = {}
        self.events = []

    def begin_substep(self) -> None:
        self._contact_this_substep = False
        self._max_penetration_this_substep = 0.0

    def finish_substep(self) -> None:
        if self._contact_this_substep:
            self.last_contact_substep = self.substep_index
        self.substep_index += 1

    def post_solve(self, arbiter: Any, _space: Any, data: dict[str, str]) -> None:
        self._contact_this_substep = True
        for point in arbiter.contact_point_set.points:
            self._max_penetration_this_substep = max(self._max_penetration_this_substep, max(0.0, -float(point.distance)))
        if bool(arbiter.is_first_contact):
            if data['kind'] == 'ball_ball':
                self.ball_ball_events += 1
            else:
                self.ball_wall_events += 1
            assert self.shape_labels is not None and self.events is not None
            labels = sorted((self.shape_labels.get(id(shape), 'unknown') for shape in arbiter.shapes))
            self.events.append({'substep': self.substep_index, 'kind': data['kind'], 'pair': labels, 'first_impulse': [float(arbiter.total_impulse.x), float(arbiter.total_impulse.y)]})
        self.total_impulse += float(arbiter.total_impulse.length)

    @property
    def total_events(self) -> int:
        return self.ball_ball_events + self.ball_wall_events

@dataclass(frozen=True)
class TrackerSnapshot:
    ball_ball_events: int
    ball_wall_events: int
    total_impulse: float
    events: tuple[dict[str, object], ...]

def _tracker_snapshot(tracker: CollisionTracker) -> TrackerSnapshot:
    return TrackerSnapshot(ball_ball_events=tracker.ball_ball_events, ball_wall_events=tracker.ball_wall_events, total_impulse=tracker.total_impulse, events=tuple((dict(event) for event in tracker.events or ())))

@dataclass(frozen=True)
class SimulationResult:
    phase: np.ndarray
    attrs: np.ndarray
    time: np.ndarray
    ball_ball_events: int
    ball_wall_events: int
    total_impulse: float
    events: tuple[dict[str, object], ...]
    attempts: int
    closure_float64_max_error: float
    closure_float32_max_error: float
    seam_quiet: bool

def _build_space(initial: InitialScene, config: GeneratorConfig, tracker: CollisionTracker) -> tuple[Any, list[Any]]:
    pymunk = _import_pymunk(config.runtime.required_pymunk_version)
    physics = config.physics
    space = pymunk.Space(threaded=False)
    space.gravity = (physics.gravity_x, physics.gravity_y)
    space.damping = physics.damping
    space.iterations = physics.iterations
    space.collision_persistence = physics.collision_persistence
    space.collision_slop = physics.collision_slop
    space.sleep_time_threshold = float('inf')
    walls = []
    if physics.collisions_enabled:
        extent = physics.box_half_extent
        wall_endpoints = (((-extent, -extent), (extent, -extent)), ((extent, -extent), (extent, extent)), ((extent, extent), (-extent, extent)), ((-extent, extent), (-extent, -extent)))
        for start, end in wall_endpoints:
            wall = pymunk.Segment(space.static_body, start, end, physics.wall_radius)
            wall.elasticity = physics.wall_restitution
            wall.friction = physics.friction
            wall.collision_type = WALL_COLLISION_TYPE
            assert tracker.shape_labels is not None
            tracker.shape_labels[id(wall)] = f'wall:{len(walls)}'
            walls.append(wall)
        space.add(*walls)
    bodies: list[Any] = []
    for object_index in range(config.sampling.num_objects):
        body = pymunk.Body(float(initial.mass[object_index]), float('inf'))
        body.position = tuple(initial.position[object_index])
        body.velocity = tuple(initial.velocity[object_index])
        body.angle = 0.0
        body.angular_velocity = 0.0
        if physics.collisions_enabled:
            shape = pymunk.Circle(body, float(initial.radius[object_index]))
            shape.elasticity = float(initial.restitution[object_index])
            shape.friction = physics.friction
            shape.collision_type = BALL_COLLISION_TYPE
            assert tracker.shape_labels is not None
            tracker.shape_labels[id(shape)] = f'ball:{object_index}'
            space.add(body, shape)
        else:
            space.add(body)
        bodies.append(body)
    if physics.collisions_enabled:
        space.on_collision(BALL_COLLISION_TYPE, BALL_COLLISION_TYPE, post_solve=tracker.post_solve, data={'kind': 'ball_ball'})
        space.on_collision(BALL_COLLISION_TYPE, WALL_COLLISION_TYPE, post_solve=tracker.post_solve, data={'kind': 'ball_wall'})
    return (space, bodies)

def _snapshot(bodies: list[Any], mass: np.ndarray, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.empty((len(bodies), 4), dtype=dtype)
    for index, body in enumerate(bodies):
        result[index, 0] = float(body.position.x)
        result[index, 1] = float(body.position.y)
        result[index, 2] = float(mass[index] * body.velocity.x)
        result[index, 3] = float(mass[index] * body.velocity.y)
        if abs(float(body.angular_velocity)) > 1e-10:
            raise SceneRejected('检测到未保存的角速度；无自旋 phase 合同已被破坏')
    return result

def _prepare_harmonic_substep(bodies: list[Any], config: GeneratorConfig, *, dt: float, center: np.ndarray) -> None:
    physics = config.physics
    if physics.resolved_smooth_integrator == 'pre_kick_symplectic_euler':
        for body in bodies:
            if physics.harmonic_k > 0.0:
                displacement = np.asarray(body.position, dtype=np.float64) - center
                force = -physics.harmonic_k * displacement
                velocity = np.asarray(body.velocity, dtype=np.float64) + dt * force / float(body.mass)
                body.velocity = (float(velocity[0]), float(velocity[1]))
            body.force = (0.0, 0.0)
        return
    if physics.harmonic_k > 0.0:
        for body in bodies:
            displacement = np.asarray(body.position, dtype=np.float64) - center
            force = -physics.harmonic_k * displacement
            body.apply_force_at_world_point(tuple(force), body.position)

def _run_continuous(initial: InitialScene, frames: int, config: GeneratorConfig, *, diagnostic_frame: int | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, CollisionTracker, np.ndarray, TrackerSnapshot]:
    if frames <= 0:
        raise ValueError('frames 必须为正')
    if diagnostic_frame is None:
        diagnostic_frame = frames - 1
    if not 0 <= diagnostic_frame < frames:
        raise ValueError('diagnostic_frame 必须落在模拟帧范围内')
    tracker = CollisionTracker()
    space, bodies = _build_space(initial, config, tracker)
    phase64 = np.empty((frames, config.sampling.num_objects, 4), dtype=np.float64)
    quiet_age = np.empty(frames, dtype=np.int64)
    phase64[0] = _snapshot(bodies, initial.mass, dtype=np.dtype(np.float64))
    quiet_age[0] = 10 ** 12
    diagnostic = _tracker_snapshot(tracker) if diagnostic_frame == 0 else None
    dt = config.physics.physics_dt
    center = np.zeros(2, dtype=np.float64)
    position_limit = config.physics.box_half_extent - config.physics.wall_radius - initial.radius + max(4.0 * config.physics.collision_slop, 1e-05)
    displacement_limit = initial.radius / 4.0
    penetration_allowance = max(4.0 * config.physics.collision_slop, 1e-05)
    for frame_index in range(1, frames):
        for _ in range(config.physics.substeps):
            tracker.begin_substep()
            previous_position = np.asarray([[float(body.position.x), float(body.position.y)] for body in bodies], dtype=np.float64)
            _prepare_harmonic_substep(bodies, config, dt=dt, center=center)
            space.step(dt)
            current_position = np.asarray([[float(body.position.x), float(body.position.y)] for body in bodies], dtype=np.float64)
            if not np.isfinite(current_position).all():
                raise SceneRejected('子步位置产生 NaN/Inf')
            substep_displacement = np.linalg.norm(current_position - previous_position, axis=-1)
            if np.any(substep_displacement > displacement_limit):
                raise SceneRejected('单个 physics substep 位移过大，存在 tunneling 风险')
            if config.physics.collisions_enabled and np.any(np.abs(current_position) > position_limit[:, None]):
                raise SceneRejected('球体在 physics substep 越过墙体允许范围')
            if tracker._max_penetration_this_substep > penetration_allowance:
                raise SceneRejected('碰撞 callback 检测到过深子步穿透')
            tracker.finish_substep()
        phase64[frame_index] = _snapshot(bodies, initial.mass, dtype=np.dtype(np.float64))
        quiet_age[frame_index] = tracker.substep_index - tracker.last_contact_substep
        if frame_index == diagnostic_frame:
            diagnostic = _tracker_snapshot(tracker)
    time = (np.arange(frames, dtype=np.float64) * config.physics.frame_dt).astype(np.float32)
    if not np.isfinite(phase64).all() or not np.isfinite(time).all():
        raise SceneRejected('模拟产生 NaN/Inf')
    phase = phase64.astype(np.float32)
    if config.physics.collisions_enabled and np.any(np.abs(phase64[..., :2]) > position_limit[None, :, None]):
        raise SceneRejected('球体越过墙体允许范围；拒绝 teleport/clamp 修补')
    if config.physics.collisions_enabled:
        for left in range(config.sampling.num_objects):
            for right in range(left + 1, config.sampling.num_objects):
                distance = np.linalg.norm(phase64[:, left, :2] - phase64[:, right, :2], axis=-1)
                if np.any(distance < initial.radius[left] + initial.radius[right] - penetration_allowance):
                    raise SceneRejected('检测到过深球体穿透；请增加 substeps 或调整速度范围')
    if diagnostic is None:
        raise RuntimeError('内部错误：未捕获 diagnostic_frame 对应的 tracker snapshot')
    return (phase, phase64, time, tracker, quiet_age, diagnostic)

def _check_handoff_boundaries(phase: np.ndarray, phase64: np.ndarray, attrs: np.ndarray, quiet_age: np.ndarray, boundaries: tuple[int, ...], config: GeneratorConfig) -> tuple[bool, float, float]:
    max_error64 = 0.0
    max_error32 = 0.0
    quiet = True
    for boundary in boundaries:
        if quiet_age[boundary] < config.sampling.seam_quiet_substeps:
            quiet = False
        end = boundary + config.sampling.closure_check_frames + 1
        if end > phase.shape[0]:
            raise RuntimeError('内部错误：handoff closure 缺少 continuous reference tail')
        _, restarted64, _, _, _, _ = _run_continuous(scene_from_phase(phase64[boundary], attrs), config.sampling.closure_check_frames + 1, config)
        _, restarted_from32, _, _, _, _ = _run_continuous(scene_from_phase(phase[boundary], attrs), config.sampling.closure_check_frames + 1, config)
        error64 = float(np.max(np.abs(restarted64 - phase64[boundary:end])))
        error32 = float(np.max(np.abs(restarted_from32 - phase64[boundary:end])))
        max_error64 = max(max_error64, error64)
        max_error32 = max(max_error32, error32)
    return (quiet, max_error64, max_error32)

def simulate_task(task: SceneTask, config: GeneratorConfig) -> SimulationResult:
    for attempt in range(config.sampling.max_scene_attempts):
        attempt_seed = stable_uint64(task.scene_seed, attempt, 'attempt')
        rng = np.random.default_rng(attempt_seed)
        try:
            initial = sample_initial_scene(rng, config)
        except RuntimeError:
            continue
        if task.frames == config.sampling.short_frames:
            simulated_frames = task.frames + config.sampling.closure_check_frames
            handoff_boundaries = (config.sampling.short_steps,)
        elif task.frames == config.sampling.long_frames:
            simulated_frames = task.frames
            handoff_boundaries = tuple((chunk_index * config.sampling.short_steps for chunk_index in range(1, config.sampling.long_chunks)))
        else:
            raise ValueError(f'task frames 不符合 short/long 合同: {task.frames}')
        try:
            phase, phase64, time, _tracker, quiet_age, diagnostic = _run_continuous(initial, simulated_frames, config, diagnostic_frame=task.frames - 1)
        except SceneRejected:
            continue
        diagnostic_events = diagnostic.ball_ball_events + diagnostic.ball_wall_events
        if task.collision_requirement == 'at_least' and diagnostic_events < config.sampling.minimum_collision_events:
            continue
        if task.collision_requirement == 'zero' and diagnostic_events != 0:
            continue
        attrs = initial.attrs
        try:
            seam_quiet, closure_error64, closure_error32 = _check_handoff_boundaries(phase, phase64, attrs, quiet_age, handoff_boundaries, config)
        except SceneRejected:
            continue
        handoff_valid = seam_quiet and (closure_error64 <= config.sampling.closure_float64_tolerance and closure_error32 <= config.sampling.closure_float32_tolerance)
        if not handoff_valid and config.sampling.resolved_handoff_validation == 'required':
            continue
        return SimulationResult(phase=phase[:task.frames], attrs=attrs, time=time[:task.frames], ball_ball_events=diagnostic.ball_ball_events, ball_wall_events=diagnostic.ball_wall_events, total_impulse=diagnostic.total_impulse, events=diagnostic.events, attempts=attempt + 1, closure_float64_max_error=closure_error64, closure_float32_max_error=closure_error32, seam_quiet=seam_quiet)
    raise RuntimeError(f'scene {task.scene_id} 在 {config.sampling.max_scene_attempts} 次内未满足接受条件；请先运行小规模 micro-smoke 检查碰撞频率与 seam 条件')
