from __future__ import annotations
import hashlib
import importlib.metadata
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from .config import HamiBalls2Config
BOUNDARY_CODE = {'ground': -1, 'top': -2, 'x_low': -3, 'x_high': -4, 'y_low': -5, 'y_high': -6}

def derived_uint64(master_seed: int, *parts: object) -> int:
    message = 'hamiballs2-v1|' + str(master_seed) + '|' + '|'.join(map(str, parts))
    return int.from_bytes(hashlib.sha256(message.encode('utf-8')).digest()[:8], 'big')

@dataclass(frozen=True)
class InitialScene:
    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    radius: np.ndarray
    restitution: np.ndarray
    spring_mask: np.ndarray
    spring_k: np.ndarray
    spring_rest: np.ndarray

@dataclass(frozen=True)
class SimulationResult:
    phase: np.ndarray
    attrs: np.ndarray
    object_mask: np.ndarray
    spring_mask: np.ndarray
    spring_k: np.ndarray
    spring_rest: np.ndarray
    time: np.ndarray
    window_start: int
    qa: dict[str, np.ndarray]
    summary: dict[str, Any]

def sample_scene(rng: np.random.Generator, n: int, cfg: HamiBalls2Config) -> InitialScene:
    s, p = (cfg.sampling, cfg.physics)
    mass = rng.uniform(s.mass_min, s.mass_max, n)
    radius = rng.uniform(s.radius_min, s.radius_max, n)
    restitution = rng.uniform(s.restitution_min, s.restitution_max, n)
    pos: list[np.ndarray] = []
    for i in range(n):
        low = np.array([-p.box_half_extent_xy, -p.box_half_extent_xy, 0.0]) + radius[i] + s.placement_margin
        high = np.array([p.box_half_extent_xy, p.box_half_extent_xy, p.box_height]) - radius[i] - s.placement_margin
        for _ in range(s.max_placement_attempts):
            candidate = rng.uniform(low, high)
            if all((np.linalg.norm(candidate - pos[j]) > radius[i] + radius[j] + s.placement_margin for j in range(i))):
                pos.append(candidate)
                break
        else:
            raise RuntimeError('failed to sample a non-overlapping 3D initial state')
    position = np.asarray(pos, dtype=np.float64)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    velocity = direction * rng.uniform(s.speed_min, s.speed_max, n)[:, None]
    edge_p = s.spring_expected_degree / (n - 1)
    while True:
        upper = rng.random((n, n)) < edge_p
        upper = np.triu(upper, 1)
        if upper.any():
            break
    mask = upper | upper.T
    k = np.zeros((n, n), dtype=np.float64)
    rest = np.zeros((n, n), dtype=np.float64)
    for i, j in np.argwhere(upper):
        log_t = rng.uniform(math.log(s.spring_period_min), math.log(s.spring_period_max))
        period = math.exp(log_t)
        reduced_mass = mass[i] * mass[j] / (mass[i] + mass[j])
        kij = reduced_mass * (2 * math.pi / period) ** 2
        dij = np.linalg.norm(position[j] - position[i])
        lij = dij * math.exp(rng.uniform(-s.spring_log_strain_abs, s.spring_log_strain_abs))
        k[i, j] = k[j, i] = kij
        rest[i, j] = rest[j, i] = lij
    return InitialScene(position, velocity, mass, radius, restitution, mask, k, rest)

def _energy(position: np.ndarray, velocity: np.ndarray, scene: InitialScene, gravity: float) -> float:
    kinetic = 0.5 * np.sum(scene.mass[:, None] * velocity ** 2)
    gravitational = np.sum(scene.mass * gravity * position[:, 2])
    spring = 0.0
    for i, j in np.argwhere(np.triu(scene.spring_mask, 1)):
        extension = np.linalg.norm(position[j] - position[i]) - scene.spring_rest[i, j]
        spring += 0.5 * scene.spring_k[i, j] * extension ** 2
    return float(kinetic + gravitational + spring)

def _spring_forces(position: np.ndarray, scene: InitialScene) -> np.ndarray:
    forces = np.zeros_like(position)
    for i, j in np.argwhere(np.triu(scene.spring_mask, 1)):
        delta = position[j] - position[i]
        distance = np.linalg.norm(delta)
        if distance <= 1e-12:
            continue
        force = scene.spring_k[i, j] * (distance - scene.spring_rest[i, j]) * delta / distance
        forces[i] += force
        forces[j] -= force
    return forces

def simulate_scene(scene: InitialScene, cfg: HamiBalls2Config, *, substeps: int | None=None, window_seed: int=0) -> SimulationResult:
    try:
        import pybullet as pb
    except ImportError as exc:
        raise RuntimeError('pybullet is required only for HamiBalls-2 generation') from exc
    actual_version = importlib.metadata.version('pybullet')
    if actual_version != cfg.physics.required_pybullet_version:
        raise RuntimeError(f'pybullet version {actual_version} != required {cfg.physics.required_pybullet_version}')
    p, s = (cfg.physics, cfg.sampling)
    steps_per_frame = substeps or p.substeps
    dt = p.frame_dt / steps_per_frame
    client = pb.connect(pb.DIRECT)
    try:
        pb.resetSimulation(physicsClientId=client)
        pb.setGravity(0, 0, -p.gravity, physicsClientId=client)
        pb.setTimeStep(dt, physicsClientId=client)
        pb.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=p.solver_iterations, deterministicOverlappingPairs=1, restitutionVelocityThreshold=p.restitution_velocity_threshold, contactERP=p.contact_erp, enableFileCaching=0, physicsClientId=client)
        h, z, t = (p.box_half_extent_xy, p.box_height, p.wall_thickness)
        wall_specs = {'ground': ([h + t, h + t, t], [0, 0, -t]), 'top': ([h + t, h + t, t], [0, 0, z + t]), 'x_low': ([t, h + t, z / 2], [-h - t, 0, z / 2]), 'x_high': ([t, h + t, z / 2], [h + t, 0, z / 2]), 'y_low': ([h + t, t, z / 2], [0, -h - t, z / 2]), 'y_high': ([h + t, t, z / 2], [0, h + t, z / 2])}
        boundary_by_body: dict[int, int] = {}
        for name, (half_extents, center) in wall_specs.items():
            shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half_extents, physicsClientId=client)
            body = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, basePosition=center, physicsClientId=client)
            pb.changeDynamics(body, -1, restitution=p.boundary_restitution, contactProcessingThreshold=p.contact_processing_threshold, lateralFriction=0, spinningFriction=0, rollingFriction=0, physicsClientId=client)
            boundary_by_body[body] = BOUNDARY_CODE[name]
        body_to_object: dict[int, int] = {}
        body_ids: list[int] = []
        for i in range(len(scene.mass)):
            shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=float(scene.radius[i]), physicsClientId=client)
            body = pb.createMultiBody(baseMass=float(scene.mass[i]), baseCollisionShapeIndex=shape, basePosition=scene.position[i], physicsClientId=client)
            pb.resetBaseVelocity(body, linearVelocity=scene.velocity[i], angularVelocity=[0, 0, 0], physicsClientId=client)
            pb.changeDynamics(body, -1, restitution=float(scene.restitution[i]), contactProcessingThreshold=p.contact_processing_threshold, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, physicsClientId=client)
            body_to_object[body] = i
            body_ids.append(body)
        n, max_n = (len(scene.mass), cfg.sampling.n_max)
        phase = np.zeros((s.episode_steps + 1, max_n, 6), dtype=np.float32)
        energies = np.zeros(s.episode_steps + 1, dtype=np.float64)
        frame_contact = np.zeros((s.episode_steps, max_n), dtype=np.uint8)
        event_rows: list[tuple[int, int, int, int, float, float]] = []
        repeated_pair = 0
        repeated_object = 0
        min_distance = 0.0

        def read_state() -> tuple[np.ndarray, np.ndarray]:
            q = np.asarray([pb.getBasePositionAndOrientation(b, physicsClientId=client)[0] for b in body_ids])
            v = np.asarray([pb.getBaseVelocity(b, physicsClientId=client)[0] for b in body_ids])
            return (q, v)
        q, v = read_state()
        phase[0, :n, :3] = q
        phase[0, :n, 3:] = scene.mass[:, None] * v
        energies[0] = _energy(q, v, scene, p.gravity)
        for frame in range(s.episode_steps):
            pair_left_after_seen: set[tuple[int, int]] = set()
            object_left_after_seen: set[int] = set()
            previous_active_pairs: set[tuple[int, int]] = set()
            previous_active_objects: set[int] = set()
            for sub in range(steps_per_frame):
                q_before, _ = read_state()
                forces = _spring_forces(q_before, scene)
                for i, body in enumerate(body_ids):
                    pb.applyExternalForce(body, -1, forces[i], q_before[i], pb.WORLD_FRAME, physicsClientId=client)
                pb.stepSimulation(physicsClientId=client)
                active_pairs: set[tuple[int, int]] = set()
                active_objects: set[int] = set()
                for cp in pb.getContactPoints(physicsClientId=client):
                    body_a, body_b = (int(cp[1]), int(cp[2]))
                    normal_force, distance = (float(cp[9]), float(cp[8]))
                    if body_a in body_to_object:
                        a = body_to_object[body_a]
                        b = body_to_object[body_b] if body_b in body_to_object else boundary_by_body.get(body_b, -99)
                    elif body_b in body_to_object:
                        a = body_to_object[body_b]
                        b = boundary_by_body.get(body_a, -99)
                    else:
                        continue
                    key = (min(a, b), max(a, b)) if b >= 0 else (a, b)
                    if distance <= 0.0001:
                        active_pairs.add(key)
                        active_objects.add(a)
                        if b >= 0:
                            active_objects.add(b)
                    if normal_force > 1e-12:
                        frame_contact[frame, a] = 1
                        if b >= 0:
                            frame_contact[frame, b] = 1
                        impulse = normal_force * dt
                        event_rows.append((frame, sub, a, b, impulse, distance))
                    min_distance = min(min_distance, distance)
                pair_left_after_seen.update(previous_active_pairs - active_pairs)
                object_left_after_seen.update(previous_active_objects - active_objects)
                repeated_pair += sum((key in pair_left_after_seen for key in active_pairs - previous_active_pairs))
                repeated_object += sum((i in object_left_after_seen for i in active_objects - previous_active_objects))
                previous_active_pairs = active_pairs
                previous_active_objects = active_objects
            q, v = read_state()
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise FloatingPointError(f'non-finite state at frame {frame + 1}')
            phase[frame + 1, :n, :3] = q
            phase[frame + 1, :n, 3:] = scene.mass[:, None] * v
            energies[frame + 1] = _energy(q, v, scene, p.gravity)
        attrs = np.zeros((max_n, 3), dtype=np.float32)
        attrs[:n] = np.stack((scene.mass, scene.radius, scene.restitution), axis=-1)
        object_mask = np.zeros(max_n, dtype=np.uint8)
        object_mask[:n] = 1
        spring_mask = np.zeros((max_n, max_n), dtype=np.uint8)
        spring_k = np.zeros((max_n, max_n), dtype=np.float32)
        spring_rest = np.zeros((max_n, max_n), dtype=np.float32)
        spring_mask[:n, :n] = scene.spring_mask
        spring_k[:n, :n] = scene.spring_k
        spring_rest[:n, :n] = scene.spring_rest
        rows = np.asarray(event_rows, dtype=np.float64).reshape(-1, 6)
        event_count = int(len(rows))
        contact_edges = int(np.any(frame_contact, axis=1).sum())
        smooth = ~np.any(frame_contact, axis=1)
        delta_e = np.diff(energies)
        scale = np.maximum(np.abs(energies[:-1]), 1e-08)
        smooth_rel = np.abs(delta_e[smooth]) / scale[smooth]
        window_rng = np.random.default_rng(window_seed)
        window_start = int(window_rng.integers(0, s.episode_steps - s.window_steps + 1))
        qa = {'contact_events': rows, 'frame_contact': frame_contact, 'mechanical_energy': energies}
        summary = {'n': n, 'spring_edges': int(np.triu(scene.spring_mask, 1).sum()), 'contact_substep_points': event_count, 'contact_frame_edges': contact_edges, 'repeated_same_pair_within_frame': int(repeated_pair), 'repeated_same_object_within_frame': int(repeated_object), 'max_penetration': float(-min_distance), 'smooth_energy_rel_median': float(np.median(smooth_rel)) if len(smooth_rel) else None, 'smooth_energy_rel_p99': float(np.quantile(smooth_rel, 0.99)) if len(smooth_rel) else None, 'window_start': window_start}
        return SimulationResult(phase=phase, attrs=attrs, object_mask=object_mask, spring_mask=spring_mask, spring_k=spring_k, spring_rest=spring_rest, time=(np.arange(s.episode_steps + 1) * p.frame_dt).astype(np.float32), window_start=window_start, qa=qa, summary=summary)
    finally:
        pb.disconnect(client)
