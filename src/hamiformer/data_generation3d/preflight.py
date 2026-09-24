from __future__ import annotations
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
from .config import HamiBalls2Config
from .core import InitialScene, derived_uint64, sample_scene, simulate_scene

def _measure_restitution(cfg: HamiBalls2Config, substeps: int) -> dict[str, float]:
    import pybullet as pb
    dt = cfg.physics.frame_dt / substeps

    def configure(client: int) -> None:
        pb.resetSimulation(physicsClientId=client)
        pb.setGravity(0, 0, 0, physicsClientId=client)
        pb.setTimeStep(dt, physicsClientId=client)
        pb.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=cfg.physics.solver_iterations, deterministicOverlappingPairs=1, restitutionVelocityThreshold=cfg.physics.restitution_velocity_threshold, contactERP=cfg.physics.contact_erp, enableFileCaching=0, physicsClientId=client)

    def dynamics(body: int, restitution: float, client: int) -> None:
        pb.changeDynamics(body, -1, restitution=restitution, contactProcessingThreshold=cfg.physics.contact_processing_threshold, linearDamping=0, angularDamping=0, lateralFriction=0, spinningFriction=0, rollingFriction=0, physicsClientId=client)
    material = 0.94
    client = pb.connect(pb.DIRECT)
    try:
        configure(client)
        shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.08, physicsClientId=client)
        left = pb.createMultiBody(baseMass=1, baseCollisionShapeIndex=shape, basePosition=[-0.3, 0, 1], physicsClientId=client)
        right = pb.createMultiBody(baseMass=1, baseCollisionShapeIndex=shape, basePosition=[0.3, 0, 1], physicsClientId=client)
        dynamics(left, material, client)
        dynamics(right, material, client)
        pb.resetBaseVelocity(left, linearVelocity=[1, 0, 0], physicsClientId=client)
        pb.resetBaseVelocity(right, linearVelocity=[-1, 0, 0], physicsClientId=client)
        incoming = outgoing = None
        was_contact = False
        for _ in range(2000):
            va = np.asarray(pb.getBaseVelocity(left, physicsClientId=client)[0])
            vb = np.asarray(pb.getBaseVelocity(right, physicsClientId=client)[0])
            before = float(va[0] - vb[0])
            pb.stepSimulation(physicsClientId=client)
            contact = bool(pb.getContactPoints(left, right, physicsClientId=client))
            if contact and (not was_contact):
                incoming = before
            if was_contact and (not contact):
                va = np.asarray(pb.getBaseVelocity(left, physicsClientId=client)[0])
                vb = np.asarray(pb.getBaseVelocity(right, physicsClientId=client)[0])
                outgoing = float(va[0] - vb[0])
                break
            was_contact = contact
        ball_ball = abs(outgoing / incoming) if incoming and outgoing is not None else float('nan')
        configure(client)
        pb.setGravity(0, 0, -cfg.physics.gravity, physicsClientId=client)
        ground_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[1, 1, 0.02], physicsClientId=client)
        ground = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, basePosition=[0, 0, -0.02], physicsClientId=client)
        sphere_shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.08, physicsClientId=client)
        ball = pb.createMultiBody(baseMass=1, baseCollisionShapeIndex=sphere_shape, basePosition=[0, 0, 0.3], physicsClientId=client)
        dynamics(ground, cfg.physics.boundary_restitution, client)
        dynamics(ball, material, client)
        pb.resetBaseVelocity(ball, linearVelocity=[0, 0, -1], physicsClientId=client)
        incoming = None
        outgoing = 0.0
        for _ in range(2000):
            before = float(pb.getBaseVelocity(ball, physicsClientId=client)[0][2])
            pb.stepSimulation(physicsClientId=client)
            contacts = pb.getContactPoints(ball, ground, physicsClientId=client)
            if incoming is None and contacts and (max((cp[9] for cp in contacts)) > 1e-12):
                incoming = before
            if incoming is not None:
                outgoing = max(outgoing, float(pb.getBaseVelocity(ball, physicsClientId=client)[0][2]))
            if incoming is not None and outgoing > 0:
                break
        ball_boundary = abs(outgoing / incoming) if incoming and outgoing > 0 else float('nan')
        return {'ball_material': material, 'boundary_material': cfg.physics.boundary_restitution, 'ball_ball_expected_product': material ** 2, 'ball_ball_realized': ball_ball, 'ball_boundary_expected_product': material * cfg.physics.boundary_restitution, 'ball_boundary_realized': ball_boundary}
    finally:
        pb.disconnect(client)

def run_preflight(cfg: HamiBalls2Config, *, scenes_per_seed: int=8) -> dict[str, Any]:
    candidates = (4, 8, 16, 32)
    max_substeps = max(candidates) * 2
    scenes: list[InitialScene] = []
    for seed in cfg.physical_seeds:
        for index in range(scenes_per_seed):
            n = cfg.sampling.n_min + index % (cfg.sampling.n_max - cfg.sampling.n_min + 1)
            rng = np.random.default_rng(derived_uint64(seed, 'preflight', index, 'scene'))
            scenes.append(sample_scene(rng, n, cfg))
    full: dict[int, list] = {}
    for substeps in (*candidates, max_substeps):
        full[substeps] = [simulate_scene(scene, cfg, substeps=substeps, window_seed=0) for scene in scenes]
        print(f'preflight full rollouts complete: substeps={substeps}', flush=True)
    results: dict[str, Any] = {}
    short_cfg = replace(cfg, sampling=replace(cfg.sampling, episode_steps=1, window_steps=1))
    probe_frames = (24, 72, 120, 168)
    total_object_edges = sum((len(scene.mass) * cfg.sampling.episode_steps for scene in scenes))
    for substeps in candidates:
        long_q_norm: list[float] = []
        long_v_norm: list[float] = []
        local_q: dict[str, list[float]] = {'all': [], 'contact': [], 'continuous': []}
        local_v: dict[str, list[float]] = {'all': [], 'contact': [], 'continuous': []}
        penetration = [item.summary['max_penetration'] for item in full[substeps]]
        repeated_pairs = [item.summary['repeated_same_pair_within_frame'] for item in full[substeps]]
        repeated_objects = [item.summary['repeated_same_object_within_frame'] for item in full[substeps]]
        energy = [item.summary['smooth_energy_rel_p99'] for item in full[substeps] if item.summary['smooth_energy_rel_p99'] is not None]
        for scene, low, high, reference in zip(scenes, full[substeps], full[substeps * 2], full[max_substeps], strict=True):
            n = len(scene.mass)
            q_scale = scene.radius[None, :, None]
            initial_speed = np.linalg.norm(scene.velocity, axis=1)[None, :, None]
            speed_scale = initial_speed + cfg.physics.gravity * cfg.physics.frame_dt
            q_delta = low.phase[:, :n, :3] - high.phase[:, :n, :3]
            low_v = low.phase[:, :n, 3:] / scene.mass[None, :, None]
            high_v = high.phase[:, :n, 3:] / scene.mass[None, :, None]
            long_q_norm.append(float(np.sqrt(np.mean((q_delta / q_scale) ** 2))))
            long_v_norm.append(float(np.sqrt(np.mean(((low_v - high_v) / speed_scale) ** 2))))
            for frame in probe_frames:
                state = InitialScene(position=reference.phase[frame, :n, :3].astype(np.float64), velocity=(reference.phase[frame, :n, 3:] / scene.mass[:, None]).astype(np.float64), mass=scene.mass, radius=scene.radius, restitution=scene.restitution, spring_mask=scene.spring_mask, spring_k=scene.spring_k, spring_rest=scene.spring_rest)
                one_low = simulate_scene(state, short_cfg, substeps=substeps, window_seed=0)
                one_high = simulate_scene(state, short_cfg, substeps=substeps * 2, window_seed=0)
                q_err = np.sqrt(np.mean(((one_low.phase[1, :n, :3] - one_high.phase[1, :n, :3]) / scene.radius[:, None]) ** 2, axis=1))
                v_low = one_low.phase[1, :n, 3:] / scene.mass[:, None]
                v_high = one_high.phase[1, :n, 3:] / scene.mass[:, None]
                v_err = np.sqrt(np.mean(((v_low - v_high) / (np.linalg.norm(state.velocity, axis=1)[:, None] + cfg.physics.gravity * cfg.physics.frame_dt)) ** 2, axis=1))
                contact_mask = np.logical_or(one_low.qa['frame_contact'][0, :n], one_high.qa['frame_contact'][0, :n])
                local_q['all'].extend(q_err.tolist())
                local_v['all'].extend(v_err.tolist())
                local_q['contact'].extend(q_err[contact_mask].tolist())
                local_v['contact'].extend(v_err[contact_mask].tolist())
                local_q['continuous'].extend(q_err[~contact_mask].tolist())
                local_v['continuous'].extend(v_err[~contact_mask].tolist())

        def quantiles(values: list[float]) -> dict[str, float | None]:
            return {'median': float(np.median(values)) if values else None, 'p95': float(np.quantile(values, 0.95)) if values else None, 'max': float(np.max(values)) if values else None}
        results[str(substeps)] = {'comparand': substeps * 2, 'one_frame_q_over_radius_rmse': {key: quantiles(value) for key, value in local_q.items()}, 'one_frame_velocity_scaled_rmse': {key: quantiles(value) for key, value in local_v.items()}, 'one_frame_probe_count': {key: len(value) for key, value in local_q.items()}, 'long_rollout_q_over_radius_rmse_median': float(np.median(long_q_norm)), 'long_rollout_velocity_scaled_rmse_median': float(np.median(long_v_norm)), 'max_penetration': float(max(penetration)), 'repeated_same_pair_within_frame_total': int(sum(repeated_pairs)), 'repeated_same_object_within_frame_total': int(sum(repeated_objects)), 'repeated_same_object_per_object_edge': float(sum(repeated_objects) / total_object_edges), 'smooth_energy_rel_p99_median': float(np.median(energy)) if energy else None}
        print(json.dumps({substeps: results[str(substeps)]}, indent=2), flush=True)
    report = {'physics_only': True, 'scenes_per_seed': scenes_per_seed, 'realized_restitution_at_selected_candidate_32': _measure_restitution(cfg, 32), 'results': results}
    path = Path(cfg.output.root) / 'preflight' / 'physics_preflight.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    return report
