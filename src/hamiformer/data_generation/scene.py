from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import GeneratorConfig

@dataclass(frozen=True)
class InitialScene:
    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    radius: np.ndarray
    restitution: np.ndarray

    @property
    def attrs(self) -> np.ndarray:
        return np.stack((self.mass, self.radius, self.restitution), axis=-1).astype(np.float32, copy=False)

def sample_initial_scene(rng: np.random.Generator, config: GeneratorConfig) -> InitialScene:
    sampling, physics = (config.sampling, config.physics)
    count = sampling.num_objects
    mass = rng.uniform(sampling.mass_min, sampling.mass_max, size=count)
    radius = rng.uniform(sampling.radius_min, sampling.radius_max, size=count)
    restitution = rng.uniform(sampling.restitution_min, sampling.restitution_max, size=count)
    positions: list[np.ndarray] = []
    for object_index in range(count):
        limit = physics.box_half_extent - physics.wall_radius - radius[object_index] - sampling.placement_margin
        if limit <= 0.0:
            raise RuntimeError('初态采样范围非正；配置的半径/墙/margin 与盒子不相容')
        placed = False
        for _ in range(sampling.max_placement_attempts):
            candidate = rng.uniform(-limit, limit, size=2)
            if all((np.linalg.norm(candidate - previous) > radius[object_index] + radius[previous_index] + sampling.placement_margin for previous_index, previous in enumerate(positions))):
                positions.append(candidate)
                placed = True
                break
        if not placed:
            raise RuntimeError('无法在 max_placement_attempts 内得到无重叠初态；请减小半径/物体数或扩大盒子')
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
    speed = rng.uniform(sampling.speed_min, sampling.speed_max, size=count)
    velocity = np.stack((np.cos(angle), np.sin(angle)), axis=-1) * speed[:, None]
    center_velocity = np.sum(mass[:, None] * velocity, axis=0) / np.sum(mass)
    velocity = velocity - center_velocity[None, :]
    mass32 = np.asarray(mass, dtype=np.float32)
    radius32 = np.asarray(radius, dtype=np.float32)
    restitution32 = np.asarray(restitution, dtype=np.float32)
    position32 = np.asarray(positions, dtype=np.float32)
    momentum32 = np.asarray(mass[:, None] * velocity, dtype=np.float32)
    velocity_closed = momentum32.astype(np.float64) / mass32.astype(np.float64)[:, None]
    return InitialScene(position=position32.astype(np.float64), velocity=velocity_closed, mass=mass32.astype(np.float64), radius=radius32.astype(np.float64), restitution=restitution32.astype(np.float64))

def scene_from_phase(phase: np.ndarray, attrs: np.ndarray) -> InitialScene:
    mass = np.asarray(attrs[:, 0], dtype=np.float64)
    return InitialScene(position=np.asarray(phase[:, :2], dtype=np.float64), velocity=np.asarray(phase[:, 2:], dtype=np.float64) / mass[:, None], mass=mass, radius=np.asarray(attrs[:, 1], dtype=np.float64), restitution=np.asarray(attrs[:, 2], dtype=np.float64))
