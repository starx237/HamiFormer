from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from .config import GeneratorConfig

def _allocated_4k(payload_bytes: int) -> int:
    return math.ceil(payload_bytes / 4096) * 4096

@dataclass(frozen=True)
class GenerationBudget:
    samples: int
    short_samples: int
    long_samples: int
    raw_array_bytes: int
    estimated_disk_bytes: int
    physics_substeps_lower_bound: int
    closure_substeps: int
    worker_ram_allowance_bytes: int
    coordinator_headroom_bytes: int
    advised_total_ram_allowance_bytes: int
    workers: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

def _sample_raw_bytes(frames: int, num_objects: int) -> int:
    phase = frames * num_objects * 4 * 4
    attrs = num_objects * 3 * 4
    time = frames * 4
    return phase + attrs + time

def estimate_generation_budget(config: GeneratorConfig) -> GenerationBudget:
    counts = asdict(config.splits)
    long_samples = counts.pop('long_test')
    short_samples = sum(counts.values())
    short_raw = _sample_raw_bytes(config.sampling.short_frames, config.sampling.num_objects)
    long_raw = _sample_raw_bytes(config.sampling.long_frames, config.sampling.num_objects)
    raw_total = short_samples * short_raw + long_samples * long_raw
    short_file = _allocated_4k(short_raw + 2048)
    long_file = _allocated_4k(long_raw + 2048)
    provenance_files = (short_samples + long_samples) * 4096
    manifests_and_metadata = (short_samples + long_samples) * 512 + 1024 * 1024
    estimated_disk = math.ceil(1.2 * (short_samples * short_file + long_samples * long_file + provenance_files + manifests_and_metadata))
    short_steps = short_samples * config.sampling.short_steps * config.physics.substeps
    long_steps = long_samples * (config.sampling.long_frames - 1) * config.physics.substeps
    seam_count = max(config.sampling.long_chunks - 1, 0)
    closure_length = config.sampling.closure_check_frames * config.physics.substeps
    closure_steps = short_samples * 3 * closure_length + long_samples * seam_count * 2 * closure_length
    worker_allowance = config.runtime.workers * 160 * 1024 ** 2
    coordinator_headroom = 512 * 1024 ** 2
    return GenerationBudget(samples=short_samples + long_samples, short_samples=short_samples, long_samples=long_samples, raw_array_bytes=raw_total, estimated_disk_bytes=estimated_disk, physics_substeps_lower_bound=short_steps + long_steps, closure_substeps=closure_steps, worker_ram_allowance_bytes=worker_allowance, coordinator_headroom_bytes=coordinator_headroom, advised_total_ram_allowance_bytes=worker_allowance + coordinator_headroom, workers=config.runtime.workers)
