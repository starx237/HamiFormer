from __future__ import annotations
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
from .config import HamiBalls2Config
from .core import derived_uint64, sample_scene, simulate_scene
SPLITS = ('train', 'calibration', 'validation', 'expansion')

def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(b'\x00')
        digest.update(path.read_bytes())
        digest.update(b'\x00')
    return digest.hexdigest()

def scene_plan(cfg: HamiBalls2Config) -> list[dict[str, Any]]:
    counts = asdict(cfg.per_seed_counts)
    n_values = np.arange(cfg.sampling.n_min, cfg.sampling.n_max + 1)
    tasks: list[dict[str, Any]] = []
    for seed in cfg.physical_seeds:
        for split in SPLITS:
            count = counts[split]
            ns = np.resize(n_values, count)
            rng = np.random.default_rng(derived_uint64(seed, split, 'n-order'))
            rng.shuffle(ns)
            for index, n in enumerate(ns.tolist()):
                tasks.append({'physical_seed': seed, 'split': split, 'index': index, 'n': int(n), 'scene_seed': derived_uint64(seed, split, index, 'scene'), 'window_seed': derived_uint64(seed, split, index, 'window'), 'sample_id': f's{seed}_{split}_{index:05d}'})
    return tasks

def _paths(root: Path, task: dict[str, Any]) -> tuple[Path, Path, Path]:
    base = root / 'samples' / task['split'] / f"seed{task['physical_seed']}" / task['sample_id']
    return (base.with_suffix('.npz'), base.with_suffix('.qa.npz'), base.with_suffix('.json'))

def _atomic_npz(path: Path, compressed: bool, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with temporary.open('wb') as handle:
        (np.savez_compressed if compressed else np.savez)(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _generate_one(task: dict[str, Any], cfg: HamiBalls2Config, source_hash: str) -> dict[str, Any]:
    root = Path(cfg.output.root)
    data_path, qa_path, meta_path = _paths(root, task)
    if data_path.is_file() and qa_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('semantic_hash') == cfg.semantic_hash and meta.get('source_fingerprint') == source_hash:
            return meta
        raise RuntimeError(f"existing artifact provenance mismatch: {task['sample_id']}")
    rng = np.random.default_rng(task['scene_seed'])
    scene = sample_scene(rng, task['n'], cfg)
    result = simulate_scene(scene, cfg, window_seed=task['window_seed'])
    _atomic_npz(data_path, cfg.output.compressed, phase=result.phase, attrs=result.attrs, object_mask=result.object_mask, spring_mask=result.spring_mask, spring_k=result.spring_k, spring_rest_length=result.spring_rest, time=result.time, window_start=np.asarray(result.window_start, dtype=np.int16), gravity=np.asarray([0.0, 0.0, -cfg.physics.gravity], dtype=np.float32), bounds=np.asarray([-cfg.physics.box_half_extent_xy, cfg.physics.box_half_extent_xy, -cfg.physics.box_half_extent_xy, cfg.physics.box_half_extent_xy, 0.0, cfg.physics.box_height], dtype=np.float32))
    _atomic_npz(qa_path, True, **result.qa)
    meta = {**task, **result.summary, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'data_path': str(data_path.relative_to(root)).replace('\\', '/'), 'qa_path': str(qa_path.relative_to(root)).replace('\\', '/'), 'data_sha256': _sha256(data_path), 'qa_sha256': _sha256(qa_path)}
    _atomic_json(meta_path, meta)
    return meta

def _worker(task: dict[str, Any], cfg: HamiBalls2Config, source_hash: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return (_generate_one(task, cfg, source_hash), None)
    except Exception:
        return (None, traceback.format_exc())

def _write_manifests(cfg: HamiBalls2Config, records: list[dict[str, Any]]) -> None:
    root = Path(cfg.output.root)
    manifest_dir = root / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        chosen = sorted((r for r in records if r['split'] == split), key=lambda r: (r['physical_seed'], r['index']))
        path = manifest_dir / f'{split}.jsonl'
        temporary = path.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            for record in chosen:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        os.replace(temporary, path)

def audit_dataset(cfg: HamiBalls2Config) -> dict[str, Any]:
    if cfg.output.storage_format == 'npz_shard_v1':
        from .sharded_pipeline import audit_sharded_dataset
        return audit_sharded_dataset(cfg)
    root = Path(cfg.output.root)
    tasks = scene_plan(cfg)
    expected_source_hash = _source_fingerprint()
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    train_scene_mean_squares: list[np.ndarray] = []
    masses: list[float] = []
    radii: list[float] = []
    restitutions: list[float] = []
    spring_edges: list[int] = []
    degrees: list[int] = []
    component_counts: list[int] = []
    cycle_surplus: list[int] = []
    spring_periods: list[float] = []
    spring_log_strains: list[float] = []
    window_starts: list[int] = []
    smooth_energy_p99: list[float] = []
    contact_object_edges = 0
    total_object_edges = 0
    for task in tasks:
        data_path, qa_path, meta_path = _paths(root, task)
        if not (data_path.is_file() and qa_path.is_file() and meta_path.is_file()):
            failures.append(f"missing:{task['sample_id']}")
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            with np.load(data_path, allow_pickle=False) as archive:
                required = {'phase', 'attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length', 'time', 'window_start', 'gravity', 'bounds'}
                if set(archive.files) != required:
                    raise ValueError('unexpected model-visible members')
                phase = archive['phase']
                attrs = archive['attrs']
                mask = archive['object_mask']
                spring_mask = archive['spring_mask']
                spring_k = archive['spring_k']
                spring_rest = archive['spring_rest_length']
                time_array = archive['time']
                window_start = int(archive['window_start'])
                if phase.shape != (cfg.sampling.episode_steps + 1, cfg.sampling.n_max, 6):
                    raise ValueError('phase shape')
                n = task['n']
                if not np.isfinite(phase).all() or int(mask.sum()) != n:
                    raise ValueError('finite/mask')
                if not np.array_equal(mask, np.r_[np.ones(n, dtype=np.uint8), np.zeros(cfg.sampling.n_max - n, dtype=np.uint8)]):
                    raise ValueError('object mask must be a prefix')
                if not np.array_equal(spring_mask, spring_mask.T) or np.diag(spring_mask).any():
                    raise ValueError('spring symmetry')
                if spring_mask.shape != (cfg.sampling.n_max, cfg.sampling.n_max):
                    raise ValueError('spring shape')
                if not np.array_equal(spring_k, spring_k.T) or not np.array_equal(spring_rest, spring_rest.T):
                    raise ValueError('spring parameter symmetry')
                if np.any(spring_k[spring_mask == 1] <= 0) or np.any(spring_rest[spring_mask == 1] <= 0):
                    raise ValueError('nonpositive active spring parameter')
                if np.any(spring_k[spring_mask == 0] != 0) or np.any(spring_rest[spring_mask == 0] != 0):
                    raise ValueError('inactive spring parameter is nonzero')
                if np.any(phase[:, mask == 0] != 0) or np.any(attrs[mask == 0] != 0):
                    raise ValueError('padding is nonzero')
                if not 0 <= window_start <= cfg.sampling.episode_steps - cfg.sampling.window_steps:
                    raise ValueError('window start')
                expected_time = (np.arange(cfg.sampling.episode_steps + 1) * cfg.physics.frame_dt).astype(np.float32)
                if not np.array_equal(time_array, expected_time):
                    raise ValueError('time grid')
                if task['split'] == 'train':
                    window = phase[window_start:window_start + cfg.sampling.window_steps + 1, :n]
                    train_scene_mean_squares.append(np.mean(window.astype(np.float64) ** 2, axis=(0, 1)))
                masses.extend(attrs[:n, 0].astype(float).tolist())
                radii.extend(attrs[:n, 1].astype(float).tolist())
                restitutions.extend(attrs[:n, 2].astype(float).tolist())
                spring_edges.append(int(np.triu(spring_mask, 1).sum()))
                local_degree = spring_mask[:n, :n].sum(axis=1).astype(int)
                degrees.extend(local_degree.tolist())
                remaining = set(range(n))
                components = 0
                while remaining:
                    components += 1
                    frontier = [remaining.pop()]
                    while frontier:
                        node = frontier.pop()
                        neighbors = set(np.flatnonzero(spring_mask[node, :n]).astype(int).tolist()) & remaining
                        remaining -= neighbors
                        frontier.extend(neighbors)
                component_counts.append(components)
                edges = int(np.triu(spring_mask[:n, :n], 1).sum())
                cycle_surplus.append(edges - n + components)
                for i, j in np.argwhere(np.triu(spring_mask[:n, :n], 1)):
                    reduced_mass = float(attrs[i, 0] * attrs[j, 0] / (attrs[i, 0] + attrs[j, 0]))
                    spring_periods.append(2 * math.pi * math.sqrt(reduced_mass / float(spring_k[i, j])))
                    initial_distance = float(np.linalg.norm(phase[0, j, :3] - phase[0, i, :3]))
                    spring_log_strains.append(math.log(float(spring_rest[i, j]) / initial_distance))
                window_starts.append(window_start)
            with np.load(qa_path, allow_pickle=False) as qa:
                if set(qa.files) != {'contact_events', 'frame_contact', 'mechanical_energy'}:
                    raise ValueError('unexpected QA members')
                if qa['frame_contact'].shape != (cfg.sampling.episode_steps, cfg.sampling.n_max):
                    raise ValueError('QA contact shape')
                if qa['mechanical_energy'].shape != (cfg.sampling.episode_steps + 1,):
                    raise ValueError('QA energy shape')
                if not all((np.isfinite(qa[key]).all() for key in qa.files)):
                    raise ValueError('nonfinite QA')
                contact_object_edges += int(qa['frame_contact'][:, :task['n']].sum())
                total_object_edges += task['n'] * cfg.sampling.episode_steps
                events = qa['contact_events']
                if events.ndim != 2 or events.shape[1] != 6:
                    raise ValueError('QA event shape')
                if len(events) and np.any(events[:, 3] == -99):
                    raise ValueError('unknown boundary code')
            if meta['data_sha256'] != _sha256(data_path) or meta['qa_sha256'] != _sha256(qa_path):
                raise ValueError('hash mismatch')
            if meta['semantic_hash'] != cfg.semantic_hash or meta['source_fingerprint'] != expected_source_hash:
                raise ValueError('provenance mismatch')
            records.append(meta)
            if meta['smooth_energy_rel_p99'] is not None:
                smooth_energy_p99.append(float(meta['smooth_energy_rel_p99']))
        except Exception as exc:
            failures.append(f"invalid:{task['sample_id']}:{exc}")
    counts = {split: sum((r['split'] == split for r in records)) for split in SPLITS}
    by_seed_n = {f'{seed}:{n}': sum((r['split'] == 'train' and r['physical_seed'] == seed and (r['n'] == n) for r in records)) for seed in cfg.physical_seeds for n in range(cfg.sampling.n_min, cfg.sampling.n_max + 1)}
    expected_per_n = cfg.per_seed_counts.train // (cfg.sampling.n_max - cfg.sampling.n_min + 1)
    if any((value != expected_per_n for value in by_seed_n.values())):
        failures.append('train N balance mismatch')
    expected_counts = {split: getattr(cfg.per_seed_counts, split) * len(cfg.physical_seeds) for split in SPLITS}
    if counts != expected_counts:
        failures.append(f'split counts mismatch: {counts} != {expected_counts}')

    def distribution(values: list[float]) -> dict[str, float | None]:
        return {'min': float(np.min(values)) if values else None, 'mean': float(np.mean(values)) if values else None, 'median': float(np.median(values)) if values else None, 'p95': float(np.quantile(values, 0.95)) if values else None, 'max': float(np.max(values)) if values else None}
    report = {'full_validation': not failures, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': _source_fingerprint(), 'samples_expected': len(tasks), 'samples_valid': len(records), 'counts': counts, 'train_by_seed_n': by_seed_n, 'failures': failures[:100], 'max_penetration': max((r['max_penetration'] for r in records), default=None), 'repeated_same_pair_within_frame': sum((r['repeated_same_pair_within_frame'] for r in records)), 'repeated_same_object_within_frame': sum((r['repeated_same_object_within_frame'] for r in records)), 'contact_frame_fraction': sum((r['contact_frame_edges'] for r in records)) / (len(records) * cfg.sampling.episode_steps) if records else None, 'contact_object_edge_fraction': contact_object_edges / total_object_edges if total_object_edges else None, 'distributions': {'mass': distribution(masses), 'radius': distribution(radii), 'restitution': distribution(restitutions), 'spring_edges_per_scene': distribution(spring_edges), 'spring_degree': distribution(degrees), 'graph_components': distribution(component_counts), 'graph_cycle_surplus': distribution(cycle_surplus), 'spring_period': distribution(spring_periods), 'spring_log_rest_over_initial_distance': distribution(spring_log_strains), 'window_start': distribution(window_starts), 'smooth_energy_rel_p99': distribution(smooth_energy_p99)}}
    if failures:
        _atomic_json(root / 'audit' / 'full_audit.json', report)
        raise RuntimeError(f'dataset audit found {len(failures)} failures; see audit/full_audit.json')
    _write_manifests(cfg, records)
    phase_rms = np.sqrt(np.mean(np.stack(train_scene_mean_squares), axis=0))
    stats = {'algorithm': 'equal-scene mean of valid-object/time mean squares on each frozen train crop', 'train_scenes': len(train_scene_mean_squares), 'phase_rms': phase_rms.tolist(), 'q_rms': phase_rms[:3].tolist(), 'p_rms': phase_rms[3:].tolist()}
    stats_path = root / 'stats' / 'phase_scales.json'
    _atomic_json(stats_path, stats)
    report['manifest_sha256'] = {split: _sha256(root / 'manifests' / f'{split}.jsonl') for split in SPLITS}
    report['stats_sha256'] = _sha256(stats_path)
    _atomic_json(root / 'audit' / 'full_audit.json', report)
    return report

def generate_dataset(cfg: HamiBalls2Config, *, limit: int | None=None) -> dict[str, Any]:
    if cfg.output.storage_format == 'npz_shard_v1':
        from .sharded_pipeline import generate_sharded_dataset
        return generate_sharded_dataset(cfg, limit=limit)
    root = Path(cfg.output.root)
    required = int(cfg.output.reserve_free_gib * 1024 ** 3)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < required:
        raise RuntimeError('free disk is below the configured reserve')
    source_hash = _source_fingerprint()
    _atomic_json(root / 'metadata' / 'dataset_config.json', {'semantic': cfg.semantic_payload(), 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'status': 'in_progress'})
    tasks = scene_plan(cfg)
    if limit is not None:
        tasks = tasks[:limit]
    records_by_id: dict[str, dict[str, Any]] = {}
    warning_count = 0
    started = time.time()
    remaining = tasks
    pass_index = 0
    while remaining:
        pass_index += 1
        retry: list[dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.output.workers) as pool:
            futures = {pool.submit(_worker, task, cfg, source_hash): task for task in remaining}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                task = futures[future]
                try:
                    record, error = future.result()
                except Exception:
                    record, error = (None, traceback.format_exc())
                if record is not None:
                    records_by_id[record['sample_id']] = record
                else:
                    warning_count += 1
                    retry.append(task)
                    warning = {'time': time.time(), 'pass': pass_index, 'sample_id': task['sample_id'], 'error': error or 'unknown'}
                    warning_path = root / 'warnings.jsonl'
                    with warning_path.open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(warning, ensure_ascii=False) + '\n')
                    print(f"WARNING {task['sample_id']} failed; generation remains alive and will retry", flush=True)
                if done == 1 or done % 64 == 0 or done == len(remaining):
                    elapsed = time.time() - started
                    completed = len(records_by_id)
                    rate = completed / max(elapsed, 1e-09)
                    eta = (len(tasks) - completed) / max(rate, 1e-09)
                    print(f'progress {completed}/{len(tasks)} pass={pass_index} warnings={warning_count} rate={rate:.2f}/s eta={eta / 60:.1f}m', flush=True)
        remaining = retry
        if remaining:
            print(f'WARNING pass {pass_index} left {len(remaining)} invalid scenes; retrying in 30s unless stopped by operator', flush=True)
            time.sleep(30)
    records = list(records_by_id.values())
    _write_manifests(cfg, records)
    summary = {'attempted': len(tasks), 'completed': len(records), 'warnings': warning_count, 'passes': pass_index, 'wall_seconds': time.time() - started, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash}
    _atomic_json(root / 'metadata' / 'generation_summary.json', summary)
    if limit is None:
        while True:
            try:
                report = audit_dataset(cfg)
                break
            except Exception:
                warning = {'time': time.time(), 'stage': 'full_audit', 'error': traceback.format_exc()}
                with (root / 'warnings.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(warning, ensure_ascii=False) + '\n')
                print('WARNING full audit failed; process remains alive for operator inspection and will retry in 30s', flush=True)
                time.sleep(30)
        _atomic_json(root / 'metadata' / 'dataset_config.json', {'semantic': cfg.semantic_payload(), 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'status': 'finalized', 'audit': report})
    return summary
