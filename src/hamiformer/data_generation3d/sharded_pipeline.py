from __future__ import annotations
import concurrent.futures
import json
import math
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
from .config import HamiBalls2Config
from .core import sample_scene, simulate_scene
from .pipeline import SPLITS, _atomic_json, _atomic_npz, _sha256, _source_fingerprint, _write_manifests, scene_plan
SHARD_SCHEMA = 'hamiformer.hamiballs2.full-episode-npz-shard.v1'

def _shard_descriptors(cfg: HamiBalls2Config) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for task in scene_plan(cfg):
        grouped.setdefault((task['split'], task['physical_seed']), []).append(task)
    descriptors = []
    for (split, seed), tasks in grouped.items():
        tasks.sort(key=lambda task: task['index'])
        for shard_index, start in enumerate(range(0, len(tasks), cfg.output.shard_size)):
            descriptors.append({'split': split, 'physical_seed': seed, 'shard_index': shard_index, 'tasks': tasks[start:start + cfg.output.shard_size]})
    return descriptors

def _shard_paths(root: Path, descriptor: dict[str, Any]) -> tuple[Path, Path]:
    base = root / 'shards' / descriptor['split'] / f"seed{descriptor['physical_seed']}" / f"shard_{descriptor['shard_index']:03d}"
    return (base.with_suffix('.npz'), base.with_suffix('.json'))

def _generate_shard(descriptor: dict[str, Any], cfg: HamiBalls2Config, source_hash: str) -> list[dict[str, Any]]:
    root = Path(cfg.output.root)
    shard_path, meta_path = _shard_paths(root, descriptor)
    if shard_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('schema') == SHARD_SCHEMA and meta.get('semantic_hash') == cfg.semantic_hash and (meta.get('source_fingerprint') == source_hash) and (meta.get('shard_sha256') == _sha256(shard_path)):
            return list(meta['records'])
        raise RuntimeError(f'existing shard provenance mismatch: {shard_path}')
    results = []
    records = []
    event_blocks = []
    event_offsets = [0]
    shard_rel = str(shard_path.relative_to(root)).replace('\\', '/')
    for row_index, task in enumerate(descriptor['tasks']):
        rng = np.random.default_rng(task['scene_seed'])
        scene = sample_scene(rng, task['n'], cfg)
        result = simulate_scene(scene, cfg, window_seed=task['window_seed'])
        results.append(result)
        events = result.qa['contact_events']
        event_blocks.append(events)
        event_offsets.append(event_offsets[-1] + len(events))
        records.append({**task, **result.summary, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'storage_format': SHARD_SCHEMA, 'shard_path': shard_rel, 'shard_row': row_index})
    count = len(results)
    arrays = {'phase': np.stack([result.phase for result in results]), 'attrs': np.stack([result.attrs for result in results]), 'object_mask': np.stack([result.object_mask for result in results]), 'spring_mask': np.stack([result.spring_mask for result in results]), 'spring_k': np.stack([result.spring_k for result in results]), 'spring_rest_length': np.stack([result.spring_rest for result in results]), 'time': np.stack([result.time for result in results]), 'window_start': np.asarray([result.window_start for result in results], dtype=np.int16), 'gravity': np.broadcast_to(np.asarray([0.0, 0.0, -cfg.physics.gravity], dtype=np.float32), (count, 3)).copy(), 'bounds': np.broadcast_to(np.asarray([-cfg.physics.box_half_extent_xy, cfg.physics.box_half_extent_xy, -cfg.physics.box_half_extent_xy, cfg.physics.box_half_extent_xy, 0.0, cfg.physics.box_height], dtype=np.float32), (count, 6)).copy(), 'qa_frame_contact': np.stack([result.qa['frame_contact'] for result in results]), 'qa_mechanical_energy': np.stack([result.qa['mechanical_energy'] for result in results]), 'qa_contact_events': np.concatenate(event_blocks, axis=0) if event_offsets[-1] else np.empty((0, 6), dtype=np.float64), 'qa_contact_event_offsets': np.asarray(event_offsets, dtype=np.int64)}
    _atomic_npz(shard_path, cfg.output.compressed, **arrays)
    shard_hash = _sha256(shard_path)
    for record in records:
        record['shard_sha256'] = shard_hash
    _atomic_json(meta_path, {'schema': SHARD_SCHEMA, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'shard_sha256': shard_hash, 'records': records})
    return records

def _worker(descriptor, cfg, source_hash):
    try:
        return (_generate_shard(descriptor, cfg, source_hash), None)
    except Exception:
        return (None, traceback.format_exc())

def generate_sharded_dataset(cfg: HamiBalls2Config, *, limit: int | None=None) -> dict[str, Any]:
    if limit is not None:
        raise ValueError('format-v2 shard generation does not support partial limit publication')
    root = Path(cfg.output.root)
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < int(cfg.output.reserve_free_gib * 1024 ** 3):
        raise RuntimeError('free disk is below the configured reserve')
    source_hash = _source_fingerprint()
    _atomic_json(root / 'metadata' / 'dataset_config.json', {'semantic': cfg.semantic_payload(), 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'storage_schema': SHARD_SCHEMA, 'status': 'in_progress'})
    descriptors = _shard_descriptors(cfg)
    remaining = descriptors
    records_by_id: dict[str, dict[str, Any]] = {}
    warning_count = 0
    pass_index = 0
    started = time.time()
    while remaining:
        pass_index += 1
        retry = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.output.workers) as pool:
            futures = {pool.submit(_worker, descriptor, cfg, source_hash): descriptor for descriptor in remaining}
            for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
                descriptor = futures[future]
                try:
                    records, error = future.result()
                except Exception:
                    records, error = (None, traceback.format_exc())
                if records is not None:
                    records_by_id.update(((record['sample_id'], record) for record in records))
                else:
                    warning_count += 1
                    retry.append(descriptor)
                    warning = {'time': time.time(), 'pass': pass_index, 'split': descriptor['split'], 'physical_seed': descriptor['physical_seed'], 'shard_index': descriptor['shard_index'], 'error': error or 'unknown'}
                    with (root / 'warnings.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(warning, ensure_ascii=False) + '\n')
                    print('WARNING shard failed; generation remains alive and will retry', flush=True)
                elapsed = time.time() - started
                completed = len(records_by_id)
                rate = completed / max(elapsed, 1e-09)
                eta = (len(scene_plan(cfg)) - completed) / max(rate, 1e-09)
                print(f'shard-progress {done}/{len(remaining)} scenes={completed}/32768 pass={pass_index} warnings={warning_count} rate={rate:.2f}/s eta={eta / 60:.1f}m', flush=True)
        remaining = retry
        if remaining:
            print(f'WARNING pass {pass_index} left {len(remaining)} invalid shards; retrying in 30s unless stopped by operator', flush=True)
            time.sleep(30)
    records = list(records_by_id.values())
    _write_manifests(cfg, records)
    summary = {'attempted': len(scene_plan(cfg)), 'completed': len(records), 'shards': len(descriptors), 'warnings': warning_count, 'passes': pass_index, 'wall_seconds': time.time() - started, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'storage_schema': SHARD_SCHEMA}
    _atomic_json(root / 'metadata' / 'generation_summary.json', summary)
    while True:
        try:
            report = audit_sharded_dataset(cfg)
            break
        except Exception:
            warning = {'time': time.time(), 'stage': 'full_audit', 'error': traceback.format_exc()}
            with (root / 'warnings.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(warning, ensure_ascii=False) + '\n')
            print('WARNING full audit failed; process remains alive for operator inspection and will retry in 30s', flush=True)
            time.sleep(30)
    _atomic_json(root / 'metadata' / 'dataset_config.json', {'semantic': cfg.semantic_payload(), 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': source_hash, 'storage_schema': SHARD_SCHEMA, 'status': 'finalized', 'audit': report})
    return summary

def audit_sharded_dataset(cfg: HamiBalls2Config) -> dict[str, Any]:
    root = Path(cfg.output.root)
    expected_source_hash = _source_fingerprint()
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    train_scene_mean_squares = []
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
    expected_time = (np.arange(cfg.sampling.episode_steps + 1) * cfg.physics.frame_dt).astype(np.float32)
    required = {'phase', 'attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length', 'time', 'window_start', 'gravity', 'bounds', 'qa_frame_contact', 'qa_mechanical_energy', 'qa_contact_events', 'qa_contact_event_offsets'}
    for descriptor in _shard_descriptors(cfg):
        shard_path, meta_path = _shard_paths(root, descriptor)
        label = f"{descriptor['split']}:{descriptor['physical_seed']}:{descriptor['shard_index']}"
        if not shard_path.is_file() or not meta_path.is_file():
            failures.append(f'missing-shard:{label}')
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('schema') != SHARD_SCHEMA:
                raise ValueError('shard schema')
            if meta.get('semantic_hash') != cfg.semantic_hash:
                raise ValueError('semantic hash')
            if meta.get('source_fingerprint') != expected_source_hash:
                raise ValueError('source fingerprint')
            shard_hash = _sha256(shard_path)
            if meta.get('shard_sha256') != shard_hash:
                raise ValueError('shard hash')
            local_records = list(meta['records'])
            tasks = descriptor['tasks']
            if len(local_records) != len(tasks):
                raise ValueError('shard record count')
            with np.load(shard_path, allow_pickle=False) as archive:
                if set(archive.files) != required:
                    raise ValueError('unexpected shard members')
                phase = archive['phase']
                attrs = archive['attrs']
                masks = archive['object_mask']
                spring_masks = archive['spring_mask']
                spring_ks = archive['spring_k']
                spring_rests = archive['spring_rest_length']
                times = archive['time']
                starts = archive['window_start']
                contacts = archive['qa_frame_contact']
                energies = archive['qa_mechanical_energy']
                events = archive['qa_contact_events']
                offsets = archive['qa_contact_event_offsets']
                count = len(tasks)
                if phase.shape != (count, cfg.sampling.episode_steps + 1, cfg.sampling.n_max, 6):
                    raise ValueError('phase shape')
                if contacts.shape != (count, cfg.sampling.episode_steps, cfg.sampling.n_max):
                    raise ValueError('contact shape')
                if energies.shape != (count, cfg.sampling.episode_steps + 1):
                    raise ValueError('energy shape')
                if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(events):
                    raise ValueError('event offsets')
                if not all((np.isfinite(archive[key]).all() for key in archive.files)):
                    raise ValueError('nonfinite shard')
                for row, (task, record) in enumerate(zip(tasks, local_records)):
                    if record['sample_id'] != task['sample_id'] or record['shard_row'] != row:
                        raise ValueError('record/task mismatch')
                    if record.get('shard_sha256') != shard_hash:
                        raise ValueError('record shard hash')
                    n = task['n']
                    mask = masks[row]
                    expected_mask = np.r_[np.ones(n, dtype=np.uint8), np.zeros(cfg.sampling.n_max - n, dtype=np.uint8)]
                    if not np.array_equal(mask, expected_mask):
                        raise ValueError('object mask')
                    spring_mask = spring_masks[row]
                    spring_k = spring_ks[row]
                    spring_rest = spring_rests[row]
                    if spring_mask.shape != (cfg.sampling.n_max, cfg.sampling.n_max) or not np.array_equal(spring_mask, spring_mask.T) or np.diag(spring_mask).any():
                        raise ValueError('spring mask')
                    if not np.array_equal(spring_k, spring_k.T) or not np.array_equal(spring_rest, spring_rest.T):
                        raise ValueError('spring symmetry')
                    if np.any(spring_k[spring_mask == 1] <= 0) or np.any(spring_rest[spring_mask == 1] <= 0):
                        raise ValueError('active spring parameter')
                    if np.any(spring_k[spring_mask == 0] != 0) or np.any(spring_rest[spring_mask == 0] != 0):
                        raise ValueError('inactive spring parameter')
                    if np.any(phase[row, :, mask == 0] != 0) or np.any(attrs[row, mask == 0] != 0):
                        raise ValueError('padding')
                    start = int(starts[row])
                    if not 0 <= start <= cfg.sampling.episode_steps - cfg.sampling.window_steps:
                        raise ValueError('window start')
                    if not np.array_equal(times[row], expected_time):
                        raise ValueError('time grid')
                    row_events = events[offsets[row]:offsets[row + 1]]
                    if row_events.ndim != 2 or row_events.shape[1] != 6:
                        raise ValueError('event shape')
                    if len(row_events) and np.any(row_events[:, 3] == -99):
                        raise ValueError('unknown boundary code')
                    if task['split'] == 'train':
                        window = phase[row, start:start + cfg.sampling.window_steps + 1, :n]
                        train_scene_mean_squares.append(np.mean(window.astype(np.float64) ** 2, axis=(0, 1)))
                    local_attrs = attrs[row, :n]
                    masses.extend(local_attrs[:, 0].astype(float).tolist())
                    radii.extend(local_attrs[:, 1].astype(float).tolist())
                    restitutions.extend(local_attrs[:, 2].astype(float).tolist())
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
                            neighbors = set(np.flatnonzero(spring_mask[node, :n]).tolist()) & remaining
                            remaining -= neighbors
                            frontier.extend(neighbors)
                    component_counts.append(components)
                    edge_count = int(np.triu(spring_mask[:n, :n], 1).sum())
                    cycle_surplus.append(edge_count - n + components)
                    for i, j in np.argwhere(np.triu(spring_mask[:n, :n], 1)):
                        reduced_mass = float(local_attrs[i, 0] * local_attrs[j, 0] / (local_attrs[i, 0] + local_attrs[j, 0]))
                        spring_periods.append(2 * math.pi * math.sqrt(reduced_mass / float(spring_k[i, j])))
                        initial_distance = float(np.linalg.norm(phase[row, 0, j, :3] - phase[row, 0, i, :3]))
                        spring_log_strains.append(math.log(float(spring_rest[i, j]) / initial_distance))
                    window_starts.append(start)
                    contact_object_edges += int(contacts[row, :, :n].sum())
                    total_object_edges += n * cfg.sampling.episode_steps
                    records.append(record)
                    if record['smooth_energy_rel_p99'] is not None:
                        smooth_energy_p99.append(float(record['smooth_energy_rel_p99']))
        except Exception as exc:
            failures.append(f'invalid-shard:{label}:{exc}')
    counts = {split: sum((r['split'] == split for r in records)) for split in SPLITS}
    expected_counts = {split: getattr(cfg.per_seed_counts, split) * len(cfg.physical_seeds) for split in SPLITS}
    if counts != expected_counts:
        failures.append(f'split counts mismatch: {counts} != {expected_counts}')
    by_seed_n = {f'{seed}:{n}': sum((r['split'] == 'train' and r['physical_seed'] == seed and (r['n'] == n) for r in records)) for seed in cfg.physical_seeds for n in range(cfg.sampling.n_min, cfg.sampling.n_max + 1)}
    expected_per_n = cfg.per_seed_counts.train // (cfg.sampling.n_max - cfg.sampling.n_min + 1)
    if any((value != expected_per_n for value in by_seed_n.values())):
        failures.append('train N balance mismatch')

    def distribution(values):
        return {'min': float(np.min(values)) if values else None, 'mean': float(np.mean(values)) if values else None, 'median': float(np.median(values)) if values else None, 'p95': float(np.quantile(values, 0.95)) if values else None, 'max': float(np.max(values)) if values else None}
    report = {'full_validation': not failures, 'semantic_hash': cfg.semantic_hash, 'source_fingerprint': expected_source_hash, 'storage_schema': SHARD_SCHEMA, 'samples_expected': len(scene_plan(cfg)), 'samples_valid': len(records), 'shards_expected': len(_shard_descriptors(cfg)), 'counts': counts, 'train_by_seed_n': by_seed_n, 'failures': failures[:100], 'max_penetration': max((r['max_penetration'] for r in records), default=None), 'repeated_same_pair_within_frame': sum((r['repeated_same_pair_within_frame'] for r in records)), 'repeated_same_object_within_frame': sum((r['repeated_same_object_within_frame'] for r in records)), 'contact_frame_fraction': sum((r['contact_frame_edges'] for r in records)) / (len(records) * cfg.sampling.episode_steps) if records else None, 'contact_object_edge_fraction': contact_object_edges / total_object_edges if total_object_edges else None, 'distributions': {'mass': distribution(masses), 'radius': distribution(radii), 'restitution': distribution(restitutions), 'spring_edges_per_scene': distribution(spring_edges), 'spring_degree': distribution(degrees), 'graph_components': distribution(component_counts), 'graph_cycle_surplus': distribution(cycle_surplus), 'spring_period': distribution(spring_periods), 'spring_log_rest_over_initial_distance': distribution(spring_log_strains), 'window_start': distribution(window_starts), 'smooth_energy_rel_p99': distribution(smooth_energy_p99)}}
    audit_path = root / 'audit' / 'full_audit.json'
    if failures:
        _atomic_json(audit_path, report)
        raise RuntimeError(f'dataset audit found {len(failures)} failures; see {audit_path}')
    _write_manifests(cfg, records)
    phase_rms = np.sqrt(np.mean(np.stack(train_scene_mean_squares), axis=0))
    stats = {'algorithm': 'equal-scene mean of valid-object/time mean squares on each frozen train crop', 'train_scenes': len(train_scene_mean_squares), 'phase_rms': phase_rms.tolist(), 'q_rms': phase_rms[:3].tolist(), 'p_rms': phase_rms[3:].tolist()}
    stats_path = root / 'stats' / 'phase_scales.json'
    _atomic_json(stats_path, stats)
    report['manifest_sha256'] = {split: _sha256(root / 'manifests' / f'{split}.jsonl') for split in SPLITS}
    report['stats_sha256'] = _sha256(stats_path)
    _atomic_json(audit_path, report)
    return report
__all__ = ['SHARD_SCHEMA', 'audit_sharded_dataset', 'generate_sharded_dataset']
