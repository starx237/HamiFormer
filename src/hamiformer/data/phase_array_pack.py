from __future__ import annotations
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from hamiformer.utils import hash_jsonable, sha256_file
from .schema import validate_phase_arrays
PHASE_ARRAY_PACK_SCHEMA = 'hamiformer.hamiballs.phase_array_pack.v1'
_SHA256_RE = re.compile('^[0-9a-f]{64}$')
_ROW_KEYS = {'sample_id', 'scene_id', 'npz_path', 'npz_sha256'}
_ARRAY_NAMES = ('phase', 'attrs', 'time')

@dataclass(frozen=True)
class SourceShard:
    root: Path
    split: str
    generator_seed: int
    sources: int

def _json_bytes(value: object, *, indent: int | None=2) -> bytes:
    suffix = '\n' if indent is not None else ''
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent) + suffix).encode('utf-8')

def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload)

def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f'{field} must be a canonical lowercase SHA-256')
    return value

def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'missing {label}: {path}')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f'{label} must be a JSON object: {path}')
    return raw

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f'missing source manifest: {path}')
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != _ROW_KEYS:
                raise ValueError(f'source manifest line {line_number} must contain exactly {sorted(_ROW_KEYS)}')
            rows.append(raw)
    if not rows:
        raise ValueError(f'source manifest is empty: {path}')
    return rows

def _source_metadata(shard: SourceShard) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    root = Path(shard.root).expanduser().resolve()
    metadata_path = root / 'metadata' / 'dataset_meta.json'
    metadata = _load_json_object(metadata_path, 'dataset metadata')
    if metadata.get('format_version') != 1:
        raise ValueError(f'dataset metadata format mismatch: {metadata_path}')
    if metadata.get('status') != 'finalized_unvalidated':
        raise ValueError(f'dataset root is not finalized: {root}')
    semantic = metadata.get('semantic_config')
    if not isinstance(semantic, dict):
        raise ValueError(f'dataset metadata lacks semantic_config: {metadata_path}')
    if semantic.get('seed') != shard.generator_seed:
        raise ValueError(f"generator seed mismatch for {root}: expected {shard.generator_seed}, got {semantic.get('seed')}")
    manifests = metadata.get('manifest_sha256')
    if not isinstance(manifests, dict) or shard.split not in manifests:
        raise ValueError(f'dataset metadata does not bind split={shard.split}: {root}')
    manifest_path = root / 'manifests' / f'{shard.split}.jsonl'
    recorded_manifest_sha = _require_sha256(manifests[shard.split], f'manifest_sha256.{shard.split}')
    if sha256_file(manifest_path) != recorded_manifest_sha:
        raise ValueError(f'source manifest drift: {manifest_path}')
    rows = _read_jsonl(manifest_path)
    if len(rows) != shard.sources:
        raise ValueError(f'registered shard must consume all rows: {manifest_path} has {len(rows)}, expected exactly {shard.sources}')
    split_counts = semantic.get('splits')
    if not isinstance(split_counts, dict) or split_counts.get(shard.split) != shard.sources:
        raise ValueError(f'semantic_config split count mismatch for {manifest_path}: {(None if not isinstance(split_counts, dict) else split_counts.get(shard.split))}')
    return (metadata, manifest_path, rows)

def _contract_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    semantic = metadata['semantic_config']
    return {'format_version': semantic.get('format_version'), 'physics': semantic.get('physics'), 'sampling': semantic.get('sampling')}

def _array_entry(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha256_file(path), 'shape': list(array.shape), 'dtype': str(array.dtype)}

def build_phase_array_pack(*, output: str | Path, pack_id: str, role: str, protocol_sha256: str, shards: Iterable[SourceShard], num_objects: int, future_steps: int, q_dim: int, attr_dim: int, source_future_steps: int | None=None, window_offsets: Iterable[int] | None=None) -> Path:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f'array-pack output already exists: {output_path}')
    if not pack_id or not role:
        raise ValueError('pack_id and role must be non-empty')
    protocol_sha256 = _require_sha256(protocol_sha256, 'protocol_sha256')
    dimensions = (int(num_objects), int(future_steps), int(q_dim), int(attr_dim))
    if min(dimensions) < 1:
        raise ValueError('array-pack dimensions must be positive')
    source_future_steps = int(future_steps) if source_future_steps is None else int(source_future_steps)
    offsets = (0,) if window_offsets is None else tuple((int(value) for value in window_offsets))
    if source_future_steps < future_steps:
        raise ValueError('source_future_steps cannot be shorter than future_steps')
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError('window_offsets must be non-empty and unique')
    if any((offset < 0 or offset + future_steps > source_future_steps for offset in offsets)):
        raise ValueError('window_offsets must fit inside the registered source horizon')
    shard_list = list(shards)
    if not shard_list:
        raise ValueError('array-pack requires at least one source shard')
    seeds = [int(shard.generator_seed) for shard in shard_list]
    if len(seeds) != len(set(seeds)):
        raise ValueError('generator seeds may not repeat within a composite pack')
    loaded: list[tuple[SourceShard, dict[str, Any], Path, list[dict[str, Any]]]] = []
    contract_hash: str | None = None
    implementation: tuple[int, str] | None = None
    for shard in shard_list:
        metadata, manifest_path, rows = _source_metadata(shard)
        current_contract = hash_jsonable(_contract_payload(metadata))
        if contract_hash is None:
            contract_hash = current_contract
        elif current_contract != contract_hash:
            raise ValueError('source shards do not share one physics/sampling contract')
        current_implementation = (int(metadata.get('implementation_version', -1)), _require_sha256(metadata.get('implementation_fingerprint'), 'implementation_fingerprint'))
        if implementation is None:
            implementation = current_implementation
        elif current_implementation != implementation:
            raise ValueError('source shards were generated by different implementations')
        loaded.append((shard, metadata, manifest_path, rows))
    source_trajectories = sum((len(item[3]) for item in loaded))
    samples = source_trajectories * len(offsets)
    phase_shape = (samples, future_steps + 1, num_objects, 2 * q_dim)
    attrs_shape = (samples, num_objects, attr_dim)
    time_shape = (samples, future_steps + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output_path.name}.building-', dir=output_path.parent) as temporary:
        temporary_path = Path(temporary)
        phase_path = temporary_path / 'phase.npy'
        attrs_path = temporary_path / 'attrs.npy'
        time_path = temporary_path / 'time.npy'
        phase_out = np.lib.format.open_memmap(phase_path, mode='w+', dtype=np.float32, shape=phase_shape)
        attrs_out = np.lib.format.open_memmap(attrs_path, mode='w+', dtype=np.float32, shape=attrs_shape)
        time_out = np.lib.format.open_memmap(time_path, mode='w+', dtype=np.float32, shape=time_shape)
        ledger_rows: list[dict[str, Any]] = []
        seen_sample_ids: set[str] = set()
        seen_scene_ids: set[str] = set()
        seen_npz_hashes: set[str] = set()
        row_index = 0
        for shard, metadata, source_manifest, rows in loaded:
            metadata_path = Path(shard.root).expanduser().resolve() / 'metadata' / 'dataset_meta.json'
            source_manifest_sha = sha256_file(source_manifest)
            source_metadata_sha = sha256_file(metadata_path)
            for source_index, row in enumerate(rows):
                source_sample_id = str(row['sample_id'])
                source_scene_id = str(row['scene_id'])
                sample_id = f'seed{int(shard.generator_seed)}:{source_sample_id}'
                scene_id = f'seed{int(shard.generator_seed)}:{source_scene_id}'
                npz_sha = _require_sha256(row['npz_sha256'], 'npz_sha256')
                if sample_id in seen_sample_ids:
                    raise ValueError(f'duplicate sample_id across source shards: {sample_id}')
                if scene_id in seen_scene_ids:
                    raise ValueError(f'duplicate scene_id across source shards: {scene_id}')
                if npz_sha in seen_npz_hashes:
                    raise ValueError(f'duplicate NPZ content across source shards: {npz_sha}')
                seen_sample_ids.add(sample_id)
                seen_scene_ids.add(scene_id)
                seen_npz_hashes.add(npz_sha)
                npz_path = Path(str(row['npz_path']))
                if not npz_path.is_absolute():
                    npz_path = (source_manifest.parent / npz_path).resolve()
                if sha256_file(npz_path) != npz_sha:
                    raise ValueError(f'NPZ content drift: {npz_path}')
                with np.load(npz_path, allow_pickle=False) as sample:
                    if set(sample.files) != set(_ARRAY_NAMES):
                        raise ValueError(f'sample must contain exactly phase/attrs/time: {npz_path}')
                    phase = np.asarray(sample['phase'])
                    attrs = np.asarray(sample['attrs'])
                    time = np.asarray(sample['time'])
                validate_phase_arrays(phase, attrs, time, num_objects=num_objects, future_steps=source_future_steps, q_dim=q_dim, attr_dim=attr_dim)
                for offset in offsets:
                    suffix = '' if source_future_steps == future_steps and offsets == (0,) else f'@e{offset:03d}'
                    phase_out[row_index] = phase[offset:offset + future_steps + 1]
                    attrs_out[row_index] = attrs
                    time_out[row_index] = time[offset:offset + future_steps + 1]
                    ledger_rows.append({'row': row_index, 'generator_seed': int(shard.generator_seed), 'split': shard.split, 'source_index': source_index, 'window_start_edge': offset, 'sample_id': sample_id + suffix, 'scene_id': scene_id, 'source_sample_id': source_sample_id, 'source_scene_id': source_scene_id, 'source_npz_sha256': npz_sha, 'source_manifest_sha256': source_manifest_sha, 'source_metadata_sha256': source_metadata_sha})
                    row_index += 1
        phase_out.flush()
        attrs_out.flush()
        time_out.flush()
        del phase_out, attrs_out, time_out
        ledger_path = temporary_path / 'rows.jsonl'
        ledger_payload = b''.join((_json_bytes(row, indent=None) + b'\n' for row in ledger_rows))
        _write_bytes(ledger_path, ledger_payload)
        arrays = {}
        for name, path in zip(_ARRAY_NAMES, (phase_path, attrs_path, time_path), strict=True):
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            arrays[name] = _array_entry(path, array)
            del array
        assert contract_hash is not None and implementation is not None
        manifest = {'schema': PHASE_ARRAY_PACK_SCHEMA, 'status': 'complete', 'pack_id': pack_id, 'role': role, 'protocol_sha256': protocol_sha256, 'samples': samples, 'source_trajectories': source_trajectories, 'dimensions': {'num_objects': num_objects, 'future_steps': future_steps, 'q_dim': q_dim, 'attr_dim': attr_dim}, 'physics_contract_sha256': contract_hash, 'implementation_version': implementation[0], 'implementation_fingerprint': implementation[1], 'composition': 'all_rows_seed_order_then_source_manifest_order_then_window_offset', 'window_view': {'source_future_steps': source_future_steps, 'future_steps': future_steps, 'offsets': list(offsets)}, 'arrays': arrays, 'row_ledger': {'file': ledger_path.name, 'bytes': ledger_path.stat().st_size, 'sha256': sha256_file(ledger_path), 'rows': len(ledger_rows)}, 'source_shards': [{'generator_seed': int(shard.generator_seed), 'split': shard.split, 'sources': len(rows), 'semantic_hash': _require_sha256(metadata.get('semantic_hash'), 'semantic_hash'), 'metadata_sha256': sha256_file(Path(shard.root).expanduser().resolve() / 'metadata' / 'dataset_meta.json'), 'manifest_sha256': sha256_file(manifest_path)} for shard, metadata, manifest_path, rows in loaded]}
        manifest_path = temporary_path / 'manifest.json'
        _write_bytes(manifest_path, _json_bytes(manifest))
        phase = np.load(phase_path, mmap_mode='r', allow_pickle=False)
        q_sumsq = np.zeros(q_dim, dtype=np.float64)
        p_sumsq = np.zeros(q_dim, dtype=np.float64)
        count = 0
        for start in range(0, samples, 1024):
            block = np.asarray(phase[start:start + 1024], dtype=np.float64)
            q_sumsq += np.sum(np.square(block[..., :q_dim]), axis=(0, 1, 2))
            p_sumsq += np.sum(np.square(block[..., q_dim:]), axis=(0, 1, 2))
            count += block.shape[0] * block.shape[1] * block.shape[2]
        q_scale = np.maximum(np.sqrt(q_sumsq / count), 1e-06)
        p_scale = np.maximum(np.sqrt(p_sumsq / count), 1e-06)
        stats = {'format_version': 1, 'train_manifest_sha256': sha256_file(manifest_path), 'q_scale': [float(value) for value in q_scale], 'p_scale': [float(value) for value in p_scale]}
        _write_bytes(temporary_path / 'phase_scales.json', _json_bytes(stats))
        del block, phase
        os.replace(temporary_path, output_path)
    return output_path / 'manifest.json'

class PhaseArrayPack:

    def __init__(self, manifest_path: str | Path, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, verify_content_hash: bool=False) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.root = self.manifest_path.parent
        self.manifest = _load_json_object(self.manifest_path, 'array-pack manifest')
        if self.manifest.get('schema') != PHASE_ARRAY_PACK_SCHEMA:
            raise ValueError(f'array-pack schema mismatch: {self.manifest_path}')
        if self.manifest.get('status') != 'complete':
            raise ValueError(f'array-pack is not complete: {self.manifest_path}')
        expected_dimensions = {'num_objects': int(num_objects), 'future_steps': int(future_steps), 'q_dim': int(q_dim), 'attr_dim': int(attr_dim)}
        if self.manifest.get('dimensions') != expected_dimensions:
            raise ValueError(f"array-pack dimension mismatch: expected {expected_dimensions}, got {self.manifest.get('dimensions')}")
        self.samples = int(self.manifest.get('samples', 0))
        if self.samples < 1:
            raise ValueError('array-pack samples must be positive')
        arrays = self.manifest.get('arrays')
        if not isinstance(arrays, dict) or set(arrays) != set(_ARRAY_NAMES):
            raise ValueError('array-pack must bind exactly phase/attrs/time arrays')
        loaded: dict[str, np.ndarray] = {}
        for name in _ARRAY_NAMES:
            item = arrays[name]
            if not isinstance(item, dict):
                raise ValueError(f'invalid array entry: {name}')
            path = self.root / str(item.get('file'))
            if not path.is_file() or path.stat().st_size != int(item.get('bytes', -1)):
                raise ValueError(f'array file missing or size changed: {path}')
            if verify_content_hash and sha256_file(path) != _require_sha256(item.get('sha256'), f'arrays.{name}.sha256'):
                raise ValueError(f'array content hash mismatch: {path}')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != item.get('shape') or str(array.dtype) != item.get('dtype'):
                raise ValueError(f'array shape/dtype mismatch: {path}')
            loaded[name] = array
        expected_shapes = {'phase': (self.samples, future_steps + 1, num_objects, 2 * q_dim), 'attrs': (self.samples, num_objects, attr_dim), 'time': (self.samples, future_steps + 1)}
        for name, expected in expected_shapes.items():
            if loaded[name].shape != expected or loaded[name].dtype != np.float32:
                raise ValueError(f'array-pack {name} must be float32 {expected}')
        ledger = self.manifest.get('row_ledger')
        if not isinstance(ledger, dict) or int(ledger.get('rows', -1)) != self.samples:
            raise ValueError('array-pack row ledger metadata mismatch')
        ledger_path = self.root / str(ledger.get('file'))
        if not ledger_path.is_file() or ledger_path.stat().st_size != int(ledger.get('bytes', -1)):
            raise ValueError('array-pack row ledger missing or size changed')
        if sha256_file(ledger_path) != _require_sha256(ledger.get('sha256'), 'row_ledger.sha256'):
            raise ValueError('array-pack row ledger content hash mismatch')
        self.records = _read_pack_ledger(ledger_path, self.samples)
        self.phase = loaded['phase']
        self.attrs = loaded['attrs']
        self.time = loaded['time']

def _read_pack_ledger(path: Path, samples: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for expected_row, line in enumerate(handle):
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get('row') != expected_row:
                raise ValueError(f'array-pack row ledger order mismatch at row {expected_row}')
            if not isinstance(raw.get('sample_id'), str) or not isinstance(raw.get('scene_id'), str):
                raise ValueError(f'array-pack row identity invalid at row {expected_row}')
            records.append(raw)
    if len(records) != samples:
        raise ValueError(f'array-pack ledger has {len(records)} rows, expected {samples}')
    return records
__all__ = ['PHASE_ARRAY_PACK_SCHEMA', 'PhaseArrayPack', 'SourceShard', 'build_phase_array_pack']
