from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from hamiformer.data import SourceShard, build_phase_array_pack
from hamiformer.utils import hash_jsonable, sha256_file
PHYSICAL_SEEDS = (40, 41, 42, 43)
ROLE_SPLITS = {'train': ('train', 3584), 'calibration': ('calibration_fit', 128), 'validation': ('dev', 128), 'reserve': ('test', 256)}
EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()

def _load_object(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f'expected JSON object: {path}')
    return raw

def _shards(source_root: Path, role: str) -> list[SourceShard]:
    split, sources = ROLE_SPLITS[role]
    return [SourceShard(source_root / f'source_seed{seed}', split, seed, sources) for seed in PHYSICAL_SEEDS]

def _build(*, source_root: Path, pack_root: Path, staging_root: Path, protocol_sha256: str, name: str, role: str, future_steps: int, source_future_steps: int=192, offsets: tuple[int, ...]=(0,)) -> Path:
    destination = pack_root / name
    existing_manifest = destination / 'manifest.json'
    if destination.exists():
        raw = _load_object(existing_manifest)
        if raw.get('status') != 'complete' or raw.get('pack_id') != f'hamiballs-canonical-v2-{name}' or raw.get('protocol_sha256') != protocol_sha256:
            raise ValueError(f'existing pack does not match publication: {destination}')
        return existing_manifest
    staged = staging_root / name
    if staged.exists():
        raise FileExistsError(f'stale local pack staging path exists: {staged}')
    staged_manifest = build_phase_array_pack(output=staged, pack_id=f'hamiballs-canonical-v2-{name}', role=role, protocol_sha256=protocol_sha256, shards=_shards(source_root, role), num_objects=5, future_steps=future_steps, source_future_steps=source_future_steps, window_offsets=offsets, q_dim=2, attr_dim=3)
    pack_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged_manifest.parent), str(destination))
    return destination / 'manifest.json'

def _publish_train_adapter(*, output: Path, train_manifest: Path, source_root: Path) -> Path:
    if output.exists():
        raise FileExistsError(f'training adapter already exists: {output}')
    manifests = output / 'manifests'
    metadata_dir = output / 'metadata'
    stats_dir = output / 'stats'
    manifests.mkdir(parents=True)
    metadata_dir.mkdir()
    stats_dir.mkdir()
    (manifests / 'train.jsonl').symlink_to(train_manifest.resolve())
    manifest_hashes = {'train': sha256_file(train_manifest)}
    for split in ('dev', 'calibration_fit', 'calibration_audit', 'test', 'long_test'):
        path = manifests / f'{split}.jsonl'
        path.write_bytes(b'')
        manifest_hashes[split] = EMPTY_SHA256
    train_stats = train_manifest.parent / 'phase_scales.json'
    shutil.copyfile(train_stats, stats_dir / 'phase_scales.json')
    source_metadata = _load_object(source_root / 'source_seed40' / 'metadata' / 'dataset_meta.json')
    semantic = copy.deepcopy(source_metadata['semantic_config'])
    if not isinstance(semantic, dict):
        raise ValueError('source semantic_config is not an object')
    semantic.pop('seed', None)
    semantic['physical_seeds'] = list(PHYSICAL_SEEDS)
    semantic['splits'] = {'train': 14336, 'dev': 512, 'calibration_fit': 512, 'calibration_audit': 0, 'test': 1024, 'long_test': 0}
    metadata = {'format_version': 1, 'dataset_id': 'hamiballs-canonical-v2-train48', 'semantic_hash': hash_jsonable(semantic), 'semantic_config': semantic, 'implementation_version': int(source_metadata['implementation_version']), 'implementation_fingerprint': str(source_metadata['implementation_fingerprint']), 'runtime': {'view': 'immutable_array_pack_symlink_v1'}, 'stats_algorithm': 'pack_window_view_coordinatewise_rms_v1', 'status': 'finalized_unvalidated', 'manifest_sha256': manifest_hashes, 'stats_sha256': sha256_file(stats_dir / 'phase_scales.json')}
    metadata_path = metadata_dir / 'dataset_meta.json'
    metadata_path.write_bytes(_json_bytes(metadata))
    return metadata_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=project_root() / 'configs/hamiballs1/packing.json')
    parser.add_argument('--staging-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    source_root = root
    pack_root = root / 'packs'
    staging_root = args.staging_root.expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = sha256_file(protocol)
    manifests = {'train192': _build(source_root=source_root, pack_root=pack_root, staging_root=staging_root, protocol_sha256=protocol_sha256, name='train192', role='train', future_steps=192), 'train48_stride48': _build(source_root=source_root, pack_root=pack_root, staging_root=staging_root, protocol_sha256=protocol_sha256, name='train48_stride48', role='train', future_steps=48, offsets=(0, 48, 96, 144)), 'calibration192': _build(source_root=source_root, pack_root=pack_root, staging_root=staging_root, protocol_sha256=protocol_sha256, name='calibration192', role='calibration', future_steps=192), 'validation192': _build(source_root=source_root, pack_root=pack_root, staging_root=staging_root, protocol_sha256=protocol_sha256, name='validation192', role='validation', future_steps=192), 'validation48_prefix': _build(source_root=source_root, pack_root=pack_root, staging_root=staging_root, protocol_sha256=protocol_sha256, name='validation48_prefix', role='validation', future_steps=48), 'reserve192': _build(source_root=source_root, pack_root=pack_root, staging_root=staging_root, protocol_sha256=protocol_sha256, name='reserve192', role='reserve', future_steps=192)}
    adapter_metadata = _publish_train_adapter(output=root / 'train_adapter48', train_manifest=manifests['train48_stride48'], source_root=source_root)
    publication = {'schema': 'hamiformer.hamiballs.canonical_v2.publication.v1', 'status': 'complete', 'physical_seeds': list(PHYSICAL_SEEDS), 'protocol': str(protocol), 'protocol_sha256': protocol_sha256, 'packs': {name: {'manifest': str(path), 'sha256': sha256_file(path)} for name, path in manifests.items()}, 'train_adapter_metadata': {'path': str(adapter_metadata), 'sha256': sha256_file(adapter_metadata)}}
    publication_path = root / 'publication.json'
    publication_path.write_bytes(_json_bytes(publication))
    print(json.dumps(publication, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
