from __future__ import annotations
from hamiformer.utils.paths import project_root
import hashlib
import json
from pathlib import Path
from typing import Any

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _metadata(root: Path) -> dict[str, Any]:
    path = root / 'metadata' / 'dataset_meta.json'
    raw = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(raw, dict):
        raise ValueError(f'dataset metadata is not an object: {path}')
    return raw

def validate_selector_dataset_contract(*, parent_root: Path, dataset_root: Path, split: str, require_independent: bool, allow_reserved_test: bool=False) -> dict[str, Any]:
    allowed_splits = {'train', 'dev'}
    if allow_reserved_test:
        allowed_splits.add('test')
    if split not in allowed_splits:
        raise ValueError('selector calibration may open only train/dev or an explicitly reserved test')
    parent_root = parent_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    parent_meta = _metadata(parent_root)
    dataset_meta = _metadata(dataset_root)
    if dataset_meta.get('implementation_fingerprint') != parent_meta.get('implementation_fingerprint'):
        raise ValueError('selector dataset implementation fingerprint drifted')
    parent_semantic = parent_meta.get('semantic_config')
    dataset_semantic = dataset_meta.get('semantic_config')
    if not isinstance(parent_semantic, dict) or not isinstance(dataset_semantic, dict):
        raise ValueError('selector dataset lacks semantic configuration')
    for name in ('physics', 'sampling', 'runtime'):
        if dataset_semantic.get(name) != parent_semantic.get(name):
            raise ValueError(f'selector dataset {name} contract drifted')
    manifest_path = dataset_root / 'manifests' / f'{split}.jsonl'
    manifest_sha256 = _sha256_file(manifest_path)
    manifest_ledger = dataset_meta.get('manifest_sha256')
    if not isinstance(manifest_ledger, dict) or manifest_ledger.get(split) != manifest_sha256:
        raise ValueError(f'selector {split} manifest hash differs from metadata')
    parent_manifest_path = parent_root / 'manifests' / f'{split}.jsonl'
    parent_manifest_sha256 = _sha256_file(parent_manifest_path)
    if require_independent and dataset_root != parent_root:
        if manifest_sha256 == parent_manifest_sha256:
            raise ValueError('selector split is a copy of the 50k parent split')
        if dataset_semantic.get('seed') == parent_semantic.get('seed'):
            raise ValueError('independent selector root did not change the sampling seed')
    return {'dataset_root': str(dataset_root), 'parent_root': str(parent_root), 'split': split, 'role': 'independent_reserve_evaluation' if split == 'test' else 'independent_selector_calibration' if dataset_root != parent_root else 'parent_expert_training_data', 'dataset_id': dataset_meta.get('dataset_id'), 'semantic_hash': dataset_meta.get('semantic_hash'), 'implementation_fingerprint': dataset_meta.get('implementation_fingerprint'), 'manifest_sha256': manifest_sha256, 'parent_manifest_sha256': parent_manifest_sha256, 'parent_scales_reused': True}
