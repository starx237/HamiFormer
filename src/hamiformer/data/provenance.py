from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
_SHA256_PATTERN = re.compile('^[0-9a-f]{64}$')
_SPLITS = {'train', 'dev', 'calibration_fit', 'calibration_audit', 'test', 'long_test'}

def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f'dataset metadata 的 {field} 必须是 64 位小写 SHA-256')
    return value

@dataclass(frozen=True)
class DatasetMetadataProvenance:
    sha256: str
    semantic_hash: str
    implementation_version: int
    implementation_fingerprint: str
    status: str
    manifest_sha256: dict[str, str]
    stats_sha256: str

def load_dataset_metadata(path: str | Path) -> DatasetMetadataProvenance:
    metadata_path = Path(path)
    payload = metadata_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    raw = json.loads(payload.decode('utf-8'))
    if not isinstance(raw, dict) or raw.get('format_version') != 1:
        raise ValueError('dataset metadata 格式或版本错误')
    semantic_hash = _require_sha256(raw.get('semantic_hash'), 'semantic_hash')
    if raw.get('implementation_version') != 1:
        raise ValueError('dataset metadata 的 implementation_version 必须为 1')
    implementation_fingerprint = _require_sha256(raw.get('implementation_fingerprint'), 'implementation_fingerprint')
    status = raw.get('status')
    if status != 'finalized_unvalidated':
        raise ValueError('dataset metadata status 必须是 finalized_unvalidated；in_progress 或其他状态不能生成正式 audit')
    manifest_raw = raw.get('manifest_sha256')
    if not isinstance(manifest_raw, dict) or set(manifest_raw) != _SPLITS:
        raise ValueError('dataset metadata 必须绑定六个且仅六个 split manifest')
    manifests = {split: _require_sha256(manifest_raw[split], f'manifest_sha256.{split}') for split in sorted(_SPLITS)}
    stats_sha256 = _require_sha256(raw.get('stats_sha256'), 'stats_sha256')
    return DatasetMetadataProvenance(sha256=digest, semantic_hash=semantic_hash, implementation_version=1, implementation_fingerprint=implementation_fingerprint, status=status, manifest_sha256=manifests, stats_sha256=stats_sha256)
