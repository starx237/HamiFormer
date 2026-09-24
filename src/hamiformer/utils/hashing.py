from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while (chunk := handle.read(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()

def hash_jsonable(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
