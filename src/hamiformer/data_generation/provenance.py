from __future__ import annotations
import hashlib
from pathlib import Path
GENERATOR_IMPLEMENTATION_VERSION = 1

def implementation_fingerprint() -> str:
    package_root = Path(__file__).resolve().parent
    source_files = sorted(package_root.glob('*.py'), key=lambda path: path.name)
    if not source_files:
        raise RuntimeError('无法定位 data_generation 源码，拒绝生成不可追溯数据')
    digest = hashlib.sha256()
    digest.update(f'implementation-version:{GENERATOR_IMPLEMENTATION_VERSION}\n'.encode('ascii'))
    for path in source_files:
        digest.update(path.name.encode('utf-8'))
        digest.update(b'\x00')
        digest.update(path.read_bytes())
        digest.update(b'\x00')
    return digest.hexdigest()
