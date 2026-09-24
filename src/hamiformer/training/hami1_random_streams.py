import hashlib
MASTER_SEED = 42
DERIVATION_PREFIX = 'hamiballs-canonical-v2'
LOCKED_STAGE_OFFSETS = {'scalar0': 64, 'scalar1': 2176, 'qp': 4288}

def derive_seed(namespace: str, counter: int=0) -> int:
    payload = f'{DERIVATION_PREFIX}|run-seed={MASTER_SEED}|{namespace}|counter={counter}'.encode('utf-8')
    value = int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')
    return value & 2147483647 or 1
