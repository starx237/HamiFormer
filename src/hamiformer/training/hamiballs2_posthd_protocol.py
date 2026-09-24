from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable
import numpy as np
from hamiformer.data_generation3d.config import load_config
DATASET_SIGNATURE = load_config(Path(__file__).resolve().parents[3] / 'configs/generator/hamiballs2_frozen_v2_g1_khalf_.yaml').semantic_hash

@dataclass(frozen=True)
class StageSpec:
    name: str
    updates: int
    carrier: str
    parent: str
    refresh_every: int
FORMAL_STAGES = (StageSpec('common_r', 200, 'fresh_random_gate_RF_N20', 'fixed_HD', 8), StageSpec('scalar0', 200, 'precomputed_mixed_RF_N', 'fixed_HDr', 0), StageSpec('scalar1', 200, 'precomputed_mixed_RF_N', 'frozen_scalar0', 0), StageSpec('final_qp', 200, 'current_policy_source_axis_mixed_RF_N', 'frozen_scalar1_tree', 8))
SCALAR_B64_RF_COUNTS = {8: 32, 12: 20, 20: 12}
FINAL_NOISE_B32_RF_COUNTS = {8: 16, 12: 10, 20: 6}
FINAL_REFRESHES = 25
NOISE_SLOTS = 2

def namespaced_seed(master_seed: int, namespace: str, index: int=0) -> int:
    digest = hashlib.sha256(f'hamiballs2-formal-posthd|{int(master_seed)}|{namespace}|{int(index)}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') & 2147483647

def mixed_rf_assignment(counts: dict[int, int], *, master_seed: int, namespace: str, index: int) -> np.ndarray:
    values = np.concatenate([np.full(count, rf_steps, dtype=np.int16) for rf_steps, count in sorted(counts.items())])
    np.random.default_rng(namespaced_seed(master_seed, namespace, index)).shuffle(values)
    return values

def scalar_pool_plan(*, master_seed: int, stage: str, blocks: int=25) -> list[dict[str, object]]:
    if stage not in {'scalar0', 'scalar1'}:
        raise ValueError('fixed scalar pool stage must be scalar0 or scalar1')
    return [{'block': block, 'rf_num_steps': mixed_rf_assignment(SCALAR_B64_RF_COUNTS, master_seed=master_seed, namespace=f'{stage}_rf_assignment', index=block).tolist(), 'source_seed': namespaced_seed(master_seed, f'{stage}_source', block), 'noise_seed': namespaced_seed(master_seed, f'{stage}_noise', block), 'route_seed': namespaced_seed(master_seed, f'{stage}_random_route', block)} for block in range(blocks)]

def common_r_refresh_plan(*, master_seed: int, blocks: int=25) -> list[dict[str, object]]:
    return [{'block': block, 'rf_num_steps': [20] * 64, 'source_seed': namespaced_seed(master_seed, 'common_r_source', block), 'noise_seed': namespaced_seed(master_seed, 'common_r_noise', block), 'route_seed': namespaced_seed(master_seed, 'common_r_random_route', block)} for block in range(blocks)]

def final_refresh_plan(*, master_seed: int) -> list[dict[str, object]]:
    result = []
    for refresh in range(FINAL_REFRESHES):
        halves = []
        for noise_slot in range(NOISE_SLOTS):
            halves.append({'noise_slot': noise_slot, 'rf_num_steps': mixed_rf_assignment(FINAL_NOISE_B32_RF_COUNTS, master_seed=master_seed, namespace=f'final_rf_assignment_noise{noise_slot}', index=refresh).tolist(), 'source_seed': namespaced_seed(master_seed, f'final_source_noise{noise_slot}', refresh), 'noise_seed': namespaced_seed(master_seed, f'final_noise_noise{noise_slot}', refresh)})
        result.append({'refresh': refresh, 'updates': 8, 'noise_halves': halves, 'optimizer_seed': namespaced_seed(master_seed, 'final_optimizer', refresh)})
    return result

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def formal_contract(master_seed: int=42) -> dict[str, object]:
    stage_updates = {stage.name: stage.updates for stage in FORMAL_STAGES}
    total = sum(stage_updates.values())
    if total != 800:
        raise AssertionError('formal post-HD schedule drifted from 800 updates')
    return {'schema': 'hamiformer.hamiballs2.formal_posthd_contract.v1', 'master_seed': int(master_seed), 'model_initialization': {'namespace': 'model_initialization', 'seed': namespaced_seed(master_seed, 'model_initialization')}, 'dataset_semantic_hash': DATASET_SIGNATURE, 'stage_updates': stage_updates, 'total_post_HD_optimizer_updates': total, 'tree_fit_count': 1, 'tree_fixed_during_final_qp': True, 'final_refreshes': FINAL_REFRESHES, 'updates_per_final_refresh': 8, 'optimizer_batch_size': 64, 'scalar_mixed_rf_counts_per_B64': SCALAR_B64_RF_COUNTS, 'final_mixed_rf_counts_per_noise_B32': FINAL_NOISE_B32_RF_COUNTS, 'q_p_model_family': 'shared_trunk_continuous_qp_heads', 'ridge': {'parameterization': 'common_plus_leaf_deviation_compiled_to_leaf', 'source_balanced': True, 'cumulative_across_final_refreshes': True, 'relative_strength': 0.01, 'intercept_penalty_multiplier': 0.01}, 'stages': [asdict(stage) for stage in FORMAL_STAGES]}

def validate_formal_contract(contract: dict[str, object]) -> None:
    expected = formal_contract(int(contract.get('master_seed', -1)))
    protected = ('dataset_semantic_hash', 'stage_updates', 'total_post_HD_optimizer_updates', 'tree_fit_count', 'tree_fixed_during_final_qp', 'final_refreshes', 'updates_per_final_refresh', 'model_initialization')
    drift = {name: (contract.get(name), expected[name]) for name in protected if contract.get(name) != expected[name]}
    if drift:
        raise ValueError(f'formal post-HD contract drift: {drift}')
__all__ = ['DATASET_SIGNATURE', 'FINAL_NOISE_B32_RF_COUNTS', 'FINAL_REFRESHES', 'FORMAL_STAGES', 'SCALAR_B64_RF_COUNTS', 'common_r_refresh_plan', 'final_refresh_plan', 'formal_contract', 'mixed_rf_assignment', 'namespaced_seed', 'scalar_pool_plan', 'sha256_file', 'validate_formal_contract']
