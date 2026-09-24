from __future__ import annotations
from hamiformer.utils.paths import project_root
from pathlib import Path
import sys
from typing import Any
ROOT = project_root()
from hamiformer.data import packing as support
from hamiformer.training.hamiballs1 import gate_training as packed
from hamiformer.training.hamiballs1 import refresh as base
from hamiformer.training.hamiballs1 import router_setup as canonical
SCHEMA = 'hamiformer.hami1.gate_run.v1'
STAGE_SCHEMA = 'hamiformer.hami1.gate_stage.v1'
SEED_NAMESPACE = 'g_phase4p_b1f_stage_parent_accumulated'
STAGES = ('scalar0', 'scalar1', 'qp')
STAGE_UPDATES = {'scalar0': 200, 'scalar1': 200, 'qp': 200}
OBJECTIVES = {'scalar0': 'global_sequence_projection_regret', 'scalar1': 'global_sequence_projection_regret', 'qp': packed.DUAL_OBSERVABLE_DISJOINT_RISK_AWARE_OBJECTIVE}
LEARNING_RATES = {'scalar0': (0.0003, 3e-06), 'scalar1': (0.0003, 3e-06), 'qp': (0.01, 0.0001)}
SOURCE_SEED = support.namespaced_seed('g_source_permutation')
STAGE_NOISE_SEEDS = {'scalar0': support.namespaced_seed('g_scalar0_noise'), 'scalar1': support.namespaced_seed('g_scalar1_noise'), 'qp': support.namespaced_seed('g_adapter_noise')}
STAGE_OPTIMIZER_SEEDS = {'scalar0': support.namespaced_seed('g_optimizer_minibatch', 0), 'scalar1': support.namespaced_seed('g_optimizer_minibatch', 1), 'qp': support.namespaced_seed('g_optimizer_minibatch', 0)}
LOCKED_STAGE_OFFSETS = {'scalar0': canonical.SOURCE_OFFSETS['scalar0'], 'scalar1': canonical.SOURCE_OFFSETS['scalar1'], 'qp': canonical.SOURCE_OFFSETS['risk']}
_BASE_PREPARE_MAIN = base.prepare.main

def _resolve_source_blocks(*, dataset_root: Path, exclusion_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    excluded, scene_ids, provenance = base._exclusion_and_scene_ledger(dataset_root=dataset_root, exclusion_path=exclusion_path)
    used_indices: set[int] = set()
    blocks: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        fit_samples = base._stage_fit_samples(stage)
        total_sources = base._stage_total_sources(stage)
        offset = LOCKED_STAGE_OFFSETS[stage]
        indices, positions = packed._sample_permutation_block_excluding_scenes(size=len(scene_ids), count=total_sources, seed=base.SOURCE_SEED, offset=offset, scene_ids=scene_ids, excluded_scenes=excluded)
        values = [int(value) for value in indices.tolist()]
        position_values = [int(value) for value in positions]
        if len(values) != total_sources or used_indices.intersection(values) or any((scene_ids[index] in excluded for index in values)):
            raise AssertionError('Gate training source blocks violate the partition')
        used_indices.update(values)
        blocks[stage] = {'sample_offset': offset, 'permutation_first_position': position_values[0], 'permutation_last_position': position_values[-1], 'sample_indices': values, 'sample_indices_sha256': support.json_digest(values), 'permutation_positions': position_values, 'permutation_positions_sha256': support.json_digest(position_values), 'fit_source_indices_sha256': support.json_digest(values[:fit_samples]), 'holdout_source_indices_sha256': support.json_digest(values[fit_samples:])}
    return (blocks, {**provenance, 'source_schedule': 'locked_stage_offsets_with_audit_exclusion', 'locked_stage_offsets': dict(LOCKED_STAGE_OFFSETS)})

def configure_runner() -> None:
    base.SCHEMA = SCHEMA
    base.STAGE_SCHEMA = STAGE_SCHEMA
    base.RUNNER_PATH = Path(__file__).resolve()
    base.SEED_NAMESPACE = SEED_NAMESPACE
    base.SOURCE_SEED = SOURCE_SEED
    base.STAGES = STAGES
    base.STAGE_UPDATES = dict(STAGE_UPDATES)
    base.STAGE_NOISE_SEEDS = dict(STAGE_NOISE_SEEDS)
    base.STAGE_OPTIMIZER_SEEDS = dict(STAGE_OPTIMIZER_SEEDS)
    base.OBJECTIVES = dict(OBJECTIVES)
    base.LEARNING_RATES = dict(LEARNING_RATES)
    base.STAGE_PARENT_CARRIER_STAGES = frozenset(('scalar1', 'qp'))
    base.AGGREGATE_REPLAY_STAGES = frozenset(STAGES)
    base.QP_DELTA_WIDTH = 16
    base._resolve_source_blocks = _resolve_source_blocks
    base.prepare.main = _BASE_PREPARE_MAIN
