from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any
import torch
ROOT = project_root()
from hamiformer.utils import sha256_file
from hamiformer.training.source_partitions import exclusion_metadata
from hamiformer.data import packing as support
from hamiformer.training.hamiballs1 import router_preparation as prepare
from hamiformer.training.hamiballs1 import gate_training as packed
from hamiformer.training.hamiballs1 import router_setup as canonical
SCHEMA = 'hamiformer.hamiballs.canonical_v2.fresh_refresh8_gate_run.v1'
STAGE_SCHEMA = 'hamiformer.hamiballs.canonical_v2.fresh_refresh8_stage.v1'
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_ROOT = canonical.ARTIFACT_ROOT.parent / 'hamiballs_canonical_v2_gate_fresh_refresh8_seed42'
EXCLUSION_CONFIGURATION = None
UPDATES = 200
REFRESH_EVERY = 8
REFRESH_SOURCES = 64
FIT_SAMPLES = 1600
HOLDOUT_SAMPLES = 64
TOTAL_SOURCES_PER_STAGE = FIT_SAMPLES + HOLDOUT_SAMPLES
SEED_NAMESPACE = 'g_fresh_refresh8'
SOURCE_SEED = support.namespaced_seed(f'{SEED_NAMESPACE}_source_permutation')
STAGES = ('scalar0', 'scalar1', 'qp')
OBJECTIVES = {'scalar0': 'global_sequence_projection_regret', 'scalar1': 'global_sequence_projection_regret', 'qp': packed.DUAL_OBSERVABLE_DISJOINT_ENDPOINT_RECURRENT_OBJECTIVE}
LEARNING_RATES = {'scalar0': (0.0003, 3e-06), 'scalar1': (0.0003, 3e-06), 'qp': (0.01, 0.0001)}
STAGE_EXTRA_ARGUMENTS: dict[str, tuple[str, ...]] = {}
STAGE_UPDATES: dict[str, int] = {}
STAGE_NOISE_SEEDS: dict[str, int] = {}
STAGE_OPTIMIZER_SEEDS: dict[str, int] = {}
STAGE_PARENT_CARRIER_STAGES: frozenset[str] = frozenset()
EMA_PARENT_CARRIER_STAGES: frozenset[str] = frozenset()
EMA_PARENT_BETA = 2.0 ** (-1.0 / 64.0)
AGGREGATE_REPLAY_STAGES: frozenset[str] = frozenset()
QP_DELTA_WIDTH = 4

def _qp_parameter_counts() -> tuple[int, int]:
    width = int(QP_DELTA_WIDTH)
    if width <= 0:
        raise ValueError('q/p delta width must be positive')
    trainable = 144 * width + 2
    return (4130 + trainable, trainable)

def _stage_updates(stage: str) -> int:
    updates = int(STAGE_UPDATES.get(stage, UPDATES))
    if updates <= 0 or updates % REFRESH_EVERY:
        raise ValueError(f'fresh refresh-8 stage {stage} has invalid update count {updates}')
    return updates

def _stage_fit_samples(stage: str) -> int:
    return _stage_updates(stage) // REFRESH_EVERY * REFRESH_SOURCES

def _stage_total_sources(stage: str) -> int:
    return _stage_fit_samples(stage) + HOLDOUT_SAMPLES

def _stage_cumulative_updates(stage: str) -> int:
    index = STAGES.index(stage)
    return sum((_stage_updates(value) for value in STAGES[:index + 1]))

def _stage_noise_seed(stage: str) -> int:
    return int(STAGE_NOISE_SEEDS.get(stage, support.namespaced_seed(f'{SEED_NAMESPACE}_{stage}_noise')))

def _stage_optimizer_seed(stage: str) -> int:
    return int(STAGE_OPTIMIZER_SEEDS.get(stage, support.namespaced_seed(f'{SEED_NAMESPACE}_{stage}_optimizer')))

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))

def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _invoke(entrypoint, arguments: list[str]) -> None:
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        entrypoint()
    finally:
        sys.argv = previous

def _fresh_checkpoint_contract(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError('fresh standard-readout gate payload is not a mapping')
    state = payload.get('gate_state_dict')
    weight = state.get('output.weight') if isinstance(state, dict) else None
    if payload.get('schema') != prepare.METRIC_GATE_SCHEMA or payload.get('trained_updates') != 0 or payload.get('role') != 'canonical-v2-observable71-fresh-standard-readout-scalar' or (not isinstance(weight, torch.Tensor)) or (weight.ndim != 2) or (int(torch.count_nonzero(weight)) == 0) or (payload.get('architecture_seed') != support.namespaced_seed('g_architecture_init')) or (payload.get('readout_seed') != support.namespaced_seed('g_readout_init')) or (payload.get('neutral_gate_digest') == payload.get('metric_gate_digest')):
        raise ValueError('fresh standard-readout gate contract drifted')
    return {'path': str(path), 'sha256': sha256_file(path), 'trained_updates': 0, 'parameter_count': int(payload['parameter_count']), 'architecture_seed': int(payload['architecture_seed']), 'readout_seed': int(payload['readout_seed']), 'readout_nonzero': True, 'metric_gate_digest': payload['metric_gate_digest']}

def _exclusion_and_scene_ledger(*, dataset_root: Path, exclusion_path: Path) -> tuple[set[str], list[str], dict[str, str]]:
    from hamiformer.training.source_partitions import load_scene_exclusions
    excluded = load_scene_exclusions(exclusion_path)
    adapter_manifest = (dataset_root / 'manifests' / 'train.jsonl').resolve()
    adapter_pack = _read_json(adapter_manifest)
    ledger_path = adapter_manifest.parent / adapter_pack['row_ledger']['file']
    if sha256_file(ledger_path) != adapter_pack['row_ledger']['sha256']:
        raise ValueError('train row ledger SHA drifted')
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if any((row.get('row') != index for index, row in enumerate(ledger))):
        raise ValueError('train row ledger ordering drifted')
    return ({str(value) for value in excluded}, [str(row['scene_id']) for row in ledger], {**exclusion_metadata(exclusion_path), 'row_ledger_path': str(ledger_path.resolve()), 'row_ledger_sha256': sha256_file(ledger_path)})

def _resolve_source_blocks(*, dataset_root: Path, exclusion_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    excluded, scene_ids, provenance = _exclusion_and_scene_ledger(dataset_root=dataset_root, exclusion_path=exclusion_path)
    offset = 0
    used_indices: set[int] = set()
    blocks: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        fit_samples = _stage_fit_samples(stage)
        total_sources = _stage_total_sources(stage)
        indices, positions = packed._sample_permutation_block_excluding_scenes(size=len(scene_ids), count=total_sources, seed=SOURCE_SEED, offset=offset, scene_ids=scene_ids, excluded_scenes=excluded)
        values = [int(value) for value in indices.tolist()]
        if len(values) != total_sources or used_indices.intersection(values) or any((scene_ids[index] in excluded for index in values)):
            raise AssertionError('fresh stage source blocks are not disjoint and excluded')
        used_indices.update(values)
        blocks[stage] = {'sample_offset': offset, 'permutation_first_position': int(positions[0]), 'permutation_last_position': int(positions[-1]), 'sample_indices': values, 'sample_indices_sha256': support.json_digest(values), 'permutation_positions': [int(value) for value in positions], 'permutation_positions_sha256': support.json_digest(positions), 'fit_source_indices_sha256': support.json_digest(values[:fit_samples]), 'holdout_source_indices_sha256': support.json_digest(values[fit_samples:])}
        offset = int(positions[-1]) + 1
    return (blocks, provenance)

def _checkpoint_path(output_root: Path, stage: str) -> Path:
    return output_root / stage / f'{OBJECTIVES[stage]}_gate.pt'

def _stage_parent(output_root: Path, stage: str) -> Path:
    if stage == 'scalar0':
        return output_root / 'prepared' / 'metric_gate.pt'
    if stage == 'scalar1':
        return _checkpoint_path(output_root, 'scalar0')
    if stage == 'qp':
        return _checkpoint_path(output_root, 'scalar1')
    if stage == 'joint':
        return _checkpoint_path(output_root, 'qp')
    raise ValueError(f'unknown fresh refresh-8 stage {stage}')

def _write_training_contract(*, output_root: Path, initial_gate: dict[str, Any], source_blocks: dict[str, dict[str, Any]], source_provenance: dict[str, str], r_registration: Path, gate_registration: Path) -> Path:
    contract_path = output_root / 'training_contract.json'
    stages = []
    for index, stage in enumerate(STAGES):
        updates = _stage_updates(stage)
        fit_samples = _stage_fit_samples(stage)
        stages.append({'stage': stage, 'stage_index': index, 'updates': updates, 'cumulative_gate_updates': _stage_cumulative_updates(stage), 'objective': OBJECTIVES[stage], 'parent_checkpoint': str(_stage_parent(output_root, stage)), 'output_checkpoint': str(_checkpoint_path(output_root, stage)), 'carrier_behavior': 'seeded_uniform_random_scalar' if stage == 'scalar0' else 'same_run_ema_teacher' if stage in EMA_PARENT_CARRIER_STAGES else 'fixed_same_run_stage_parent_target' if stage in STAGE_PARENT_CARRIER_STAGES else 'current_candidate', 'ema_beta': EMA_PARENT_BETA if stage in EMA_PARENT_CARRIER_STAGES else None, 'ema_half_life_updates': 64 if stage in EMA_PARENT_CARRIER_STAGES else None, 'refresh_every': REFRESH_EVERY, 'refresh_sources': REFRESH_SOURCES, 'refresh_blocks': updates // REFRESH_EVERY, 'fit_samples': fit_samples, 'holdout_samples': HOLDOUT_SAMPLES, 'aggregate_replay': stage in AGGREGATE_REPLAY_STAGES, 'noise_seed': _stage_noise_seed(stage), 'optimizer_sample_seed': _stage_optimizer_seed(stage), 'observable_disjoint_qp_expansion': stage == 'qp', 'observable_disjoint_delta_width': int(QP_DELTA_WIDTH) if stage == 'qp' else None, 'trainable_parameter_owner': f'width{int(QP_DELTA_WIDTH)}_qp_delta_towers_only' if stage == 'qp' else 'full_observable_disjoint_qp_gate' if stage == 'joint' else 'full_scalar_gate', 'source_block': source_blocks[stage]})
    contract = {'schema': SCHEMA, 'status': 'PREPARED', 'initial_gate': initial_gate, 'r_registration': {'path': str(r_registration.resolve()), 'sha256': sha256_file(r_registration)}, 'gate_registration': {'path': str(gate_registration.resolve()), 'sha256': sha256_file(gate_registration)}, 'source_seed': SOURCE_SEED, 'source_provenance': source_provenance, 'source_exclusions': exclusion_metadata(EXCLUSION_CONFIGURATION), 'stages': stages}
    _atomic_json(contract_path, contract)
    return contract_path

def _validate_refresh_terminal(*, stage: str, output_root: Path, contract_path: Path, source_block: dict[str, Any]) -> dict[str, Any]:
    output = output_root / stage
    summary_path = output / 'summary.json'
    checkpoint = _checkpoint_path(output_root, stage)
    parent = _stage_parent(output_root, stage)
    if not ((output / 'COMPLETE').is_file() and summary_path.is_file() and checkpoint.is_file() and parent.is_file()):
        raise RuntimeError(f'gate stage {stage} did not publish a terminal')
    summary = _read_json(summary_path)
    updates = _stage_updates(stage)
    fit_samples = _stage_fit_samples(stage)
    if (summary.get('updates') != updates or summary.get('objectives') != [OBJECTIVES[stage]]
            or summary.get('fit_samples') != fit_samples
            or summary.get('holdout_samples') != HOLDOUT_SAMPLES
            or summary.get('noise_seeds') != [_stage_noise_seed(stage)]):
        raise ValueError(f'gate stage summary mismatch for {stage}')
    steps = summary.get('actual_optimizer_steps', {}).get(OBJECTIVES[stage])
    if steps != updates:
        raise ValueError(f'gate stage {stage} completed {steps!r} optimizer steps; expected {updates}')
    record = {'schema': STAGE_SCHEMA, 'status': 'COMPLETE', 'stage': stage, 'updates': updates, 'cumulative_gate_updates': _stage_cumulative_updates(stage), 'objective': OBJECTIVES[stage], 'parent_checkpoint': str(parent), 'parent_checkpoint_sha256': sha256_file(parent), 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint), 'summary': str(summary_path), 'summary_sha256': sha256_file(summary_path), 'training_contract': str(contract_path), 'training_contract_sha256': sha256_file(contract_path), 'sample_indices_sha256': source_block['sample_indices_sha256']}
    record_path = output / 'stage.json'
    if record_path.is_file():
        if _read_json(record_path) != record:
            raise ValueError(f'existing stage record differs for {stage}')
    else:
        _atomic_json(record_path, record)
    record['stage_path'] = str(record_path)
    record['stage_sha256'] = sha256_file(record_path)
    return record

def _run_stage(*, stage: str, output_root: Path, gate_registration: Path, contract_path: Path, source_block: dict[str, Any]) -> dict[str, Any]:
    output = output_root / stage
    parent = _stage_parent(output_root, stage)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite fresh stage {output}')
    if not parent.is_file() or output_root.resolve() not in parent.resolve().parents:
        raise ValueError('formal gate parent is not an existing same-run checkpoint')
    maximum_lr, minimum_lr = LEARNING_RATES[stage]
    updates = _stage_updates(stage)
    fit_samples = _stage_fit_samples(stage)
    arguments = ['--registration', str(gate_registration.resolve()), '--dataset-root', str(canonical.DATASET_ROOT), '--output-dir', str(output), '--fit-samples', str(fit_samples), '--holdout-samples', str(HOLDOUT_SAMPLES), '--collection-batch', str(REFRESH_SOURCES), '--sequence-batch', '128', '--updates', str(updates), '--self-policy-refresh-every', str(REFRESH_EVERY), '--self-policy-refresh-sources', str(REFRESH_SOURCES), '--field-indices', ','.join((str(index) for index in range(18))), '--sample-seed', str(SOURCE_SEED), '--sample-offset', str(source_block['sample_offset']), '--noise-seeds', str(_stage_noise_seed(stage)), '--optimizer-sample-seed', str(_stage_optimizer_seed(stage)), '--maximum-lr', str(maximum_lr), '--minimum-lr', str(minimum_lr), '--arms', OBJECTIVES[stage], '--gate-checkpoint', str(parent), '--online-recurrence', '--device', 'cuda']
    if EXCLUSION_CONFIGURATION is not None:
        arguments.extend(['--sample-excluded-scene-configuration', str(EXCLUSION_CONFIGURATION)])
    if stage == 'scalar0':
        arguments.append('--self-policy-uniform-random-carriers')
    if stage in STAGE_PARENT_CARRIER_STAGES:
        if stage == 'scalar0':
            raise ValueError('scalar0 cannot mix random and stage-parent carriers')
        arguments.append('--self-policy-stage-parent-carriers')
    if stage in EMA_PARENT_CARRIER_STAGES:
        if stage == 'scalar0' or stage in STAGE_PARENT_CARRIER_STAGES:
            raise ValueError('EMA carriers require a non-random, non-frozen-parent stage')
        arguments.extend(['--self-policy-ema-parent-carriers', '--self-policy-ema-beta', str(EMA_PARENT_BETA)])
    if stage in AGGREGATE_REPLAY_STAGES:
        arguments.append('--self-policy-aggregate-replay')
    if stage == 'qp':
        arguments.append('--observable-disjoint-component-gate-expansion')
        if int(QP_DELTA_WIDTH) != 4:
            arguments.extend(['--observable-disjoint-delta-width', str(int(QP_DELTA_WIDTH))])
    if stage == 'joint':
        arguments.append('--observable-disjoint-component-gate-joint-resume')
    arguments.extend(STAGE_EXTRA_ARGUMENTS.get(stage, ()))
    _invoke(packed.main, arguments)
    return _validate_refresh_terminal(stage=stage, output_root=output_root, contract_path=contract_path, source_block=source_block)

def _publish_terminal(*, output_root: Path, contract_path: Path, records: list[dict[str, Any]], resumed_from_scalar_stages: bool) -> None:
    terminal_stage = STAGES[-1]
    terminal_checkpoint = _checkpoint_path(output_root, terminal_stage)
    terminal = {'schema': SCHEMA, 'status': 'COMPLETE', 'cumulative_gate_updates': sum((_stage_updates(stage) for stage in STAGES)), 'terminal_stage': terminal_stage, 'terminal_checkpoint': str(terminal_checkpoint), 'terminal_checkpoint_sha256': sha256_file(terminal_checkpoint), 'training_contract': str(contract_path), 'training_contract_sha256': sha256_file(contract_path), 'stage_records': records, 'resumed_from_scalar_stages': resumed_from_scalar_stages}
    _atomic_json(output_root / 'terminal.json', terminal)
    (output_root / 'COMPLETE').write_text('COMPLETE\n', encoding='utf-8')
    print(json.dumps(terminal, sort_keys=True))

def _resume_from_scalar_stages(*, r_registration: Path, gate_registration: Path, output_root: Path) -> None:
    contract_path = output_root / 'training_contract.json'
    if not output_root.is_dir() or (output_root / 'COMPLETE').exists() or (output_root / 'terminal.json').exists() or (output_root / 'qp').exists() or (not contract_path.is_file()):
        raise ValueError('fresh q/p resume requires an incomplete pre-update run')
    contract = _read_json(contract_path)
    source_blocks, source_provenance = _resolve_source_blocks(dataset_root=canonical.DATASET_ROOT, exclusion_path=EXCLUSION_CONFIGURATION)
    contract_stages = {row.get('stage'): row for row in contract.get('stages', []) if isinstance(row, dict)}
    fresh = _fresh_checkpoint_contract(output_root / 'prepared' / 'metric_gate.pt')
    if contract.get('schema') != SCHEMA or contract.get('status') != 'PREPARED' or (contract.get('r_registration') != {'path': str(r_registration), 'sha256': sha256_file(r_registration)}) or (contract.get('gate_registration') != {'path': str(gate_registration), 'sha256': sha256_file(gate_registration)}) or (contract.get('initial_gate') != fresh) or (contract.get('source_provenance') != source_provenance) or (set(contract_stages) != set(STAGES)) or any((contract_stages[stage].get('source_block') != source_blocks[stage] for stage in STAGES)):
        raise ValueError('q/p resume training contract differs')
    canonical.PREPARED = output_root / 'prepared'
    canonical._install(r_registration=r_registration, gate_registration=gate_registration)
    records = [_validate_refresh_terminal(stage=stage, output_root=output_root, contract_path=contract_path, source_block=source_blocks[stage]) for stage in ('scalar0', 'scalar1')]
    records.append(_run_stage(stage='qp', output_root=output_root, gate_registration=gate_registration, contract_path=contract_path, source_block=source_blocks['qp']))
    _publish_terminal(output_root=output_root, contract_path=contract_path, records=records, resumed_from_scalar_stages=True)

def _run(*, r_registration: Path, gate_registration: Path, output_root: Path, resume_from_scalar_stages: bool=False, stop_after: str='qp') -> None:
    output_root = output_root.expanduser().resolve()
    r_registration = r_registration.expanduser().resolve()
    gate_registration = gate_registration.expanduser().resolve()
    if stop_after not in STAGES:
        raise ValueError('unknown gate stage: ' + stop_after)
    if resume_from_scalar_stages and stop_after != 'qp':
        raise ValueError('resuming scalar stages requires the qp target')
    if resume_from_scalar_stages:
        _resume_from_scalar_stages(r_registration=r_registration, gate_registration=gate_registration, output_root=output_root)
        return
    if output_root.exists():
        raise FileExistsError(f'refusing to overwrite formal run {output_root}')
    if EXCLUSION_CONFIGURATION is not None and (not EXCLUSION_CONFIGURATION.is_file()):
        raise FileNotFoundError('source exclusion file is missing')
    source_blocks, source_provenance = _resolve_source_blocks(dataset_root=canonical.DATASET_ROOT, exclusion_path=EXCLUSION_CONFIGURATION)
    prepared = output_root / 'prepared'
    _invoke(prepare.main, ['--r-registration', str(r_registration), '--gate-registration', str(gate_registration), '--output-dir', str(prepared), '--device', 'cuda'])
    fresh = _fresh_checkpoint_contract(prepared / 'metric_gate.pt')
    contract_path = _write_training_contract(output_root=output_root, initial_gate=fresh, source_blocks=source_blocks, source_provenance=source_provenance, r_registration=r_registration, gate_registration=gate_registration)
    canonical.PREPARED = prepared
    canonical._install(r_registration=r_registration, gate_registration=gate_registration)
    records = [_run_stage(stage=stage, output_root=output_root, gate_registration=gate_registration, contract_path=contract_path, source_block=source_blocks[stage]) for stage in STAGES[:STAGES.index(stop_after) + 1]]
    if stop_after != 'qp':
        _atomic_json(output_root / 'stage_summary.json', {'completed_stages': list(STAGES[:STAGES.index(stop_after) + 1]), 'stage_records': records})
        (output_root / 'SCALAR_STAGES_COMPLETE').write_text('COMPLETE\n')
        return
    _publish_terminal(output_root=output_root, contract_path=contract_path, records=records, resumed_from_scalar_stages=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r-registration', type=Path, required=True)
    parser.add_argument('--gate-registration', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--stop-after', choices=STAGES, default='qp')
    parser.add_argument('--resume-from-scalar-stages', action='store_true', help='Start q/p training from completed scalar0/scalar1 stages; the q/p output directory must be empty.')
    args = parser.parse_args()
    _run(r_registration=args.r_registration, gate_registration=args.gate_registration, output_root=args.output_root, resume_from_scalar_stages=args.resume_from_scalar_stages, stop_after=args.stop_after)
