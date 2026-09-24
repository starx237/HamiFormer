from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
ROOT = project_root()
from hamiformer.training.hamiballs_recovery import module_digest, set_trainable
from hamiformer.utils import sha256_file
from hamiformer.training.hamiballs1 import frozen_residual as residual_training
from hamiformer.training.hamiballs1 import router_models as gate_training
REGISTERED_RESIDUAL_LR: dict[str, float | int] | None = None
CARRIER_OBSERVER: Any | None = None

def _preview_stream_indices(state: dict[str, Any]) -> torch.Tensor:
    size = int(state['size'])
    needed = int(state['batch_size'])
    offset = int(state['offset'])
    permutation = state['permutation'].clone()
    generator = torch.Generator(device='cpu')
    generator.set_state(state['generator_state'].clone())
    chunks: list[torch.Tensor] = []
    while needed:
        take = min(needed, size - offset)
        chunks.append(permutation[offset:offset + take])
        offset += take
        needed -= take
        if offset == size and needed:
            permutation = torch.randperm(size, generator=generator)
            offset = 0
    return torch.cat(chunks)

def _patch_residual_training() -> None:
    residual_training.SCHEMA = gate_training.SCHEMA
    residual_training.ROLE = gate_training.ROLE
    residual_training._load_registration = _load_registration
    residual_training._source_manifest = gate_training._source_manifest
    residual_training._set_marker = gate_training._set_marker
    residual_training._build_frozen_experts_and_zero_r = gate_training._build_per_object_residual
    residual_training._collect_external_carrier = gate_training._collect_per_object_external_carrier
    gate_training.stage_a._collect_external_carrier = gate_training._collect_per_object_external_carrier
    residual_training.recovery_full_no_regret_residual_update = gate_training.joint_residual_joint_residual_update

def _load_registration(path):
    from hamiformer.training.residual import _load_registration as load
    return load(path)

def _carrier_blocks(expected: dict[str, int], refresh: int) -> list[tuple[int, int]]:
    if refresh <= 0:
        raise ValueError('refresh_every_updates must be positive')
    blocks: list[tuple[int, int]] = []
    for n in (8, 12, 20):
        remaining = int(expected[f'n{n}'])
        while remaining:
            length = min(refresh, remaining)
            blocks.append((n, length))
            remaining -= length
    return blocks

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    return p.parse_args()

def _set_frozen_experts_eval(d: torch.nn.Module, hamiltonian: torch.nn.Module | None) -> None:
    d.eval()
    if hamiltonian is not None:
        hamiltonian.eval()

def _clear_frozen_expert_gradients(d: torch.nn.Module, hamiltonian: torch.nn.Module | None) -> None:
    if hamiltonian is None:
        residual_training.clear_frozen_gradients(d)
    else:
        residual_training.clear_frozen_gradients(d, hamiltonian)

def _frozen_expert_gradients_absent(d: torch.nn.Module, hamiltonian: torch.nn.Module | None) -> bool:
    if hamiltonian is None:
        return residual_training.frozen_gradients_absent(d)
    return residual_training.frozen_gradients_absent(d, hamiltonian)

def _frozen_expert_digests(d: torch.nn.Module, hamiltonian: torch.nn.Module | None) -> dict[str, str]:
    return {'d': module_digest(d), 'h': 'identity_affine_map' if hamiltonian is None else module_digest(hamiltonian)}

def main() -> None:
    _patch_residual_training()
    args = _args()
    registration = _load_registration(args.registration.expanduser().resolve())
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    parent, _wide, parent_contract = residual_training._load_parents(registration)
    config = copy.deepcopy(parent['config'])
    gate_training.stage_a._validate_runtime_config(config, smoke=False)
    config_sha256 = residual_training._canonical_hash(config)
    if config_sha256 != parent['config_sha256']:
        raise ValueError('resolved parent config drift')
    train_contract = gate_training.stage_a._train_contract(config)
    if train_contract != parent['dataset_contract']:
        raise ValueError('train contract drift')
    parent_contract = {**parent_contract, 'train_contract': train_contract}
    source_manifest = gate_training._source_manifest()
    source_manifest['src/hamiformer/training/hamiballs1/router_state.py'] = sha256_file(Path(__file__).resolve())
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    d, hamiltonian, residual, frozen_digests = gate_training._build_per_object_residual(parent, device=device)
    set_trainable(residual, True)
    r_zero_digest = module_digest(residual)
    wd = float(config['optimization']['weight_decay'])
    if wd < 0.0:
        raise ValueError('--weight-decay must be nonnegative')
    registered_lr = copy.deepcopy(REGISTERED_RESIDUAL_LR)
    if registered_lr is None or set(registered_lr) != {'maximum', 'minimum', 'warmup_updates'}:
        raise ValueError('Residual learning-rate schedule is missing or invalid')
    maximum = float(registered_lr['maximum'])
    minimum = float(registered_lr['minimum'])
    warmup = int(registered_lr['warmup_updates'])
    opt = torch.optim.AdamW(residual.parameters(), lr=maximum, weight_decay=wd)
    r_initial_digest = module_digest(residual)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.mkdir(parents=True)
    gate_training.stage_a._atomic_json(output / 'registration.json', registration)
    gate_training.stage_a._atomic_json(output / 'parent_contract.json', parent_contract)
    gate_training.stage_a._atomic_json(output / 'source_manifest.json', source_manifest)
    counts = {'total': 0, 'n8': 0, 'n12': 0, 'n20': 0}
    expected = residual_training._expected_counts(registration)
    training = registration['training']
    blocks = _carrier_blocks(expected, int(training['refresh_every_updates']))
    schedule_seed = int(training['sampling_seed']) ^ 2781030144
    schedule_rng = random.Random(schedule_seed)
    schedule_rng.shuffle(blocks)
    schedule_digest = residual_training._canonical_hash({'blocks': blocks})
    gate_training.stage_a._atomic_json(output / 'preflight.json', {'schema': gate_training.SCHEMA, 'role': gate_training.ROLE, 'registration': str(args.registration.resolve()), 'config_sha256': config_sha256, 'r_initial_digest': r_initial_digest, 'r_zero_digest': r_zero_digest, 'frozen_digests': frozen_digests, 'random_n_blocks': blocks, 'random_n_schedule_sha256': schedule_digest, 'registered_residual_learning_rate': copy.deepcopy(REGISTERED_RESIDUAL_LR), 'carrier_observer': None if CARRIER_OBSERVER is None else type(CARRIER_OBSERVER).__name__})
    gate_training._set_marker(output, 'RUNNING', f"GateTraining stability r training ({expected['total']} updates) running\n")
    train = gate_training.stage_a._load_train_cache(config, device=device)
    attr_scale = gate_training.stage_a._attribute_scale(train.attrs).to(device=device, dtype=torch.float32)
    state_scale = parent['state_scale'].to(device=device, dtype=torch.float32)
    if not torch.equal(attr_scale, parent['attr_scale'].to(device=device, dtype=torch.float32)):
        raise ValueError('attribute scale drift')
    stream, rngs = residual_training._new_sampling(train_size=train.size, registration=registration, device=device)
    history = output / 'history.jsonl'
    history_rows = 0
    started = time.perf_counter()
    if not (0.0 < minimum < maximum and 0 <= warmup < int(expected['total'])):
        raise ValueError('Residual learning-rate schedule values are invalid')
    residual_plan = {'learning_rate': {'maximum': maximum, 'minimum': minimum, 'warmup_updates': warmup}}
    lr_values = [gate_training.stage_a._lr(residual_plan, update=i, total=expected['total']) for i in range(1, expected['total'] + 1)]
    checkpoint_updates = {max(1, expected['total'] // 4), max(1, expected['total'] // 2), expected['total']}
    try:
        with history.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps({'phase': 'random_n_schedule', 'blocks': blocks, 'schedule_sha256': schedule_digest}, sort_keys=True) + '\n')
            history_rows += 1
            for block_index, (n, block_len) in enumerate(blocks):
                carrier = None
                payload = None
                updates_on_carrier = int(training['refresh_every_updates'])
                for _ in range(block_len):
                    _set_frozen_experts_eval(d, hamiltonian)
                    residual.eval()
                    collect_new = carrier is None or updates_on_carrier >= int(training['refresh_every_updates'])
                    stream_state = stream.state_dict() if collect_new else None
                    carrier, x0, target_state, attrs, physical_time = gate_training._collect_per_object_external_carrier(config=config, contract=train_contract, d=d, hamiltonian=hamiltonian, residual=residual, train=train, stream=stream, state_scale=state_scale, attr_scale=attr_scale, num_steps=n, batch_size=int(training['carrier_batch_size']), source_rng=rngs['source'], reset_tau_rng=rngs['reset_tau'], reset_rng=rngs['reset'], device=device) if collect_new else (carrier, *payload)
                    payload = (x0, target_state, attrs, physical_time)
                    if collect_new:
                        updates_on_carrier = 0
                        if CARRIER_OBSERVER is not None:
                            assert stream_state is not None
                            CARRIER_OBSERVER.observe(carrier=carrier, x0=x0, target=target_state, attrs=attrs, physical_time=physical_time, source_indices=_preview_stream_indices(stream_state), residual=residual, state_scale=state_scale, q_dim=int(config['dataset']['q_dim']))
                        handle.write(json.dumps({'phase': 'external_r_carrier', 'n': n, 'block_index': block_index, 'accepted_fields': int(carrier.accepted_fields), 'cold_left_intervals': list(carrier.cold_left_intervals), 'pure_rows': int(carrier.trace.pure_rows), 'reset_edges': int(carrier.trace.reset_edges), 'gate_forwarded': False, 'schedule_sha256': schedule_digest, 'sampling_after_collection': residual_training._sampling_digest(stream, rngs), 'collector_wall_seconds': float(carrier.wall_seconds)}, sort_keys=True) + '\n')
                        history_rows += 1
                    counts['total'] += 1
                    counts[f'n{n}'] += 1
                    lr = gate_training.stage_a._lr(residual_plan, update=counts['total'], total=expected['total'])
                    residual_training.set_optimizer_learning_rate(opt, lr)
                    residual.train()
                    _clear_frozen_expert_gradients(d, hamiltonian)
                    detail = gate_training.joint_residual_joint_residual_update(residual=residual, optimizer=opt, carrier=carrier, update_index=counts['total'], x0=payload[0], attrs=payload[2], physical_time=payload[3], target=payload[1], state_scale=state_scale, q_dim=int(config['dataset']['q_dim']), grad_clip=float(config['optimization']['grad_clip']))
                    if not _frozen_expert_gradients_absent(d, hamiltonian) or _frozen_expert_digests(d, hamiltonian) != frozen_digests:
                        raise AssertionError('frozen D/H changed')
                    updates_on_carrier += 1
                    handle.write(json.dumps({'phase': 'full_qp_no_regret_r', 'update': counts['total'], 'n': n, 'block_index': block_index, 'updates_on_carrier': updates_on_carrier, 'lr': float(opt.param_groups[0]['lr']), 'lr_schedule': 'explicit_full_cosine', 'lr_max_override': None, 'lr_min_override': None, 'warmup_updates_override': None, 'grad_clip_override': None, 'weight_decay_override': None, 'counts': dict(counts), 'gate_forwarded': False, **detail}, sort_keys=True) + '\n')
                    history_rows += 1
                if counts['total'] in checkpoint_updates:
                    ck = residual_training._checkpoint(output=output, total=counts['total'], config=config, config_sha256=config_sha256, parent_contract=parent_contract, source_manifest=source_manifest, residual=residual, optimizer=opt, stream=stream, rngs=rngs, counts=counts, frozen_digests=frozen_digests, history_rows=history_rows)
                    handle.write(json.dumps({'phase': 'checkpoint', 'path': ck.name, 'counts': dict(counts)}, sort_keys=True) + '\n')
                    history_rows += 1
        if counts != expected:
            raise AssertionError(f'ledger mismatch {counts} != {expected}')
        if CARRIER_OBSERVER is not None:
            CARRIER_OBSERVER.finalize(output=output, residual=residual, state_scale=state_scale, attr_scale=attr_scale, config=config, counts=counts)
        torch.cuda.synchronize(device)
        terminal = {'schema': gate_training.SCHEMA, 'role': gate_training.ROLE, 'status': 'PASS', 'counts': counts, 'parent_contract': parent_contract, 'frozen_digests': frozen_digests, 'r_initial_digest': r_initial_digest, 'r_terminal_digest': module_digest(residual), 'optimizer_digest': gate_training.stage_a._state_digest(opt.state_dict()), 'gate_constructed': False, 'gate_forwarded': False, 'gate_trained': False, 'training_split': 'parent_train_only', 'training_split_only': True, 'random_n_schedule_sha256': schedule_digest, 'lr_schedule': 'explicit_full_cosine', 'registered_residual_learning_rate': registered_lr, 'lr_sum': float(sum(lr_values)), 'lr_first': float(lr_values[0]), 'lr_peak': float(max(lr_values)), 'lr_last': float(lr_values[-1]), 'segment_updates': int(expected['total']), 'effective_total_updates': int(expected['total']), 'wall_seconds': time.perf_counter() - started}
        gate_training.stage_a._atomic_torch(output / 'r_terminal.pt', {'schema': gate_training.SCHEMA, 'role': gate_training.ROLE, 'config': config, 'config_sha256': config_sha256, 'parent_contract': parent_contract, 'source_manifest': source_manifest, 'r_state_dict': residual.state_dict(), 'terminal': terminal})
        gate_training.stage_a._atomic_json(output / 'terminal_summary.json', terminal)
        gate_training._set_marker(output, 'COMPLETE', f"GateTraining stability r terminal ({expected['total']} updates) complete\n")
        print(json.dumps(terminal, ensure_ascii=False))
    except Exception:
        gate_training._set_marker(output, 'FAILED', f"GateTraining stability r training ({expected['total']} updates) failed\n")
        raise
