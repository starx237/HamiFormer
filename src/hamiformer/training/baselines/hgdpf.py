from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Iterator
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = project_root()
from hamiformer.baselines.hgdpf import HGDPFConfig, HGDPFPerceiver, SeparableHamiltonianNetwork, cosine_alpha_bar, hnn_derivative_loss
from hamiformer.baselines.hgdpf_hamiballs import compute_attribute_normalization, compute_state_normalization, dpf_noise_prediction_loss, flatten_hamiballs_attributes, hamiballs_hgdpf_config, make_dpf_training_batch, pack_hamiballs_phase, published_parameter_ledger, scaled_query_cardinalities
from hamiformer.data.dataset import PhaseWindowDataset, collate_phase_windows
from hamiformer.utils import sha256_file
SCHEMA = 'hamiformer.baseline.hgdpf.hamiballs.training.v1'
EXPECTED_TRAIN_WINDOWS = 57344
SOURCE_TRAJECTORIES = 14336
FUTURE_EDGES = 48
NUM_OBJECTS = 5
Q_DIM_PER_OBJECT = 2
ATTR_DIM_PER_OBJECT = 3
STATE_DIM = NUM_OBJECTS * 2 * Q_DIM_PER_OBJECT
CANONICAL_Q_DIM = NUM_OBJECTS * Q_DIM_PER_OBJECT
ATTRIBUTE_DIM = NUM_OBJECTS * ATTR_DIM_PER_OBJECT
ACTION_DIM = CANONICAL_Q_DIM
WIDE_D_PARAMETERS = 1173068

@dataclass(frozen=True)
class TrainingContract:
    budget_mode: str = 'paper_epochs'
    parameter_mode: str = 'paper'
    dpf_target_updates: int | None = None
    hnn_target_updates: int | None = None
    master_seed: int = 42
    dpf_epochs: int = 3000
    dpf_batch_size: int = 128
    dpf_learning_rate: float = 0.0001
    dpf_min_learning_rate: float = 0.0
    dpf_warmup_updates: int = 1000
    hnn_epochs: int = 1000
    hnn_batch_size: int = 8192
    hnn_source_chunk: int = 256
    hnn_learning_rate: float = 0.0003
    hnn_min_learning_rate: float = 1e-06
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    diffusion_steps: int = 1000
    checkpoint_every_epochs: int = 25
    immutable_every_epochs: int = 100

def _derived_seed(master_seed: int, namespace: str) -> int:
    payload = f'hgdpf-v1|master={int(master_seed)}|stream={namespace}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & (1 << 63) - 1

def _seed_ledger(contract: TrainingContract) -> dict[str, object]:
    fixed = {name: _derived_seed(contract.master_seed, name) for name in ('dpf_init', 'hnn_init', 'stats_loader')}
    return {'master_seed': contract.master_seed, 'derivation': "uint64_be(SHA256('hgdpf-v1|master=<seed>|stream=<namespace>')[:8]) & (2^63-1)", 'fixed_streams': fixed, 'epoch_streams': {'dpf_order': 'namespace dpf_order_epoch_<zero-based epoch>', 'dpf_noise': 'namespace dpf_noise_epoch_<zero-based epoch>', 'hnn_order': 'namespace hnn_order_epoch_<zero-based epoch>', 'hnn_within_chunk': 'namespace hnn_within_epoch_<zero-based epoch>'}}

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)

def _complete_phase(batch) -> torch.Tensor:
    return torch.cat([batch.x0[:, None], batch.future], dim=1)

def _dataset(root: Path, *, verify_content_hash: bool) -> PhaseWindowDataset:
    dataset = PhaseWindowDataset(root / 'manifests' / 'train.jsonl', num_objects=NUM_OBJECTS, future_steps=FUTURE_EDGES, q_dim=Q_DIM_PER_OBJECT, attr_dim=ATTR_DIM_PER_OBJECT, verify_content_hash=verify_content_hash)
    if len(dataset) != EXPECTED_TRAIN_WINDOWS:
        raise ValueError(f'formal HG-DPF requires {EXPECTED_TRAIN_WINDOWS} stride-48 train windows, got {len(dataset)}')
    return dataset

def _loader(dataset: PhaseWindowDataset, *, batch_size: int, shuffle_seed: int | None) -> DataLoader:
    generator = None
    if shuffle_seed is not None:
        generator = torch.Generator().manual_seed(int(shuffle_seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_seed is not None, generator=generator, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collate_phase_windows)

def _model_config(contract: TrainingContract) -> HGDPFConfig:
    return hamiballs_hgdpf_config(contract.parameter_mode)

def _construct_models(contract: TrainingContract) -> tuple[HGDPFPerceiver, SeparableHamiltonianNetwork]:
    with torch.random.fork_rng():
        torch.manual_seed(_derived_seed(contract.master_seed, 'dpf_init'))
        dpf = HGDPFPerceiver(_model_config(contract))
    with torch.random.fork_rng():
        torch.manual_seed(_derived_seed(contract.master_seed, 'hnn_init'))
        hnn = SeparableHamiltonianNetwork(CANONICAL_Q_DIM, attribute_dim=ATTRIBUTE_DIM, width=248 if contract.parameter_mode == 'matched_wide_d' else 1024, hidden_layers=4)
    return (dpf, hnn)

def _normalization(dataset: PhaseWindowDataset, output: Path, contract: TrainingContract) -> dict[str, torch.Tensor]:
    path = output / 'normalization.pt'
    if path.is_file():
        raw = torch.load(path, map_location='cpu', weights_only=True)
        required = {'state_mean', 'state_scale', 'attribute_mean', 'attribute_scale'}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError('HG-DPF normalization checkpoint has an invalid schema')
        return raw
    state_rows: list[torch.Tensor] = []
    attribute_rows: list[torch.Tensor] = []
    loader = _loader(dataset, batch_size=128, shuffle_seed=_derived_seed(contract.master_seed, 'stats_loader'))
    for batch in loader:
        state_rows.append(pack_hamiballs_phase(_complete_phase(batch)).reshape(-1, STATE_DIM))
        attribute_rows.append(flatten_hamiballs_attributes(batch.attrs))
    states = torch.cat(state_rows, dim=0)
    attributes = torch.cat(attribute_rows, dim=0)
    state_mean, state_scale = compute_state_normalization(states[:, None])
    attribute_mean, attribute_scale = compute_attribute_normalization(attributes)
    result = {'state_mean': state_mean, 'state_scale': state_scale, 'attribute_mean': attribute_mean, 'attribute_scale': attribute_scale}
    _atomic_torch(path, result)
    return result

def _learning_rate(completed_updates: int, *, total_updates: int, warmup_updates: int, maximum: float, minimum: float) -> float:
    next_update = completed_updates + 1
    if warmup_updates > 0 and next_update <= warmup_updates:
        return maximum * next_update / warmup_updates
    denominator = max(1, total_updates - warmup_updates)
    progress = min(1.0, max(0.0, (next_update - warmup_updates) / denominator))
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))

def _set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = float(learning_rate)

def _gradient_step(*, loss: torch.Tensor, model: torch.nn.Module, optimizer: torch.optim.Optimizer, clip_norm: float, warning_log: Path, context: dict[str, object]) -> tuple[bool, float | None]:
    if not bool(torch.isfinite(loss)):
        with warning_log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({**context, 'kind': 'nonfinite_loss'}) + '\n')
        print(json.dumps({'warning': 'nonfinite_loss', **context}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        return (False, None)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=False)
    if not bool(torch.isfinite(norm)):
        with warning_log.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({**context, 'kind': 'nonfinite_gradient'}) + '\n')
        print(json.dumps({'warning': 'nonfinite_gradient', **context}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        return (False, float('nan'))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return (True, float(norm.detach().cpu()))

class _StopRequest:
    requested = False

def _install_signal_handlers() -> None:

    def request_stop(signum, _frame) -> None:
        _StopRequest.requested = True
        print(json.dumps({'warning': 'signal_received', 'signal': int(signum)}), flush=True)
    for name in ('SIGINT', 'SIGTERM'):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, request_stop)

def _checkpoint_payload(*, stage: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, batch_in_epoch: int, completed_updates: int, skipped_updates: int, contract: TrainingContract, noise_generator: torch.Generator | None=None) -> dict[str, object]:
    return {'schema': SCHEMA, 'stage': stage, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state': {'completed_updates': completed_updates, 'implementation': 'stateless linear-warmup/cosine function in train_hamiballs_hgdpf.py'}, 'epoch': epoch, 'batch_in_epoch': batch_in_epoch, 'completed_updates': completed_updates, 'skipped_updates': skipped_updates, 'contract': asdict(contract), 'cpu_rng_state': torch.get_rng_state(), 'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'noise_generator_state': None if noise_generator is None else noise_generator.get_state()}

def _save_checkpoint(output: Path, stage: str, payload: dict[str, object], *, immutable: bool) -> None:
    _atomic_torch(output / f'{stage}_latest.pt', payload)
    if immutable:
        epoch = int(payload['epoch'])
        target = output / 'checkpoints' / f'{stage}_epoch{epoch:04d}.pt'
        if not target.exists():
            _atomic_torch(target, payload)

def _resume(path: Path, *, stage: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, contract: TrainingContract) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location='cpu', weights_only=False)
    saved_contract = payload.get('contract') if isinstance(payload, dict) else None
    current_contract = asdict(contract)
    resume_irrelevant = {'checkpoint_every_epochs', 'immutable_every_epochs'}
    comparable_saved = {key: value for key, value in saved_contract.items() if key not in resume_irrelevant} if isinstance(saved_contract, dict) else None
    comparable_current = {key: value for key, value in current_contract.items() if key not in resume_irrelevant}
    if not isinstance(payload, dict) or payload.get('schema') != SCHEMA or payload.get('stage') != stage or (comparable_saved != comparable_current):
        raise ValueError(f'incompatible {stage} resume checkpoint: {path}')
    model.load_state_dict(payload['model_state_dict'])
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    if 'cpu_rng_state' in payload:
        torch.set_rng_state(payload['cpu_rng_state'])
    if torch.cuda.is_available() and payload.get('cuda_rng_state_all'):
        torch.cuda.set_rng_state_all(payload['cuda_rng_state_all'])
    return payload

def train_dpf(*, dataset: PhaseWindowDataset, model: HGDPFPerceiver, normalization: dict[str, torch.Tensor], output: Path, device: torch.device, contract: TrainingContract) -> None:
    if (output / 'DPF_COMPLETE').is_file():
        print(json.dumps({'stage': 'dpf', 'status': 'already_complete'}), flush=True)
        return
    model.to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract.dpf_learning_rate, weight_decay=contract.weight_decay)
    resume = _resume(output / 'dpf_latest.pt', stage='dpf', model=model, optimizer=optimizer, contract=contract)
    start_epoch = int(resume['epoch']) if resume else 0
    start_batch = int(resume['batch_in_epoch']) if resume else 0
    completed_updates = int(resume['completed_updates']) if resume else 0
    skipped_updates = int(resume['skipped_updates']) if resume else 0
    batches_per_epoch = math.ceil(len(dataset) / contract.dpf_batch_size)
    total_updates = int(contract.dpf_target_updates) if contract.dpf_target_updates is not None else batches_per_epoch * contract.dpf_epochs
    alpha_bar = cosine_alpha_bar(contract.diffusion_steps)
    cardinalities = scaled_query_cardinalities(FUTURE_EDGES)
    warning_log = output / 'warnings.jsonl'
    stage_started = time.perf_counter()
    epoch = start_epoch
    while completed_updates < total_updates:
        loader = _loader(dataset, batch_size=contract.dpf_batch_size, shuffle_seed=_derived_seed(contract.master_seed, f'dpf_order_epoch_{epoch}'))
        noise_generator = torch.Generator(device=device).manual_seed(_derived_seed(contract.master_seed, f'dpf_noise_epoch_{epoch}'))
        first_batch = start_batch if epoch == start_epoch else 0
        if resume and epoch == start_epoch and (resume.get('noise_generator_state') is not None):
            noise_generator.set_state(resume['noise_generator_state'])
        for batch_index, batch in enumerate(loader):
            if batch_index < first_batch:
                continue
            physical = pack_hamiballs_phase(_complete_phase(batch)).to(device=device, dtype=torch.float32, non_blocking=True)
            times = batch.time.to(device=device, dtype=torch.float32, non_blocking=True)
            attributes = flatten_hamiballs_attributes(batch.attrs).to(device=device, dtype=torch.float32, non_blocking=True)
            cardinality_index = int(torch.randint(0, len(cardinalities), (), device=device, generator=noise_generator).item())
            training_batch = make_dpf_training_batch(model=model, phase=physical, times=times, attributes=attributes, state_mean=normalization['state_mean'], state_scale=normalization['state_scale'], attribute_mean=normalization['attribute_mean'], attribute_scale=normalization['attribute_scale'], alpha_bar=alpha_bar, query_points=cardinalities[cardinality_index], generator=noise_generator)
            lr = _learning_rate(completed_updates, total_updates=total_updates, warmup_updates=contract.dpf_warmup_updates, maximum=contract.dpf_learning_rate, minimum=contract.dpf_min_learning_rate)
            _set_lr(optimizer, lr)
            loss = dpf_noise_prediction_loss(model, training_batch)
            accepted, gradient_norm = _gradient_step(loss=loss, model=model, optimizer=optimizer, clip_norm=contract.gradient_clip_norm, warning_log=warning_log, context={'stage': 'dpf', 'epoch': epoch, 'batch': batch_index})
            completed_updates += int(accepted)
            skipped_updates += int(not accepted)
            if completed_updates % 20 == 0 or batch_index + 1 == batches_per_epoch:
                print(json.dumps({'stage': 'dpf', 'epoch': epoch + 1, 'batch': batch_index + 1, 'updates': completed_updates, 'loss': float(loss.detach().cpu()), 'gradient_norm': gradient_norm, 'lr': lr, 'elapsed_seconds': time.perf_counter() - stage_started}), flush=True)
            if _StopRequest.requested:
                payload = _checkpoint_payload(stage='dpf', model=model, optimizer=optimizer, epoch=epoch, batch_in_epoch=batch_index + 1, completed_updates=completed_updates, skipped_updates=skipped_updates, contract=contract, noise_generator=noise_generator)
                _save_checkpoint(output, 'dpf', payload, immutable=False)
                print(json.dumps({'stage': 'dpf', 'status': 'paused_after_signal'}), flush=True)
                return
            if completed_updates >= total_updates:
                payload = _checkpoint_payload(stage='dpf', model=model, optimizer=optimizer, epoch=epoch, batch_in_epoch=batch_index + 1, completed_updates=completed_updates, skipped_updates=skipped_updates, contract=contract, noise_generator=noise_generator)
                _save_checkpoint(output, 'dpf', payload, immutable=True)
                (output / 'DPF_COMPLETE').write_text('COMPLETE\n', encoding='utf-8')
                return
        start_batch = 0
        resume = None
        epoch += 1
        payload = _checkpoint_payload(stage='dpf', model=model, optimizer=optimizer, epoch=epoch, batch_in_epoch=0, completed_updates=completed_updates, skipped_updates=skipped_updates, contract=contract)
        if epoch % contract.checkpoint_every_epochs == 0:
            _save_checkpoint(output, 'dpf', payload, immutable=epoch % contract.immutable_every_epochs == 0)

def _hnn_batches(dataset: PhaseWindowDataset, *, epoch: int, contract: TrainingContract, normalization: dict[str, torch.Tensor]) -> Iterator[dict[str, torch.Tensor]]:
    loader = _loader(dataset, batch_size=contract.hnn_source_chunk, shuffle_seed=_derived_seed(contract.master_seed, f'hnn_order_epoch_{epoch}'))
    within = torch.Generator().manual_seed(_derived_seed(contract.master_seed, f'hnn_within_epoch_{epoch}'))
    buffer: dict[str, torch.Tensor] = {}
    for source_batch in loader:
        phase = pack_hamiballs_phase(_complete_phase(source_batch)).to(torch.float32)
        times = source_batch.time.to(torch.float32)
        attributes = (flatten_hamiballs_attributes(source_batch.attrs).to(torch.float32) - normalization['attribute_mean']) / normalization['attribute_scale']
        p = phase[..., :CANONICAL_Q_DIM]
        q = phase[..., CANONICAL_Q_DIM:]
        dt = (times[:, 2:] - times[:, :-2]).unsqueeze(-1)
        rows = {'p': p[:, 1:-1].reshape(-1, CANONICAL_Q_DIM), 'q': q[:, 1:-1].reshape(-1, CANONICAL_Q_DIM), 'p_dot': ((p[:, 2:] - p[:, :-2]) / dt).reshape(-1, CANONICAL_Q_DIM), 'q_dot': ((q[:, 2:] - q[:, :-2]) / dt).reshape(-1, CANONICAL_Q_DIM), 'attributes': attributes[:, None].expand(-1, phase.shape[1] - 2, -1).reshape(-1, ATTRIBUTE_DIM)}
        order = torch.randperm(rows['p'].shape[0], generator=within)
        rows = {name: value[order] for name, value in rows.items()}
        if buffer:
            rows = {name: torch.cat([buffer[name], value], dim=0) for name, value in rows.items()}
        offset = 0
        while rows['p'].shape[0] - offset >= contract.hnn_batch_size:
            right = offset + contract.hnn_batch_size
            yield {name: value[offset:right] for name, value in rows.items()}
            offset = right
        buffer = {name: value[offset:] for name, value in rows.items()}
    if buffer and buffer['p'].shape[0] > 0:
        yield buffer

def train_hnn(*, dataset: PhaseWindowDataset, model: SeparableHamiltonianNetwork, normalization: dict[str, torch.Tensor], output: Path, device: torch.device, contract: TrainingContract) -> None:
    if (output / 'HNN_COMPLETE').is_file():
        print(json.dumps({'stage': 'hnn', 'status': 'already_complete'}), flush=True)
        return
    model.to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=contract.hnn_learning_rate, weight_decay=contract.weight_decay)
    resume = _resume(output / 'hnn_latest.pt', stage='hnn', model=model, optimizer=optimizer, contract=contract)
    start_epoch = int(resume['epoch']) if resume else 0
    start_batch = int(resume['batch_in_epoch']) if resume else 0
    completed_updates = int(resume['completed_updates']) if resume else 0
    skipped_updates = int(resume['skipped_updates']) if resume else 0
    transitions_per_epoch = len(dataset) * (FUTURE_EDGES - 1)
    batches_per_epoch = math.ceil(transitions_per_epoch / contract.hnn_batch_size)
    total_updates = int(contract.hnn_target_updates) if contract.hnn_target_updates is not None else batches_per_epoch * contract.hnn_epochs
    warning_log = output / 'warnings.jsonl'
    stage_started = time.perf_counter()
    epoch = start_epoch
    while completed_updates < total_updates:
        first_batch = start_batch if epoch == start_epoch else 0
        for batch_index, rows in enumerate(_hnn_batches(dataset, epoch=epoch, contract=contract, normalization=normalization)):
            if batch_index < first_batch:
                continue
            rows = {name: value.to(device=device, dtype=torch.float32, non_blocking=True) for name, value in rows.items()}
            lr = _learning_rate(completed_updates, total_updates=total_updates, warmup_updates=0, maximum=contract.hnn_learning_rate, minimum=contract.hnn_min_learning_rate)
            _set_lr(optimizer, lr)
            loss = hnn_derivative_loss(model, p=rows['p'], q=rows['q'], p_dot_target=rows['p_dot'], q_dot_target=rows['q_dot'], attributes=rows['attributes'])
            accepted, gradient_norm = _gradient_step(loss=loss, model=model, optimizer=optimizer, clip_norm=contract.gradient_clip_norm, warning_log=warning_log, context={'stage': 'hnn', 'epoch': epoch, 'batch': batch_index})
            completed_updates += int(accepted)
            skipped_updates += int(not accepted)
            if completed_updates % 20 == 0 or batch_index + 1 == batches_per_epoch:
                print(json.dumps({'stage': 'hnn', 'epoch': epoch + 1, 'batch': batch_index + 1, 'updates': completed_updates, 'loss': float(loss.detach().cpu()), 'gradient_norm': gradient_norm, 'lr': lr, 'elapsed_seconds': time.perf_counter() - stage_started}), flush=True)
            if _StopRequest.requested:
                payload = _checkpoint_payload(stage='hnn', model=model, optimizer=optimizer, epoch=epoch, batch_in_epoch=batch_index + 1, completed_updates=completed_updates, skipped_updates=skipped_updates, contract=contract)
                _save_checkpoint(output, 'hnn', payload, immutable=False)
                print(json.dumps({'stage': 'hnn', 'status': 'paused_after_signal'}), flush=True)
                return
            if completed_updates >= total_updates:
                payload = _checkpoint_payload(stage='hnn', model=model, optimizer=optimizer, epoch=epoch, batch_in_epoch=batch_index + 1, completed_updates=completed_updates, skipped_updates=skipped_updates, contract=contract)
                _save_checkpoint(output, 'hnn', payload, immutable=True)
                (output / 'HNN_COMPLETE').write_text('COMPLETE\n', encoding='utf-8')
                return
        start_batch = 0
        resume = None
        epoch += 1
        payload = _checkpoint_payload(stage='hnn', model=model, optimizer=optimizer, epoch=epoch, batch_in_epoch=0, completed_updates=completed_updates, skipped_updates=skipped_updates, contract=contract)
        if epoch % contract.checkpoint_every_epochs == 0:
            _save_checkpoint(output, 'hnn', payload, immutable=epoch % contract.immutable_every_epochs == 0)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('dpf', 'hnn', 'all', 'preflight'), default='all')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verify-content-hash', action='store_true')
    parser.add_argument('--budget-mode', choices=('paper_epochs', 'matched_hamiballs'), default='paper_epochs', help='paper_epochs retains the published epoch counts; matched_hamiballs uses the locked Main role budgets: DPF=50k and HNN=40k updates')
    args = parser.parse_args()
    _install_signal_handlers()
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but is unavailable')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = TrainingContract()
    if args.budget_mode == 'matched_hamiballs':
        contract = replace(contract, budget_mode='matched_hamiballs', parameter_mode='matched_wide_d', dpf_target_updates=50000, hnn_target_updates=40000, dpf_batch_size=64, hnn_batch_size=64 * (FUTURE_EDGES - 1), hnn_source_chunk=64)
    dataset_root = args.dataset_root.expanduser().resolve()
    dataset = _dataset(dataset_root, verify_content_hash=args.verify_content_hash)
    dpf, hnn = _construct_models(contract)
    ledger = published_parameter_ledger(dpf, hnn, wide_d_parameters=WIDE_D_PARAMETERS)
    if contract.parameter_mode == 'matched_wide_d' and (not (ledger['guided_total_parameters'] <= WIDE_D_PARAMETERS and WIDE_D_PARAMETERS - ledger['guided_total_parameters'] <= 2000)):
        raise ValueError(f'HG-DPF parameter budget is not matched to wide-D: {ledger}')
    manifest = {'schema': SCHEMA, 'status': 'configured', 'dataset_root': str(dataset_root), 'dataset_manifest': str((dataset_root / 'manifests' / 'train.jsonl').resolve()), 'dataset_windows': len(dataset), 'source_trajectories': SOURCE_TRAJECTORIES, 'future_edges': FUTURE_EDGES, 'paper_state_order': '[p_all,q_all]', 'action_adaptation': 'ten-channel identically-zero torque; action pathway retained', 'attribute_adaptation': '15 visible mass/radius/restitution values condition both DPF and HNN', 'query_cardinalities': list(scaled_query_cardinalities(FUTURE_EDGES)), 'hnn_epoch_semantics': "one optimizer update consumes 64 trajectory windows and all 47 valid central-difference rows per window (3008 transition rows), matching Main-H's 64-window whole-trajectory exposure" if contract.parameter_mode == 'matched_wide_d' else 'one full pass over every central-difference transition in all stride-48 windows with the published nominal B8192', 'optimization_budget_alignment': 'DPF follows the locked D budget of 50000 optimizer updates and HNN follows the locked H budget of 40000 optimizer updates' if contract.budget_mode == 'matched_hamiballs' else 'published literal epoch counts', 'dpf_under_specified_defaults': {'attention_norm_order': 'pre-norm', 'ffn_activation': 'GELU', 'dropout': 0.0, 'weight_decay': contract.weight_decay, 'cosine_end_lr': contract.dpf_min_learning_rate, 'learned_latent_initialization': 'normal std=1/sqrt(hidden_dim)'}, 'contract': asdict(contract), 'seed_ledger': _seed_ledger(contract), 'parameter_ledger': ledger, 'content_hash_verification_at_runtime': bool(args.verify_content_hash)}
    _atomic_json(output / 'manifest.json', manifest)
    normalization = _normalization(dataset, output, contract)
    _atomic_json(output / 'normalization_summary.json', {name: value.tolist() for name, value in normalization.items()})
    print(json.dumps({'status': 'PREFLIGHT_COMPLETE', **ledger}), flush=True)
    if args.stage == 'preflight':
        return
    if args.stage in {'dpf', 'all'}:
        train_dpf(dataset=dataset, model=dpf, normalization=normalization, output=output, device=device, contract=contract)
    if _StopRequest.requested:
        return
    if args.stage in {'hnn', 'all'}:
        train_hnn(dataset=dataset, model=hnn, normalization=normalization, output=output, device=device, contract=contract)
    if (output / 'DPF_COMPLETE').is_file() and (output / 'HNN_COMPLETE').is_file():
        final_manifest = dict(manifest)
        final_manifest['status'] = 'complete'
        final_manifest['dpf_checkpoint_sha256'] = sha256_file(output / 'dpf_latest.pt')
        final_manifest['hnn_checkpoint_sha256'] = sha256_file(output / 'hnn_latest.pt')
        final_manifest['warning_count'] = len((output / 'warnings.jsonl').read_text(encoding='utf-8').splitlines()) if (output / 'warnings.jsonl').is_file() else 0
        _atomic_json(output / 'manifest.json', final_manifest)
        (output / 'COMPLETE').write_text('COMPLETE\n', encoding='utf-8')
if __name__ == '__main__':
    main()
