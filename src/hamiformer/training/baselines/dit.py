from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from hamiformer.data_generation3d.signature import dataset_signature
import torch
import yaml
from torch.utils.data import DataLoader
from hamiformer.data.hamiballs2 import HamiBalls2CropPackDataset, build_hamiballs2_crop_pack
from hamiformer.baselines.dit import HamiBalls2DiT
from hamiformer.baselines.physiformer import parameter_count

def derived_seed(master: int, label: str) -> int:
    digest = hashlib.sha256(f'hamiballs2-wide-d-v1|{master}|{label}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 63 - 1)

def sample_tau(batch: int, *, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    choose = torch.rand(batch, device=device, generator=generator)
    gaussian = torch.randn(batch, device=device, generator=generator)
    logit = torch.sigmoid(-0.8 + 0.8 * gaussian)
    uniform = torch.rand(batch, device=device, generator=generator) * 0.98
    return torch.where(choose < 0.9, logit, uniform).clamp(0.0, 0.98)

def make_pair(clean: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = sample_tau(clean.shape[0], device=clean.device, generator=generator)
    noise = 0.1 * scale.view(1, 1, 1, 6) * torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    valid = mask[:, None, :, None].to(clean)
    noise = noise * valid
    tau_view = tau[:, None, None, None]
    noisy = (tau_view * clean + (1.0 - tau_view) * noise) * valid
    target_velocity = clean - noise
    return (tau, noisy, target_velocity)

def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

def model_forward(model: HamiBalls2DiT, batch: dict[str, torch.Tensor], noisy: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return model(noisy, tau, x0=batch['phase'][:, 0], attrs=batch['attrs'], physical_time=batch['time'], object_mask=batch['object_mask'].bool(), spring_mask=batch['spring_mask'], spring_k=batch['spring_k'], spring_rest_length=batch['spring_rest_length'])

def masked_mean(square: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask[:, :, :, None].to(square)
    return (square * weight).sum() / (weight.sum() * square.shape[-1]).clamp_min(1.0)

@torch.no_grad()
def validate(model: HamiBalls2DiT, loader: DataLoader, *, device: torch.device, scale: torch.Tensor, validation_seed: int) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(validation_seed)
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, float] = defaultdict(float)

    def add(name: str, values: torch.Tensor, selection: torch.Tensor) -> None:
        weight = selection[..., None].to(values)
        totals[name] += float((values * weight).sum().cpu())
        counts[name] += float(weight.sum().cpu()) * values.shape[-1]
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        clean = batch['phase'][:, 1:].float()
        object_mask = batch['object_mask'].bool()
        tau, noisy, target_velocity = make_pair(clean, object_mask, scale, generator=generator)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            predicted_clean = model_forward(model, batch, noisy, tau)
        predicted_clean = predicted_clean.float()
        denominator = (1.0 - tau[:, None, None, None]).clamp_min(0.05)
        predicted_velocity = (predicted_clean - noisy) / denominator
        valid = object_mask[:, None, :].expand_as(batch['contact'].bool())
        contact = batch['contact'].bool() & valid
        continuous = ~batch['contact'].bool() & valid
        clean_scaled = ((predicted_clean - clean) / scale.view(1, 1, 1, 6)).square()
        velocity_scaled = ((predicted_velocity - target_velocity) / scale.view(1, 1, 1, 6)).square()
        add('clean/all', clean_scaled, valid)
        add('clean/contact', clean_scaled, contact)
        add('clean/continuous', clean_scaled, continuous)
        add('velocity/all', velocity_scaled, valid)
        for seed in (40, 41, 42, 43):
            chosen = batch['physical_seed'] == seed
            if bool(chosen.any()):
                add(f'clean/seed{seed}', clean_scaled[chosen], valid[chosen])
        add('clean_q/all', clean_scaled[..., :3], valid)
        add('clean_p/all', clean_scaled[..., 3:], valid)
        add('clean_q/contact', clean_scaled[..., :3], contact)
        add('clean_p/contact', clean_scaled[..., 3:], contact)
        add('clean_q/continuous', clean_scaled[..., :3], continuous)
        add('clean_p/continuous', clean_scaled[..., 3:], continuous)
    return {name: totals[name] / max(counts[name], 1.0) for name in sorted(totals)}

def update_ema(model: torch.nn.Module, ema: dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].lerp_(value.detach(), 1.0 - decay)
            else:
                ema[name].copy_(value)

def learning_rate(step: int, total: int) -> float:
    if step <= 1000:
        return 0.0003 * step / 1000
    progress = (step - 1000) / (total - 1000)
    return 3e-07 + 0.5 * (0.0003 - 3e-07) * (1.0 + math.cos(math.pi * progress))

def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-steps', type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    dataset_root = Path(cfg['dataset_root'])
    audit = json.loads((dataset_root / 'audit/full_audit.json').read_text(encoding='utf-8'))
    metadata = json.loads((dataset_root / 'metadata/dataset_config.json').read_text(encoding='utf-8'))
    if audit.get('full_validation') is not True or metadata.get('status') != 'finalized':
        raise RuntimeError('HamiBalls-2 dataset is not finalized and fully audited')
    if metadata.get('semantic_hash') != dataset_signature(cfg):
        raise RuntimeError('dataset identity differs from configuration')
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    train_pack = build_hamiballs2_crop_pack(dataset_root / 'manifests/train.jsonl', cfg['train_pack'], include_contact_labels=False)
    val_pack = build_hamiballs2_crop_pack(dataset_root / 'manifests/validation.jsonl', cfg['validation_pack'], include_contact_labels=True)
    master_seed = int(cfg['master_seed'])
    random.seed(master_seed)
    np.random.seed(master_seed)
    torch.manual_seed(master_seed)
    torch.cuda.manual_seed_all(master_seed)
    device = torch.device('cuda')
    model = HamiBalls2DiT(**cfg['model']).to(device)
    if cfg.get('expected_parameters') != parameter_count(model):
        raise RuntimeError('DiT parameter count does not match frozen contract')
    params = parameter_count(model)
    if not 5000000 <= params <= 6000000:
        raise RuntimeError(f'wide-D parameter count {params} is outside configured 5--6M range')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001, fused=True)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    step = 0
    target_steps = int(args.max_steps or cfg['steps'])
    latest = output / 'latest.pt'
    if latest.is_file():
        raise RuntimeError('Refusing implicit resume: use a fresh output directory; exact data-stream resume requires explicit implementation.')
    train_generator = torch.Generator(device=device).manual_seed(derived_seed(master_seed, 'train-rf'))
    shuffle_generator = torch.Generator().manual_seed(derived_seed(master_seed, 'train-order'))
    train_loader = DataLoader(HamiBalls2CropPackDataset(train_pack), batch_size=64, shuffle=True, generator=shuffle_generator, num_workers=int(cfg['num_workers']), pin_memory=True, persistent_workers=int(cfg['num_workers']) > 0, drop_last=True)
    val_loader = DataLoader(HamiBalls2CropPackDataset(val_pack), batch_size=64, shuffle=False, num_workers=int(cfg['num_workers']), pin_memory=True, persistent_workers=int(cfg['num_workers']) > 0)
    stats = json.loads((dataset_root / 'stats/phase_scales.json').read_text(encoding='utf-8'))
    scale = torch.tensor(stats['phase_rms'], device=device, dtype=torch.float32)
    validation_seed = derived_seed(master_seed, 'validation-rf')
    seed_record = {'scheme': "sha256('hamiballs2-wide-d-v1|<master>|<label>')[:8] mod (2^63-1)", 'master_seed': master_seed, 'train_rf_seed': derived_seed(master_seed, 'train-rf'), 'train_order_seed': derived_seed(master_seed, 'train-order'), 'validation_rf_seed': validation_seed}
    (output / 'run_contract.json').write_text(json.dumps({'schema': 'hamiformer.hamiballs2.dit.train.v1', 'config': cfg, 'parameters': params, 'seeds': seed_record, 'architecture': 'full spatiotemporal adaLN-Zero DiT; matched PF-RF clean prediction', 'reference_wide_d_parameters': 5615598, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    metrics_path = output / 'metrics.jsonl'
    start_time = time.perf_counter()
    running_loss = 0.0
    running_scaled_loss = 0.0
    running_count = 0

    def save_and_validate() -> None:
        nonlocal running_loss, running_scaled_loss, running_count
        online_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        model.load_state_dict(ema)
        validation = validate(model, val_loader, device=device, scale=scale, validation_seed=validation_seed)
        model.load_state_dict(online_state)
        record = {'step': step, 'elapsed_seconds': time.perf_counter() - start_time, 'train_velocity_raw_mse_recent': running_loss / max(running_count, 1), 'train_velocity_scaled_mse_recent': running_scaled_loss / max(running_count, 1), 'validation_ema': validation}
        with metrics_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)
        atomic_save({'step': step, 'model': model.state_dict(), 'ema': {k: v.cpu() for k, v in ema.items()}, 'optimizer': optimizer.state_dict(), 'contract': seed_record, 'train_rf_generator_state': train_generator.get_state().cpu(), 'train_order_generator_state': shuffle_generator.get_state()}, latest)
        running_loss = 0.0
        running_scaled_loss = 0.0
        running_count = 0
    if step == 0:
        save_and_validate()
    while step < target_steps:
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            clean = batch['phase'][:, 1:].float()
            object_mask = batch['object_mask'].bool()
            tau, noisy, target_velocity = make_pair(clean, object_mask, scale, generator=train_generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted_clean = model_forward(model, batch, noisy, tau)
                predicted_velocity = (predicted_clean - noisy) / (1.0 - tau[:, None, None, None]).clamp_min(0.05)
                valid = object_mask[:, None, :].expand(clean.shape[:3])
                loss = masked_mean((predicted_velocity - target_velocity).square(), valid)
                scaled_loss = masked_mean(((predicted_velocity - target_velocity) / scale.view(1, 1, 1, 6)).square(), valid)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('non-finite wide-D loss')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError('non-finite wide-D gradient')
            step += 1
            lr = learning_rate(step, int(cfg['steps']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            update_ema(model, ema, 0.9999)
            running_loss += float(loss.detach().cpu())
            running_scaled_loss += float(scaled_loss.detach().cpu())
            running_count += 1
            if step % 100 == 0:
                print(json.dumps({'step': step, 'loss': running_loss / running_count, 'lr': lr, 'elapsed_seconds': time.perf_counter() - start_time}), flush=True)
            if step % int(cfg['validation_every']) == 0 or step == target_steps:
                save_and_validate()
            if step >= target_steps:
                break
    atomic_save(torch.load(latest, map_location='cpu', weights_only=False), output / f'step_{step:06d}.pt')
    (output / 'completed.json').write_text(json.dumps({'step': step, 'parameters': params, 'elapsed_seconds': time.perf_counter() - start_time}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
