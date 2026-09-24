from hamiformer.utils.paths import project_root
import argparse
import json
import random
from pathlib import Path
import sys
import numpy as np
from hamiformer.data_generation3d.signature import dataset_signature
import torch
import yaml
ROOT = project_root()
from hamiformer.training import ExplicitEpochBatchStream
from hamiformer.training.training_schedule import cosine_learning_rate
TOTAL_STEPS = 50000
ENDPOINT_START = 40001

def h_objective(dataset, step):
    return 'type2_endpoint' if dataset == 'h2' and step >= ENDPOINT_START else 'midpoint_field'

def set_joint_schedule(adapter, optimizers, dataset, step, stage, device):
    wanted = h_objective(dataset, step)
    if wanted != stage:
        optimizers['h'] = torch.optim.AdamW(adapter.h.parameters(), lr=1e-05, weight_decay=0.0001, fused=device.type == 'cuda')
    for name in ('d', 'h'):
        lr = 1e-05 if name == 'h' and wanted == 'type2_endpoint' else cosine_learning_rate(step, total_steps=TOTAL_STEPS, warmup_steps=1000, maximum=0.0003 if name == 'd' else 0.0001, minimum=3e-07 if name == 'd' else 1e-06)
        for group in optimizers[name].param_groups:
            group['lr'] = lr
    return wanted

def joint_update(d_loss, h_loss, d, h, d_optimizer, h_optimizer):
    if not bool(torch.isfinite(d_loss) and torch.isfinite(h_loss)):
        raise FloatingPointError('Nonfinite joint loss')
    d_optimizer.zero_grad(set_to_none=True)
    h_optimizer.zero_grad(set_to_none=True)
    (d_loss + h_loss).backward()
    d_norm = torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
    h_norm = torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
    if not bool(torch.isfinite(d_norm) and torch.isfinite(h_norm)):
        raise FloatingPointError('Nonfinite joint gradient; neither optimizer was stepped')
    d_optimizer.step()
    h_optimizer.step()
    return (float(d_norm), float(h_norm))

def atomic_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)

class H1:

    def __init__(self, cfg, device, cache_limit=None, equation_scale=None):
        from hamiformer.training import base as base
        from hamiformer.training.hamiballs1 import hamiltonian as continuous
        from hamiformer.training.hamiballs1 import hamiltonian_objective as objective
        self.base, self.continuous, self.objective = (base, continuous, objective)
        self.cfg, self.device = (cfg, device)
        self.contract = base._train_contract(cfg)
        self.scale = base.load_phase_scales(Path(self.contract['root']) / 'stats/phase_scales.json', int(cfg['dataset']['q_dim']), expected_train_manifest_sha256=self.contract['train_manifest_sha256']).state(device=device, dtype=torch.float32)
        self.train = base._load_train_cache(cfg, device=device, limit=cache_limit)
        self.attr_scale = base._attribute_scale(self.train.attrs)
        seed = int(cfg['seed'])
        self.d = base.build_hamiballs_d(cfg['model'], q_dim=cfg['dataset']['q_dim'], attr_dim=cfg['dataset']['attr_dim'], seed=seed, device=device)
        torch.manual_seed(seed + 151)
        self.h = continuous._build_continuous_h(cfg, self.scale, device=device)
        objective._prepare_relation_objective(self.train, self.scale, q_dim=cfg['dataset']['q_dim'])
        self.eq = objective._equation_scale.clone()
        self.streams = {name: ExplicitEpochBatchStream(self.train.size, cfg['optimization']['batch_size'], seed=seed + 17) for name in ('d', 'h')}
        self.generators = {name: torch.Generator(device=device).manual_seed(seed + offset) for name, offset in (('tau', 23), ('noise', 29))}
        self.ema = None

    def losses(self, step):
        b = self.base
        ids = self.streams['d'].next_indices().to(self.device)
        _, _, x0, clean, attrs, time = b._normalised_batch(self.train, ids, self.scale)
        rf = self.cfg['rectified_flow']
        tau = b.sample_pf_rf_v1_tau(len(ids), mean=rf['train_tau_p_mean'], std=rf['train_tau_p_std'], device=self.device, dtype=torch.float32, generator=self.generators['tau'])
        pair = b.make_pf_rf_v1_pair(clean, tau, noise_scale=rf['phase_noise_scale'], t_eps=rf['t_eps'], generator=self.generators['noise'])
        predicted, _ = b.d_clean_and_tokens(self.d, pair.noisy, tau, x0=x0, attrs=attrs, physical_time=time)
        velocity = (predicted - pair.noisy) / (1 - tau[:, None, None, None]).clamp_min(rf['t_eps'])
        d_loss = (velocity - pair.target_velocity).square().mean()
        ids = self.streams['h'].next_indices().to(self.device)
        phase = torch.cat((self.train.raw_x0[ids, None], self.train.raw_future[ids]), 1)
        self.objective._equation_scale = self.eq
        context = (self.train.attrs[ids] / self.attr_scale.reshape(1, 1, -1)).unsqueeze(-2)
        h_loss = self.objective._relation_loss(self.h, phase, context, state_scale=self.scale, frame_dt=self.contract['frame_dt'], dof=self.cfg['hamiltonian']['student_t_dof'])
        return (d_loss, h_loss)

    def export(self, output, step):
        from hamiformer.training.hamiballs_recovery import module_digest
        from hamiformer.utils import sha256_file
        parent_path = output / f'checkpoint_{step:06d}.pt'
        parent = dict(schema=self.base.CHECKPOINT_SCHEMA, role='main', step=step, config=self.cfg, dataset_contract=self.contract, state_scale=self.scale.cpu(), attr_scale=self.attr_scale.cpu(), d_state_dict=self.d.state_dict(), h_state_dict=self.h.state_dict(), hamiltonian_kind='continuous', joint_training=True, config_sha256=self.base._canonical_hash(self.cfg), source_manifest={'joint_experts': sha256_file(Path(__file__))}, h_freeze_digest=module_digest(self.h))
        atomic_save(parent, parent_path)
        atomic_save(dict(schema='hamiformer.hami1.continuous_h.relation_training.v1', role='hami1-continuous-h-relation-training', status='COMPLETE', updates=step, config=self.cfg, dataset_contract=self.contract, state_scale=self.scale.cpu(), attr_scale=self.attr_scale.cpu(), h_state_dict=self.h.state_dict(), h_digest=module_digest(self.h), parent_checkpoint_sha256=sha256_file(parent_path), joint_training=True), output / 'continuous_h_terminal.pt')

class H2:

    def __init__(self, cfg, device, cache_limit=None, equation_scale=None):
        from hamiformer.cli import train_hamiballs2_wide_d as diffusion
        from hamiformer.cli import train_hamiballs2_h_capacity as physics
        self.diffusion, self.physics = (diffusion, physics)
        self.cfg, self.device = (cfg, device)
        dc, hc = (cfg['d'], cfg['h'])
        if dataset_signature(dc) != dataset_signature(hc) or Path(dc['train_pack']).resolve() != Path(hc['train_pack']).resolve():
            raise ValueError('D/H must use the same training data')
        root = Path(dc['dataset_root'])
        metadata = json.loads((root / 'metadata/dataset_config.json').read_text())
        audit = json.loads((root / 'audit/full_audit.json').read_text())
        if metadata['semantic_hash'] != dataset_signature(dc) or audit.get('full_validation') is not True:
            raise ValueError('Dataset identity or full audit mismatch')
        diffusion.build_hamiballs2_crop_pack(root / 'manifests/train.jsonl', dc['train_pack'], include_contact_labels=False)
        self.train = physics.HamiBalls2CropPackDataset(dc['train_pack'])
        stats = json.loads((root / 'stats/phase_scales.json').read_text())
        self.scale = torch.tensor(stats['phase_rms'], device=device, dtype=torch.float32)
        values = json.loads(Path(equation_scale).read_text())['values'] if equation_scale else physics.compute_equation_scale(self.train, self.scale.cpu().numpy())
        self.eq = torch.as_tensor(values, device=device, dtype=torch.float32)
        seed = int(dc['master_seed'])
        torch.manual_seed(seed)
        self.d = diffusion.HamiBalls2WideD(**dc['model']).to(device)
        torch.manual_seed(physics.derived_seed(int(hc['master_seed']), 'model-init'))
        self.h = physics.HamiBalls2ContinuousHamiltonian(**hc['model'], q_scale=tuple(self.scale[:3].tolist()), p_scale=tuple(self.scale[3:].tolist())).to(device)
        count = min(cache_limit, len(self.train)) if cache_limit else len(self.train)
        self.streams = {'d': ExplicitEpochBatchStream(count, 64, seed=diffusion.derived_seed(seed, 'train-order')), 'h': ExplicitEpochBatchStream(count, int(hc['batch_size']), seed=physics.derived_seed(int(hc['master_seed']), 'train-order'))}
        self.generators = {'noise': torch.Generator(device=device).manual_seed(diffusion.derived_seed(seed, 'train-rf'))}
        self.ema = {k: v.detach().clone() for k, v in self.d.state_dict().items()}

    def losses(self, step):
        df = self.diffusion
        rows = self.streams['d'].next_indices()
        batch = self.physics.fetch(self.train, rows, self.device)
        batch['time'] = torch.as_tensor(np.array(self.train.arrays['time'][rows.cpu().numpy()], copy=True), device=self.device)
        clean = batch['phase'][:, 1:].float()
        mask = batch['object_mask'].bool()
        tau, noisy, target = df.make_pair(clean, mask, self.scale, generator=self.generators['noise'])
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            predicted = df.model_forward(self.d, batch, noisy, tau)
            velocity = (predicted - noisy) / (1 - tau[:, None, None, None]).clamp_min(0.05)
            d_loss = df.masked_mean((velocity - target).square(), mask[:, None, :].expand(clean.shape[:3]))
        batch = self.physics.fetch(self.train, self.streams['h'].next_indices(), self.device)
        h_loss, _, _ = self.physics.relation_terms(self.h, batch, self.scale, self.eq, frame_dt=float(self.cfg['h']['frame_dt']), dof=4.0, create_graph=True, robust_granularity='object_edge', objective_mode=h_objective('h2', step))
        return (d_loss, h_loss)

    def export(self, output, step):
        for name in ('d', 'h'):
            (output / name).mkdir(exist_ok=True)
        atomic_save({'step': step, 'model': self.d.state_dict(), 'ema': self.ema, 'joint_training': True}, output / 'd' / f'step_{step:06d}.pt')
        atomic_save({'step': step, 'model': self.h.state_dict(), 'phase_scale': self.scale.cpu(), 'equation_scale': self.eq.cpu(), 'contract': {'model': self.cfg['h']['model'], 'steps': step, 'objective_mode': h_objective('h2', step), 'joint_training': True}}, output / 'h' / f'step_{step:06d}.pt')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('h1', 'h2'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--h-config', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-steps', type=int, default=TOTAL_STEPS)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--cache-limit', type=int)
    parser.add_argument('--equation-scale', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not 1 <= args.max_steps <= TOTAL_STEPS or args.save_every < 1:
        parser.error('invalid step budget')
    if args.cache_limit is not None and args.cache_limit < 1:
        parser.error('cache-limit must be positive')
    if args.cache_limit and args.max_steps == TOTAL_STEPS:
        parser.error('cache-limit is for short diagnostic runs')
    if args.dataset == 'h1':
        from hamiformer.training import base as base
        cfg = base._load_config(args.config)
    else:
        if args.h_config is None:
            parser.error('--h-config is required for h2')
        cfg = {'d': yaml.safe_load(args.config.read_text()), 'h': yaml.safe_load(args.h_config.read_text())}
    output = args.output_dir.resolve()
    if args.resume is None and output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    seed = int(cfg['seed'] if args.dataset == 'h1' else cfg['d']['master_seed'])
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    adapter = (H1 if args.dataset == 'h1' else H2)(cfg, device, args.cache_limit, args.equation_scale)
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    optimizers = {name: torch.optim.AdamW(getattr(adapter, name).parameters(), lr=lr, weight_decay=0.0001, fused=device.type == 'cuda') for name, lr in (('d', 0.0003), ('h', 0.0001))}
    step = 0
    stage = 'midpoint_field'
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state['dataset'] != args.dataset or state['config'] != cfg or state['cache_limit'] != args.cache_limit:
            raise ValueError('Resume configuration differs')
        if not torch.equal(adapter.eq, state['equation_scale']) or not torch.equal(adapter.scale, state['phase_scale']):
            raise ValueError('Resume normalization differs')
        step = state['step']
        stage = state['h_objective']
        for name in ('d', 'h'):
            getattr(adapter, name).load_state_dict(state['models'][name], strict=True)
            optimizers[name].load_state_dict(state['optimizers'][name])
            adapter.streams[name].load_state_dict(state['streams'][name])
        for name, g in adapter.generators.items():
            g.set_state(state['generators'][name].cpu())
        adapter.ema = state['ema']
        adapter.eq = state['equation_scale']
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
    if step >= args.max_steps:
        raise ValueError('Resume step already reaches requested budget')
    for next_step in range(step + 1, args.max_steps + 1):
        stage = set_joint_schedule(adapter, optimizers, args.dataset, next_step, stage, device)
        adapter.d.train()
        adapter.h.train()
        dl, hl = adapter.losses(next_step)
        dn, hn = joint_update(dl, hl, adapter.d, adapter.h, optimizers['d'], optimizers['h'])
        if adapter.ema is not None:
            adapter.diffusion.update_ema(adapter.d, adapter.ema, 0.9999)
        step = next_step
        print(json.dumps({'step': step, 'd_loss': float(dl.detach()), 'h_loss': float(hl.detach()), 'd_grad': dn, 'h_grad': hn, 'h_objective': stage}), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            atomic_save({'schema': 'hamiformer.joint_experts.v1', 'dataset': args.dataset, 'config': cfg, 'cache_limit': args.cache_limit, 'step': step, 'h_objective': stage, 'models': {n: getattr(adapter, n).state_dict() for n in ('d', 'h')}, 'optimizers': {n: o.state_dict() for n, o in optimizers.items()}, 'streams': {n: s.state_dict() for n, s in adapter.streams.items()}, 'generators': {n: g.get_state() for n, g in adapter.generators.items()}, 'ema': adapter.ema, 'equation_scale': adapter.eq, 'phase_scale': adapter.scale, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, output / 'joint_latest.pt')
    if step == TOTAL_STEPS:
        adapter.export(output, step)
if __name__ == '__main__':
    main()
