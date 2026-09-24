from hamiformer.utils.paths import project_root
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
from hamiformer.inference.weights import load_h1, load_h2, load_hgdpf
from hamiformer.integrators.solver import Solver, METHODS as SOLVERS
SEEDS = (1518115115, 530564752)
METHODS = {'h1': ('ours', 'physiformer', 'hgdpf'), 'h2': ('ours', 'physiformer', 'dit', 'transformer_ar1', 'transformer_ar4')}

def metrics(prediction, target, contact, mask, qdim):
    error = (prediction.astype(np.float64) - target[None]) ** 2
    valid = np.broadcast_to(mask[None, :, None, :, None], error.shape)
    result = {}
    for label, left, right in [('total', 0, 192), ('1-48', 0, 48), ('49-96', 48, 96), ('97-192', 96, 192)]:
        result[label] = {}
        for name, cols in [('z', slice(None)), ('q', slice(0, qdim)), ('p', slice(qdim, None))]:
            e = error[:, :, left:right, :, cols]
            v = valid[:, :, left:right, :, cols]
            result[label][name] = float(e[v].mean())
    for label, c in [('contact', contact), ('continuous', ~contact)]:
        result[label] = {}
        for name, cols in [('q', slice(0, qdim)), ('p', slice(qdim, None))]:
            v = valid[:, :, :, :, cols] & c[None, :, :, :, None]
            result[label][name] = float(error[:, :, :, :, cols][v].mean()) if v.any() else None
    return result

def rf_noise(dataset, count, seed, scale, device):
    g = torch.Generator(device=device).manual_seed(seed)
    if dataset == 'h1':
        tape = torch.empty((count, 4, 48, 5, 4), device=device)
        for chunk in range(4):
            for begin in range(0, count, 64):
                end = min(begin + 64, count)
                tape[begin:end, chunk] = torch.randn((end - begin, 48, 5, 4), generator=g, device=device)
    else:
        tape = torch.empty((count, 4, 48, 10, 6), device=device)
        for begin in range(0, count, 8):
            end = min(begin + 8, count)
            for chunk in range(4):
                tape[begin:end, chunk] = 0.1 * scale * torch.randn((end - begin, 48, 10, 6), generator=g, device=device)
    return tape

def hgdpf_generators(seed, device):

    def derive(text):
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') & (1 << 63) - 1
    result = []
    for chunk in range(4):
        sampling = seed if chunk == 0 else derive(f'hgdpf-v1|sampling-seed={seed}|stream=sampling|rolling-chunk={chunk}')
        proposal = derive(f'hgdpf-v1|sampling-seed={seed}|stream=snis-proposal' + ('' if chunk == 0 else f'|rolling-chunk={chunk}'))
        result.append((torch.Generator(device=device).manual_seed(sampling), torch.Generator(device=device).manual_seed(proposal)))
    return result

def predict(args, data, total):
    device = torch.device(args.device)
    method = args.method
    qdim = 2 if args.dataset == 'h1' else 3
    count = len(data['phase'])
    outputs = []
    if args.dataset == 'h1':
        models = load_h1(args.weights, device)
        scale = models['state_scale']
        if method == 'ours':
            Solver(args.solver).install_h1(models, compiled=device.type == 'cuda')
        if method == 'hgdpf':
            from hamiformer.baselines.hgdpf import HamiltonianGuidedDDIM
            from hamiformer.baselines.hgdpf_hamiballs import pack_hamiballs_phase, unpack_hamiballs_phase, flatten_hamiballs_attributes
            d, h, norm = load_hgdpf(args.weights, device)
            sampler = HamiltonianGuidedDDIM(d, h, state_mean=norm['state_mean'], state_scale=norm['state_scale'], diffusion_steps=1000, inference_steps=20, context_ratio=0.5)
    else:
        models = load_h2(args.weights, method, device)
        scale = models['model'].phase_scale if method == 'ours' else models['state_scale']
        if method == 'ours':
            from hamiformer.inference.mixed_rollout import MixedRollout
            from hamiformer.evaluation.precision import configure_fp32_evaluation
            cfg = yaml.safe_load((project_root() / 'configs/hamiballs2/main.yaml').read_text())
            collector = MixedRollout(cfg=cfg, dataset=None, wide=models['d'], hamiltonian=models['h'], model=models['model'], device=device)
            configure_fp32_evaluation(models['d'], models['h'], models['model'], collector=collector)
            if device.type == 'cuda':
                from hamiformer.inference.hamiballs2 import sample_hami2_stateful_pf_rf_heun
                runtime = Solver(args.solver).install_h2(collector)
    for seed in args.seeds:
        tape = rf_noise(args.dataset, total, seed, scale, device)[:count]
        generators = hgdpf_generators(seed, device) if method == 'hgdpf' else None
        rows = []
        for begin in range(0, count, args.batch_size):
            end = min(begin + args.batch_size, count)
            batch = {k: torch.as_tensor(v[begin:end].copy(), device=device) for k, v in data.items() if k not in ('contact',)}
            batch['phase'] = batch['phase'].float()
            batch['attrs'] = batch['attrs'].float()
            batch['time'] = batch['time'].float()
            if args.dataset == 'h1':
                x0 = batch['phase'][:, 0] / scale
            else:
                x0 = batch['phase'][:, 0]
            parts = []
            if method.startswith('transformer_ar'):
                model = models['model']
                state = x0 / scale
                history = state[:, None]
                with torch.no_grad():
                    for edge in range(192):
                        dt = batch['time'][:, edge + 1] - batch['time'][:, edge]
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                            state = model(history if model.context > 1 else state, batch, dt).float()
                        history = torch.cat((history, state[:, None]), 1)[:, -model.context:]
                        parts.append(state * scale)
                prediction = torch.stack(parts, 1)
            else:
                if method == 'hgdpf':
                    x0 = (pack_hamiballs_phase(batch['phase'][:, 0]) - norm['state_mean']) / norm['state_scale']
                    attrs = (flatten_hamiballs_attributes(batch['attrs']) - norm['attribute_mean']) / norm['attribute_scale']
                for chunk in range(4):
                    times = batch['time'][:, chunk * 48:chunk * 48 + 49]
                    if args.dataset == 'h1' and method != 'hgdpf':
                        from hamiformer.training.hamiballs1.sampling import expert_kwargs
                        from hamiformer.evaluation.hamiballs_formal import sample_hamiballs_expert_chunk, sample_hamiballs_d_chunk
                        with torch.no_grad():
                            if method == 'ours':
                                p = sample_hamiballs_expert_chunk(models['d'], models['h'], models['r'], models['gate'], tape[begin:end, chunk], x0=x0, attrs=batch['attrs'], physical_time=times, mode='main', **expert_kwargs(models)).trajectory
                            else:
                                p = sample_hamiballs_d_chunk(models['wide_d'], tape[begin:end, chunk], x0=x0, attrs=batch['attrs'], physical_time=times, num_steps=20, t_eps=0.05)
                        x0 = p[:, -1]
                        parts.append(p * scale)
                    elif method == 'hgdpf':
                        p = sampler.sample(query_times=times[:, 1:], observed_initial=x0, initial_time=times[:, 0], actions=torch.zeros(end - begin, 48, 10, device=device), attributes=attrs, guidance='snis', gradient_eta=0.01, snis_candidates=4, snis_lambda=1.0, snis_ddim_stochasticity=1.0, snis_recompute_clean=True, generator=generators[chunk][0], proposal_generator=generators[chunk][1])
                        x0 = (p[:, -1] - norm['state_mean']) / norm['state_scale']
                        parts.append(unpack_hamiballs_phase(p))
                    else:
                        from hamiformer.flow.pf_rf_v1 import sample_pf_rf_v1_heun
                        b = dict(batch, phase=torch.cat((x0[:, None], torch.zeros_like(batch['phase'][:, 1:49])), 1), time=times)
                        noise = tape[begin:end, chunk] * batch['object_mask'][:, None, :, None]
                        if method == 'ours':
                            if device.type == 'cuda':
                                p = sample_hami2_stateful_pf_rf_heun(runtime, noise, b, num_steps=20)
                            else:
                                p, _ = collector.sample_stateful(noise, b, rf_steps=20, parent_mode='learned')
                        else:

                            def field(z, tau):
                                return models['model'](z, tau, x0=x0, attrs=batch['attrs'], physical_time=times, object_mask=batch['object_mask'].bool(), spring_mask=batch['spring_mask'], spring_k=batch['spring_k'], spring_rest_length=batch['spring_rest_length'])
                            with torch.no_grad():
                                p = sample_pf_rf_v1_heun(field, noise, num_steps=20, t_eps=0.05).trajectory
                        x0 = p[:, -1]
                        parts.append(p)
                prediction = torch.cat(parts, 1)
            if not bool(prediction.isfinite().all()):
                raise FloatingPointError(f'{method}: nonfinite prediction')
            rows.append(prediction.detach().cpu().numpy())
            print(json.dumps({'method': method, 'seed': seed, 'completed': end}), flush=True)
        outputs.append(np.concatenate(rows))
    return (np.stack(outputs), scale.detach().cpu().numpy())

def main():
    p = argparse.ArgumentParser(description='Table 1 and Table 2 trajectory evaluation')
    p.add_argument('--dataset', choices=('h1', 'h2'), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--solver', choices=SOLVERS, default='plas')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--limit', type=int)
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.batch_size is None:
        args.batch_size = 64 if args.dataset == 'h1' or args.method.startswith('transformer_ar') else 8
    if args.method not in METHODS[args.dataset]:
        p.error('unknown method for dataset')
    if args.solver != 'plas' and (args.dataset != 'h1' or args.method != 'ours'):
        p.error('solver variants apply to HamiBalls-1 Ours evaluation')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error('positive batch size and limit required')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.mha.set_fastpath_enabled(False)
    with h5py.File(args.data, 'r') as f:
        group = f['validation']
        total = len(group['phase'])
        data = {k: v[:args.limit] for k, v in group.items()}
    predictions, scale = predict(args, data, total)
    mask = data.get('object_mask', np.ones(data['phase'].shape[::2][:2], dtype=bool)).astype(bool)
    target = data['phase'][:, 1:193]
    report = metrics(predictions.astype(np.float64) / scale, target.astype(np.float64) / scale, data['contact'].astype(bool), mask, 2 if args.dataset == 'h1' else 3)
    args.output.mkdir(parents=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / 'trajectories.npz', prediction=predictions, target=target, initial=data['phase'][:, 0], state_scale=scale, object_mask=mask, attrs=data['attrs'], time=data['time'])
    print(json.dumps(report), flush=True)
if __name__ == '__main__':
    main()
