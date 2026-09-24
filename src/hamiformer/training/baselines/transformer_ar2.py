from hamiformer.utils.paths import project_root
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import numpy as np
from hamiformer.data_generation3d.signature import dataset_signature
import torch
from torch import nn
import yaml
from hamiformer.data.hamiballs2 import HamiBalls2CropPackDataset
from hamiformer.baselines.physiformer import HamiBalls2WideD
from hamiformer.cli.train_hamiballs2_wide_d import derived_seed, learning_rate
from hamiformer.baselines.graph_transformer import GraphTransformerAR

def save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)

def load_pack(path):
    ds = HamiBalls2CropPackDataset(path)
    keys = ('phase', 'attrs', 'time', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length', 'contact', 'physical_seed')
    return {k: torch.from_numpy(np.array(v)).cuda() for k, v in ds.arrays.items() if k in keys}

def tf_loss(model, batch, scale):
    z = batch['phase'] / scale
    B, T, K, _ = z.shape
    static = {k: v[:, None].expand(-1, T - 1, *v.shape[1:]).reshape(B * (T - 1), *v.shape[1:]) for k, v in batch.items() if k in ('attrs', 'object_mask', 'spring_mask', 'spring_k', 'spring_rest_length')}
    inputs = z[:, :-1].reshape(-1, K, 6)
    if model.context > 1:
        indices = torch.arange(T - 1, device=z.device)[:, None] + torch.arange(1 - model.context, 1, device=z.device)[None, :]
        static['history_valid'] = (indices >= 0)[None].expand(B, -1, -1).reshape(-1, model.context)
        inputs = z[:, indices.clamp_min(0)].reshape(-1, model.context, K, 6)
    pred = model(inputs, static, batch['time'].diff(dim=1).reshape(-1))
    valid = static['object_mask'][..., None]
    square = (pred.float() - z[:, 1:].reshape(-1, K, 6)).square()
    return (square * valid).sum() / (valid.sum() * 6)

@torch.no_grad()
def evaluate(model, data, scale, out, step):
    model.eval()
    predictions = []
    for start in range(0, len(data['phase']), 64):
        b = {k: v[start:start + 64] for k, v in data.items()}
        state = b['phase'][:, 0] / scale
        history = state[:, None]
        states = []
        for dt in b['time'].diff(dim=1).unbind(1):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                state = model(history if model.context > 1 else state, b, dt).float()
            history = torch.cat((history, state[:, None]), 1)[:, -model.context:]
            states.append(state * scale)
        predictions.append(torch.stack(states, 1))
    pred = torch.cat(predictions)
    if not torch.isfinite(pred).all():
        raise FloatingPointError('nonfinite autoregressive rollout')
    square = ((pred - data['phase'][:, 1:]) / scale).double().square()
    valid = data['object_mask'][:, None].bool().expand(square.shape[:3])
    contact = data['contact'].bool() & valid
    metrics = {}
    for region, m in [('all', valid), ('continuous', valid & ~contact), ('contact', contact)]:
        for name, sl in [('all', slice(None)), ('q', slice(0, 3)), ('p', slice(3, 6))]:
            metrics[name + '/' + region] = float(square[..., sl][m].mean())
    record = dict(step=step, metrics=metrics, sources=len(pred), deterministic_replicas=True, noise_labels=[1518115115, 530564752], independent_predictions=len(pred))
    (out / f'validation_{step:06d}.json').write_text(json.dumps(record, indent=2))
    if step == 50000:
        np.savez_compressed(out / 'formal_validation512x2.npz', predictions=np.repeat(pred.cpu().numpy()[None], 2, axis=0), target=data['phase'][:, 1:].cpu().numpy(), object_mask=data['object_mask'].cpu().numpy(), contact=data['contact'].cpu().numpy(), physical_seed=data['physical_seed'].cpu().numpy(), state_scale=scale.cpu().numpy())
    print(json.dumps(record), flush=True)
    model.train()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/hamiballs2/physiformer.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--context', type=int, choices=[1, 4], default=1)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg['dataset_root'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'latest.pt').exists():
        raise FileExistsError('Use a fresh output directory')
    meta = json.loads((root / 'metadata/dataset_config.json').read_text())
    audit = json.loads((root / 'audit/full_audit.json').read_text())
    assert meta['semantic_hash'] == dataset_signature(cfg) and meta['status'] == 'finalized' and audit['full_validation']
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(cfg['master_seed'])
    np.random.seed(cfg['master_seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    train = load_pack(cfg['train_pack'])
    val = load_pack(cfg['validation_pack'])
    assert train['phase'].shape[1] == 49 and len(val['phase']) == 512
    scale = torch.tensor(json.loads((root / 'stats/phase_scales.json').read_text())['phase_rms'], device='cuda')
    model = GraphTransformerAR(context=args.context).cuda()
    count = sum((p.numel() for p in model.parameters()))
    assert abs(count / 5615598 - 1) < 0.01, count
    b = {k: v[:2] for k, v in train.items()}
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        z = b['phase'][:, 0] / scale
        dt = b['time'][:, 1] - b['time'][:, 0]
        y = model(z, b, dt)
        perm = torch.arange(z.shape[1] - 1, -1, -1, device='cuda')
        bp = {k: v for k, v in b.items()}
        for k in ('attrs', 'object_mask'):
            bp[k] = b[k][:, perm]
        for k in ('spring_mask', 'spring_k', 'spring_rest_length'):
            bp[k] = b[k][:, perm][:, :, perm]
        torch.testing.assert_close(model(z[:, perm], bp, dt), y[:, perm], atol=2e-05, rtol=2e-05)
        zp = z.clone()
        zp[~b['object_mask'].bool()] = 123
        torch.testing.assert_close(model(zp, b, dt), y, atol=2e-05, rtol=2e-05)
    torch.backends.cuda.matmul.allow_tf32 = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001, fused=True)
    order = torch.Generator().manual_seed(derived_seed(cfg['master_seed'], 'train-order'))
    steps = 10 if args.smoke else 50000
    contract = dict(parameters=count, reference_parameters=5615598, model=dict(width=340, depth=4, heads=4, context=args.context), config=cfg, batch_windows=64, edges=48, steps=steps, master_seed=cfg['master_seed'], order_seed=derived_seed(cfg['master_seed'], 'train-order'), precision='bf16 autocast; float32 loss and parameters', objective='masked normalized next-state MSE, teacher forcing over all 48 pairs', inputs='ordered recent context q,p with explicit missing-history mask when context>1; static physical graph and dt; evaluation starts from one GT state only', evaluation='terminal online weights; deterministic recursive rollout; 512 sources; two identical noise-label replicas', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), train_manifest_sha256=hashlib.sha256((root / 'manifests/train.jsonl').read_bytes()).hexdigest(), validation_manifest_sha256=hashlib.sha256((root / 'manifests/validation.jsonl').read_bytes()).hexdigest())
    (out / 'run_contract.json').write_text(json.dumps(contract, indent=2))
    print(json.dumps(contract), flush=True)
    start = time.perf_counter()
    step = 0
    while step < steps:
        permutation = torch.randperm(len(train['phase']), generator=order)
        for ids in permutation[:len(permutation) // 64 * 64].split(64):
            ids = ids.cuda()
            batch = {k: v[ids] for k, v in train.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = tf_loss(model, batch, scale)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            step += 1
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(step, 50000)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == steps:
                row = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), seconds=time.perf_counter() - start)
                print(json.dumps(row), flush=True)
                with (out / 'history.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
            if step % 5000 == 0 or step == steps:
                save(out / 'latest.pt', dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(), contract=contract, scale=scale, order_rng=order.get_state(), permutation=permutation, next_offset=(permutation == ids[-1].cpu()).nonzero()[0].item() + 1, seconds=time.perf_counter() - start))
                evaluate(model, val, scale, out, step)
            if step == steps:
                break
    (out / 'COMPLETE').write_text(str(steps))
if __name__ == '__main__':
    main()
