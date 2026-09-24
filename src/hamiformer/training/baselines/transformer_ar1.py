from __future__ import annotations
from hamiformer.utils.paths import project_root
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = project_root()
from hamiformer.baselines.transformer_ar import TransformerAR, TransformerARConfig
from hamiformer.data import load_phase_scales
from hamiformer.data.dataset import PhaseWindowDataset, collate_phase_windows
from hamiformer.training.training_schedule import ExplicitEpochBatchStream, cosine_learning_rate
from hamiformer.utils import sha256_file

def save(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def recursive_inputs(model, initial, attrs, dt):
    with torch.no_grad():
        state = initial
        inputs = [state]
        for edge in range(dt.shape[1] - 1):
            state = model(state, attrs, dt[:, edge])
            inputs.append(state)
        return torch.stack(inputs, 1)

def batched_bptt1_loss(model, z, attrs, dt):
    inputs = recursive_inputs(model, z[:, 0], attrs, dt)
    batch, edges, objects, dims = inputs.shape
    a = attrs[:, None].expand(-1, edges, -1, -1).reshape(batch * edges, objects, -1)
    prediction = model(inputs.reshape(batch * edges, objects, dims), a, dt.reshape(-1))
    return (prediction - z[:, 1:].reshape(batch * edges, objects, dims)).square().mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=50000)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--closed-loop-bptt1', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(4101)
    np.random.seed(4101)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda'
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'latest.pt').exists() and (not args.resume):
        raise FileExistsError('checkpoint exists; use --resume')
    manifest = args.dataset_root / 'manifests/train.jsonl'
    scale = load_phase_scales(args.dataset_root / 'stats/phase_scales.json', 2, expected_train_manifest_sha256=sha256_file(manifest)).state(device=device, dtype=torch.float32)
    ds = PhaseWindowDataset(manifest, num_objects=5, future_steps=48, q_dim=2, attr_dim=3)
    assert len(ds) == 57344, len(ds)
    loader = DataLoader(ds, batch_size=512, num_workers=0, collate_fn=collate_phase_windows)
    phases, attributes, times = ([], [], [])
    for b in loader:
        phases.append(torch.cat((b.x0[:, None], b.future), 1))
        attributes.append(b.attrs)
        times.append(b.time)
        if args.smoke:
            break
    phase = torch.cat(phases).to(device) / scale
    attrs = torch.cat(attributes).to(device)
    attr_scale = attrs.double().square().mean((0, 1)).sqrt().float()
    attrs = attrs / attr_scale
    dt = torch.diff(torch.cat(times).to(device), dim=1)
    assert phase.shape[1:] == (49, 5, 4) and dt.shape[1] == 48
    config = TransformerARConfig()
    model = TransformerAR(config).to(device)
    count = sum((x.numel() for x in model.parameters()))
    assert abs(count / 1173068 - 1) < 0.01, count
    model.eval()
    with torch.no_grad():
        z, a, d = (phase[:2, 0], attrs[:2], dt[:2, 0])
        pred = model(z, a, d)
        perm = torch.tensor([3, 1, 4, 0, 2], device=device)
        torch.testing.assert_close(model(z[:, perm], a[:, perm], d), pred[:, perm], atol=2e-05, rtol=2e-05)
        torch.testing.assert_close(model(z[:1], a[:1], d[:1]), pred[:1], atol=2e-05, rtol=2e-05)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001, fused=True)
    stream = ExplicitEpochBatchStream(len(phase), 64, seed=4101 + 17)
    contract = dict(schema='hamiformer.transformer_ar.ctx1.train.v1', config=asdict(config), parameters=count, reference_parameters=1173068, seed=4101, order_seed=4118, order_seed_derivation='master_seed + 17, same canonical epoch stream as wide-D', batch_windows=64, edges_per_window=48, target_steps=args.steps, loss='mean normalized q/p squared error over 48 recursive predictions; detach between frames; one update per window batch' if args.closed_loop_bptt1 else 'mean normalized next-state q/p squared error over all 48 teacher-forced pairs', closed_loop_bptt1=args.closed_loop_bptt1, execution='detached recursive inputs then batched local gradients', optimizer='AdamW', weight_decay=0.0001, lr_max=0.0003, lr_min=3e-07, warmup_steps=1000, dataset_manifest=str(manifest.resolve()), dataset_sha256=sha256_file(manifest), model_sha256=sha256_file(ROOT / 'src/hamiformer/baselines/transformer_ar.py'), runner_sha256=sha256_file(Path(__file__)), smoke=args.smoke)
    (args.output / 'manifest.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(json.dumps(contract), flush=True)
    first, previous_seconds = (1, 0.0)
    if args.resume:
        ck = torch.load(args.output / 'latest.pt', map_location=device, weights_only=False)
        assert ck['contract'] == contract
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        stream.load_state_dict(ck['stream'])
        first, previous_seconds = (ck['step'] + 1, ck['training_seconds'])
    start = time.perf_counter()
    if args.closed_loop_bptt1:
        static_z = phase[:64].clone()
        static_a = attrs[:64].clone()
        static_dt = dt[:64].clone()
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                batched_bptt1_loss(model, static_z, static_a, static_dt).backward()
        torch.cuda.current_stream().wait_stream(warmup)
        optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_loss = batched_bptt1_loss(model, static_z, static_a, static_dt)
            graph_loss.backward()
    for step in range(first, args.steps + 1):
        idx = stream.next_indices().to(device)
        z = phase[idx]
        a = attrs[idx, None].expand(-1, 48, -1, -1).reshape(-1, 5, 3)
        learning_rate = cosine_learning_rate(step, total_steps=args.steps, warmup_steps=min(1000, args.steps - 1), maximum=0.0003, minimum=3e-07)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        if args.closed_loop_bptt1:
            static_z.copy_(z)
            static_a.copy_(attrs[idx])
            static_dt.copy_(dt[idx])
            graph.replay()
            loss = graph_loss
        else:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(z[:, :-1].reshape(-1, 5, 4), a, dt[idx].reshape(-1))
            loss = (prediction - z[:, 1:].reshape(-1, 5, 4)).square().mean()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            torch.cuda.synchronize()
            row = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), lr=learning_rate, training_seconds=previous_seconds + time.perf_counter() - start)
            print(json.dumps(row), flush=True)
            with (args.output / 'history.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
        if step % 5000 == 0 or step == args.steps:
            save(args.output / 'latest.pt', dict(contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(), stream=stream.state_dict(), step=step, state_scale=scale, attr_scale=attr_scale, training_seconds=previous_seconds + time.perf_counter() - start))
    (args.output / 'COMPLETE').write_text(str(args.steps) + '\n')
if __name__ == '__main__':
    main()
