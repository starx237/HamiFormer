from __future__ import annotations
from hamiformer.utils.paths import project_root
from hamiformer.training.hamiballs1 import refresh as fresh
from hamiformer.training import router as component_router
from hamiformer.training.hamiballs1 import gate_training as packed
import copy
import json
from pathlib import Path
import sys
import time
import types
import torch
ROOT = project_root()
from hamiformer.training.hamiballs1 import onpolicy_refinement as refiner
COMPONENT_ROUTER_ROOT = ROOT / 'artifacts/hami1_gate_v1'
OUTPUT = ROOT / 'outputs/hami1/qp_integrated_ridge_stage_parent'
QP_OUTPUT = OUTPUT / 'qp'
RIDGE_OUTPUT = OUTPUT / 'ridge_terminal.pt'
INPUT_DIM = 33
RIDGE_RELATIVE = 0.01
MIN_TEST_BLOCKS = 4
MIN_BLOCK_SUCCESS = 0.8
MIN_SOURCE_WIN = 0.5
QP_MAXIMUM_LR = 0.01
QP_MINIMUM_LR = 0.0001
FINAL_FIT_SAMPLES = 800
FINAL_HOLDOUT_SAMPLES = 32
FINAL_COLLECTION_BATCH = 32
FINAL_REFRESH_SOURCES = 32
FINAL_GATE_PARENT_OVERRIDE: Path | None = None
FINAL_GATE_RESUME = False
R_REGISTRATION_OVERRIDE: Path | None = None
GATE_REGISTRATION_OVERRIDE: Path | None = None
FINAL_EXTRA_ARGUMENTS: tuple[str, ...] = ()

def _componentize(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value[..., None].expand(*value.shape, 2)
    if value.ndim == 3 and value.shape[-1] != 2:
        return value[..., None].expand(*value.shape, 2)
    if value.ndim == 3 and value.shape[-1] == 2:
        return value
    if value.ndim == 4 and value.shape[-1] == 2:
        return value
    raise ValueError(f'expected scalar or q/p gate history, got {tuple(value.shape)}')

class IntegratedRidge:

    def __init__(self) -> None:
        self.leaf_capacity = 8
        self.encoder = None
        self.collector = None
        self.terminal_weight = None
        self.xtx = self.xty = self.pending = self.last_evaluated = None
        self.history: list[dict] = []
        self.evidence = {(leaf, component): {'tests': 0, 'successes': 0, 'source_wins': 0, 'sources': 0, 'noise_gain': [0.0, 0.0]} for leaf in range(self.leaf_capacity) for component in range(2)}
        self.started = time.perf_counter()

    def _initialize(self, candidate, collector) -> None:
        device = next(candidate.parameters()).device
        self.encoder = refiner.TrainableRegimeRefiner(copy.deepcopy(candidate)).to(device).eval()
        self.xtx = torch.zeros(self.leaf_capacity, INPUT_DIM, INPUT_DIM, device=device)
        self.xty = torch.zeros(self.leaf_capacity, 2, INPUT_DIM, 2, device=device)
        collector.register_buffer('_integrated_ridge_weight', torch.zeros(self.leaf_capacity, 2, INPUT_DIM, 2, device=device))

        def refine_candidates(module, *, edge, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, previous_g, residual_hidden, base_gate, **_unused):
            if int(edge) == 0:
                return (hr_candidate, base_gate)
            gate_qp = _componentize(base_gate)
            previous_qp = _componentize(previous_g)
            features = self.encoder._features(previous_mixed=previous_mixed, h_candidate=h_candidate, hr_candidate=hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_qp, residual_hidden=residual_hidden, base_gate=gate_qp)
            leaf = self.encoder._leaves(features)
            x = torch.stack([features[name] for name in self.encoder.feature_names], dim=-1)
            x = ((x - self.encoder.feature_mean) / self.encoder.feature_scale).clamp(-8.0, 8.0)
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            delta = torch.zeros_like(hr_candidate)
            for regime in range(self.leaf_capacity):
                use = leaf == regime
                if bool(use.any()):
                    delta[..., :2][use] = x[use] @ module._integrated_ridge_weight[regime, 0]
                    delta[..., 2:][use] = x[use] @ module._integrated_ridge_weight[regime, 1]
            return (hr_candidate + delta, base_gate)
        collector.refine_candidates = types.MethodType(refine_candidates, collector)
        self.collector = collector

    def adapt_collector(self, *, block, candidate, collector):
        if self.collector is None:
            self._initialize(candidate, collector)
        elif collector is not self.collector:
            raise RuntimeError('stage-parent collector identity changed')
        return collector

    @torch.no_grad()
    def _decode(self, candidate, cache):
        base_gate = _componentize(cache['base_gate'])
        previous_g = _componentize(cache['previous_g'])
        features = self.encoder._features(previous_mixed=cache['previous_mixed'], h_candidate=cache['h_candidate'], hr_candidate=cache['base_hr_candidate'], d_candidate=cache['d_candidate'], attrs=cache['attrs'], tau=cache['tau'], previous_g=previous_g, residual_hidden=cache['residual_hidden'], base_gate=base_gate)
        leaf = self.encoder._leaves(features)
        x = torch.stack([features[name] for name in self.encoder.feature_names], dim=-1)
        x = ((x - self.encoder.feature_mean) / self.encoder.feature_scale).clamp(-8.0, 8.0)
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        source = cache['source_index'][:, None, None].expand(leaf.shape)
        object_index = torch.arange(leaf.shape[-1], device=leaf.device, dtype=torch.long)[None, None, :].expand(leaf.shape)
        return {'x': x.reshape(-1, INPUT_DIM), 'leaf': leaf.reshape(-1), 'source': source.reshape(-1), 'object': object_index.reshape(-1), 'base_hr': cache['base_hr_candidate'].reshape(-1, 4), 'target': cache['target'].reshape(-1, 4)}

    def _accumulate_data(self, data) -> None:
        for item in data:
            for leaf in range(self.leaf_capacity):
                use = item['leaf'] == leaf
                x = item['x'][use]
                self.xtx[leaf] += x.T @ x
                for component in range(2):
                    sl = slice(0, 2) if component == 0 else slice(2, 4)
                    y = item['target'][use, sl] - item['base_hr'][use, sl]
                    self.xty[leaf, component] += x.T @ y

    def _solve(self):
        result = torch.zeros(self.leaf_capacity, 2, INPUT_DIM, 2, device=self.xtx.device)
        eye = torch.eye(INPUT_DIM, device=self.xtx.device)
        for leaf in range(self.leaf_capacity):
            ridge = RIDGE_RELATIVE * self.xtx[leaf].diagonal().mean().clamp_min(1e-08)
            matrix = self.xtx[leaf] + ridge * eye
            for component in range(2):
                result[leaf, component] = torch.linalg.solve(matrix, self.xty[leaf, component])
        return result

    @staticmethod
    def _source_counts(source, old, new):
        wins = total = 0
        for value in source.unique():
            use = source == value
            wins += int(new[use].sum() < old[use].sum())
            total += 1
        return (wins, total)

    @torch.no_grad()
    def _admit(self, data, weight, leaf, component, alpha):
        use = data['leaf'] == leaf
        x = data['x'][use]
        sl = slice(0, 2) if component == 0 else slice(2, 4)
        delta = alpha * (x @ weight[leaf, component])
        base, target = (data['base_hr'][use, sl], data['target'][use, sl])
        old = (base - target).square().sum(-1)
        new = (base + delta - target).square().sum(-1)
        wins, sources = self._source_counts(data['source'][use], old, new)
        return (float(old.sum() - new.sum()), wins, sources)

    @torch.no_grad()
    def _admission_alpha(self, data, weight, leaf, component):
        use = data['leaf'] == leaf
        x = data['x'][use]
        sl = slice(0, 2) if component == 0 else slice(2, 4)
        prediction = x @ weight[leaf, component]
        target = data['target'][use, sl] - data['base_hr'][use, sl]
        dot = (prediction * target).sum()
        p2 = prediction.square().sum().clamp_min(1e-30)
        return float((dot / p2).clamp(0.0, 1.0))

    @torch.no_grad()
    def __call__(self, *, block, candidate, caches) -> None:
        if len(caches) != 2:
            raise ValueError('integrated ridge requires paired-noise caches')
        data = [self._decode(candidate, cache) for cache in caches]
        row = {'block': int(block)}
        if self.pending is not None:
            self.last_evaluated = self.pending.clone()
            admission = {}
            deployed = torch.zeros_like(self.pending)
            for leaf in range(self.leaf_capacity):
                admission[str(leaf)] = {}
                for component, label in enumerate(('q', 'p')):
                    alphas = []
                    for item in data:
                        alphas.append(self._admission_alpha(item, self.pending, leaf, component))
                    common = min(alphas)
                    metrics = [self._admit(item, self.pending, leaf, component, common) for item in data]
                    evidence = self.evidence[leaf, component]
                    evidence['tests'] += 1
                    success = common > 0.0 and all((value[0] > 0.0 for value in metrics))
                    evidence['successes'] += int(success)
                    evidence['source_wins'] += sum((value[1] for value in metrics))
                    evidence['sources'] += sum((value[2] for value in metrics))
                    for noise_index, value in enumerate(metrics):
                        evidence['noise_gain'][noise_index] += value[0]
                    success_rate = evidence['successes'] / evidence['tests']
                    source_win = evidence['source_wins'] / max(evidence['sources'], 1)
                    accepted = evidence['tests'] >= MIN_TEST_BLOCKS and success_rate >= MIN_BLOCK_SUCCESS and (source_win >= MIN_SOURCE_WIN) and all((value > 0.0 for value in evidence['noise_gain'])) and (common > 0.0)
                    if accepted:
                        deployed[leaf, component] = common * self.pending[leaf, component]
                    admission[str(leaf)][label] = {'accepted': accepted, 'common_shrink': common, 'paired_gain_sse': [value[0] for value in metrics], 'tests': evidence['tests'], 'success_rate': success_rate, 'source_win_fraction': source_win, 'cumulative_noise_gain_sse': list(evidence['noise_gain'])}
            self.collector._integrated_ridge_weight.copy_(deployed)
            row['admission'] = admission
        self._accumulate_data(data)
        self.pending = self._solve()
        self.history.append(row)
        accepted_count = int((self.collector._integrated_ridge_weight.square().sum(dim=(-1, -2)) > 0.0).sum().item()) if block > 0 else 0
        if int(block) == 24:
            self.terminal_weight = self.collector._integrated_ridge_weight.detach().cpu().clone()
            del self.collector._buffers['_integrated_ridge_weight']
            delattr(self.collector, 'refine_candidates')
            self.collector = None
        print(json.dumps({'integrated_ridge': {'block': block, 'accepted': accepted_count}}, default=lambda value: value.tolist()), flush=True)

    def finalize(self, gate_checkpoint: Path) -> None:
        if self.last_evaluated is None or len(self.history) != 25:
            raise RuntimeError('integrated ridge did not complete 25 refreshes')
        if self.terminal_weight is None:
            raise RuntimeError('terminal ridge weight was not sealed before frozen-model audit')
        weight = self.terminal_weight
        accepted = weight.square().sum(dim=(-1, -2)) > 0.0
        payload = {'schema': 'hamiformer.hamiballs.COMPONENT_ROUTER.qp_integrated_ridge_stage_parent.v1', 'status': 'COMPLETE', 'training_updates_added': 0, 'extra_rf_forwards': 0, 'source_noise_per_refresh': 64, 'physical_sources_per_refresh': 32, 'paired_noise_count': 2, 'ridge_relative': RIDGE_RELATIVE, 'reliability_rule': {'minimum_test_blocks': MIN_TEST_BLOCKS, 'minimum_block_success_fraction': MIN_BLOCK_SUCCESS, 'minimum_source_win_fraction': MIN_SOURCE_WIN, 'both_cumulative_noise_gains_positive': True}, 'gate_checkpoint': str(gate_checkpoint), 'ridge_weight': weight, 'accepted_mask': accepted, 'evidence': {f'{leaf}_{component}': value for (leaf, component), value in self.evidence.items()}, 'tree': self.encoder.tree, 'feature_names': self.encoder.feature_names, 'feature_mean': self.encoder.feature_mean.cpu(), 'feature_scale': self.encoder.feature_scale.cpu(), 'history': self.history, 'wall_seconds': time.perf_counter() - self.started}
        torch.save(payload, RIDGE_OUTPUT)
        (OUTPUT / 'ridge_summary.json').write_text(json.dumps({key: value for key, value in payload.items() if key not in {'ridge_weight', 'feature_mean', 'feature_scale', 'history'}}, indent=2, sort_keys=True, default=lambda value: value.tolist()) + '\n')

def main(controller_factory=IntegratedRidge) -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    component_router.configure_runner()
    r_registration = R_REGISTRATION_OVERRIDE if R_REGISTRATION_OVERRIDE is not None else ROOT / 'configs/hamiballs1/residual.json'
    gate_registration = GATE_REGISTRATION_OVERRIDE if GATE_REGISTRATION_OVERRIDE is not None else ROOT / 'configs/hamiballs1/gate.json'
    fresh.canonical.PREPARED = COMPONENT_ROUTER_ROOT / 'prepared'
    fresh.canonical._install(r_registration=r_registration, gate_registration=gate_registration)
    controller = controller_factory()
    packed.SELF_POLICY_REFRESH_COLLECTOR_ADAPTER = controller.adapt_collector
    packed.SELF_POLICY_REFRESH_CACHE_OBSERVER = controller
    parent = FINAL_GATE_PARENT_OVERRIDE if FINAL_GATE_PARENT_OVERRIDE is not None else COMPONENT_ROUTER_ROOT / 'scalar1/global_sequence_projection_regret_with_source_object_qp_no_harm_gate.pt'
    args = ['--registration', str(gate_registration), '--dataset-root', str(fresh.canonical.DATASET_ROOT), '--output-dir', str(QP_OUTPUT), '--fit-samples', str(FINAL_FIT_SAMPLES), '--holdout-samples', str(FINAL_HOLDOUT_SAMPLES), '--collection-batch', str(FINAL_COLLECTION_BATCH), '--sequence-batch', '128', '--updates', '200', '--self-policy-refresh-every', '8', '--self-policy-refresh-sources', str(FINAL_REFRESH_SOURCES), '--field-indices', ','.join((str(i) for i in range(18))), '--sample-seed', '1570077070', '--sample-offset', '4288', '--noise-seeds', '1942634267,2035767743', '--optimizer-sample-seed', '1301159805', '--maximum-lr', str(QP_MAXIMUM_LR), '--minimum-lr', str(QP_MINIMUM_LR), '--arms', component_router.QP_OBJECTIVE, '--gate-checkpoint', str(parent), '--online-recurrence', '--self-policy-stage-parent-carriers', '--self-policy-aggregate-replay', '--sample-excluded-scene-configuration', str(fresh.EXCLUSION_CONFIGURATION), '--observable-disjoint-component-gate-expansion', '--observable-disjoint-delta-width', str(component_router.QP_WIDTH), *FINAL_EXTRA_ARGUMENTS, '--device', 'cuda']
    if fresh.EXCLUSION_CONFIGURATION is None:
        index = args.index('--sample-excluded-scene-configuration')
        del args[index:index + 2]
    old = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *args]
    try:
        packed.main()
    finally:
        sys.argv = old
        packed.SELF_POLICY_REFRESH_COLLECTOR_ADAPTER = None
        packed.SELF_POLICY_REFRESH_CACHE_OBSERVER = None
    checkpoints = sorted(QP_OUTPUT.glob('*.pt'))
    if len(checkpoints) != 1:
        raise RuntimeError(f'expected one qp checkpoint, found {checkpoints}')
    controller.finalize(checkpoints[0])
    (OUTPUT / 'COMPLETE').write_text('COMPLETE\n')
    print(json.dumps({'status': 'COMPLETE', 'gate_checkpoint': str(checkpoints[0]), 'ridge_terminal': str(RIDGE_OUTPUT)}), flush=True)
