from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
import numpy as np
import torch
from hamiformer.training.hamiballs1 import tree_features as tree_tools
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1.tree_encoder import CausalObservableTreeEncoder
from hamiformer.training.hamiballs1.tree_candidate import install_pre_gate_refiner
FEATURE_COUNT = 32
INPUT_DIM = FEATURE_COUNT + 1
TREE_FIT_BLOCK = 4
CELLS_PER_SOURCE_NOISE = 128
RIDGE_RELATIVE = 0.01
MIN_TESTS = 4
MIN_SUCCESS = 0.8
MIN_SOURCE_WIN = 0.5

def _componentize(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value[..., None].expand(*value.shape, 2)
    if value.ndim == 3 and value.shape[-1] != 2:
        return value[..., None].expand(*value.shape, 2)
    if value.ndim == 3 and value.shape[-1] == 2:
        return value
    if value.ndim == 4 and value.shape[-1] == 2:
        return value
    raise ValueError(f'unexpected q/p history shape {tuple(value.shape)}')

class SameRunETrController:

    def __init__(self, *, gate_mode: str='constant') -> None:
        if gate_mode not in {'constant', 'linear'}:
            raise ValueError('gate_mode must be constant or linear')
        self.gate_mode = gate_mode
        self.encoder = CausalObservableTreeEncoder()
        self.active_weight = torch.zeros(8, 2, 2, 2)
        self.pending_weight: torch.Tensor | None = None
        self.stage = ''
        self.global_block = -1
        self.tree_fitted = False
        self.bootstrap_records: list[dict[str, np.ndarray]] = []
        self.xtx: torch.Tensor | None = None
        self.xty: torch.Tensor | None = None
        self.evidence = {(leaf, component): {'tests': 0, 'successes': 0, 'source_wins': 0, 'sources': 0, 'gain': 0.0} for leaf in range(8) for component in range(2)}
        self.history: list[dict[str, object]] = []

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage)

    def _cell_sample_seed(self, noise_slot: int) -> int:
        return 905043 + 1009 * (self.global_block + 1) + 9176 * int(noise_slot)

    def _gate_install(self, candidate: torch.nn.Module):
        if self.stage != 'qp':
            return (None, None, 0)
        device = next(candidate.parameters()).device
        if self.gate_mode == 'constant':
            candidate.register_parameter('etrg_leaf_gate_bias', torch.nn.Parameter(torch.zeros(8, 2, device=device)))
            return (lambda: candidate.etrg_leaf_gate_bias, None, 16)
        design_dim = 72
        candidate.register_parameter('etrg_leaf_gate_linear', torch.nn.Parameter(torch.zeros(8, 2, design_dim, device=device)))

        def adjust(observable: torch.Tensor, leaf: torch.Tensor) -> torch.Tensor:
            design = torch.cat((observable, torch.ones_like(observable[..., :1])), dim=-1)
            local = candidate.etrg_leaf_gate_linear[leaf.long()]
            return torch.matmul(local, design.unsqueeze(-1)).squeeze(-1)
        return (None, adjust, 8 * 2 * design_dim)

    def _install(self, module: torch.nn.Module, *, trainable_gate: bool) -> None:
        if bool(getattr(module, '_etrg_pre_gate_installed', False)):
            return
        device = next(module.parameters()).device
        self.encoder.to(device).eval()
        self.active_weight = self.active_weight.to(device)
        bias_getter = adjuster = None
        extra = 0
        if trainable_gate:
            bias_getter, adjuster, extra = self._gate_install(module)
            if extra:
                module._declared_extra_trainable_parameters = extra
        install_pre_gate_refiner(module, self.encoder, lambda: self.active_weight, bias_getter, replace_base_hr=True, gate_logit_adjuster=adjuster)

    def adapt_collector(self, *, block, candidate, collector):
        del block
        self._install(candidate, trainable_gate=True)
        if collector is not candidate:
            self._install(collector, trainable_gate=False)
        return collector

    @torch.no_grad()
    def _sample_cache(self, candidate: torch.nn.Module, cache: dict[str, torch.Tensor], *, noise_slot: int) -> dict[str, np.ndarray]:
        base_gate = _componentize(cache['base_gate'])
        previous_g = _componentize(cache['previous_g'])
        features = self.encoder._features(previous_mixed=cache['previous_mixed'], h_candidate=cache['h_candidate'], hr_candidate=cache['base_hr_candidate'], d_candidate=cache['d_candidate'], attrs=cache['attrs'], tau=cache['tau'], previous_g=previous_g, residual_hidden=cache['residual_hidden'], base_gate=base_gate)
        names = list(features)
        x = torch.stack([features[name] for name in names], dim=-1).reshape(-1, len(names))
        source = cache['source_index'][:, None, None].expand(cache['h_candidate'].shape[:-1]).reshape(-1)
        source_cpu = source.detach().cpu().numpy()
        rng = np.random.default_rng(self._cell_sample_seed(noise_slot))
        chosen = []
        for value in np.unique(source_cpu):
            available = np.flatnonzero(source_cpu == value)
            chosen.append(rng.choice(available, min(CELLS_PER_SOURCE_NOISE, len(available)), replace=False))
        index_np = np.sort(np.concatenate(chosen)).astype(np.int64, copy=False)
        index = torch.from_numpy(index_np).to(x.device)
        return {'x': x[index].float().cpu().numpy(), 'names': np.asarray(names), 'source': source[index].long().cpu().numpy(), 'target': cache['target'].reshape(-1, 4)[index].float().cpu().numpy(), 'h_candidate': cache['h_candidate'].reshape(-1, 4)[index].float().cpu().numpy(), 'd_candidate': cache['d_candidate'].reshape(-1, 4)[index].float().cpu().numpy()}

    @staticmethod
    def _combine(records: list[dict[str, np.ndarray]]) -> dict[str, object]:
        if not records:
            raise ValueError('same-run ETr controller has no records')
        names = records[0]['names'].tolist()
        if any((row['names'].tolist() != names for row in records)):
            raise ValueError('same-run feature order drifted')
        return {'x': np.concatenate([row['x'] for row in records]), 'names': names, 'source': np.concatenate([row['source'] for row in records]), 'target': np.concatenate([row['target'] for row in records]), 'h_candidate': np.concatenate([row['h_candidate'] for row in records]), 'd_candidate': np.concatenate([row['d_candidate'] for row in records])}

    @staticmethod
    def _fold_masks(data: dict[str, object]) -> dict[str, np.ndarray]:
        fold = clean._fold(np.asarray(data['source'], np.int64))
        return {'grow': fold < 2, 'head': fold == 2, 'safety': fold == 3}

    @staticmethod
    def _selected_features(data: dict[str, object], grow: np.ndarray) -> list[str]:
        x = np.asarray(data['x'], np.float32)
        names = list(data['names'])
        allowed = clean._clean_columns(names)
        a, b, _c = clean._pure_quadratics(data)
        selected = tree_tools._select_features_quad(x[:, allowed], a, b, np.flatnonzero(grow), min(FEATURE_COUNT, len(allowed)))
        return [names[index] for index in allowed[selected]]

    @staticmethod
    def _design(data: dict[str, object], names: list[str], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        lookup = {name: index for index, name in enumerate(data['names'])}
        x = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in names]]
        x = np.clip((x - mean) / scale, -8.0, 8.0)
        return np.concatenate((x, np.ones((len(x), 1), np.float32)), axis=-1).astype(np.float64)

    @staticmethod
    def _leaf_ids(tree: dict[str, object], data: dict[str, object]) -> np.ndarray:
        return clean._assign(tree, np.asarray(data['x'], np.float32), list(data['names']))

    @staticmethod
    def _solve_np(design: np.ndarray, target: np.ndarray, leaf: np.ndarray, fit: np.ndarray) -> np.ndarray:
        result = np.zeros((8, 2, design.shape[-1], 2), np.float64)
        for region in range(int(leaf.max()) + 1):
            use = fit & (leaf == region)
            local_x = design[use]
            if len(local_x) == 0:
                continue
            gram = local_x.T @ local_x
            lam = RIDGE_RELATIVE * max(float(np.diag(gram).mean()), 1e-12)
            penalty = np.eye(len(gram)) * lam
            penalty[-1, -1] = lam * 0.01
            for component, sl in ((0, slice(0, 2)), (1, slice(2, 4))):
                result[region, component] = np.linalg.solve(gram + penalty, local_x.T @ target[use, sl])
        return result

    @staticmethod
    def _admit_np(weight: np.ndarray, design: np.ndarray, target: np.ndarray, leaf: np.ndarray, source: np.ndarray, evaluate: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        deployed = np.zeros_like(weight)
        report: dict[str, object] = {}
        for region in range(int(leaf.max()) + 1):
            report[str(region)] = {}
            for component, label, sl in ((0, 'q', slice(0, 2)), (1, 'p', slice(2, 4))):
                use = evaluate & (leaf == region)
                prediction = design[use] @ weight[region, component]
                truth = target[use, sl]
                p2 = float(np.square(prediction).sum())
                alpha = float(np.clip(float((prediction * truth).sum()) / max(p2, 1e-30), 0.0, 1.0))
                old = np.square(truth).sum(-1)
                new = np.square(truth - alpha * prediction).sum(-1)
                values = np.unique(source[use])
                wins = 0
                for value in values:
                    local = use & (source == value)
                    local_prediction = design[local] @ weight[region, component]
                    local_truth = target[local, sl]
                    wins += int(np.square(local_truth - alpha * local_prediction).sum() < np.square(local_truth).sum())
                gain = float(old.sum() - new.sum())
                source_win = wins / max(len(values), 1)
                accepted = gain > 0.0 and source_win >= MIN_SOURCE_WIN and (alpha > 0.0)
                if accepted:
                    deployed[region, component] = alpha * weight[region, component]
                report[str(region)][label] = {'accepted': accepted, 'alpha': alpha, 'gain_sse': gain, 'source_win_fraction': source_win, 'sources': int(len(values))}
        return (deployed, report)

    def _fit_common(self) -> dict[str, object]:
        data = self._combine(self.bootstrap_records)
        masks = self._fold_masks(data)
        names = self._selected_features(data, masks['grow'])
        lookup = {name: i for i, name in enumerate(data['names'])}
        raw = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in names]]
        mean = raw[masks['head']].mean(0)
        scale = np.maximum(raw[masks['head']].std(0), 1e-05)
        design = self._design(data, names, mean, scale)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        leaf = np.zeros(len(design), np.int64)
        fitted = self._solve_np(design, residual, leaf, masks['head'])
        deployed, admission = self._admit_np(fitted, design, residual, leaf, np.asarray(data['source'], np.int64), masks['safety'])
        self.encoder.set_fitted_state(tree={'depth': 0, 'leaf': 0}, feature_names=names, feature_mean=torch.from_numpy(mean), feature_scale=torch.from_numpy(scale))
        self.active_weight = torch.from_numpy(deployed).float().to(self.encoder.feature_mean.device)
        return {'mode': 'one_leaf', 'sources': int(len(np.unique(data['source']))), 'admission': admission}

    def _fit_tree(self) -> dict[str, object]:
        data = self._combine(self.bootstrap_records)
        masks = self._fold_masks(data)
        tree, leaf, names = clean._tree(data, masks['grow'])
        lookup = {name: i for i, name in enumerate(data['names'])}
        raw = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in names]]
        mean = raw[masks['head']].mean(0)
        scale = np.maximum(raw[masks['head']].std(0), 1e-05)
        design = self._design(data, names, mean, scale)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        fitted = self._solve_np(design, residual, leaf, masks['head'])
        deployed, admission = self._admit_np(fitted, design, residual, leaf, np.asarray(data['source'], np.int64), masks['safety'])
        self.encoder.set_fitted_state(tree=tree, feature_names=names, feature_mean=torch.from_numpy(mean), feature_scale=torch.from_numpy(scale))
        device = self.encoder.feature_mean.device
        self.active_weight = torch.from_numpy(deployed).float().to(device)
        self.pending_weight = torch.from_numpy(fitted).float().to(device)
        self.xtx = torch.zeros(8, INPUT_DIM, INPUT_DIM, dtype=torch.float64, device=device)
        self.xty = torch.zeros(8, 2, INPUT_DIM, 2, dtype=torch.float64, device=device)
        self._accumulate_torch(torch.from_numpy(design).to(device), torch.from_numpy(residual).to(device), torch.from_numpy(leaf).to(device))
        self.pending_weight = self._solve_torch().float()
        self.tree_fitted = True
        self.bootstrap_records.clear()
        return {'mode': 'depth3_tree', 'sources': int(len(np.unique(data['source']))), 'leaf_count': int(leaf.max()) + 1, 'selected_features': names, 'tree': tree, 'admission': admission}

    def _encode_record(self, record: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = {'x': record['x'], 'names': record['names'].tolist()}
        design = self._design(data, self.encoder.feature_names, self.encoder.feature_mean.detach().cpu().numpy(), self.encoder.feature_scale.detach().cpu().numpy())
        full = {'x': record['x'], 'names': record['names'].tolist()}
        leaf = self._leaf_ids(self.encoder.tree, full)
        residual = record['target'].astype(np.float64) - record['h_candidate'].astype(np.float64)
        device = self.encoder.feature_mean.device
        return (torch.from_numpy(design).to(device), torch.from_numpy(residual).to(device), torch.from_numpy(leaf).to(device), torch.from_numpy(record['source']).to(device))

    def _accumulate_torch(self, design: torch.Tensor, target: torch.Tensor, leaf: torch.Tensor) -> None:
        if self.xtx is None or self.xty is None:
            raise RuntimeError('tree statistics are uninitialized')
        for region in range(8):
            use = leaf == region
            if not bool(use.any()):
                continue
            local = design[use]
            self.xtx[region] += local.T @ local
            for component, sl in ((0, slice(0, 2)), (1, slice(2, 4))):
                self.xty[region, component] += local.T @ target[use, sl]

    def _solve_torch(self) -> torch.Tensor:
        if self.xtx is None or self.xty is None:
            raise RuntimeError('tree statistics are uninitialized')
        result = torch.zeros_like(self.xty)
        penalty_scale = torch.ones(INPUT_DIM, dtype=self.xtx.dtype, device=self.xtx.device)
        penalty_scale[-1] = 0.01
        penalty = torch.diag(penalty_scale)
        for region in range(8):
            lam = RIDGE_RELATIVE * self.xtx[region].diagonal().mean().clamp_min(1e-08)
            matrix = self.xtx[region] + lam * penalty
            for component in range(2):
                result[region, component] = torch.linalg.solve(matrix, self.xty[region, component])
        return result

    @torch.no_grad()
    def _test_and_update(self, records: list[dict[str, np.ndarray]]) -> dict[str, object]:
        if self.pending_weight is None:
            raise RuntimeError('same-run pending ridge is unavailable')
        rows = [self._encode_record(record) for record in records]
        report: dict[str, object] = {}
        deployed = torch.zeros_like(self.pending_weight)
        for region in range(8):
            report[str(region)] = {}
            for component, label, sl in ((0, 'q', slice(0, 2)), (1, 'p', slice(2, 4))):
                gains = []
                wins = sources = 0
                alphas = []
                for design, target, leaf, source in rows:
                    use = leaf == region
                    prediction = design[use] @ self.pending_weight[region, component].double()
                    truth = target[use, sl]
                    alpha = ((prediction * truth).sum() / prediction.square().sum().clamp_min(1e-30)).clamp(0.0, 1.0)
                    alphas.append(float(alpha.cpu()))
                    old = truth.square().sum(-1)
                    new = (truth - alpha * prediction).square().sum(-1)
                    gains.append(float((old.sum() - new.sum()).cpu()))
                    for value in source[use].unique():
                        local = use & (source == value)
                        local_prediction = design[local] @ self.pending_weight[region, component].double()
                        local_truth = target[local, sl]
                        wins += int((local_truth - alpha * local_prediction).square().sum() < local_truth.square().sum())
                        sources += 1
                common = min(alphas) if alphas else 0.0
                evidence = self.evidence[region, component]
                evidence['tests'] += 1
                success = common > 0.0 and all((gain > 0.0 for gain in gains))
                evidence['successes'] += int(success)
                evidence['source_wins'] += wins
                evidence['sources'] += sources
                evidence['gain'] += sum(gains)
                success_rate = evidence['successes'] / evidence['tests']
                source_win = evidence['source_wins'] / max(evidence['sources'], 1)
                accepted = evidence['tests'] >= MIN_TESTS and success_rate >= MIN_SUCCESS and (source_win >= MIN_SOURCE_WIN) and (evidence['gain'] > 0.0) and (common > 0.0)
                if accepted:
                    deployed[region, component] = common * self.pending_weight[region, component]
                report[str(region)][label] = {'accepted': accepted, 'common_alpha': common, 'paired_gain_sse': gains, 'tests': evidence['tests'], 'success_rate': success_rate, 'source_win_fraction': source_win, 'cumulative_gain_sse': evidence['gain']}
        if any((value['accepted'] for region in report.values() for value in region.values())):
            self.active_weight = deployed.float()
        for design, target, leaf, _source in rows:
            self._accumulate_torch(design, target, leaf)
        self.pending_weight = self._solve_torch().float()
        return report

    @torch.no_grad()
    def observe(self, *, block, candidate, caches) -> None:
        self.global_block += 1
        records = [self._sample_cache(candidate, cache, noise_slot=index) for index, cache in enumerate(caches)]
        row: dict[str, object] = {'global_block': self.global_block, 'stage': self.stage, 'stage_block': int(block), 'noise_count': len(caches), 'sampled_rows': int(sum((len(value['source']) for value in records)))}
        if not self.tree_fitted:
            self.bootstrap_records.extend(records)
            if self.global_block < TREE_FIT_BLOCK:
                row['fit'] = self._fit_common()
            else:
                row['fit'] = self._fit_tree()
        else:
            row['admission'] = self._test_and_update(records)
        row['active_slots'] = int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())
        row['active_abs_max'] = float(self.active_weight.abs().max().cpu())
        self.history.append(copy.deepcopy(row))
        print(json.dumps({'same_run_etr': row}), flush=True)

    def payload(self) -> dict[str, object]:
        if not self.tree_fitted:
            raise RuntimeError('same-run ETr tree was never fitted')
        return {'schema': 'hamiformer.hamiballs.etrg.same_run_controller.v1', 'status': 'COMPLETE', 'tree': copy.deepcopy(self.encoder.tree), 'feature_names': list(self.encoder.feature_names), 'feature_mean': self.encoder.feature_mean.detach().cpu(), 'feature_scale': self.encoder.feature_scale.detach().cpu(), 'ridge_weight': self.active_weight.detach().cpu(), 'gate_mode': self.gate_mode, 'tree_fit_global_block': TREE_FIT_BLOCK, 'cells_per_source_noise': CELLS_PER_SOURCE_NOISE, 'ridge_relative': RIDGE_RELATIVE, 'history': self.history, 'evidence': {f'{leaf}_{component}': value for (leaf, component), value in self.evidence.items()}, 'training_updates_added': 0, 'extra_rf_forwards': 0, 'validation_or_event_labels_read': False}
__all__ = ['SameRunETrController']
