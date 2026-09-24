from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
import time
import numpy as np
import torch
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1.tree_controller import RIDGE_RELATIVE, SameRunETrController

class StageBoundaryHierarchicalController(SameRunETrController):

    def __init__(self, *, gate_mode: str='linear') -> None:
        super().__init__(gate_mode=gate_mode)
        self.tree_boundary_complete = False
        self.residual_boundary_complete = False
        self.tree_fit_global_block_actual: int | None = None
        self.fit_wall_seconds = 0.0

    @staticmethod
    def _hierarchical_solve(design: np.ndarray, target: np.ndarray, leaf: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        d = design.shape[-1]
        groups = 9
        width = groups * d
        result = np.zeros((8, 2, d, 2), np.float64)
        diagnostics: dict[str, object] = {}
        active_leaves = sorted((int(value) for value in np.unique(leaf)))
        for component, label, sl in ((0, 'q', slice(0, 2)), (1, 'p', slice(2, 4))):
            gram = np.zeros((width, width), np.float64)
            rhs = np.zeros((width, 2), np.float64)
            for region in active_leaves:
                use = fit & (leaf == region)
                x = design[use]
                y = target[use, sl]
                local_gram = x.T @ x
                local_rhs = x.T @ y
                common = slice(0, d)
                local = slice((region + 1) * d, (region + 2) * d)
                gram[common, common] += local_gram
                gram[common, local] += local_gram
                gram[local, common] += local_gram
                gram[local, local] += local_gram
                rhs[common] += local_rhs
                rhs[local] += local_rhs
            common_diag = np.diag(gram[:d, :d])
            lam = RIDGE_RELATIVE * max(float(common_diag.mean()), 1e-12)
            penalty = np.ones(width, np.float64) * lam
            for block in range(groups):
                penalty[(block + 1) * d - 1] *= 0.01
            coefficient = np.linalg.solve(gram + np.diag(penalty), rhs).reshape(groups, d, 2)
            shared = coefficient[0]
            deviation_norms = []
            for region in active_leaves:
                deviation = coefficient[region + 1]
                result[region, component] = shared + deviation
                deviation_norms.append(float(np.linalg.norm(deviation)))
            diagnostics[label] = {'lambda': lam, 'common_l2': float(np.linalg.norm(shared)), 'leaf_deviation_l2': deviation_norms}
        return (result, diagnostics)

    @staticmethod
    def _global_calibrate(weight: np.ndarray, design: np.ndarray, target: np.ndarray, leaf: np.ndarray, source: np.ndarray, evaluate: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        deployed = np.zeros_like(weight)
        report: dict[str, object] = {}
        for component, label, sl in ((0, 'q', slice(0, 2)), (1, 'p', slice(2, 4))):
            use_index = np.flatnonzero(evaluate)
            prediction = np.stack([design[index] @ weight[int(leaf[index]), component] for index in use_index], axis=0)
            truth = target[use_index, sl]
            alpha = float(np.clip(float((prediction * truth).sum()) / max(float(np.square(prediction).sum()), 1e-30), 0.0, 1.0))
            old_sse = float(np.square(truth).sum())
            new_sse = float(np.square(truth - alpha * prediction).sum())
            source_values = np.unique(source[use_index])
            source_wins = 0
            for value in source_values:
                local = source[use_index] == value
                source_wins += int(np.square(truth[local] - alpha * prediction[local]).sum() < np.square(truth[local]).sum())
            accepted = alpha > 0.0 and new_sse < old_sse
            if accepted:
                deployed[:, component] = alpha * weight[:, component]
            report[label] = {'accepted': accepted, 'alpha': alpha, 'gain_sse': old_sse - new_sse, 'relative_sse_percent': 100.0 * (old_sse - new_sse) / max(old_sse, 1e-30), 'source_win_fraction': source_wins / max(len(source_values), 1), 'sources': int(len(source_values))}
        return (deployed, report)

    def _fit_fixed_tree_residual(self, *, fit_tree: bool) -> dict[str, object]:
        started = time.perf_counter()
        data = self._combine(self.bootstrap_records)
        masks = self._fold_masks(data)
        if fit_tree:
            tree, leaf, names = clean._tree(data, masks['grow'])
            lookup = {name: i for i, name in enumerate(data['names'])}
            raw = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in names]]
            mean = raw[masks['head']].mean(0)
            scale = np.maximum(raw[masks['head']].std(0), 1e-05)
            self.encoder.set_fitted_state(tree=tree, feature_names=names, feature_mean=torch.from_numpy(mean), feature_scale=torch.from_numpy(scale))
        else:
            names = list(self.encoder.feature_names)
            tree = copy.deepcopy(self.encoder.tree)
            leaf = self._leaf_ids(tree, data)
            mean = self.encoder.feature_mean.detach().cpu().numpy()
            scale = self.encoder.feature_scale.detach().cpu().numpy()
        design = self._design(data, names, mean, scale)
        if fit_tree:
            leaf = self._leaf_ids(tree, data)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        independent_fit = self._solve_np(design, residual, leaf, masks['head'])
        independent, independent_admission = self._admit_np(independent_fit, design, residual, leaf, np.asarray(data['source'], np.int64), masks['safety'])
        fitted, hierarchy = self._hierarchical_solve(design, residual, leaf, masks['head'])
        deployed, calibration = self._global_calibrate(fitted, design, residual, leaf, np.asarray(data['source'], np.int64), masks['safety'])
        device = self.encoder.feature_mean.device
        self.active_weight = torch.from_numpy(deployed).float().to(device)
        elapsed = time.perf_counter() - started
        self.fit_wall_seconds += elapsed

        def heldout_score(weight: np.ndarray) -> dict[str, float]:
            index = np.flatnonzero(masks['safety'])
            result: dict[str, float] = {}
            new_total = old_total = 0.0
            for component, label, sl in ((0, 'q', slice(0, 2)), (1, 'p', slice(2, 4))):
                local_prediction = np.stack([design[row] @ weight[int(leaf[row]), component] for row in index], axis=0)
                truth = residual[index, sl]
                old = float(np.square(truth).sum())
                new = float(np.square(truth - local_prediction).sum())
                result[label] = 100.0 * (old - new) / max(old, 1e-30)
                old_total += old
                new_total += new
            result['total'] = 100.0 * (old_total - new_total) / max(old_total, 1e-30)
            return result
        return {'mode': 'tree_and_hierarchical_r' if fit_tree else 'fixed_tree_hierarchical_r', 'sources': int(len(np.unique(data['source']))), 'leaf_count': int(leaf.max()) + 1, 'selected_features': names, 'tree': tree if fit_tree else None, 'hierarchy': hierarchy, 'calibration': calibration, 'same_cache_control': {'independent_leaf_hard_source_majority': heldout_score(independent), 'hierarchical_global_calibration': heldout_score(deployed), 'independent_admission': independent_admission}, 'fit_wall_seconds': elapsed, 'active_slots': int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())}

    @torch.no_grad()
    def observe(self, *, block, candidate, caches) -> None:
        self.global_block += 1
        row: dict[str, object] = {'global_block': self.global_block, 'stage': self.stage, 'stage_block': int(block), 'noise_count': len(caches)}
        if self.stage == 'qp' and self.residual_boundary_complete:
            row.update({'mode': 'fixed_final_etr', 'sampled_rows': 0, 'active_slots': int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())})
        else:
            records = [self._sample_cache(candidate, cache, noise_slot=index) for index, cache in enumerate(caches)]
            self.bootstrap_records.extend(records)
            row['sampled_rows'] = int(sum((len(value['source']) for value in records)))
            if self.global_block == 0:
                row['fit'] = self._fit_common()
            else:
                row['mode'] = 'collect_for_tree' if not self.tree_boundary_complete else 'collect_for_fixed_tree_refit'
            row['active_slots'] = int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())
            row['active_abs_max'] = float(self.active_weight.abs().max().cpu())
        self.history.append(copy.deepcopy(row))
        print(json.dumps({'stage_boundary_etr': row}), flush=True)

    @torch.no_grad()
    def finish_stage(self, stage: str) -> None:
        if stage == 'scalar0':
            report = self._fit_fixed_tree_residual(fit_tree=True)
            self.tree_fitted = True
            self.tree_boundary_complete = True
            self.tree_fit_global_block_actual = self.global_block
        elif stage == 'scalar1':
            if not self.tree_boundary_complete:
                raise RuntimeError('scalar1 ended before the E/T boundary fit')
            report = self._fit_fixed_tree_residual(fit_tree=False)
            self.residual_boundary_complete = True
        else:
            return
        row = {'boundary_after': stage, 'fit': report}
        self.history.append(copy.deepcopy(row))
        print(json.dumps({'stage_boundary_etr': row}), flush=True)

    def payload(self) -> dict[str, object]:
        value = super().payload()
        if not self.residual_boundary_complete:
            raise RuntimeError('final stage-boundary residual was not fitted')
        value.update({'schema': 'hamiformer.hamiballs.etrg.stage_boundary_hierarchical.v1', 'tree_fit_schedule': 'scalar0_boundary_once', 'tree_fit_global_block': self.tree_fit_global_block_actual, 'residual_refit_schedule': 'scalar0_and_scalar1_boundaries', 'qp_residual_frozen': True, 'hierarchical_parameterization': 'common_plus_leaf_deviation_compiled_to_leaf', 'source_majority_used_as_hard_gate': False, 'fit_wall_seconds': self.fit_wall_seconds})
        return value
__all__ = ['StageBoundaryHierarchicalController']
