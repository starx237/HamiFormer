from __future__ import annotations
from hamiformer.utils.paths import project_root
import copy
import json
import time
import numpy as np
import torch
from hamiformer.training.hamiballs1 import tree_readout as clean
from hamiformer.training.hamiballs1.tree_controller import SameRunETrController
from hamiformer.training.hamiballs1.stage_tree import StageBoundaryHierarchicalController
TREE_FIT_QP_BLOCK = 8

class QPLateTreeController(SameRunETrController):

    def __init__(self, *, gate_mode: str='linear') -> None:
        super().__init__(gate_mode=gate_mode)
        self.fit_wall_seconds = 0.0
        self.tree_fit_qp_block_actual: int | None = None
        self.final_r_frozen = False

    @staticmethod
    @torch.no_grad()
    def _clone_gate_parent_to_children(module: torch.nn.Module) -> float:
        name = 'etrg_leaf_gate_linear' if hasattr(module, 'etrg_leaf_gate_linear') else 'etrg_leaf_gate_bias'
        value = getattr(module, name, None)
        if not isinstance(value, torch.Tensor) or value.shape[0] != 8:
            raise ValueError('late-tree gate readout is absent or malformed')
        parent = value[0].detach().clone()
        value[1:].copy_(parent.unsqueeze(0).expand_as(value[1:]))
        return float((value - value[0:1]).abs().max().cpu())

    def _install(self, module: torch.nn.Module, *, trainable_gate: bool) -> None:
        already = bool(getattr(module, '_etrg_pre_gate_installed', False))
        super()._install(module, trainable_gate=trainable_gate)
        if self.tree_fitted and (not already):
            error = self._clone_gate_parent_to_children(module)
            if error != 0.0:
                raise AssertionError('late-tree collector clone changed by leaf')

    @staticmethod
    def _heldout_score(weight: np.ndarray, design: np.ndarray, residual: np.ndarray, leaf: np.ndarray, evaluate: np.ndarray) -> dict[str, float]:
        index = np.flatnonzero(evaluate)
        result: dict[str, float] = {}
        old_total = new_total = 0.0
        for component, label, sl in ((0, 'q', slice(0, 2)), (1, 'p', slice(2, 4))):
            prediction = np.stack([design[row] @ weight[int(leaf[row]), component] for row in index], axis=0)
            truth = residual[index, sl]
            old = float(np.square(truth).sum())
            new = float(np.square(truth - prediction).sum())
            result[label] = 100.0 * (old - new) / max(old, 1e-30)
            old_total += old
            new_total += new
        result['total'] = 100.0 * (old_total - new_total) / max(old_total, 1e-30)
        return result

    def _fit_standard(self, *, fit_tree: bool) -> dict[str, object]:
        started = time.perf_counter()
        data = self._combine(self.bootstrap_records)
        masks = self._fold_masks(data)
        if fit_tree:
            tree, leaf, names = clean._tree(data, masks['grow'])
        else:
            names = self._selected_features(data, masks['grow'])
            tree = {'depth': 0, 'leaf': 0}
            leaf = np.zeros(len(np.asarray(data['source'])), np.int64)
        lookup = {name: index for index, name in enumerate(data['names'])}
        raw = np.asarray(data['x'], np.float32)[:, [lookup[name] for name in names]]
        mean = raw[masks['head']].mean(0)
        scale = np.maximum(raw[masks['head']].std(0), 1e-05)
        design = self._design(data, names, mean, scale)
        if fit_tree:
            leaf = self._leaf_ids(tree, data)
        residual = np.asarray(data['target'], np.float64) - np.asarray(data['h_candidate'], np.float64)
        fitted = self._solve_np(design, residual, leaf, masks['head'])
        deployed, calibration = StageBoundaryHierarchicalController._global_calibrate(fitted, design, residual, leaf, np.asarray(data['source'], np.int64), masks['safety'])
        self.encoder.set_fitted_state(tree=tree, feature_names=names, feature_mean=torch.from_numpy(mean), feature_scale=torch.from_numpy(scale))
        device = self.encoder.feature_mean.device
        self.active_weight = torch.from_numpy(deployed).float().to(device)
        self.bootstrap_records.clear()
        elapsed = time.perf_counter() - started
        self.fit_wall_seconds += elapsed
        return {'mode': 'depth3_cart_independent_ridge' if fit_tree else 'one_leaf_ridge', 'sources': int(len(np.unique(data['source']))), 'leaf_count': int(leaf.max()) + 1, 'selected_features': names, 'tree': copy.deepcopy(tree), 'calibration': calibration, 'heldout': self._heldout_score(deployed, design, residual, leaf, masks['safety']), 'fit_wall_seconds': elapsed, 'active_slots': int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())}

    @torch.no_grad()
    def observe(self, *, block, candidate, caches) -> None:
        self.global_block += 1
        row: dict[str, object] = {'global_block': self.global_block, 'stage': self.stage, 'stage_block': int(block), 'noise_count': len(caches)}
        if self.stage == 'qp' and self.final_r_frozen:
            row.update({'mode': 'fixed_final_tree_and_r', 'sampled_rows': 0, 'active_slots': int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())})
        else:
            records = [self._sample_cache(candidate, cache, noise_slot=index) for index, cache in enumerate(caches)]
            self.bootstrap_records.extend(records)
            row['sampled_rows'] = int(sum((len(value['source']) for value in records)))
            if self.stage == 'qp' and int(block) == TREE_FIT_QP_BLOCK:
                row['fit'] = self._fit_standard(fit_tree=True)
                self.tree_fitted = True
                self.final_r_frozen = True
                self.tree_fit_qp_block_actual = int(block)
                clone_error = self._clone_gate_parent_to_children(candidate)
                if clone_error != 0.0:
                    raise AssertionError('late-tree candidate split was not exact')
                row['function_preserving_gate_split_max_abs'] = clone_error
            else:
                row['mode'] = 'collect_provisional_qp_parent_history' if self.stage == 'qp' else 'collect_for_one_leaf_residual'
            row['active_slots'] = int((self.active_weight.square().sum(dim=(-1, -2)) > 0).sum().item())
            row['active_abs_max'] = float(self.active_weight.abs().max().cpu())
        self.history.append(copy.deepcopy(row))
        print(json.dumps({'qp_late_tree_etr': row}), flush=True)

    @torch.no_grad()
    def finish_stage(self, stage: str) -> None:
        if stage in {'scalar0', 'scalar1'}:
            report = self._fit_standard(fit_tree=False)
            row = {'boundary_after': stage, 'fit': report}
        elif stage == 'qp_bootstrap':
            released = int(sum((len(value['source']) for value in self.bootstrap_records)))
            self.bootstrap_records.clear()
            row = {'boundary_after': stage, 'mode': 'release_scalar1_parent_records', 'released_sampled_rows': released}
        elif stage == 'qp':
            if not self.final_r_frozen or not self.tree_fitted:
                raise RuntimeError('final q/p stage ended before T/r fit')
            row = {'boundary_after': stage, 'mode': 'retain_fixed_terminal_tree_and_r', 'tree_fit_stage_block': self.tree_fit_qp_block_actual}
        else:
            return
        self.history.append(copy.deepcopy(row))
        print(json.dumps({'qp_late_tree_etr': row}), flush=True)

    def payload(self) -> dict[str, object]:
        value = super().payload()
        value.update({'schema': 'hamiformer.hamiballs.etrg.qp_late_tree.v1', 'tree_fit_schedule': 'once_in_final_qp_from_provisional_qp_stage_parent', 'tree_fit_qp_block': self.tree_fit_qp_block_actual, 'residual_fit': 'ordinary_independent_leaf_ridge_global_qp_holdout_shrink', 'final_r_frozen_after_tree_fit': self.final_r_frozen, 'fit_wall_seconds': self.fit_wall_seconds})
        return value

    def boundary_payload(self) -> dict[str, object]:
        return {'schema': 'hamiformer.hamiballs.etrg.qp_late_tree_boundary.v1', 'gate_mode': self.gate_mode, 'stage': self.stage, 'global_block': self.global_block, 'tree_fitted': self.tree_fitted, 'tree': copy.deepcopy(self.encoder.tree), 'feature_names': list(self.encoder.feature_names), 'feature_mean': self.encoder.feature_mean.detach().cpu(), 'feature_scale': self.encoder.feature_scale.detach().cpu(), 'active_weight': self.active_weight.detach().cpu(), 'pending_weight': None if self.pending_weight is None else self.pending_weight.detach().cpu(), 'history': copy.deepcopy(self.history), 'evidence': copy.deepcopy(self.evidence), 'fit_wall_seconds': self.fit_wall_seconds, 'tree_fit_qp_block_actual': self.tree_fit_qp_block_actual, 'final_r_frozen': self.final_r_frozen}

    def load_boundary_payload(self, value: dict[str, object]) -> None:
        if value.get('schema') != 'hamiformer.hamiballs.etrg.qp_late_tree_boundary.v1' or value.get('gate_mode') != self.gate_mode:
            raise ValueError('late-tree controller boundary contract drifted')
        self.stage = str(value['stage'])
        self.global_block = int(value['global_block'])
        self.tree_fitted = bool(value['tree_fitted'])
        self.encoder.set_fitted_state(tree=copy.deepcopy(value['tree']), feature_names=list(value['feature_names']), feature_mean=value['feature_mean'], feature_scale=value['feature_scale'])
        self.active_weight = value['active_weight'].clone()
        pending = value['pending_weight']
        self.pending_weight = None if pending is None else pending.clone()
        self.history = copy.deepcopy(value['history'])
        self.evidence = copy.deepcopy(value['evidence'])
        self.fit_wall_seconds = float(value['fit_wall_seconds'])
        actual = value['tree_fit_qp_block_actual']
        self.tree_fit_qp_block_actual = None if actual is None else int(actual)
        self.final_r_frozen = bool(value['final_r_frozen'])
        self.bootstrap_records.clear()
__all__ = ['QPLateTreeController', 'TREE_FIT_QP_BLOCK']
