from __future__ import annotations
from hamiformer.utils.paths import project_root
import json
import math
from pathlib import Path
import sys
import torch
from hamiformer.training.hamiballs1 import ridge_training as base
import types
from hamiformer.training.hamiballs1.tree_candidate import install_pre_gate_refiner
from hamiformer.training.hamiballs1.tree_candidate import conditional_delta
import numpy as np
ROOT = project_root()

def _majority_probability(wins: int, sources: int) -> float:
    if sources < 1:
        return 0.0
    probability = (wins + 0.5) / (sources + 1.0)
    variance = probability * (1.0 - probability) / (sources + 1.0)
    z = (probability - 0.5) / max(math.sqrt(variance), 1e-12)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

class SoftIntegratedRidge(base.IntegratedRidge):

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

    @torch.no_grad()
    def __call__(self, *, block, candidate, caches) -> None:
        if len(caches) != 2:
            raise ValueError('soft integrated ridge requires paired-noise caches')
        data = [self._decode(candidate, cache) for cache in caches]
        row = {'block': int(block)}
        if self.pending is not None:
            self.last_evaluated = self.pending.clone()
            continuous = {}
            deployed = torch.zeros_like(self.pending)
            for leaf in range(self.leaf_capacity):
                continuous[str(leaf)] = {}
                for component, label in enumerate(('q', 'p')):
                    alphas = []
                    for item in data:
                        use = item['leaf'] == leaf
                        x = item['x'][use]
                        sl = slice(0, 2) if component == 0 else slice(2, 4)
                        prediction = x @ self.pending[leaf, component]
                        target = item['target'][use, sl] - item['base_hr'][use, sl]
                        dot = (prediction * target).sum()
                        p2 = prediction.square().sum().clamp_min(1e-30)
                        alphas.append(float((dot / p2).clamp(0.0, 1.0)))
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
                    reliability = positive_confidence(evidence['source_wins'], evidence['sources'])
                    scale = common * reliability
                    deployed[leaf, component] = scale * self.pending[leaf, component]
                    continuous[str(leaf)][label] = {'common_shrink': common, 'source_majority_probability': reliability, 'deployed_scale': scale, 'paired_gain_sse': [value[0] for value in metrics], 'tests': evidence['tests'], 'source_win_fraction': evidence['source_wins'] / max(evidence['sources'], 1), 'cumulative_noise_gain_sse': list(evidence['noise_gain'])}
            self.collector._integrated_ridge_weight.copy_(deployed)
            row['continuous_reliability'] = continuous
        self._accumulate_data(data)
        self.pending = self._solve()
        self.history.append(row)
        if int(block) == 24:
            self.terminal_weight = self.collector._integrated_ridge_weight.detach().cpu().clone()
            del self.collector._buffers['_integrated_ridge_weight']
            delattr(self.collector, 'refine_candidates')
            self.collector = None
        print(json.dumps({'soft_integrated_ridge': {'block': block}}), flush=True)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_ROUTER.qp_soft_reliability_ridge.v1'
        payload.pop('accepted_mask', None)
        payload.pop('reliability_rule', None)
        payload['continuous_reliability_rule'] = {'local_scale': 'minimum paired-noise current-block quadratic optimum in [0,1]', 'reliability': 'Jeffreys-smoothed normal posterior P(source win rate > 0.5)', 'binary_admission': False}
        torch.save(payload, base.RIDGE_OUTPUT)
        (base.OUTPUT / 'ridge_summary.json').write_text(json.dumps({key: value for key, value in payload.items() if key not in {'ridge_weight', 'feature_mean', 'feature_scale', 'history'}}, indent=2, sort_keys=True, default=lambda value: value.tolist()) + '\n')
ROOT = project_root()
positive_ridge_posterior_probability = _majority_probability

def positive_confidence(wins: int, sources: int) -> float:
    return max(0.0, 2.0 * positive_ridge_posterior_probability(wins, sources) - 1.0)

class PositiveConfidenceRidge(SoftIntegratedRidge):

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_ROUTER.qp_positive_confidence_ridge.v1'
        for row in payload['history']:
            for leaf in row.get('continuous_reliability', {}).values():
                for component in leaf.values():
                    component['positive_posterior_confidence'] = component.pop('source_majority_probability')
        payload['continuous_reliability_rule'] = {'local_scale': 'minimum paired-noise current-block quadratic optimum in [0,1]', 'reliability': 'max(0, 2*Jeffreys posterior P(source win rate > 0.5)-1)', 'uninformative_posterior_scale': 0.0, 'binary_admission': False}
        torch.save(payload, base.RIDGE_OUTPUT)
        (base.OUTPUT / 'ridge_summary.json').write_text(json.dumps({key: value for key, value in payload.items() if key not in {'ridge_weight', 'feature_mean', 'feature_scale', 'history'}}, indent=2, sort_keys=True, default=lambda value: value.tolist()) + '\n')
ROOT = project_root()
ridge_rank_PHYSICAL_SOURCES_PER_BLOCK = 32

class RankReadyConfidenceRidge(PositiveConfidenceRidge):

    @torch.no_grad()
    def __call__(self, *, block, candidate, caches) -> None:
        super().__call__(block=block, candidate=candidate, caches=caches)
        independent_fit_sources = int(block) * ridge_rank_PHYSICAL_SOURCES_PER_BLOCK
        rank_ready = independent_fit_sources >= base.INPUT_DIM
        if not rank_ready and self.collector is not None:
            self.collector._integrated_ridge_weight.zero_()
            for leaf in self.history[-1].get('continuous_reliability', {}).values():
                for component in leaf.values():
                    component['pre_readiness_scale'] = component['deployed_scale']
                    component['deployed_scale'] = 0.0
        self.history[-1]['rank_readiness'] = {'independent_fit_sources': independent_fit_sources, 'readout_dimension': base.INPUT_DIM, 'ready': rank_ready}

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_ROUTER.qp_rank_ready_confidence_ridge.v1'
        payload['rank_readiness_rule'] = {'criterion': 'independent_physical_sources >= conditional_readout_dimension', 'physical_sources_per_block': ridge_rank_PHYSICAL_SOURCES_PER_BLOCK, 'conditional_readout_dimension': base.INPUT_DIM, 'first_ready_block': (base.INPUT_DIM + ridge_rank_PHYSICAL_SOURCES_PER_BLOCK - 1) // ridge_rank_PHYSICAL_SOURCES_PER_BLOCK, 'per_leaf_threshold': False}
        torch.save(payload, base.RIDGE_OUTPUT)
        (base.OUTPUT / 'ridge_summary.json').write_text(json.dumps({key: value for key, value in payload.items() if key not in {'ridge_weight', 'feature_mean', 'feature_scale', 'history'}}, indent=2, sort_keys=True, default=lambda value: value.tolist()) + '\n')
ROOT = project_root()
routing_inputs_TREE_REPORT = ROOT / 'outputs/hami1/tree_input_information' / 'arm_no_current_gate.json'

class NoCurrentGateTreeRidge(RankReadyConfidenceRidge):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        report = json.loads(routing_inputs_TREE_REPORT.read_text(encoding='utf-8'))
        if report['name'] != 'no_current_gate':
            raise ValueError('unexpected CART report arm')
        self.encoder.tree = report['tree']

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = base.torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.TreeInputs.no_current_gate_tree.v1'
        payload['tree_report'] = str(routing_inputs_TREE_REPORT)
        base.torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()

def mask_current_gate_features(encoder) -> None:
    names = list(encoder.feature_names)
    blocked = ('gate_q', 'gate_p')
    if any((name not in names for name in blocked)):
        raise ValueError('current-g coordinates missing from conditional readout')
    means = {name: float(encoder.feature_mean[names.index(name)].detach().cpu()) for name in blocked}
    original = encoder._features

    def causal_features(_module, *args, **kwargs):
        features = original(*args, **kwargs)
        for name, mean in means.items():
            features[name] = torch.full_like(features[name], mean)
        return features
    encoder._features = types.MethodType(causal_features, encoder)

class NoCurrentGateReadoutRidge(NoCurrentGateTreeRidge):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        mask_current_gate_features(self.encoder)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.ReadoutInputs.no_current_gate_readout.v1'
        payload['blocked_readout_features'] = ['gate_q', 'gate_p']
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()
precorrection_readout_BLOCKED_R_FEATURES = ('r_0', 'r_1', 'r_2', 'r_3', 'r_q_norm', 'r_p_norm', 'r_gap_q_cos', 'r_gap_p_cos')

def mask_current_r_features(encoder) -> None:
    names = list(encoder.feature_names)
    if any((name not in names for name in precorrection_readout_BLOCKED_R_FEATURES)):
        missing = [name for name in precorrection_readout_BLOCKED_R_FEATURES if name not in names]
        raise ValueError(f'current-r coordinates missing: {missing}')
    means = {name: float(encoder.feature_mean[names.index(name)].detach().cpu()) for name in precorrection_readout_BLOCKED_R_FEATURES}
    original = encoder._features

    def pre_rg_features(_module, *args, **kwargs):
        features = original(*args, **kwargs)
        for name, mean in means.items():
            features[name] = torch.full_like(features[name], mean)
        return features
    encoder._features = types.MethodType(pre_rg_features, encoder)

class PreRgReadoutRidge(NoCurrentGateReadoutRidge):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        mask_current_r_features(self.encoder)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.PreCorrectionReadout.pre_rg_readout.v1'
        payload['blocked_readout_features'] = ['gate_q', 'gate_p', *precorrection_readout_BLOCKED_R_FEATURES]
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()

class MatchedFinalHrGateRidge(PreRgReadoutRidge):

    def __init__(self) -> None:
        super().__init__()
        self.candidate = None

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        self.candidate = candidate
        candidate.register_buffer('_etrg_active_ridge_weight', torch.zeros_like(collector._integrated_ridge_weight), persistent=False)
        gate_bias_getter = self._install_optional_gate_bias(candidate)
        gate_logit_adjuster = self._install_optional_gate_adjuster(candidate)
        install_pre_gate_refiner(candidate, self.encoder, lambda: candidate._etrg_active_ridge_weight, gate_bias_getter, replace_base_hr=self._replace_base_hr(), gate_logit_adjuster=gate_logit_adjuster, candidate_blend_alpha=self._candidate_blend_alpha())

    def _install_optional_gate_bias(self, candidate):
        del candidate
        return None

    def _install_optional_gate_adjuster(self, candidate):
        del candidate
        return None

    def _replace_base_hr(self) -> bool:
        return False

    def _candidate_blend_alpha(self):
        return None

    @torch.no_grad()
    def __call__(self, *, block, candidate, caches) -> None:
        super().__call__(block=block, candidate=candidate, caches=caches)
        if candidate is not self.candidate:
            raise RuntimeError('CandidateGate candidate identity changed')
        if self.collector is not None:
            active = self.collector._integrated_ridge_weight
        elif self.terminal_weight is not None:
            active = self.terminal_weight.to(candidate._etrg_active_ridge_weight)
        else:
            active = torch.zeros_like(candidate._etrg_active_ridge_weight)
        candidate._etrg_active_ridge_weight.copy_(active)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.CandidateGate.matched_final_hr_gate.v1'
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()

class UnifiedTreeResidual(MatchedFinalHrGateRidge):

    def _replace_base_hr(self) -> bool:
        return True

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)

        def replace_candidates(module, *, edge, previous_mixed, h_candidate, hr_candidate, d_candidate, attrs, tau, previous_g, residual_hidden, base_gate, **_unused):
            delta, _leaf = conditional_delta(self.encoder, module._integrated_ridge_weight, edge=int(edge), previous_mixed=previous_mixed, h_candidate=h_candidate, base_hr_candidate=hr_candidate, d_candidate=d_candidate, attrs=attrs, tau=tau, previous_g=previous_g, residual_hidden=residual_hidden)
            return (h_candidate + delta, base_gate)
        collector.refine_candidates = types.MethodType(replace_candidates, collector)

    @torch.no_grad()
    def _decode(self, candidate, cache):
        data = super()._decode(candidate, cache)
        data['base_hr'] = cache['h_candidate'].reshape(-1, 4)
        return data

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.UnifiedRefiner.unified_tree_r.v1'
        payload['residual_target'] = 'clean_minus_H_on_current_mixed_predecessor'
        payload['deployment_expert'] = 'H_plus_single_tree_residual'
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()

class StableUnifiedTreeResidual(UnifiedTreeResidual):

    def __init__(self) -> None:
        super().__init__()
        self.debug_block = -1
        self._gradient_hooks = []

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)

        def check_gradient(name):

            def hook(gradient):
                if not bool(torch.isfinite(gradient).all()):
                    print(json.dumps({'nonfinite_gradient': {'block': self.debug_block, 'parameter': name}}), flush=True)
                    raise FloatingPointError(f'non-finite ScaledRefiner gradient at block {self.debug_block}: {name}')
                return gradient
            return hook
        self._gradient_hooks = [parameter.register_hook(check_gradient(name)) for name, parameter in candidate.named_parameters() if parameter.requires_grad]

    @torch.no_grad()
    def __call__(self, *, block, candidate, caches) -> None:
        self.debug_block = int(block)
        super().__call__(block=block, candidate=candidate, caches=caches)
        trainable = [parameter.detach() for parameter in candidate.parameters() if parameter.requires_grad]
        print(json.dumps({'scaled_refiner_support_numeric_audit': {'block': int(block), 'parameter_abs_max': max((float(value.abs().amax().cpu()) for value in trainable)), 'ridge_weight_abs_max': float(candidate._etrg_active_ridge_weight.abs().amax().cpu()), 'all_parameters_finite': all((bool(torch.isfinite(value).all()) for value in trainable))}}), flush=True)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.ScaledRefiner.unified_r_lr003.v1'
        payload['maximum_lr'] = base.QP_MAXIMUM_LR
        payload['minimum_lr'] = base.QP_MINIMUM_LR
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()
aligned_readout_TREE_REPORT = ROOT / 'outputs/hami1/clean_tree_unified_r' / 'report.json'

class StableCleanAlignedUnifiedTreeResidual(StableUnifiedTreeResidual):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        report = json.loads(aligned_readout_TREE_REPORT.read_text(encoding='utf-8'))
        if int(report['leaf_count']) != 7:
            raise ValueError('clean-H-aligned CART must have seven leaves')
        if bool(report['tree_reads_current_r_or_gate']):
            raise ValueError('clean-H-aligned CART unexpectedly reads current r/g')
        self.encoder.tree = report['tree']

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hami1.aligned_readout.v1'
        payload['tree_report'] = str(aligned_readout_TREE_REPORT)
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()
readout_scaling_TREE_ROOT = ROOT / 'outputs/hami1/clean_tree_unified_r'
readout_scaling_SELECTED_DIM = 32

class StableCompactCleanTreeUnifiedResidual(StableCleanAlignedUnifiedTreeResidual):

    def _initialize(self, candidate, collector) -> None:
        super()._initialize(candidate, collector)
        report = json.loads((readout_scaling_TREE_ROOT / 'report.json').read_text(encoding='utf-8'))
        arrays = np.load(readout_scaling_TREE_ROOT / 'model.npz')
        names = [str(value) for value in report['selected_features']]
        if len(names) != readout_scaling_SELECTED_DIM or len(set(names)) != readout_scaling_SELECTED_DIM:
            raise ValueError('expected 32 unique train-selected causal features')
        if any((name.startswith(('gate_', 'r_')) for name in names)):
            raise ValueError('compact clean readout unexpectedly uses current r/g')
        device = self.encoder.feature_mean.device
        dtype = self.encoder.feature_mean.dtype
        self.encoder.feature_names = names
        self.encoder.feature_mean = torch.from_numpy(arrays['feature_mean']).to(device=device, dtype=dtype)
        self.encoder.feature_scale = torch.from_numpy(arrays['feature_scale']).to(device=device, dtype=dtype)

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.CompactScale.clean_tree_compact_unified_r_lr003.v1'
        payload['readout_feature_count'] = readout_scaling_SELECTED_DIM
        payload['readout_columns_with_bias'] = base.INPUT_DIM
        payload['readout_learned_scalars_7_leaves'] = 7 * 2 * base.INPUT_DIM * 2
        torch.save(payload, base.RIDGE_OUTPUT)
ROOT = project_root()

class CompactUnifiedResidualGateOffset(StableCompactCleanTreeUnifiedResidual):

    def _install_optional_gate_bias(self, candidate):
        device = next(candidate.parameters()).device
        candidate.register_parameter('etrg_leaf_gate_bias', torch.nn.Parameter(torch.zeros(8, 2, device=device)))
        candidate._declared_extra_trainable_parameters = 16
        return lambda: candidate.etrg_leaf_gate_bias

    def finalize(self, gate_checkpoint: Path) -> None:
        super().finalize(gate_checkpoint)
        payload = torch.load(base.RIDGE_OUTPUT, map_location='cpu', weights_only=False)
        payload['schema'] = 'hamiformer.hamiballs.COMPONENT_REFINER.CompactGateOffset.compact_unified_r_gate_offset_lr003.v1'
        payload['added_gate_parameters'] = 16
        torch.save(payload, base.RIDGE_OUTPUT)
