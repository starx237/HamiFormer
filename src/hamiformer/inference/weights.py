from hamiformer.utils.paths import project_root
from pathlib import Path
import torch
from torch import nn
H1_FEATURES = ['Dstep_p_norm', 'gap_p_norm', 'gap_2', 'Dstep_2', 'observable71_40', 'gap_3', 'Dstep_3', 'observable71_36', 'Hstep_p_norm', 'observable71_37', 'observable71_39', 'observable71_41', 'Hstep_Dstep_p_cos', 'observable71_34', 'Hstep_1', 'observable71_33', 'Hstep_0', 'observable71_35', 'Hstep_q_norm', 'Dstep_1', 'Dstep_0', 'rf_trace', 'observable71_38', 'mass', 'observable71_42', 'gap_q_norm', 'attr2', 'gap_0', 'observable71_31', 'Hstep_2', 'observable71_32', 'Hstep_3']

def load_state(path):
    state = torch.load(Path(path), map_location='cpu', weights_only=True)

    def check(value):
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif not torch.is_tensor(value):
            raise ValueError('Expected tensor model state')
    check(state)
    return state

def h1_diffusion(wide=False):
    from hamiformer.models.phase_dit import PhaseDiT
    return PhaseDiT(state_dim=4, q_dim=2, attr_dim=3, hidden_size=128, depth=4, num_heads=4, mlp_ratio=2.0, mlp_inner_dim=265 if wide else 256, num_register_tokens=2, block_attn_pattern=('spatial', 'temporal', 'object', 'temporal'))

def load_h1(root, device):
    from hamiformer.physics.continuous_hamiltonian import TokenConditionalContinuousHamiltonian
    from hamiformer.models.hamiballs_observable_sidecar_physical_features_observable_sidecar_r import HamiBallsPhysicalFeaturesObservableSidecarResidual
    from hamiformer.models.hamiballs_gate_dual_channel import HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate
    from hamiformer.models.hami1_routed_expert import HamiBalls1RoutedExpert
    root = Path(root)
    state = load_state(root / 'hami1_ours.pt')
    scale = state['state_scale']
    models = {'d': h1_diffusion(), 'wide_d': h1_diffusion(True), 'h': TokenConditionalContinuousHamiltonian(num_objects=5, spatial_tokens=1, coordinate_dim=2, token_context_dim=3, hidden_size=16, depth=2, heads=4, expansion=2.0, q_scale=tuple(scale[:2].tolist()), p_scale=tuple(scale[2:].tolist())), 'r': HamiBallsPhysicalFeaturesObservableSidecarResidual(token_dim=128, state_dim=4, attr_dim=3), 'base_gate': HamiBallsObservableDisjointDeltaPerObjectCompactCommittedGate(delta_width=16, token_dim=128, state_dim=4, attr_dim=3, residual_hidden_dim=71, rank=12)}
    models['r'].per_object_previous_g = True
    models['r'].register_buffer('statistics_fitted', torch.ones((), dtype=torch.uint8), persistent=False)
    models['base_gate'].register_parameter('etrg_leaf_gate_linear', nn.Parameter(torch.empty_like(state['base_gate']['etrg_leaf_gate_linear'])))
    for name in ('d', 'h', 'r', 'base_gate'):
        models[name].load_state_dict(state[name], strict=True)
    models['wide_d'].load_state_dict(load_state(root / 'hami1_physiformer.pt')['model'], strict=True)
    routing = state['routing']
    nodes = routing['tree']

    def tree(i):
        f, t, l, r, leaf = nodes[i].tolist()
        if leaf >= 0:
            return {'leaf': int(leaf)}
        return {'feature': H1_FEATURES[int(f)], 'threshold': t, 'left': tree(int(l)), 'right': tree(int(r))}
    models['gate'] = HamiBalls1RoutedExpert(models.pop('base_gate'), dict(routing, tree=tree(0), feature_names=H1_FEATURES))
    for model in models.values():
        model.to(device).eval().requires_grad_(False)
    models.update(state_scale=scale.to(device), attr_scale=state['attr_scale'].to(device), config={'dataset': {'q_dim': 2}, 'rectified_flow': {'t_eps': 0.05, 'phase_noise_scale': 1.0}, 'hamiltonian': {'mixed_singular_floor': 0.2, 'mixed_condition_limit': 10.0, 'tangent_spectral_norm_limit': 5.0}})
    return models

def load_h2(root, method, device):
    from hamiformer.baselines.physiformer import HamiBalls2WideD
    state = load_state(Path(root) / f'hami2_{method}.pt')
    if method.startswith('transformer_ar'):
        from hamiformer.baselines.graph_transformer import GraphTransformerAR
        model = GraphTransformerAR(context=int(method[-1]))
    elif method == 'dit':
        from hamiformer.baselines.dit import HamiBalls2DiT
        weights = state['model']
        width = weights['input.weight'].shape[0]
        depth = len({k.split('.')[1] for k in weights if k.startswith('blocks.')})
        model = HamiBalls2DiT(hidden_size=width, depth=depth, num_heads=weights['edge_bias.weight'].shape[0])
    elif method == 'physiformer':
        model = HamiBalls2WideD(hidden_size=168, depth=12, num_heads=8, mlp_inner_dim=336, num_register_tokens=2)
    elif method == 'ours':
        from hamiformer.models.hamiballs2_hamiltonian import HamiBalls2ContinuousHamiltonian
        from hamiformer.models.hamiballs2_posthd_formal import HamiBalls2FormalPostHD
        scale = state['model']['phase_scale']
        model = HamiBalls2FormalPostHD(phase_scale=scale.tolist(), d_token_dim=168, residual_width=64, gate_rank=24, gate_hidden=24, gate_qp_width=32, max_leaves=8, leaf_gate_standardized=True)
        model.load_state_dict(state['model'], strict=True)
        tree = {k: v.tolist() for k, v in state['tree'].items()}

        def depth(i):
            return 0 if tree['children_left'][i] < 0 else 1 + max(depth(tree['children_left'][i]), depth(tree['children_right'][i]))
        tree['depth'] = depth(0)
        model.tree = tree
        d = HamiBalls2WideD(hidden_size=168, depth=12, num_heads=8, mlp_inner_dim=336, num_register_tokens=2)
        h = HamiBalls2ContinuousHamiltonian(hidden_size=32, depth=2, heads=4, expansion=2.0, pair_energy_hidden_size=56, explicit_relations_exclusive_to_pair_energy=True, num_objects=10, q_scale=tuple(scale[:3].tolist()), p_scale=tuple(scale[3:].tolist()))
        d.load_state_dict(state['d'], strict=True)
        h.load_state_dict(state['h'], strict=True)
        return {k: v.to(device).eval().requires_grad_(False) for k, v in {'model': model, 'd': d, 'h': h}.items()}
    else:
        raise ValueError(method)
    model.load_state_dict(state['model'], strict=True)
    return {'model': model.to(device).eval().requires_grad_(False), 'state_scale': state['state_scale'].to(device)}

def load_hgdpf(root, device):
    from hamiformer.baselines.hgdpf_hamiballs import make_hamiballs_hgdpf_models
    state = load_state(Path(root) / 'hami1_hgdpf.pt')
    d, h = make_hamiballs_hgdpf_models('matched_wide_d')
    for name, model in [('dpf', d), ('hnn', h)]:
        model.load_state_dict(state[name], strict=True)
        model.to(device).eval().requires_grad_(False)
    return (d, h, {k: v.to(device) for k, v in state['normalization'].items()})
