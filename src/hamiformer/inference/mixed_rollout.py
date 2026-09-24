from __future__ import annotations
import math
import numpy as np
import torch
from hamiformer.flow.rectified_flow import clean_to_velocity
from hamiformer.models.hamiballs2_dual_expert import hamiballs2_node_graph_features
from hamiformer.models.hamiballs2_hamiltonian import graph_context
from hamiformer.physics.generic_gfjp import generic_gfjp_scan
from hamiformer.physics.hamiballs_type2 import flatten_hamiballs_phase, unflatten_hamiballs_phase
from hamiformer.physics.pgf_scan import apply_prefix, parallel_doubling_prefix, serial_prefix

def _canonical_flat(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[..., :3].flatten(1), state[..., 3:].flatten(1)), -1)

def _compose_affine_substeps(matrix: torch.Tensor, offset: torch.Tensor, *, edges: int, substeps: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, internal_edges, width, second_width = matrix.shape
    if width != second_width or internal_edges != edges * substeps:
        raise ValueError('substep affine-jet shape mismatch')
    if offset.shape != (batch, internal_edges, width):
        raise ValueError('substep affine-offset shape mismatch')
    matrices = matrix.reshape(batch, edges, substeps, width, width)
    offsets = offset.reshape(batch, edges, substeps, width)
    total_matrix = torch.eye(width, device=matrix.device, dtype=matrix.dtype)[None, None].expand(batch, edges, -1, -1).clone()
    total_offset = torch.zeros(batch, edges, width, device=offset.device, dtype=offset.dtype)
    for substep in range(substeps):
        local_matrix = matrices[:, :, substep]
        local_offset = offsets[:, :, substep]
        total_offset = (local_matrix @ total_offset.unsqueeze(-1)).squeeze(-1) + local_offset
        total_matrix = local_matrix @ total_matrix
    return (total_matrix, total_offset)

def _replace_unhealthy_jets_with_d_reset(matrix: torch.Tensor, offset: torch.Tensor, reset_state: torch.Tensor, healthy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.shape[:-2] != healthy.shape or offset.shape[:-1] != healthy.shape:
        raise ValueError('jet health mask does not align with affine maps')
    if reset_state.shape != offset.shape:
        raise ValueError('D-reset state does not align with affine offsets')
    safe_matrix = torch.where(healthy[..., None, None], matrix, torch.zeros_like(matrix))
    safe_offset = torch.where(healthy[..., None], offset, reset_state)
    return (safe_matrix, safe_offset)

def _tangent_health_with_certified_bounds(matrix: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('tangent spectral norm limit must be finite and positive')
    finite = torch.isfinite(matrix).all(dim=(-2, -1))
    gram = matrix.transpose(-1, -2) @ matrix
    gram_diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
    lower_square = gram_diagonal.amax(dim=-1)
    upper_square = gram.abs().sum(dim=-1).amax(dim=-1)
    limit_square = limit * limit
    certified_healthy = finite & (upper_square <= limit_square)
    certified_unhealthy = ~finite | (lower_square > limit_square)
    ambiguous = ~(certified_healthy | certified_unhealthy)
    healthy = certified_healthy.clone()
    if bool(ambiguous.any()):
        largest_eigenvalue = torch.linalg.eigvalsh(gram[ambiguous], UPLO='U')[..., -1]
        healthy[ambiguous] = torch.isfinite(largest_eigenvalue) & (largest_eigenvalue <= limit_square)
    return (healthy, ambiguous)

class MixedRollout:

    def __init__(self, *, cfg, dataset, wide, hamiltonian, model, device, exact_tangent_gate: bool | None=None):
        self.cfg, self.dataset, self.wide, self.h, self.model, self.device = (cfg, dataset, wide, hamiltonian, model, device)
        self.scale = torch.tensor(cfg['model']['phase_scale'], device=device, dtype=torch.float32)
        self.frame_dt = float(cfg['hamiltonian']['frame_dt'])
        self.t_eps = 0.05
        self.exact_tangent_gate = cfg['hamiltonian'].get('carrier_tangent_diagnostic') == 'exact_threshold_certified_gram' if exact_tangent_gate is None else bool(exact_tangent_gate)

    @staticmethod
    def _cold_start_intervals(rf_steps: int) -> int:
        if int(rf_steps) < 2:
            raise ValueError('stateful H2 sampling requires at least two RF steps')
        return min(2, int(rf_steps) - 1)

    def _wide(self, state, tau, batch):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            d, tokens = self.wide.forward_with_tokens(state, tau, x0=batch['phase'][:, 0].float(), attrs=batch['attrs'].float(), physical_time=batch['time'].float(), object_mask=batch['object_mask'].bool(), spring_mask=batch['spring_mask'], spring_k=batch['spring_k'].float(), spring_rest_length=batch['spring_rest_length'].float())
        return (d.float(), tokens.float())

    @torch.no_grad()
    def _d_only_field(self, state, tau, batch):
        d, tokens = self._wide(state, tau, batch)
        next_anchor = (torch.cat((batch['phase'][:, :1, :, :3].float(), d[:, :-1, :, :3]), 1).detach(), d[..., 3:].detach())
        return (d, tokens, next_anchor)

    def _jets_once(self, d, batch, *, substeps: int, anchor=None):
        phase = batch['phase'].float()
        objects = phase.shape[2]
        source = torch.cat((phase[:, :1], d[:, :-1]), 1)
        if anchor is None:
            anchor_q, anchor_p = (source[..., :3], d[..., 3:])
        else:
            if substeps != 1:
                raise ValueError('stateful H2 anchors require one PLAS substep per frame')
            anchor_q, anchor_p = anchor
            expected = (*d.shape[:3], 3)
            if anchor_q.shape != expected or anchor_p.shape != expected:
                raise ValueError('H2 anchor shape does not match D candidate')
        q_rows, p_rows, reset_rows = ([], [], [])
        for substep in range(substeps):
            left = float(substep) / float(substeps)
            right = float(substep + 1) / float(substeps)
            q_rows.append((anchor_q + left * (d[..., :3] - anchor_q)).reshape(len(d), d.shape[1], 3 * objects))
            p_rows.append((anchor_p + (right - 1.0) * (d[..., 3:] - anchor_p)).reshape(len(d), d.shape[1], 3 * objects))
            reset_rows.append(source + right * (d - source))
        q_anchor = torch.stack(q_rows, 2).reshape(len(d), d.shape[1] * substeps, 3 * objects)
        p_anchor = torch.stack(p_rows, 2).reshape(len(d), d.shape[1] * substeps, 3 * objects)
        reset = torch.stack(reset_rows, 2).reshape(len(d), d.shape[1] * substeps, objects, 6)
        reset_flat = torch.cat((reset[..., :3].flatten(2), reset[..., 3:].flatten(2)), -1)
        initial = flatten_hamiballs_phase(phase[:, 0], q_dim=3)
        context = graph_context(batch['attrs'].float(), batch['object_mask'].bool(), batch['spring_mask'], batch['spring_k'].float(), batch['spring_rest_length'].float())[:, None].expand(-1, d.shape[1] * substeps, -1, -1, -1)
        result = generic_gfjp_scan(self.h, q_anchor, p_anchor, context, initial, step_size=self.frame_dt / substeps, mixed_singular_floor=1e-12, mixed_condition_limit=1000000000000.0, tangent_spectral_norm_limit=float(np.finfo(np.float32).max), method=str(self.cfg['hamiltonian']['plas_method']), differentiable=False, compute_tangent_spectral_norm=False)
        mixed_min = result.jets.mixed_singular_min_per_map
        mixed_condition = result.jets.mixed_condition_per_map
        finite_matrix = torch.isfinite(result.jets.matrix).all(dim=(-2, -1))
        healthy = torch.isfinite(mixed_min) & torch.isfinite(mixed_condition) & finite_matrix & (mixed_min >= float(self.cfg['hamiltonian']['mixed_singular_floor'])) & (mixed_condition <= float(self.cfg['hamiltonian']['mixed_condition_limit']))
        tangent_exact_eigensolve = torch.zeros_like(healthy)
        if self.exact_tangent_gate:
            tangent_healthy, tangent_exact_eigensolve = _tangent_health_with_certified_bounds(result.jets.matrix, float(self.cfg['hamiltonian']['tangent_diagnostic_limit']))
            healthy = healthy & tangent_healthy
        matrix_internal, offset_internal = _replace_unhealthy_jets_with_d_reset(result.jets.matrix, result.jets.offset, reset_flat, healthy)
        prefix = serial_prefix(matrix_internal, offset_internal) if str(self.cfg['hamiltonian']['plas_method']) == 'serial' else parallel_doubling_prefix(matrix_internal, offset_internal)
        states = apply_prefix(prefix, initial)
        internal = unflatten_hamiballs_phase(states, num_objects=objects, q_dim=3)
        h_raw = internal[:, substeps - 1::substeps]
        matrix, offset = _compose_affine_substeps(matrix_internal, offset_internal, edges=d.shape[1], substeps=substeps)
        accepted_tangent = torch.zeros(len(d), device=d.device)
        d_reset_mask = (~healthy).reshape(len(d), d.shape[1], substeps).any(-1)
        d_reset_edges = d_reset_mask.sum(-1).to(torch.int16)
        tangent_exact_eigensolve_edges = tangent_exact_eigensolve.reshape(len(d), d.shape[1], substeps).any(-1).sum(-1).to(torch.int16)
        return (h_raw, matrix, offset, accepted_tangent, d_reset_edges, d_reset_mask, tangent_exact_eigensolve_edges)

    def _jets(self, d, batch, *, anchor=None):
        return self._jets_once(d, batch, substeps=int(self.cfg['hamiltonian']['plas_substeps_per_frame']), anchor=anchor)

    @torch.no_grad()
    def _field(self, state, tau, batch, *, parent_mode, route_generator=None, anchor=None, return_affine_jets=False):
        d, tokens = self._wide(state, tau, batch)
        h_raw, matrix, offset, tangent, d_reset_edges, d_reset_mask, tangent_exact_eigensolve_edges = self._jets(d, batch, anchor=anchor)
        node_graph = hamiballs2_node_graph_features(batch['object_mask'].bool(), batch['spring_mask'], batch['spring_k'].float(), batch['spring_rest_length'].float())
        forced = None
        if parent_mode == 'random':
            if route_generator is None:
                raise ValueError('random carrier requires its registered route generator')
            forced = torch.rand(state.shape[0], state.shape[1], state.shape[2], 2, device=self.device, generator=route_generator)
        result = self.model(state, d, h_raw, tokens, x0=batch['phase'][:, 0].float(), attrs=batch['attrs'].float(), node_graph=node_graph, physical_time=batch['time'][:, 1:].float(), tau=tau, object_mask=batch['object_mask'].bool(), jet_matrix=matrix, jet_offset=offset, force_gate=forced)
        result['next_anchor'] = (torch.cat((batch['phase'][:, :1, :, :3].float(), result['mixed'][:, :-1, :, :3]), 1), result['h_candidate'][..., 3:].detach())
        target = batch['phase'][:, 1:].float()
        true_previous = torch.cat((batch['phase'][:, :1], batch['phase'][:, 1:-1]), 1).float()
        flat_previous = torch.stack([_canonical_flat(true_previous[:, edge]) for edge in range(true_previous.shape[1])], 1)
        local_flat = torch.matmul(matrix, flat_previous.unsqueeze(-1)).squeeze(-1) + offset
        objects = state.shape[2]
        local_h = torch.cat((local_flat[..., :3 * objects].reshape(len(state), state.shape[1], objects, 3), local_flat[..., 3 * objects:].reshape(len(state), state.shape[1], objects, 3)), -1)
        result_tuple = (result, d, tokens, local_h, forced, tangent, d_reset_edges, d_reset_mask, tangent_exact_eigensolve_edges)
        return (*result_tuple, matrix, offset) if return_affine_jets else result_tuple

    @torch.no_grad()
    def sample_stateful(self, source, batch, *, rf_steps, parent_mode='learned', route_generator=None):
        if int(self.cfg['hamiltonian']['plas_substeps_per_frame']) != 1:
            raise ValueError('stateful H2 sampling requires one PLAS substep per frame')
        state = source.clone()
        grid = torch.linspace(0.0, 1.0, int(rf_steps) + 1, device=state.device, dtype=state.dtype)
        committed = None
        diagnostics = []
        cold_intervals = self._cold_start_intervals(int(rf_steps))
        for step in range(int(rf_steps)):
            left, right = (grid[step], grid[step + 1])
            tau = left.expand(len(state))
            is_cold = step < cold_intervals
            if is_cold:
                left_clean, left_tokens, next_anchor = self._d_only_field(state, tau, batch)
                velocity_left = clean_to_velocity(left_clean, state, tau, t_eps=self.t_eps)
                diagnostics.append({'is_cold': True, 'd_reset_mask': torch.empty((len(state), 0), device=state.device, dtype=torch.bool), 'tangent_exact_eigensolve_edges': torch.zeros(len(state), device=state.device, dtype=torch.int64)})
            else:
                left_value = self._field(state, tau, batch, parent_mode=parent_mode, route_generator=route_generator, anchor=committed)
                left_output = left_value[0]
                next_anchor = tuple((value.detach() for value in left_output['next_anchor']))
                velocity_left = clean_to_velocity(left_output['mixed'], state, tau, t_eps=self.t_eps)
                diagnostics.append({'is_cold': False, 'd_reset_mask': left_value[7], 'tangent_exact_eigensolve_edges': left_value[8]})
            if step == int(rf_steps) - 1:
                state = state + (right - left) * velocity_left
                break
            proposal = state + (right - left) * velocity_left
            right_tau = right.expand(len(state))
            if is_cold:
                right_clean, _, _ = self._d_only_field(proposal, right_tau, batch)
                velocity_right = clean_to_velocity(right_clean, proposal, right_tau, t_eps=self.t_eps)
            else:
                right_value = self._field(proposal, right_tau, batch, parent_mode=parent_mode, route_generator=route_generator, anchor=committed)
                velocity_right = clean_to_velocity(right_value[0]['mixed'], proposal, right_tau, t_eps=self.t_eps)
            state = state + 0.5 * (right - left) * (velocity_left + velocity_right)
            committed = next_anchor
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError('stateful H2 sampler produced NaN/Inf')
        return (state, diagnostics)
