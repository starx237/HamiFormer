from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from .block_hamiltonian_expert import BoundaryConditionMode, BlockHamiltonianOccurrences, ParallelBlockHamiltonianExpert
AssemblyMode = Literal['incoming', 'odd_even', 'mean', 'ci', 'tangent']
OddEvenOrder = Literal['even_odd', 'odd_even']

@dataclass
class G1YAssemblyOutput:
    clean: torch.Tensor
    occurrences: BlockHamiltonianOccurrences
    disagreement: torch.Tensor
    relation_plus_before: torch.Tensor | None = None
    relation_minus_before: torch.Tensor | None = None
    relation_plus_after: torch.Tensor | None = None
    relation_minus_after: torch.Tensor | None = None
    tangent_alpha: torch.Tensor | None = None
    tangent_step: torch.Tensor | None = None
    tangent_success: torch.Tensor | None = None
    tangent_displacement_normalized: torch.Tensor | None = None
    tangent_line_search_trials: int = 0
    log_variances: dict[str, torch.Tensor] | None = None
    factor_calls: int = 0
    sequential_depth: int = 1

class CorruptionOnlyReliabilityCalibrator(nn.Module):
    _PROPOSAL_TYPES = ('q_plus', 'p_plus', 'q_minus', 'p_minus')

    def __init__(self, *, hidden_size: int=16, log_variance_min: float=-6.0, log_variance_max: float=4.0) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError('hidden_size 必须为正')
        if not log_variance_min < log_variance_max:
            raise ValueError('log-variance 上界必须大于下界')
        self.network = nn.Sequential(nn.Linear(9, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)

    def _bounded_log_variance(self, raw: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (self.log_variance_min + self.log_variance_max)
        radius = 0.5 * (self.log_variance_max - self.log_variance_min)
        return midpoint + radius * torch.tanh(raw)

    def _one_type(self, *, tau: torch.Tensor, first_signal: torch.Tensor, second_signal: torch.Tensor, first_known: torch.Tensor, second_known: torch.Tensor, proposal_type: str) -> torch.Tensor:
        if proposal_type not in self._PROPOSAL_TYPES:
            raise ValueError(f'未知 proposal_type: {proposal_type}')
        batch, windows, block = first_signal.shape
        if block != 1:
            raise ValueError('G1Y reliability calibrator 只支持 b=1')
        type_index = self._PROPOSAL_TYPES.index(proposal_type)
        one_hot = first_signal.new_zeros(batch, windows, block, 4)
        one_hot[..., type_index] = 1.0
        tau_feature = tau[:, None, None, None].expand(batch, windows, block, 1)
        features = torch.cat([tau_feature, first_signal[..., None], second_signal[..., None], first_known.to(first_signal.dtype)[..., None], second_known.to(first_signal.dtype)[..., None], one_hot], dim=-1)
        raw = self.network(features).unsqueeze(-2)
        return self._bounded_log_variance(raw)

    def forward(self, *, tau: torch.Tensor, signal: torch.Tensor, known: torch.Tensor, source_indices: torch.Tensor, target_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        q_source_signal = signal[:, source_indices]
        p_source_signal = signal[:, source_indices]
        q_target_signal = signal[:, target_indices]
        p_target_signal = signal[:, target_indices]
        source_known = known[:, source_indices]
        target_known = known[:, target_indices]
        return {'q_plus': self._one_type(tau=tau, first_signal=q_source_signal, second_signal=p_target_signal, first_known=source_known, second_known=target_known, proposal_type='q_plus'), 'p_plus': self._one_type(tau=tau, first_signal=q_source_signal, second_signal=p_target_signal, first_known=source_known, second_known=target_known, proposal_type='p_plus'), 'q_minus': self._one_type(tau=tau, first_signal=q_target_signal, second_signal=p_source_signal, first_known=target_known, second_known=source_known, proposal_type='q_minus'), 'p_minus': self._one_type(tau=tau, first_signal=q_target_signal, second_signal=p_source_signal, first_known=target_known, second_known=source_known, proposal_type='p_minus')}

class G1YAssemblyHamiltonianExpert(nn.Module):

    def __init__(self, *, num_objects: int, future_steps: int, q_dim: int, attr_dim: int, q_scale: torch.Tensor, p_scale: torch.Tensor, assembly_mode: AssemblyMode, hidden_size: int=128, depth: int=2, num_heads: int=4, mlp_ratio: float=4.0, dropout: float=0.0, odd_even_order: OddEvenOrder='odd_even', tangent_initial_step: float=0.25, tangent_backtracking_steps: int=4, tangent_decrease_tolerance: float=1e-07, boundary_condition_mode: BoundaryConditionMode='none', boundary_anneal_end: float=1.0, force_float32: bool=True) -> None:
        super().__init__()
        if assembly_mode not in {'incoming', 'odd_even', 'mean', 'ci', 'tangent'}:
            raise ValueError(f'未知 assembly_mode: {assembly_mode}')
        if odd_even_order not in {'even_odd', 'odd_even'}:
            raise ValueError('odd_even_order 只能是 even_odd 或 odd_even')
        if tangent_initial_step <= 0:
            raise ValueError('tangent_initial_step 必须为正')
        if tangent_backtracking_steps <= 0:
            raise ValueError('tangent_backtracking_steps 必须为正')
        if tangent_decrease_tolerance < 0:
            raise ValueError('tangent_decrease_tolerance 不能为负')
        self.num_objects = int(num_objects)
        self.future_steps = int(future_steps)
        self.q_dim = int(q_dim)
        self.state_dim = 2 * self.q_dim
        self.assembly_mode: AssemblyMode = assembly_mode
        self.odd_even_order: OddEvenOrder = odd_even_order
        self.tangent_initial_step = float(tangent_initial_step)
        self.tangent_backtracking_steps = int(tangent_backtracking_steps)
        self.tangent_decrease_tolerance = float(tangent_decrease_tolerance)
        self.edge_model = ParallelBlockHamiltonianExpert(num_objects=num_objects, future_steps=future_steps, q_dim=q_dim, attr_dim=attr_dim, block_size=1, block_step=1, q_scale=q_scale, p_scale=p_scale, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout, chart_coupling='independent', boundary_condition_mode=boundary_condition_mode, boundary_anneal_end=boundary_anneal_end, force_float32=force_float32)
        self.reliability = CorruptionOnlyReliabilityCalibrator()

    @property
    def q_scale(self) -> torch.Tensor:
        return self.edge_model.q_scale

    @property
    def p_scale(self) -> torch.Tensor:
        return self.edge_model.p_scale

    @property
    def source_indices(self) -> torch.Tensor:
        return self.edge_model.source_indices

    @property
    def target_indices(self) -> torch.Tensor:
        return self.edge_model.target_indices

    def set_assembly_mode(self, assembly_mode: AssemblyMode) -> None:
        if assembly_mode not in {'incoming', 'odd_even', 'mean', 'ci', 'tangent'}:
            raise ValueError(f'未知 assembly_mode: {assembly_mode}')
        self.assembly_mode = assembly_mode

    def _signal_and_known(self, tau: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tau.shape[0]
        signal = tau[:, None].expand(batch, self.future_steps + 1).to(device=device, dtype=dtype).clone()
        signal[:, 0] = 1.0
        known = torch.zeros(batch, self.future_steps + 1, device=device, dtype=torch.bool)
        known[:, 0] = True
        return (signal, known)

    @staticmethod
    def _squeeze_block(value: torch.Tensor) -> torch.Tensor:
        if value.shape[2] != 1:
            raise ValueError('G1Y 只接受 block 维为1的 occurrences')
        return value[:, :, 0]

    def _edge_state_proposals(self, occurrences: BlockHamiltonianOccurrences) -> tuple[torch.Tensor, torch.Tensor]:
        source = torch.cat([self._squeeze_block(occurrences.q_minus), self._squeeze_block(occurrences.p_plus)], dim=-1)
        target = torch.cat([self._squeeze_block(occurrences.q_plus), self._squeeze_block(occurrences.p_minus)], dim=-1)
        return (source, target)

    def _normalized_disagreement(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError('left/right proposal shape 必须一致')
        state_scale = torch.cat([self.q_scale, self.p_scale]).to(left)
        normalized = (left - right) / state_scale.view(1, 1, 1, -1)
        return normalized.square().mean(dim=(1, 2, 3)).sqrt()

    def _assemble_incoming(self, occurrences: BlockHamiltonianOccurrences) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self._edge_state_proposals(occurrences)
        disagreement = self._normalized_disagreement(target[:, :-1], source[:, 1:])
        return (target, disagreement)

    def _assemble_mean(self, occurrences: BlockHamiltonianOccurrences) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self._edge_state_proposals(occurrences)
        clean = target.clone()
        clean[:, :-1] = 0.5 * (target[:, :-1] + source[:, 1:])
        disagreement = self._normalized_disagreement(target[:, :-1], source[:, 1:])
        return (clean, disagreement)

    def _assemble_ci(self, occurrences: BlockHamiltonianOccurrences, *, tau: torch.Tensor, signal: torch.Tensor, known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source, target = self._edge_state_proposals(occurrences)
        log_variances = self.reliability(tau=tau, signal=signal, known=known, source_indices=occurrences.source_indices, target_indices=occurrences.target_indices)
        q_left_precision = torch.exp(-log_variances['q_plus'][:, :-1, 0])
        q_right_precision = torch.exp(-log_variances['q_minus'][:, 1:, 0])
        p_left_precision = torch.exp(-log_variances['p_minus'][:, :-1, 0])
        p_right_precision = torch.exp(-log_variances['p_plus'][:, 1:, 0])
        q_weight = (q_left_precision / (q_left_precision + q_right_precision).clamp_min(1e-08)).detach()
        p_weight = (p_left_precision / (p_left_precision + p_right_precision).clamp_min(1e-08)).detach()
        clean = target.clone()
        clean[:, :-1, :, :self.q_dim] = q_weight * target[:, :-1, :, :self.q_dim] + (1.0 - q_weight) * source[:, 1:, :, :self.q_dim]
        clean[:, :-1, :, self.q_dim:] = p_weight * target[:, :-1, :, self.q_dim:] + (1.0 - p_weight) * source[:, 1:, :, self.q_dim:]
        disagreement = self._normalized_disagreement(target[:, :-1], source[:, 1:])
        return (clean, disagreement, log_variances)

    @staticmethod
    def _merge_partitioned_occurrences(parts: list[BlockHamiltonianOccurrences]) -> BlockHamiltonianOccurrences:
        if len(parts) != 2:
            raise ValueError('red-black 必须恰好包含两个 half-sweeps')
        source_indices = torch.cat([part.source_indices for part in parts], dim=0)
        target_indices = torch.cat([part.target_indices for part in parts], dim=0)
        order = torch.argsort(source_indices[:, 0])

        def merge(name: str) -> torch.Tensor:
            return torch.cat([getattr(part, name) for part in parts], dim=1)[:, order]
        return BlockHamiltonianOccurrences(q_plus=merge('q_plus'), p_plus=merge('p_plus'), q_minus=merge('q_minus'), p_minus=merge('p_minus'), source_indices=source_indices[order], target_indices=target_indices[order])

    def _red_black_half_sweep(self, full_state: torch.Tensor, *, edge_indices: torch.Tensor, signal: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, x0: torch.Tensor, boundary_strength: torch.Tensor | None, create_graph: bool) -> tuple[torch.Tensor, BlockHamiltonianOccurrences]:
        occurrences = self.edge_model.chart_occurrences(full_state, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=False, create_graph=create_graph, window_indices=edge_indices, detach_state_inputs=True, boundary_strength=boundary_strength)
        source, target = self._edge_state_proposals(occurrences)
        source_nodes = occurrences.source_indices[:, 0]
        target_nodes = occurrences.target_indices[:, 0]
        updated = full_state.index_copy(1, source_nodes, source)
        updated = updated.index_copy(1, target_nodes, target)
        updated = torch.cat([x0[:, None], updated[:, 1:]], dim=1)
        return (updated, occurrences)

    def _assemble_odd_even(self, noisy_future: torch.Tensor, *, x0: torch.Tensor, signal: torch.Tensor, tau: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, BlockHamiltonianOccurrences, torch.Tensor]:
        device = noisy_future.device
        all_edges = torch.arange(self.future_steps, device=device)
        even_edges = all_edges[all_edges.remainder(2) == 0]
        odd_edges = all_edges[all_edges.remainder(2) == 1]
        if odd_edges.numel() == 0:
            odd_edges = even_edges
        order = (even_edges, odd_edges) if self.odd_even_order == 'even_odd' else (odd_edges, even_edges)
        full = torch.cat([x0[:, None], noisy_future], dim=1)
        boundary_strength = self.edge_model.rf_boundary_strength(tau)
        parts: list[BlockHamiltonianOccurrences] = []
        for color_index, edge_indices in enumerate(order):
            if color_index == 1 and torch.equal(edge_indices, order[0]):
                continue
            full, occurrences = self._red_black_half_sweep(full, edge_indices=edge_indices, signal=signal, attrs=attrs, physical_time=physical_time, x0=x0, boundary_strength=boundary_strength, create_graph=create_graph)
            parts.append(occurrences)
        if len(parts) == 1:
            merged = parts[0]
        else:
            merged = self._merge_partitioned_occurrences(parts)
        source, target = self._edge_state_proposals(merged)
        disagreement = self._normalized_disagreement(target[:, :-1], source[:, 1:])
        return (full[:, 1:], merged, disagreement)

    def _clean_relation_objectives(self, candidate_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, preserve_state_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        batch = candidate_future.shape[0]
        full = torch.cat([x0[:, None], candidate_future], dim=1)
        signal = torch.ones(batch, self.future_steps + 1, device=full.device, dtype=full.dtype)
        occurrences = self.edge_model.chart_occurrences(full, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=False, create_graph=preserve_state_graph, detach_state_inputs=not preserve_state_graph)
        q = full[..., :self.q_dim]
        p = full[..., self.q_dim:]
        q_target = q[:, occurrences.target_indices]
        p_target = p[:, occurrences.target_indices]
        q_source = q[:, occurrences.source_indices]
        p_source = p[:, occurrences.source_indices]
        q_scale = self.q_scale.to(full).view(1, 1, 1, 1, -1)
        p_scale = self.p_scale.to(full).view(1, 1, 1, 1, -1)
        plus_residuals = torch.cat([((occurrences.q_plus - q_target) / q_scale).flatten(1), ((occurrences.p_plus - p_source) / p_scale).flatten(1)], dim=1)
        minus_residuals = torch.cat([((occurrences.q_minus - q_source) / q_scale).flatten(1), ((occurrences.p_minus - p_target) / p_scale).flatten(1)], dim=1)
        f_plus = 0.5 * plus_residuals.square().sum(dim=1)
        f_minus = 0.5 * minus_residuals.square().sum(dim=1)
        return (f_plus, f_minus)

    def _line_search_tangent_step(self, anchor: torch.Tensor, direction_raw: torch.Tensor, *, f_plus_before: torch.Tensor, f_minus_before: torch.Tensor, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch = anchor.shape[0]
        selected_step = anchor.new_zeros(batch)
        selected_plus = f_plus_before.detach().clone()
        selected_minus = f_minus_before.detach().clone()
        accepted = torch.zeros(batch, device=anchor.device, dtype=torch.bool)
        anchor_detached = anchor.detach()
        direction_detached = direction_raw.detach()
        plus_reference = f_plus_before.detach()
        minus_reference = f_minus_before.detach()
        tolerance = self.tangent_decrease_tolerance
        already_satisfied = (plus_reference <= tolerance) & (minus_reference <= tolerance)
        accepted = accepted | already_satisfied
        trials_executed = 0
        for index in range(self.tangent_backtracking_steps):
            trials_executed += 1
            step_value = self.tangent_initial_step * 0.5 ** index
            trial = (anchor_detached + step_value * direction_detached).requires_grad_(True)
            with torch.enable_grad():
                trial_plus, trial_minus = self._clean_relation_objectives(trial, x0=x0, attrs=attrs, physical_time=physical_time, preserve_state_graph=False)
            trial_plus = trial_plus.detach()
            trial_minus = trial_minus.detach()
            plus_ok = trial_plus <= plus_reference - tolerance
            minus_ok = trial_minus <= minus_reference - tolerance
            newly_accepted = ~accepted & plus_ok & minus_ok
            selected_step = torch.where(newly_accepted, selected_step.new_full((), step_value), selected_step)
            selected_plus = torch.where(newly_accepted, trial_plus, selected_plus)
            selected_minus = torch.where(newly_accepted, trial_minus, selected_minus)
            accepted = accepted | newly_accepted
            if bool(accepted.all().item()):
                break
        return (selected_step, accepted, selected_plus, selected_minus, trials_executed)

    def _assemble_tangent(self, occurrences: BlockHamiltonianOccurrences, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        anchor, disagreement = self._assemble_mean(occurrences)
        if create_graph:
            geometry_anchor = anchor
            if not geometry_anchor.requires_grad:
                geometry_anchor = geometry_anchor.requires_grad_(True)
        else:
            geometry_anchor = anchor.detach().requires_grad_(True)
        with torch.enable_grad():
            f_plus, f_minus = self._clean_relation_objectives(geometry_anchor, x0=x0, attrs=attrs, physical_time=physical_time, preserve_state_graph=True)
            g_plus_raw = torch.autograd.grad(f_plus.sum(), geometry_anchor, create_graph=create_graph, retain_graph=True)[0]
            g_minus_raw = torch.autograd.grad(f_minus.sum(), geometry_anchor, create_graph=create_graph, retain_graph=create_graph)[0]
        state_scale = torch.cat([self.q_scale, self.p_scale]).to(geometry_anchor)
        scale_view = state_scale.view(1, 1, 1, -1)
        g_plus = g_plus_raw * scale_view
        g_minus = g_minus_raw * scale_view
        difference = g_plus - g_minus
        reduce_dims = tuple(range(1, difference.ndim))
        denominator = difference.square().sum(dim=reduce_dims)
        numerator = g_minus.square().sum(dim=reduce_dims) - (g_plus * g_minus).sum(dim=reduce_dims)
        alpha = torch.where(denominator > 1e-12, numerator / denominator.clamp_min(1e-12), numerator.new_full((), 0.5)).clamp(0.0, 1.0)
        alpha_for_update = alpha.detach()
        alpha_view = alpha_for_update.view(alpha_for_update.shape[0], *[1] * (g_plus.ndim - 1))
        common_gradient = alpha_view * g_plus + (1.0 - alpha_view) * g_minus
        direction_raw = -common_gradient * scale_view
        step, success, f_plus_after, f_minus_after, line_search_trials = self._line_search_tangent_step(geometry_anchor, direction_raw, f_plus_before=f_plus, f_minus_before=f_minus, x0=x0, attrs=attrs, physical_time=physical_time)
        step_view = step.view(step.shape[0], *[1] * (direction_raw.ndim - 1))
        clean = geometry_anchor + step_view * direction_raw
        displacement = ((clean - geometry_anchor) / state_scale.view(1, 1, 1, -1)).square().mean(dim=(1, 2, 3)).sqrt()
        if not create_graph:
            clean = clean.detach()
        return (clean, disagreement, f_plus.detach(), f_minus.detach(), f_plus_after, f_minus_after, alpha.detach(), step.detach(), success.detach(), displacement.detach(), line_search_trials)

    def forward(self, noisy_future: torch.Tensor, tau: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool | None=None) -> G1YAssemblyOutput:
        self.edge_model._validate_inputs(noisy_future, tau, x0, attrs, physical_time)
        if create_graph is None:
            create_graph = self.training
        signal, known = self._signal_and_known(tau, dtype=noisy_future.dtype, device=noisy_future.device)
        if self.assembly_mode == 'odd_even':
            clean, occurrences, disagreement = self._assemble_odd_even(noisy_future, x0=x0, signal=signal, tau=tau, attrs=attrs, physical_time=physical_time, create_graph=create_graph)
            if not create_graph:
                clean = clean.detach()
                occurrences = occurrences.detached()
            return G1YAssemblyOutput(clean=clean, occurrences=occurrences, disagreement=disagreement.detach(), factor_calls=self.future_steps, sequential_depth=2 if self.future_steps > 1 else 1)
        full = torch.cat([x0[:, None], noisy_future], dim=1)
        occurrences = self.edge_model.chart_occurrences(full, q_signal=signal, p_signal=signal, attrs=attrs, physical_time=physical_time, sequential=False, create_graph=create_graph, boundary_strength=self.edge_model.rf_boundary_strength(tau))
        if self.assembly_mode == 'incoming':
            clean, disagreement = self._assemble_incoming(occurrences)
            output = G1YAssemblyOutput(clean=clean, occurrences=occurrences, disagreement=disagreement, factor_calls=self.future_steps)
        elif self.assembly_mode == 'mean':
            clean, disagreement = self._assemble_mean(occurrences)
            output = G1YAssemblyOutput(clean=clean, occurrences=occurrences, disagreement=disagreement, factor_calls=self.future_steps)
        elif self.assembly_mode == 'ci':
            clean, disagreement, log_variances = self._assemble_ci(occurrences, tau=tau, signal=signal, known=known)
            output = G1YAssemblyOutput(clean=clean, occurrences=occurrences, disagreement=disagreement, log_variances=log_variances, factor_calls=self.future_steps)
        else:
            clean, disagreement, plus_before, minus_before, plus_after, minus_after, alpha, step, success, displacement, line_search_trials = self._assemble_tangent(occurrences, x0=x0, attrs=attrs, physical_time=physical_time, create_graph=create_graph)
            output = G1YAssemblyOutput(clean=clean, occurrences=occurrences, disagreement=disagreement, relation_plus_before=plus_before, relation_minus_before=minus_before, relation_plus_after=plus_after, relation_minus_after=minus_after, tangent_alpha=alpha, tangent_step=step, tangent_success=success, tangent_displacement_normalized=displacement, tangent_line_search_trials=line_search_trials, factor_calls=self.future_steps * (2 + line_search_trials), sequential_depth=2 + line_search_trials)
        if not create_graph:
            output.clean = output.clean.detach()
            output.occurrences = output.occurrences.detached()
            output.disagreement = output.disagreement.detach()
            if output.log_variances is not None:
                output.log_variances = {name: value.detach() for name, value in output.log_variances.items()}
        return output

    def clean_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, create_graph: bool) -> BlockHamiltonianOccurrences:
        return self.edge_model.clean_relation_occurrences(clean_future, x0=x0, attrs=attrs, physical_time=physical_time, create_graph=create_graph)

    def noisy_relation_occurrences(self, clean_future: torch.Tensor, *, x0: torch.Tensor, attrs: torch.Tensor, physical_time: torch.Tensor, noise_scale: float, create_graph: bool, generator: torch.Generator | None=None) -> BlockHamiltonianOccurrences:
        return self.edge_model.noisy_relation_occurrences(clean_future, x0=x0, attrs=attrs, physical_time=physical_time, noise_scale=noise_scale, create_graph=create_graph, generator=generator)

    def reliability_nll(self, output: G1YAssemblyOutput, clean_full: torch.Tensor) -> torch.Tensor:
        if output.log_variances is None:
            return clean_full.new_zeros(())
        occurrences = output.occurrences
        q = clean_full[..., :self.q_dim]
        p = clean_full[..., self.q_dim:]
        targets = {'q_plus': q[:, occurrences.target_indices], 'p_plus': p[:, occurrences.source_indices], 'q_minus': q[:, occurrences.source_indices], 'p_minus': p[:, occurrences.target_indices]}
        proposals = {'q_plus': occurrences.q_plus, 'p_plus': occurrences.p_plus, 'q_minus': occurrences.q_minus, 'p_minus': occurrences.p_minus}
        scales = {'q_plus': self.q_scale, 'p_plus': self.p_scale, 'q_minus': self.q_scale, 'p_minus': self.p_scale}
        losses: list[torch.Tensor] = []
        for name in targets:
            scale = scales[name].to(clean_full).view(1, 1, 1, 1, -1)
            squared_error = ((proposals[name].detach() - targets[name]) / scale).square().mean(dim=(-1, -2), keepdim=True)
            log_variance = output.log_variances[name]
            losses.append(0.5 * (torch.exp(-log_variance) * squared_error + log_variance).mean())
        return torch.stack(losses).mean()
