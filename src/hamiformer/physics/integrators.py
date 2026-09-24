from __future__ import annotations
from collections.abc import Callable
from typing import Literal
import torch
IntegratorName = Literal['explicit_euler', 'symplectic_euler', 'leapfrog']

def _validate_state(state: torch.Tensor) -> None:
    if state.ndim < 1 or state.shape[-1] != 2:
        raise ValueError('state 最后一维必须按 (q,p) 排列且长度为 2')

def _pendulum_force(q: torch.Tensor, theta_sys: torch.Tensor | float) -> torch.Tensor:
    theta = torch.as_tensor(theta_sys, device=q.device, dtype=q.dtype)
    return theta * torch.sin(q)

def explicit_euler_step(state: torch.Tensor, *, step_size: float, theta_sys: torch.Tensor | float=1.0) -> torch.Tensor:
    _validate_state(state)
    h = torch.as_tensor(step_size, device=state.device, dtype=state.dtype)
    q, p = state.unbind(dim=-1)
    q_next = q + h * p
    p_next = p - h * _pendulum_force(q, theta_sys)
    return torch.stack([q_next, p_next], dim=-1)

def symplectic_euler_step(state: torch.Tensor, *, step_size: float, theta_sys: torch.Tensor | float=1.0) -> torch.Tensor:
    _validate_state(state)
    h = torch.as_tensor(step_size, device=state.device, dtype=state.dtype)
    q, p = state.unbind(dim=-1)
    p_next = p - h * _pendulum_force(q, theta_sys)
    q_next = q + h * p_next
    return torch.stack([q_next, p_next], dim=-1)

def leapfrog_step(state: torch.Tensor, *, step_size: float, theta_sys: torch.Tensor | float=1.0) -> torch.Tensor:
    _validate_state(state)
    h = torch.as_tensor(step_size, device=state.device, dtype=state.dtype)
    q, p = state.unbind(dim=-1)
    p_half = p - 0.5 * h * _pendulum_force(q, theta_sys)
    q_next = q + h * p_half
    p_next = p_half - 0.5 * h * _pendulum_force(q_next, theta_sys)
    return torch.stack([q_next, p_next], dim=-1)

def get_integrator(name: IntegratorName | str) -> Callable[..., torch.Tensor]:
    table: dict[str, Callable[..., torch.Tensor]] = {'explicit_euler': explicit_euler_step, 'symplectic_euler': symplectic_euler_step, 'leapfrog': leapfrog_step}
    try:
        return table[str(name)]
    except KeyError as exc:
        raise ValueError(f'未知 integrator: {name}') from exc

def forced_integrator_rollout(initial_state: torch.Tensor, innovation: torch.Tensor | None, *, steps: int | None=None, step_size: float, integrator: IntegratorName | str, theta_sys: torch.Tensor | float=1.0) -> torch.Tensor:
    if initial_state.ndim != 2 or initial_state.shape[-1] != 2:
        raise ValueError('initial_state 必须为 [B,2]')
    _validate_state(initial_state)
    if innovation is not None:
        if innovation.ndim != 3 or innovation.shape[:1] != initial_state.shape[:1]:
            raise ValueError('innovation 必须为 [B,T,2] 且 batch 与 initial_state 一致')
        if innovation.shape[-1] != 2:
            raise ValueError('innovation 最后一维必须为 2')
        rollout_steps = int(innovation.shape[1])
        if steps is not None and int(steps) != rollout_steps:
            raise ValueError('steps 与 innovation 的时间长度不一致')
    else:
        if steps is None:
            raise ValueError('innovation=None 时必须显式提供 steps')
        rollout_steps = int(steps)
    if rollout_steps < 1:
        raise ValueError('rollout steps 必须为正')
    step_fn = get_integrator(integrator)
    current = initial_state
    states: list[torch.Tensor] = []
    theta = torch.as_tensor(theta_sys, device=current.device, dtype=current.dtype)
    for edge in range(rollout_steps):
        current = step_fn(current, step_size=step_size, theta_sys=theta)
        if innovation is not None:
            current = current + innovation[:, edge]
        states.append(current)
    return torch.stack(states, dim=1)

def transition_residuals(initial_state: torch.Tensor, trajectory: torch.Tensor, *, step_size: float, integrator: IntegratorName | str, theta_sys: torch.Tensor | float=1.0) -> torch.Tensor:
    if trajectory.ndim != 3 or trajectory.shape[0] != initial_state.shape[0]:
        raise ValueError('trajectory 必须为 [B,T,2] 且 batch 一致')
    if trajectory.shape[-1] != 2 or trajectory.shape[1] < 1:
        raise ValueError('trajectory 必须至少包含一个 [q,p] future state')
    sources = torch.cat([initial_state[:, None], trajectory[:, :-1]], dim=1)
    step_fn = get_integrator(integrator)
    theta = torch.as_tensor(theta_sys, device=sources.device, dtype=sources.dtype)
    if theta.ndim == 1:
        theta = theta[:, None]
    proposals = step_fn(sources, step_size=step_size, theta_sys=theta)
    return trajectory - proposals
__all__ = ['IntegratorName', 'explicit_euler_step', 'forced_integrator_rollout', 'get_integrator', 'leapfrog_step', 'symplectic_euler_step', 'transition_residuals']
