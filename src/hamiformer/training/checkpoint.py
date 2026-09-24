from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import torch

def _model_contract(config: dict[str, Any]) -> dict[str, Any]:
    data = config['data']
    return {'data': {key: data[key] for key in ('num_objects', 'future_steps', 'q_dim', 'attr_dim')}, 'rectified_flow': config['rectified_flow'], 'phase_d': config['phase_d'], 'hamiltonian_h': config['hamiltonian_h'], 'matched_s': config['matched_s']}

def _resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    training = config['training']
    return {'seed': config['seed'], 'model': _model_contract(config), 'training': {key: training[key] for key in ('batch_size', 'max_steps', 'learning_rate', 'warmup_steps', 'min_learning_rate', 'weight_decay', 'grad_clip', 'ema_decay', 'mixed_precision_d', 'mixed_precision_h')}}

def validate_checkpoint_payload(payload: dict[str, Any], *, expected_config: dict[str, Any], expected_q_scale: torch.Tensor, expected_p_scale: torch.Tensor, require_optimizer: bool) -> None:
    if payload.get('format_version') != 1:
        raise ValueError('checkpoint format_version 不受支持')
    saved_config = payload.get('config')
    if not isinstance(saved_config, dict):
        raise ValueError('checkpoint 缺少可审计的完整 config')
    saved_contract = _resume_contract(saved_config) if require_optimizer else _model_contract(saved_config)
    expected_contract = _resume_contract(expected_config) if require_optimizer else _model_contract(expected_config)
    if saved_contract != expected_contract:
        kind = 'resume' if require_optimizer else 'model'
        raise ValueError(f'checkpoint 的 {kind} contract 与当前配置不一致')
    required = {'models', 'ema', 'phase_scales', 'step'}
    if require_optimizer:
        required.add('optimizer')
    missing = required - set(payload)
    if missing:
        raise ValueError(f'checkpoint 缺少字段: {sorted(missing)}')
    models, ema, scales = (payload['models'], payload['ema'], payload['phase_scales'])
    if not isinstance(models, dict) or set(models) != {'d', 'h', 's'}:
        raise ValueError('checkpoint models 字段结构错误')
    if not isinstance(ema, dict) or set(ema) != {'d', 'h', 's'}:
        raise ValueError('checkpoint ema 字段结构错误')
    if not isinstance(scales, dict) or set(scales) != {'q', 'p'}:
        raise ValueError('checkpoint phase_scales 字段结构错误')
    saved_q = torch.as_tensor(scales['q'], dtype=torch.float32, device='cpu')
    saved_p = torch.as_tensor(scales['p'], dtype=torch.float32, device='cpu')
    expected_q = expected_q_scale.detach().float().cpu()
    expected_p = expected_p_scale.detach().float().cpu()
    if not torch.equal(saved_q, expected_q) or not torch.equal(saved_p, expected_p):
        raise ValueError('checkpoint 的 q/p train scales 与当前 stats 不一致')

def save_checkpoint_atomic(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, output)

def load_checkpoint(path: str | Path, *, map_location: str | torch.device='cpu') -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
