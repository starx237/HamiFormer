import copy
import torch
import pytest

def test_joint_parent_has_no_residual_or_gate_requirement(tmp_path, monkeypatch):
    from hamiformer.training.hamiballs1 import frozen_residual as runner
    cfg = {'model': {}}
    contract = {'train_only': True}
    main = dict(schema=runner.stage_a.CHECKPOINT_SCHEMA, role='main', step=50000, config=cfg, config_sha256=runner._canonical_hash(cfg), dataset_contract=contract, state_scale=torch.ones(4), attr_scale=torch.ones(3), joint_training=True, d_state_dict={'weight': torch.ones(1)}, h_state_dict={'weight': torch.ones(1)}, source_manifest={})
    main['h_freeze_digest'] = runner.stage_a.module_digest_from_state(main['h_state_dict'])
    architecture = {'role': 'wide-d', 'nested_initialization': {'function_preserving': True}, 'initial_forward_equivalence': {'within_tolerance': True}}
    wide = copy.deepcopy(main)
    wide.update(schema=runner.stage_a.WIDE_CHECKPOINT_SCHEMA, role='wide-d', architecture=architecture, architecture_sha256=runner._canonical_hash(architecture))
    mp, wp = (tmp_path / 'main.pt', tmp_path / 'wide.pt')
    torch.save(main, mp)
    torch.save(wide, wp)
    registration = {'parents': {'main_checkpoint': str(mp), 'wide_checkpoint': str(wp), 'main_checkpoint_sha256': runner.sha256_file(mp), 'wide_checkpoint_sha256': runner.sha256_file(wp), 'terminal_step': 50000}}
    monkeypatch.setattr(runner, '_terminal_summary', lambda *args, **kwargs: None)
    loaded, _, _ = runner._load_parents(registration)
    assert loaded['joint_training']
    main['d_state_dict']['weight'].fill_(float('nan'))
    torch.save(main, mp)
    registration['parents']['main_checkpoint_sha256'] = runner.sha256_file(mp)
    with pytest.raises(ValueError, match='nonfinite'):
        runner._load_parents(registration)


def test_non_joint_parent_is_rejected(tmp_path, monkeypatch):
    from hamiformer.training.hamiballs1 import frozen_residual as runner

    cfg = {'model': {}}
    contract = {'train_only': True}
    main = dict(
        schema=runner.stage_a.CHECKPOINT_SCHEMA,
        role='main',
        step=50000,
        config=cfg,
        config_sha256=runner._canonical_hash(cfg),
        dataset_contract=contract,
        state_scale=torch.ones(4),
        attr_scale=torch.ones(3),
        joint_training=False,
        d_state_dict={'weight': torch.ones(1)},
        h_state_dict={'weight': torch.ones(1)},
        source_manifest={},
    )
    main['h_freeze_digest'] = runner.stage_a.module_digest_from_state(main['h_state_dict'])
    architecture = {
        'role': 'wide-d',
        'nested_initialization': {'function_preserving': True},
        'initial_forward_equivalence': {'within_tolerance': True},
    }
    wide = copy.deepcopy(main)
    wide.update(
        schema=runner.stage_a.WIDE_CHECKPOINT_SCHEMA,
        role='wide-d',
        architecture=architecture,
        architecture_sha256=runner._canonical_hash(architecture),
    )
    main_path, wide_path = tmp_path / 'main.pt', tmp_path / 'wide.pt'
    torch.save(main, main_path)
    torch.save(wide, wide_path)
    registration = {
        'parents': {
            'main_checkpoint': str(main_path),
            'wide_checkpoint': str(wide_path),
            'main_checkpoint_sha256': runner.sha256_file(main_path),
            'wide_checkpoint_sha256': runner.sha256_file(wide_path),
            'terminal_step': 50000,
        }
    }
    monkeypatch.setattr(runner, '_terminal_summary', lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match='joint D/H training'):
        runner._load_parents(registration)
