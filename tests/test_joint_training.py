import copy
import torch
import pytest
from hamiformer.training.joint import joint_update, h_objective
from hamiformer.training.joint import set_joint_schedule
from types import SimpleNamespace

def test_objective_boundary():
    assert h_objective('h2', 40000) == 'midpoint_field'
    assert h_objective('h2', 40001) == 'type2_endpoint'
    assert h_objective('h1', 50000) == 'midpoint_field'

def test_h1_relation_objective_gradient(monkeypatch):
    from hamiformer.training.hamiballs1 import hamiltonian_objective as objective
    coefficient = torch.nn.Parameter(torch.tensor(0.4))
    monkeypatch.setattr(objective, '_equation_scale', torch.ones(4))
    monkeypatch.setattr(objective, 'canonical_vector_field', lambda model, state, context, create_graph: coefficient * state)
    phase = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4) / 48
    context = torch.ones(2, 2, 1, 3)
    loss = objective._relation_loss(None, phase, context, state_scale=torch.ones(4), frame_dt=1 / 30, dof=4)
    loss.backward()
    assert torch.isfinite(loss) and coefficient.grad is not None
    assert torch.isfinite(coefficient.grad) and coefficient.grad != 0

def test_h1_tree_collection_dependencies():
    import inspect
    from hamiformer.training.hamiballs1 import tree_training as tree
    from hamiformer.training import final as final
    assert 'load_models' in inspect.signature(tree._collect_parent_records).parameters
    assert callable(final._scalar1_models)
    assert callable(tree.final.main)

def test_h1_gate_schedule_is_idempotent():
    from hamiformer.training import router as gate
    from hamiformer.training.hamiballs1 import gate_schedule as schedule
    gate.configure_runner()
    runner = gate.fresh
    before = (dict(runner.STAGE_NOISE_SEEDS), dict(runner.STAGE_OPTIMIZER_SEEDS), dict(runner.LEARNING_RATES), dict(runner.OBJECTIVES), runner.SOURCE_SEED)
    gate.configure_runner()
    after = (dict(runner.STAGE_NOISE_SEEDS), dict(runner.STAGE_OPTIMIZER_SEEDS), dict(runner.LEARNING_RATES), dict(runner.OBJECTIVES), runner.SOURCE_SEED)
    assert before == after
    assert runner.STAGE_PARENT_CARRIER_STAGES == frozenset(('scalar1', 'qp'))
    assert runner.AGGREGATE_REPLAY_STAGES == frozenset(('scalar0', 'scalar1', 'qp'))
    assert runner.QP_DELTA_WIDTH == 16
    assert runner.STAGE_UPDATES == {'scalar0': 200, 'scalar1': 200, 'qp': 200}
    assert runner._resolve_source_blocks is schedule._resolve_source_blocks

def test_empty_scene_exclusions_preserve_sampling():
    from hamiformer.training.hamiballs1.gate_training import _sample_permutation_block_excluding_scenes
    ids, positions = _sample_permutation_block_excluding_scenes(size=12, count=4, seed=42, offset=2, scene_ids=[str(i) for i in range(12)], excluded_scenes=set())
    expected = torch.randperm(12, generator=torch.Generator().manual_seed(42))[2:6]
    assert torch.equal(ids, expected)
    assert positions == [2, 3, 4, 5]

def test_joint_update_steps_both_models():
    d = torch.nn.Linear(2, 1)
    h = torch.nn.Linear(2, 1)
    od = torch.optim.AdamW(d.parameters())
    oh = torch.optim.AdamW(h.parameters())
    before = [p.clone() for p in (d.weight, h.weight)]
    x = torch.ones(2, 2)
    joint_update(d(x).square().mean(), h(x).square().mean(), d, h, od, oh)
    assert not torch.equal(before[0], d.weight)
    assert not torch.equal(before[1], h.weight)
    assert int(od.state[d.weight]['step']) == int(oh.state[h.weight]['step']) == 1

def test_nonfinite_h_does_not_step_d():
    d = torch.nn.Linear(2, 1)
    h = torch.nn.Linear(2, 1)
    od = torch.optim.AdamW(d.parameters())
    oh = torch.optim.AdamW(h.parameters())
    before = copy.deepcopy(d.state_dict())
    x = torch.ones(2, 2)
    with pytest.raises(FloatingPointError):
        joint_update(d(x).sum(), h(x).sum() * float('nan'), d, h, od, oh)
    assert all((torch.equal(before[k], v) for k, v in d.state_dict().items()))
    assert not od.state

def test_endpoint_resets_only_h_optimizer_once():
    d = torch.nn.Linear(2, 1)
    h = torch.nn.Linear(2, 1)
    optimizers = {'d': torch.optim.AdamW(d.parameters()), 'h': torch.optim.AdamW(h.parameters())}
    x = torch.ones(2, 2)
    joint_update(d(x).square().mean(), h(x).square().mean(), d, h, optimizers['d'], optimizers['h'])
    od, oh = (optimizers['d'], optimizers['h'])
    adapter = SimpleNamespace(h=h)
    stage = set_joint_schedule(adapter, optimizers, 'h2', 40000, 'midpoint_field', torch.device('cpu'))
    assert optimizers['h'] is oh
    stage = set_joint_schedule(adapter, optimizers, 'h2', 40001, stage, torch.device('cpu'))
    assert optimizers['d'] is od and len(od.state) > 0
    assert optimizers['h'] is not oh and (not optimizers['h'].state)
    assert optimizers['h'].param_groups[0]['lr'] == 1e-05
    endpoint_optimizer = optimizers['h']
    set_joint_schedule(adapter, optimizers, 'h2', 40002, stage, torch.device('cpu'))
    assert optimizers['h'] is endpoint_optimizer
