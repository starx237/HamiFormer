import torch
from hamiformer.models.hamiballs_observable_sidecar_physical_features_observable_sidecar_r import HamiBallsPhysicalFeaturesObservableSidecarResidual

def test_residual_statistics_state():
    model = HamiBallsPhysicalFeaturesObservableSidecarResidual(token_dim=128, state_dim=4, attr_dim=3)
    assert 'statistics_fitted' in model.state_dict()
    assert not any((k in model.state_dict() for k in ('axis_low_cut', 'axis_high_cut', 'provenance_tertiles')))
    model.register_buffer('statistics_fitted', torch.ones((), dtype=torch.uint8), persistent=False)
    assert 'statistics_fitted' not in model.state_dict()
    assert model.statistics_fitted.item() == 1

def test_generator_signature():
    from pathlib import Path
    import yaml
    from hamiformer.data_generation3d.signature import dataset_signature
    from hamiformer.training.hamiballs2_posthd_protocol import DATASET_SIGNATURE
    root = Path(__file__).resolve().parents[1]
    for name in ('physiformer', 'hamiltonian', 'main', 'dit'):
        cfg = yaml.safe_load((root / 'configs' / 'hamiballs2' / (name + '.yaml')).read_text())
        cfg['generator_config'] = root / cfg['generator_config']
        assert dataset_signature(cfg) == DATASET_SIGNATURE
