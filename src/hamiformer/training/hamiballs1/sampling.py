from hamiformer.utils.paths import project_root
import torch

def expert_kwargs(models):
    config = models['config']
    hcfg = config['hamiltonian']
    return {'state_scale': models['state_scale'], 'attr_scale': models['attr_scale'], 'q_dim': int(config['dataset']['q_dim']), 'step_size': 1.0 / 30.0, 'num_steps': 20, 't_eps': float(config['rectified_flow']['t_eps']), 'mixed_singular_floor': float(hcfg['mixed_singular_floor']), 'mixed_condition_limit': float(hcfg['mixed_condition_limit']), 'tangent_spectral_norm_limit': float(hcfg['tangent_spectral_norm_limit'])}

def chunk_major_sources(*, source_count, batch_size, noise_seed, noise_scale, device):
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    result = [[] for _ in range(4)]
    for chunk_index in range(4):
        for left in range(0, source_count, batch_size):
            count = min(batch_size, source_count - left)
            result[chunk_index].append(noise_scale * torch.randn((count, 48, 5, 4), device=device, generator=generator))
    return result
_expert_kwargs = expert_kwargs
_chunk_major_sources = chunk_major_sources
