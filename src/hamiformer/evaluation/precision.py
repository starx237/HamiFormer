from hamiformer.utils.paths import project_root
import types
import torch
EVALUATION_PRECISION = {'dtype': 'float32', 'autocast': False, 'tf32': False, 'protocol': 'hamiballs2-formal-eval-fp32-v1'}

def configure_fp32_evaluation(*models, collector=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    for model in models:
        model.float().eval()
    if collector is not None:

        @torch.no_grad()
        def wide_fp32(self, state, tau, batch):
            with torch.autocast(device_type=self.device.type, enabled=False):
                d, tokens = self.wide.forward_with_tokens(state.float(), tau.float(), x0=batch['phase'][:, 0].float(), attrs=batch['attrs'].float(), physical_time=batch['time'].float(), object_mask=batch['object_mask'].bool(), spring_mask=batch['spring_mask'], spring_k=batch['spring_k'].float(), spring_rest_length=batch['spring_rest_length'].float())
            return (d.float(), tokens.float())
        collector._wide = types.MethodType(wide_fp32, collector)
    return dict(EVALUATION_PRECISION)
