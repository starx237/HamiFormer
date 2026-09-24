import torch
from torch import nn
from hamiformer.baselines.physiformer import HamiBalls2WideD

class GraphTransformerAR(nn.Module):

    def __init__(self, width=340, depth=4, heads=4, context=1):
        super().__init__()
        self.context = context
        self.num_heads = heads
        self.edge_bias = nn.Linear(3, heads, bias=False)
        nn.init.zeros_(self.edge_bias.weight)
        self.embed = nn.Linear(14 if context == 1 else 8 + 7 * context, width)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 6)

    def forward(self, z, b, dt):
        if self.context > 1:
            if z.ndim == 3:
                z = z[:, None]
            z = z[:, -self.context:]
            current = z[:, -1]
            length = z.shape[1]
            history = torch.nn.functional.pad(z, (0, 0, 0, 0, self.context - length, 0))
            present = torch.arange(self.context, device=z.device) >= self.context - length
            present = present[None, :, None].expand(z.shape[0], -1, z.shape[2])
            if 'history_valid' in b:
                present = present & b['history_valid'][:, :, None]
            history = history * present[..., None]
            z_features = torch.cat((history.permute(0, 2, 1, 3).flatten(2), present.permute(0, 2, 1).to(z.dtype)), -1)
            z = current
        else:
            z_features = z
        attrs, bias = HamiBalls2WideD._graph_inputs(self, b['attrs'], b['object_mask'].bool(), b['spring_mask'], b['spring_k'], b['spring_rest_length'])
        mask = b['object_mask'].bool()
        bias = bias.masked_fill(~mask[:, None, None, :], float('-inf'))
        h = self.embed(torch.cat((z_features, attrs, dt[:, None, None].expand(-1, z.shape[1], 1)), -1))
        for layer in self.blocks:
            h = layer(h, src_mask=bias.flatten(0, 1))
        return (z + self.head(self.norm(h))) * mask[..., None]
