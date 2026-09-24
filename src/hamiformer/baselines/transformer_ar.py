from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class TransformerARConfig:
    width: int = 156
    depth: int = 4
    heads: int = 4
    expansion: int = 4
    state_dim: int = 4
    attr_dim: int = 3

class TransformerAR(nn.Module):

    def __init__(self, config=TransformerARConfig()):
        super().__init__()
        self.config = config
        self.embed = nn.Linear(config.state_dim + config.attr_dim + 1, config.width)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(config.width, config.heads, config.width * config.expansion, dropout=0.0, activation='gelu', batch_first=True, norm_first=True) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.state_dim)

    def forward(self, state, attrs, dt):
        dt = dt.reshape(-1, 1, 1).expand(-1, state.shape[1], 1)
        h = self.embed(torch.cat((state, attrs, dt), dim=-1))
        for block in self.blocks:
            h = block(h)
        return state + self.head(self.norm(h))
