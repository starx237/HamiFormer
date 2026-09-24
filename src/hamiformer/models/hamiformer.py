from torch import nn

class HamiFormer(nn.Module):

    def __init__(self, components, dataset):
        super().__init__()
        if dataset not in ('h1', 'h2'):
            raise ValueError(dataset)
        self.dataset = dataset
        self.networks = nn.ModuleDict({k: v for k, v in components.items() if isinstance(v, nn.Module)})
        self.settings = {k: v for k, v in components.items() if not isinstance(v, nn.Module)}

    @classmethod
    def from_pretrained(cls, weights, dataset, device='cpu'):
        from hamiformer.inference.weights import load_h1, load_h2
        if dataset not in ('h1', 'h2'):
            raise ValueError(dataset)
        components = load_h1(weights, device) if dataset == 'h1' else load_h2(weights, 'ours', device)
        return cls(components, dataset)

    def components(self):
        return dict(self.settings, **dict(self.networks.items()))
