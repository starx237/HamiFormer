import pytest
import torch

from hamiformer.integrators.solver import Solver


class Energy(torch.nn.Module):
    def __init__(self, objects):
        super().__init__()
        self.state_dim = 2*objects
        self.weight = torch.nn.Parameter(torch.linspace(.5,1.5,4*objects,dtype=torch.float64))

    def forward(self,q,p,c):
        z = torch.cat((q,p),-1)
        return (z.square()*self.weight).sum(-1)/2+.1*torch.sin(z.sum(-1))


@pytest.mark.parametrize('objects', (2,5,12))
def test_window_dlr_matches_scalar_reference_and_keeps_linear_factor_storage(objects):
    model = Energy(objects)
    g = torch.Generator().manual_seed(19)
    q = torch.randn(2,3,objects,2,generator=g,dtype=torch.float64)*.1
    p = torch.randn(2,3,objects,2,generator=g,dtype=torch.float64)*.1
    attrs = torch.ones(2,objects,3,dtype=torch.float64)
    backend = Solver('plas_dlr').bind_dlr(model,torch.ones(3),.03,compiled=False)
    factors = backend.build(q,p,attrs)
    previous = torch.cat((q[:,0]+.05,p[:,0]-.02),-1)
    actual = backend.apply(factors,1,previous)
    for b in range(2):
        flat = lambda z: torch.cat((z[...,:2].flatten(),z[...,2:].flatten()))
        energy = lambda z: model(z[:2*objects][None],z[2*objects:][None],None).sum()
        expected = Solver('plas_dlr').step(energy,flat(previous[b]),.03,torch.cat((q[b,1].flatten(),p[b,1].flatten())))
        torch.testing.assert_close(flat(actual[b]),expected,atol=1e-11,rtol=1e-11)
    m, r = objects*2, 2
    assert (factors.matrix.numel()+factors.offset.numel())//6 == 3*m*r+6*m+r*r
    repeated = backend.build(q,p,attrs)
    torch.testing.assert_close(backend.apply(repeated,1,previous),actual,atol=0,rtol=0)
    q.add_(.1)
    changed = backend.build(q,p,attrs)
    assert not torch.equal(backend.apply(changed,1,previous),actual), 'mutated anchors must rebuild factors'
