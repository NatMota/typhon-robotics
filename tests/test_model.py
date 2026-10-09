import numpy as np
import torch
from supervisor.model import TemporalActorCritic


def test_candidate_permutation_equivariance_and_mask():
    torch.manual_seed(9);torch.set_num_threads(2)
    model=TemporalActorCritic().eval();x=torch.randn(2,8,4,12)
    mask=torch.ones(2,7,dtype=torch.bool);mask[:,1]=False
    p=torch.tensor([2,0,3,1]);permuted=torch.cat([mask[:,:4][:,p],mask[:,4:]],dim=-1)
    with torch.no_grad():
        a,v=model(x,mask);b,w=model(x[:,:,p],permuted)
    assert torch.allclose(a[:,:4][:,p],b[:,:4],atol=1e-6)
    assert torch.allclose(a[:,4:],b[:,4:],atol=1e-6) and torch.allclose(v,w,atol=1e-6)
    assert a[:,1].max() < -1e8
