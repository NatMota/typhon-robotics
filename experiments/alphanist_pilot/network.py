"""Compact point-cloud encoders plus separate residual actor/critic networks."""
import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

class Residual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*2),nn.SiLU(),nn.Linear(width*2,width))
    def forward(self,x): return x+self.net(x)

class Branch(nn.Module):
    def __init__(self,out):
        super().__init__()
        self.clouds=nn.ModuleList([nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU()) for _ in range(3)])
        self.net=nn.Sequential(nn.Linear(3*64+24,128),nn.LayerNorm(128),nn.SiLU(),Residual(128),Residual(128),nn.LayerNorm(128),nn.Linear(128,out))
    def forward(self,x):
        cloud=x[:,:864].reshape(-1,3,96,3)
        z=[enc(cloud[:,i]).max(1).values for i,enc in enumerate(self.clouds)]
        return self.net(torch.cat(z+[x[:,864:]],dim=1))

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor,self.critic=Branch(8),Branch(1)
        self.logstd=nn.Parameter(torch.full((8,),-1.))
        self.register_buffer("obs_mean",torch.zeros(888))
        self.register_buffer("obs_var",torch.ones(888))
        self.register_buffer("obs_count",torch.tensor(1e-4,dtype=torch.float64))
        nn.init.orthogonal_(self.actor.net[-1].weight,.01)
        nn.init.zeros_(self.actor.net[-1].bias)

    @torch.no_grad()
    def update_normalization(self,x):
        # Freeze these statistics throughout each rollout and its PPO optimization.
        mean,var,n=x.mean(0),x.var(0,unbiased=False),x.shape[0]
        count=self.obs_count.to(x.dtype);total=count+n;delta=mean-self.obs_mean
        merged=self.obs_var*count+var*n+delta.square()*count*n/total
        self.obs_mean.add_(delta*n/total);self.obs_var.copy_(merged/total);self.obs_count.add_(n)

    def distribution(self,x):
        x=((x-self.obs_mean)/torch.sqrt(self.obs_var+1e-5)).clamp(-10,10)
        return Normal(self.actor(x),self.logstd.clamp(-4,1).exp()),self.critic(x).squeeze(-1)

    def sample(self,x,raw=None,deterministic=False):
        dist,value=self.distribution(x)
        if raw is None: raw=dist.mean if deterministic else dist.sample()
        action=raw.tanh()
        logp=(dist.log_prob(raw)-torch.log(1-action.square()+1e-6)).sum(-1)
        return action,raw,logp,dist.entropy().sum(-1),value
