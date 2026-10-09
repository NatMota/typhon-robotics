"""Shared per-candidate temporal transformer + actor/critic heads."""
import torch
from torch import nn
from .core import FEATURES


class TemporalActorCritic(nn.Module):
    def __init__(self, width=64, heads=4, layers=2, history=8):
        super().__init__()
        self.history = history
        self.embed = nn.Linear(FEATURES, width)
        self.position = nn.Parameter(torch.zeros(1, history, width))
        layer = nn.TransformerEncoderLayer(width, heads, width*2, dropout=0,
                                            activation='gelu', batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.move = nn.Linear(width,1)
        self.special = nn.Linear(width,3)
        self.value = nn.Linear(width,1)

    def forward(self, history, mask=None):
        b,t,n,f = history.shape
        assert t == self.history
        z = self.embed(history.permute(0,2,1,3).reshape(b*n,t,f)) + self.position
        causal = torch.triu(torch.ones(t,t,device=z.device,dtype=torch.bool), diagonal=1)
        z = self.temporal(z,mask=causal)[:,-1].reshape(b,n,-1)
        pooled = z.mean(1)
        logits = torch.cat((self.move(z).squeeze(-1),self.special(pooled)),dim=-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits, self.value(pooled).squeeze(-1)
