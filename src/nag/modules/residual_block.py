import torch
from torch import nn
from torch.nn import functional as F
from .bert_layernorm import BertLayerNorm


class ResidualBlock(nn.Module):
    def __init__(self, layer, d_model, dropout=0.1, no_norm=False, postnorm=True):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.postnorm = postnorm
        self.no_norm = no_norm
        if no_norm:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            #self.layernorm = nn.LayerNorm(d_model)
            self.layernorm = BertLayerNorm(d_model)

    def forward(self, x, *param, **kwargs):
        residual = x
        if self.no_norm:
            return residual + self.dropout(self.layer(x, *param, **kwargs)) * self.alpha
        elif self.postnorm:
            return self.layernorm(residual + self.dropout(self.layer(x, *param, **kwargs)))
        else:
            return residual + self.dropout(self.layer(self.layernorm(x), *param, **kwargs))
