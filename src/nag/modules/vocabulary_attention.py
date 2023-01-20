
import torch
from torch import nn
from torch.nn import functional as F

from .operators import GumbelSoftmax


class VocabularyAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, gumbels=False):
        super(VocabularyAttention, self).__init__()
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)
        self.mlp = nn.Linear(embed_dim*2, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, embedding):
        A = self.soft_max(
            x.matmul(embedding.transpose(0, 1))).matmul(embedding)
        out = self.dropout(self.mlp(torch.cat([x, A], -1)))
        return out
