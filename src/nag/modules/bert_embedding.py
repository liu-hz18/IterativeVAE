
import torch
from torch import nn


class BertEmbedding(nn.Module):
    """docstring for BertEmbedding"""
    def __init__(self, d_model, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.d_model = d_model
        self.dropout = dropout

    def forward(self, src, src_lengths=None):
        pass
