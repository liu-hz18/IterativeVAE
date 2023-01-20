
from torch import nn
from torch.nn import functional as F
from .gelu import gelu


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, negative_slope=0.01, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        if activation == 'relu':
            self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        else:
            self.activation = gelu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
