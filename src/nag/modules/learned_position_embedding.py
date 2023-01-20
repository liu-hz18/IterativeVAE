import math
import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=128, residual=True):
        super(LearnedPositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual
        self.d_model = d_model
        self.factor = math.sqrt(self.d_model)
        max_len = max_len + 2
        self.position_encoding = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def forward(self, x, input_lens=None, use_dropout=True):
        max_len = x.shape[1]
        if input_lens is not None:
            input_pos = torch.LongTensor(
                [list(range(1, lenx + 1)) + [0] * (max_len - lenx)\
                 for lenx in input_lens.cpu().numpy()]).to(x.device)
        else:
            B = x.shape[0]
            input_pos = torch.LongTensor(
                [list(range(1, max_len + 1)) for _ in range(B)]).to(x.device)
        if self.residual:
            out = self.position_encoding(input_pos) + x * self.factor
        else:
            out = self.position_encoding(input_pos)
        if use_dropout:
            return self.dropout(out)
        else:
            return out

    def reset_parameters(self):
        # nn.init.normal_(
        #     self.position_encoding.weight, mean=0., std=self.d_model ** -0.5)
        nn.init.constant_(self.position_encoding.weight[0], 0.)
