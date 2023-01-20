import math
import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128,
                 device=None, residual=True, requires_grad=False):
        super(PositionalEncoding, self).__init__()
        """
        implementation of `PositionalEncoding` description in Section 3.5 of "Attention Is All You Need".
        """
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.residual = residual
        '''
        self.multier = -math.log(10000.0)
        max_len = max_len + 1
        pe = torch.zeros(max_len+1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (self.multier / d_model))
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        pe[0, :] = 0
        '''
        """
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        max_len += 2
        half_dim = d_model // 2
        pe = math.log(10000) / (half_dim - 1)
        pe = torch.exp(torch.arange(half_dim, dtype=torch.float) * -pe)
        pe = torch.arange(max_len, dtype=torch.float).unsqueeze(1) * pe.unsqueeze(0)
        pe = torch.cat([torch.sin(pe), torch.cos(pe)], dim=1).view(max_len, -1)
        if d_model % 2 == 1:
            pe = torch.cat([pe, torch.zeros(max_len, 1)], dim=1)  # zero pad
        pe[0, :] = 0
        self.position_encoding = nn.Embedding.from_pretrained(
            pe.to(self.device), freeze=True)

    def forward(self, x, input_lens=None):
        '''
        :param x: B x L x E
        return: B x L x E
        '''
        max_len = x.shape[1]
        if input_lens is not None:
            input_pos = torch.LongTensor(
                [list(range(1, lenx + 1)) + [0] * (max_len - lenx) for lenx in input_lens.cpu().numpy()]).to(self.device)
        else:
            B = x.shape[0]
            input_pos = torch.LongTensor(
                [list(range(1, max_len + 1)) for _ in range(B)]).to(self.device)
        if self.residual:
            out = self.position_encoding(input_pos) + x
        else:
            out = self.position_encoding(input_pos)
        return self.dropout(out)
