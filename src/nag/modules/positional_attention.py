import math
from torch import nn
from torch.nn import functional as F

from .multihead_attention import MultiHeadAttention
from .bert_layernorm import BertLayerNorm


class PositionalAttention(nn.Module):

    def __init__(self, d_model, nhead, position_encoding_layer,
                 dropout=0.1, device=None, relative_clip=0,
                 gumbels=False, postnorm=True, batch_first=False):
        super(PositionalAttention, self).__init__()
        self.position_encoding_layer = position_encoding_layer
        self.pos_selfattn = MultiHeadAttention(
            d_model, nhead=nhead,
            dropout=0.1, bias=True, gumbels=gumbels,
            relative_clip=relative_clip, device=device,
            use_wo=True, batch_first=batch_first)
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.factor = math.sqrt(d_model)
        self.postnorm = postnorm

    def forward(self, tgt_emb, attn_mask=None, key_padding_mask=None, tgt_lengths=None):
        if not self.postnorm:
            tgt_emb = self.layernorm(tgt_emb)
        pos_encoding_out = self.position_encoding_layer(tgt_emb, tgt_lengths)
        if not self.batch_first:
            pos_encoding_out = pos_encoding_out.transpose(0, 1)
        pos_attn_out = self.pos_selfattn(
            pos_encoding_out, pos_encoding_out, tgt_emb,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask)  # B x l_tar x d_tar
        if self.postnorm:
            pos_attn_out = self.layernorm(tgt_emb * self.factor + self.dropout(pos_attn_out))
        else:
            pos_attn_out = tgt_emb * self.factor + self.dropout(pos_attn_out)
        return pos_attn_out
