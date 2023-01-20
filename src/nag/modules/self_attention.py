from torch import nn
from torch.nn import functional as F

from .residual_block import ResidualBlock
from .multihead_attention import MultiHeadAttention, FaireseqMultiHeadAttention


class SelfAttention(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, device=None,
                 relative_clip=0, gumbels=False, postnorm=True, batch_first=False):
        super(SelfAttention, self).__init__()
        self.self_attn = ResidualBlock(
            MultiHeadAttention(d_model, nhead=nhead,
                               dropout=0.1, bias=True, gumbels=gumbels,
                               relative_clip=relative_clip, device=device,
                               use_wo=True, batch_first=batch_first),
            d_model, dropout, no_norm=False, postnorm=postnorm)

    def forward(self, emb, attn_mask=None, key_padding_mask=None):
        return self.self_attn(
            emb, emb, emb, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
