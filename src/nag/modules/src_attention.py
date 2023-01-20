from torch import nn
from torch.nn import functional as F

from .residual_block import ResidualBlock
from .multihead_attention import MultiHeadAttention


class EncoderDecoderAttention(nn.Module):

    def __init__(self, d_src, d_tar, nhead,
                 dropout=0.1, device=None, relative_clip=0,
                 gumbels=False, postnorm=True, batch_first=False):
        super(EncoderDecoderAttention, self).__init__()
        self.src_attn = ResidualBlock(
            MultiHeadAttention(d_tar, kdim=d_src, vdim=d_src, nhead=nhead,
                               dropout=0.1, bias=True, gumbels=gumbels,
                               relative_clip=relative_clip, device=device,
                               use_wo=True, batch_first=batch_first),
            d_tar, dropout, no_norm=False, postnorm=postnorm)

    def forward(self, src_emb, tgt_emb, attn_mask=None, key_padding_mask=None):
        src_attn_out = self.src_attn(
            tgt_emb, src_emb, src_emb,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask)  # B x l_tar x d_tar
        return src_attn_out
