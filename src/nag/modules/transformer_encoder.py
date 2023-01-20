import copy
from torch import nn
from torch.nn import functional as F

from .residual_block import ResidualBlock
from .highway import HighwayBlock
from .feedforward import FeedForward
from ..utils import generate_key_padding_mask
from .self_attention import SelfAttention


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_src, nhead, dropout=0.1, ff_dropout=0.1, activation="relu", postnorm=True,
                 dim_feedforward=2048, relative_clip=4, use_wo=True, no_norm=False,
                 gumbels=False, device=None, highway=False, batch_first=False):
        super(TransformerEncoderLayer, self).__init__()
        self.d_src = d_src
        self.batch_first = batch_first
        self.self_attn = SelfAttention(
            d_src, nhead, dropout=dropout, device=device,
            relative_clip=relative_clip, gumbels=gumbels, postnorm=postnorm,
            batch_first=batch_first)

        if highway:
            self.feedforward = HighwayBlock(
                FeedForward(d_src, dim_feedforward, dropout=ff_dropout, activation=activation),
                d_src, dropout, no_norm=no_norm)
        else:
            self.feedforward = ResidualBlock(
                FeedForward(d_src, dim_feedforward, dropout=ff_dropout, activation=activation),
                d_src, dropout, no_norm=no_norm, postnorm=postnorm)

    def forward(self, src, embedding=None, src_mask=None, src_key_padding_mask=None):
        '''
        :attn_mask: L_q x L_k, Tensor(bool)
        :key_padding_mask: B x L_k, Tensor(bool)
            value `0` is masked !!!
        '''
        self_attn_out = self.self_attn(
            src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        out = self.feedforward(self_attn_out)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, nlayers=6, dropout=0.):
        super(TransformerEncoder, self).__init__()
        self.nlayers = nlayers
        self.transformer_encoder = nn.ModuleList([
            copy.deepcopy(encoder_layer) for i in range(nlayers)])

    def forward(self, src, embedding=None, src_mask=None, src_lengths=None,
                src_key_padding_mask=None):
        if self.transformer_encoder[0].batch_first:
            seq_len = src.shape[1]
        else:
            seq_len = src.shape[0]
        if src_lengths is not None and src_key_padding_mask is None:
            src_key_padding_mask = generate_key_padding_mask(seq_len, src_lengths)
        x = src
        xs = []
        for layer in self.transformer_encoder:
            x = layer(x, embedding=embedding, src_mask=src_mask,
                      src_key_padding_mask=src_key_padding_mask)
            xs.append(x)
        return xs, x
