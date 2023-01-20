import copy
from torch import nn
from torch.nn import functional as F

from .self_attention import SelfAttention
from .positional_attention import PositionalAttention
from .src_attention import EncoderDecoderAttention

from .residual_block import ResidualBlock
from .highway import HighwayBlock
from .feedforward import FeedForward
from .vocabulary_attention import VocabularyAttention
from .sinusoidal_position_embedding import PositionalEncoding
from ..utils import generate_key_padding_mask


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_tar, d_src, nhead, gumbels=False, no_norm=False,
                 dim_feedforward=2048, dropout=0.1, ff_dropout=0.1, activation="relu",
                 relative_clip=4, device=None, use_wo=True,
                 use_pos_attn=False, use_vocab_attn=False, highway=False,
                 postnorm=True, position_encoding_layer=None,
                 batch_first=False):
        super(TransformerDecoderLayer, self).__init__()
        self.use_pos_attn = use_pos_attn
        self.use_vocab_attn = use_vocab_attn
        self.d_tar = d_tar
        self.batch_first = batch_first
        if use_vocab_attn:
            self.vocab_attn_layer = ResidualBlock(
                VocabularyAttention(d_tar, gumbels=gumbels, dropout=dropout),
                d_tar, dropout=dropout, no_norm=no_norm)

        self.self_attn = SelfAttention(
            d_src, nhead, dropout=dropout, device=device,
            relative_clip=relative_clip, gumbels=gumbels, postnorm=postnorm,
            batch_first=batch_first)

        if use_pos_attn and position_encoding_layer is not None:
            self.pos_selfattn = PositionalAttention(
                d_tar, nhead, position_encoding_layer,
                dropout=dropout, device=device, relative_clip=relative_clip,
                gumbels=gumbels, postnorm=postnorm,
                batch_first=batch_first)

        self.src_attn = EncoderDecoderAttention(
            d_src, d_tar, nhead,
            dropout=dropout, device=device, relative_clip=relative_clip,
            gumbels=gumbels, postnorm=postnorm,
            batch_first=batch_first)

        if highway:
            self.feedforward = HighwayBlock(
                FeedForward(d_tar, dim_feedforward, dropout=ff_dropout, activation=activation),
                d_tar, dropout=dropout, no_norm=no_norm)
        else:
            self.feedforward = ResidualBlock(
                FeedForward(d_tar, dim_feedforward, dropout=ff_dropout, activation=activation),
                d_tar, dropout=dropout, no_norm=no_norm, postnorm=postnorm)

    def forward(self, tgt, src, embedding=None, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_lengths=None):
        if self.use_vocab_attn and embedding is not None:
            tgt = self.vocab_attn_layer(tgt, embedding)  # B x l_tar x d_tar
        self_attn_out = self.self_attn(
            emb=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)  # B x l_tar x d_tar
        if self.use_pos_attn:
            self_attn_out = self.pos_selfattn(
                tgt_emb=self_attn_out,
                attn_mask=None,
                key_padding_mask=tgt_key_padding_mask,
                tgt_lengths=tgt_lengths)  # B x l_tar x d_tar
        src_attn_out = self.src_attn(
            src_emb=src, tgt_emb=self_attn_out,
            attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)  # B x l_tar x d_tar
        out = self.feedforward(src_attn_out)  # B x l_tar x d_tar
        return out  # B x l_tar x d_tar


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, nlayers=6, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.nlayers = nlayers
        self.transformer_decoder = nn.ModuleList([
            copy.deepcopy(decoder_layer) for i in range(nlayers)])

    def forward(self, tgt, memory, embedding=None, tgt_mask=None, memory_mask=None,
                tgt_lengths=None, tgt_key_padding_mask=None,
                src_lengths=None, memory_key_padding_mask=None):
        if self.transformer_decoder[0].batch_first:
            src_len = memory.shape[1]
            tgt_len = tgt.shape[1]
        else:
            src_len = memory.shape[0]
            tgt_len = tgt.shape[0]
        if src_lengths is not None and memory_key_padding_mask is None:
            memory_key_padding_mask = generate_key_padding_mask(
                src_len, src_lengths)
        if tgt_lengths is not None and tgt_key_padding_mask is None:
            tgt_key_padding_mask = generate_key_padding_mask(
                tgt_len, tgt_lengths)
        x = tgt
        xs = []
        for layer in self.transformer_decoder:
            x = layer(x, memory, embedding=embedding,
                      tgt_mask=tgt_mask,
                      memory_mask=memory_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      tgt_lengths=tgt_lengths)
            xs.append(x)
        return xs, x
