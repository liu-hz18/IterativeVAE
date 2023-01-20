import math
import torch
from torch import nn
from torch.nn import functional as F

from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from .sinusoidal_position_embedding import PositionalEncoding
from .learned_position_embedding import LearnedPositionalEmbedding


class BERT(nn.Module):
    """
        `Bidirectional Encoder Representations from Transformers`
    """
    def __init__(self, ntoken, num_segments, d_model=768, nhead=8, dim_feedforward=3072,
                 nlayers=6, dropout=0.1, activation="relu", postnorm=True,
                 relative_clip=0, use_wo=True, no_norm=False,
                 gumbels=False, device=None, highway=False,
                 learned_pos_embedding=False, max_sent_length=64,
                 segment_padding_idx=0):
        super(BERT, self).__init__()
        # pos embedding
        if learned_pos_embedding:
            self.pos_encoder = LearnedPositionalEmbedding(
                d_model, dropout=dropout, max_len=max_sent_length, residual=True)
        else:
            self.pos_encoder = PositionalEncoding(
                d_model, dropout=dropout, max_len=max_sent_length, residual=True,
                device=device, requires_grad=False)
        # token embedding
        self.factor = math.sqrt(d_model)
        self.src_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # segment embedding
        self.segment_padding_idx = segment_padding_idx
        self.segment_embedding = nn.Embedding(
            num_segments, d_model, padding_idx=segment_padding_idx)
        # CLS token embedding
        self.cls_emb = nn.Parameter(torch.FloatTensor(d_model))
        nn.init.normal_(
            self.cls_emb.weight, mean=0., std=self.d_model ** -0.5)
        # encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_src=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            gumbels=gumbels,
            relative_clip=relative_clip,
            use_wo=True,
            no_norm=False,
            device=device,
            highway=highway,
            postnorm=postnorm,
        )
        self.encoder = TransformerEncoder(encoder_layer, nlayers)

    def forward(self, src: torch.Tensor, segment_labels: torch.Tensor = None, src_lengths=None):
        bz = src.shape[0]
        src_emb = self.src_embedding(src) * self.factor
        src_emb = torch.cat([self.cls_emb.repeat(bz, 1, 1), src_emb], dim=1)
        src_lengths += 1

        src_emb = self.pos_encoder(src_emb, input_lens=src_lengths)

        if segment_labels:
            segment_labels = torch.cat(
                [
                    torch.LongTensor([self.segment_padding_idx], device=src.device).repeat(bz, 1),
                    segment_labels
                ], dim=1)
            src_emb += self.segment_embedding(segment_labels)

        encoder_hidden, encoder_output = self.encoder(
            src_emb, src_mask=None, src_lengths=src_lengths)

        sentence_rep = encoder_output[:, 0, :].squeeze()
        encoder_output = encoder_output[:, 1:, :]

        return encoder_output, sentence_rep
