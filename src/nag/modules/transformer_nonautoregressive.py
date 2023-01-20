import torch
from torch import nn
from torch.nn import functional as F

from .transformer import Transformer
from .length_predictor import LengthPredictor
from ..utils import summary


class TransformerNonAutoRegressive(Transformer):

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, learned_pos_embedding=True,
                 min_length_change=-20, max_length_change=20, use_src_to_tgt=False,
                 batch_first=False):
        use_src_mask = use_tgt_mask = use_memory_mask = False
        super(TransformerNonAutoRegressive, self).__init__(
            ntoken, d_model, nhead, num_encoder_layers, num_decoder_layers,
            dim_feedforward, postnorm, dropout, dropout, gumbels,
            use_src_mask, use_tgt_mask, use_memory_mask,
            activation, use_vocab_attn, use_pos_attn,
            relative_clip, highway, device, max_sent_length,
            share_input_output_embedding, share_encoder_decoder_embedding,
            share_vocab_embedding,
            learned_pos_embedding=learned_pos_embedding, need_tgt_embed=False,
            batch_first=batch_first)
        self.min_length_change = min_length_change
        self.max_length_change = max_length_change
        self.length_predictor = LengthPredictor(
            d_model, min_value=min_length_change, max_value=max_length_change,
            output_transform=False, device=self.device)
        self.use_src_to_tgt = use_src_to_tgt

    def forward(self, src, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        '''
        use length-predictor to predict target length
        '''
        src_embed_prev = self.src_embedding(src)
        src_embed = self.pos_encoder(
            src_embed_prev, input_lens=src_lengths)
        if self.use_vocab_attn:
            if self.share_vocab_embedding:
                embedding = self.src_embedding.weight
            else:
                embedding = self.vocab_embed
        else:
            embedding = None
        # forward
        if not self.batch_first:
            src_embed = src_embed.transpose(0, 1)
        encoder_hidden, encoder_output = self.encoder(
            src_embed, embedding=embedding, src_mask=None, src_lengths=src_lengths,
            src_key_padding_mask=src_key_padding_mask)
        if not self.batch_first:
            encoder_output = encoder_output.transpose(0, 1)

        if self.use_src_to_tgt:
            decoder_input, delta_length_probs, tgt_lengths = self.length_predictor(
                src_embed_prev+encoder_output, src_lengths, tgt_lengths)  # B x L x E
        else:
            decoder_input, delta_length_probs, tgt_lengths = self.length_predictor(
                encoder_output, src_lengths, tgt_lengths)  # B x L x E

        decoder_input_probs = None
        # if not self.share_input_output_embedding:
        #     decoder_input_probs = self.out_projection(decoder_input)
        # else:
        #     decoder_input_probs = F.linear(decoder_input, self.src_embedding.weight)

        tgt_embed = self.pos_decoder(decoder_input, input_lens=tgt_lengths)
        if not self.batch_first:
            tgt_embed = tgt_embed.transpose(0, 1)
            encoder_output = encoder_output.transpose(0, 1)
        decoder_hidden, decoder_output = self.decoder(
            tgt_embed, encoder_output, embedding=embedding, tgt_mask=None,
            memory_mask=None, tgt_lengths=tgt_lengths,
            tgt_key_padding_mask=tgt_key_padding_mask)
        if not self.batch_first:
            decoder_output = decoder_output.transpose(0, 1)
        if not self.share_input_output_embedding:
            output = self.out_projection(decoder_output)
        else:
            output = F.linear(decoder_output, self.src_embedding.weight)
        return output, delta_length_probs, decoder_input_probs  # need transpose for CE Loss ! ! ! e.g. output.permute(0, 2, 1)

    def show_graph(self):
        from torchviz import make_dot
        summary(self, type_size=4)
        dummy_src = torch.rand(12, 20).long().to(self.device)
        dummy_src_lens = (20 * torch.rand(12)).long().to(self.device)
        dummy_tgt_lens = (16 * torch.rand(12)).long().to(self.device)
        decoder_output_probs, pred_lengths_probs, decoder_input_probs = self.forward(
            dummy_src, src_lengths=dummy_src_lens, tgt_lengths=dummy_tgt_lens)
        g = make_dot(
            (decoder_output_probs, pred_lengths_probs,),
            params=dict(self.named_parameters()))
        g.render('transformer_arch', view=False)
