
import torch
from torch import nn
from torch.nn import functional as F
from .transformer import Transformer


class TransformerContinuousDecoder(Transformer):
    """docstring for TransformerContinuousDecoder"""
    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, learned_pos_embedding=False,
                 bos_token=3):
        super(TransformerContinuousDecoder, self).__init__(
            ntoken, d_model, nhead=nhead, gumbels=gumbels,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, postnorm=postnorm, dropout=dropout,
            use_src_mask=use_src_mask, use_tgt_mask=use_tgt_mask, use_memory_mask=use_memory_mask,
            activation=activation, use_vocab_attn=use_vocab_attn, use_pos_attn=use_pos_attn,
            relative_clip=relative_clip, highway=highway, device=device, max_sent_length=max_sent_length,
            share_input_output_embedding=share_input_output_embedding,
            share_encoder_decoder_embedding=share_encoder_decoder_embedding,
            share_vocab_embedding=share_vocab_embedding,
            learned_pos_embedding=learned_pos_embedding, need_tgt_embed=True)
        self.bos_token = bos_token

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        '''
        src: LongTensor of shape (B x L)
        tgt: FloatTensor of shape (B x L x V)
        output: FloatTensor of shape (B x L x V)
        '''
        bz = src.shape[0]
        src_embed = self.src_embedding(src)
        src_embed = self.pos_encoder(src_embed, input_lens=src_lengths)
        if self.need_tgt_embed is True:
            # tgt = self.input_operator(tgt)
            bos_token = torch.LongTensor(bz, 1).fill_(self.bos_token).to(src.device)
            if not self.share_encoder_decoder_embedding:
                bos_embed = self.tgt_embedding(bos_token)
                tgt_embed = torch.matmul(tgt[:, :-1, :], self.tgt_embedding.weight)
            else:
                bos_embed = self.src_embedding(bos_token)
                tgt_embed = torch.matmul(tgt[:, :-1, :], self.src_embedding.weight)
            tgt_embed = torch.cat((bos_embed, tgt_embed), 1)
            tgt_embed = self.pos_decoder(tgt_embed, input_lens=tgt_lengths)
        return self.forward_after_embed(
            src_embed, tgt_embed, src_lengths, tgt_lengths,
            src_key_padding_mask, tgt_key_padding_mask)


class TransformerContinuousEncoder(Transformer):
    """docstring for TransformerContinuousEncoder"""
    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, learned_pos_embedding=False,
                 bos_token=3):
        super(TransformerContinuousEncoder, self).__init__(
            ntoken, d_model, nhead=nhead, gumbels=gumbels,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, postnorm=postnorm, dropout=dropout,
            use_src_mask=use_src_mask, use_tgt_mask=use_tgt_mask, use_memory_mask=use_memory_mask,
            activation=activation, use_vocab_attn=use_vocab_attn, use_pos_attn=use_pos_attn,
            relative_clip=relative_clip, highway=highway, device=device, max_sent_length=max_sent_length,
            share_input_output_embedding=share_input_output_embedding,
            share_encoder_decoder_embedding=share_encoder_decoder_embedding,
            share_vocab_embedding=share_vocab_embedding,
            learned_pos_embedding=learned_pos_embedding, need_tgt_embed=True)
        self.name = 'TransformerContinuousEncoder'
        self.bos_token = bos_token

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        '''
        src: FloatTensor of shape (B x L x V)
        tgt: LongTensor of shape (B x L)
        output: FloatTensor of shape (B x L x V)
        '''
        bz = src.shape[0]

        bos_token = torch.LongTensor(bz, 1).fill_(self.bos_token).to(src.device)
        bos_embed = self.src_embedding(bos_token)
        src_embed = torch.matmul(src, self.src_embedding.weight)
        src_embed = self.pos_decoder(
            torch.cat((bos_embed, src_embed), 1),
            input_lens=src_lengths
        )

        tgt_embed = self.pos_decoder(
            self.tgt_embedding(tgt),
            input_lens=tgt_lengths
        )

        return self.forward_after_embed(
            src_embed, tgt_embed, src_lengths, tgt_lengths,
            src_key_padding_mask, tgt_key_padding_mask)
