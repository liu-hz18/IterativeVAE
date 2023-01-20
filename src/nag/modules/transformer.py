import math
import copy
import torch
from torch import nn
from torch.nn import functional as F

from .sinusoidal_position_embedding import PositionalEncoding
from .learned_position_embedding import LearnedPositionalEmbedding
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from ..utils import generate_key_padding_mask, generate_triu_mask, summary


class TransformerBase(nn.Module):
    """docstring for TransformerBase"""
    def __init__(self):
        super(TransformerBase, self).__init__()

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        raise NotImplementedError

    def show_graph(self):
        summary(self, type_size=4)

    def beam_search(self, src, tgt_begin, src_length, eos_token_id, beam_size=2, max_length=32):  # for eval mode, bz = 1
        '''
        src: 1 x L, torch.LongTensor()
        tgt_begin: 1 x 1, torch.LongTensor()
        src_length: 1, torch.LongTensor()
        '''
        # init step: 1 -> beam_size
        out_probs = []
        select_path = []
        candidates = []
        encoder_output = self.encode(src, src_lengths=src_length)
        outputs = self.decode(encoder_output, tgt_begin, src_lengths=src_length) # 1 x L x V
        out_prob = outputs[:, -1, :]  # 1 x V
        out_probs.append(out_prob)
        pred_probs, pred_tokens = torch.topk(
            F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
        pred_tokens = torch.flatten(pred_tokens)
        for indice in pred_tokens:
            candidates.append(torch.cat((tgt_begin[0], indice.unsqueeze(0)), dim=-1))
        tgts = torch.stack(candidates, dim=0)
        srcs = src.repeat(beam_size, 1)
        accumulate_probs = pred_probs.repeat(beam_size, 1)
        src_lengths = src_length.repeat(beam_size)
        encoder_output = self.encode(srcs, src_lengths=src_lengths)
        # next step: beam_size -> beam_size^2
        for i in range(max_length):  # O(beam_size x length)
            candidates = []
            outputs = self.decode(encoder_output, tgts, src_lengths=src_lengths)
            out_prob = outputs[:, -1, :]  # beam_size x V
            out_probs.append(out_prob)
            pred_probs, pred_tokens = torch.topk(
                F.log_softmax(out_prob, dim=-1), dim=1, k=beam_size)  # beam_size x beam_size
            pred_tokens = torch.flatten(pred_tokens)
            accumulate_probs += pred_probs
            topk_probs, topk_indices = torch.topk(torch.flatten(accumulate_probs), dim=0, k=beam_size)
            accumulate_probs = topk_probs.repeat(beam_size, 1)
            for indice in topk_indices:
                new_tgt = torch.cat((tgts[indice.item()//beam_size], pred_tokens[indice.item()].unsqueeze(0)), dim=-1)
                candidates.append(new_tgt)
            select_path.append(topk_indices[0]//beam_size)
            tgts = torch.stack(candidates, dim=0)
            if pred_tokens[0].item() == eos_token_id:
                break
        out_probs = torch.stack([out_probs[0][0]] + [out_prob[path] for out_prob, path in zip(out_probs[1:], select_path)], dim=0)
        return tgts[0][1:].unsqueeze(0), out_probs.unsqueeze(0)

    def greedy(self, src, tgt_begin, src_length, eos_token_id, max_length=32):  # for eval mode, bz = 1
        '''
        src: 1 x L, torch.LongTensor()
        tgt_begin: 1 x 1, torch.LongTensor()
        src_length: 1, torch.LongTensor()
        '''
        tgt = tgt_begin
        out_probs = []
        for i in range(max_length):
            output, _ = self.forward(src, tgt, src_lengths=src_length)  # 1 x L x V
            out_prob = output[:, -1, :]
            out_probs.append(out_prob[0])
            pred_token = torch.argmax(out_prob, dim=1)
            tgt = torch.cat((tgt, pred_token.unsqueeze(0)), dim=1)  # 1 x (L+1)
            if pred_token.item() == eos_token_id:
                break
        return tgt[:, 1:], torch.stack(out_probs, dim=0).unsqueeze(0)

    def encode(self, src, src_lengths=None, src_key_padding_mask=None):
        raise NotImplementedError

    def decode(self, encoder_output, decoder_input,
               src_lengths=None, tgt_lengths=None,
               src_key_padding_mask=None, tgt_key_padding_mask=None):
        raise NotImplementedError


class Transformer(TransformerBase):

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, ff_dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, learned_pos_embedding=False, need_tgt_embed=True,
                 batch_first=False):
        super(Transformer, self).__init__()
        self.use_src_mask = use_src_mask
        self.use_tgt_mask = use_tgt_mask
        self.use_memory_mask = use_memory_mask
        self.device = device
        self.batch_first = batch_first
        self.factor = math.sqrt(d_model)
        # src embedding
        self.src_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # output embedding
        self.share_input_output_embedding = share_input_output_embedding
        if not share_input_output_embedding:
            self.out_projection = nn.Linear(d_model, ntoken, bias=False)
        # tgt embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.need_tgt_embed = need_tgt_embed
        if (not share_encoder_decoder_embedding) and need_tgt_embed:
            self.tgt_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # vocab attention
        self.use_vocab_attn = use_vocab_attn
        self.share_vocab_embedding = share_vocab_embedding
        if use_vocab_attn and (not share_vocab_embedding):
            self.vocab_embed = nn.Parameter(torch.Tensor(ntoken, d_model))
            nn.init.xavier_uniform_(self.vocab_embed)
        # pos embedding
        if learned_pos_embedding:
            self.pos_encoder = LearnedPositionalEmbedding(
                d_model, dropout=0.1, max_len=max_sent_length, residual=True)
            self.pos_decoder = LearnedPositionalEmbedding(
                d_model, dropout=0.1, max_len=max_sent_length, residual=True)
        else:
            self.pos_encoder = PositionalEncoding(
                d_model, dropout=dropout, max_len=max_sent_length, residual=True,
                device=self.device, requires_grad=False)
            self.pos_decoder = PositionalEncoding(
                d_model, dropout=dropout, max_len=max_sent_length, residual=True,
                device=self.device, requires_grad=False)
        if use_pos_attn:
            self.position_encoding_layer = PositionalEncoding(
                d_model, max_len=max_sent_length, device=self.device,
                residual=False, requires_grad=False)
        else:
            self.position_encoding_layer = None
        # build model
        encoder_layer = TransformerEncoderLayer(
            d_src=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            ff_dropout=ff_dropout,
            activation=activation,
            gumbels=gumbels,
            relative_clip=relative_clip,
            use_wo=True,
            no_norm=False,
            device=device,
            highway=highway,
            postnorm=postnorm,
            batch_first=batch_first
        )
        decoder_layer = TransformerDecoderLayer(
            d_tar=d_model,
            d_src=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            ff_dropout=ff_dropout,
            activation=activation,
            gumbels=gumbels,
            relative_clip=relative_clip,
            use_wo=True,
            no_norm=False,
            device=device,
            use_vocab_attn=use_vocab_attn,
            use_pos_attn=use_pos_attn,
            position_encoding_layer=self.position_encoding_layer,
            highway=highway,
            postnorm=postnorm,
            batch_first=batch_first
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, dropout=dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, dropout=dropout)

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embed = self.src_embedding(src)
        src_embed = self.pos_encoder(src_embed, input_lens=src_lengths)
        if self.need_tgt_embed is True:
            if self.share_encoder_decoder_embedding:
                tgt_embed = self.src_embedding(tgt)
            else:
                tgt_embed = self.tgt_embedding(tgt)
            tgt_embed = self.pos_decoder(tgt_embed, input_lens=tgt_lengths)
        return self.forward_after_embed(
            src_embed, tgt_embed, src_lengths, tgt_lengths,
            src_key_padding_mask, tgt_key_padding_mask)

    def forward_after_embed(self, src_embed, tgt_embed, src_lengths=None, tgt_lengths=None,
                            src_key_padding_mask=None, tgt_key_padding_mask=None):
        if self.use_vocab_attn:
            if self.share_vocab_embedding:
                embedding = self.src_embedding.weight
            else:
                embedding = self.vocab_embed
        else:
            embedding = None
        encoder_output = self.encode(
            src_embed, src_lengths, src_key_padding_mask, embedding,
            no_embed=True)
        decoder_output = self.decode(
            encoder_output, tgt_embed, src_lengths, tgt_lengths,
            embedding=embedding, src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, no_embed=True)
        return decoder_output, encoder_output

    def _subsequent_mask(self, src_len, tgt_len, use_mask, device=None):
        if use_mask:
            return generate_triu_mask(src_len, tgt_len, device=device)
        else:
            return None

    def encode(self, src, src_lengths=None, src_key_padding_mask=None,
               embedding=None, no_embed=False):
        if no_embed:
            src_embed = src
        else:
            src_embed = self.src_embedding(src)
            src_embed = self.pos_encoder(src_embed, input_lens=src_lengths)
        src_len = src_embed.shape[1]
        src_mask = self._subsequent_mask(
            src_len, src_len, self.use_src_mask, self.device)
        if not self.batch_first:
            src_embed = src_embed.transpose(0, 1)
        encoder_hidden, encoder_output = self.encoder(
            src_embed, embedding=embedding, src_mask=src_mask, src_lengths=src_lengths,
            src_key_padding_mask=src_key_padding_mask)
        if not self.batch_first:
            encoder_output = encoder_output.transpose(0, 1)
        return encoder_output

    def decode(self, encoder_output, decoder_input,
               src_lengths=None, tgt_lengths=None, embedding=None,
               src_key_padding_mask=None, tgt_key_padding_mask=None,
               no_embed=False):
        if no_embed:
            tgt_embed = decoder_input
        elif self.need_tgt_embed is True:
            if self.share_encoder_decoder_embedding:
                tgt_embed = self.src_embedding(decoder_input)
            else:
                tgt_embed = self.tgt_embedding(decoder_input)
            tgt_embed = self.pos_decoder(tgt_embed, input_lens=tgt_lengths)
        src_len, tgt_len = encoder_output.shape[1], tgt_embed.shape[1]
        tgt_mask = self._subsequent_mask(
            tgt_len, tgt_len, self.use_tgt_mask, self.device)
        memory_mask = self._subsequent_mask(
            src_len, tgt_len, self.use_memory_mask, self.device)
        if not self.batch_first:
            tgt_embed = tgt_embed.transpose(0, 1)
            encoder_output = encoder_output.transpose(0, 1)
        decoder_hidden, decoder_output = self.decoder(
            tgt_embed, encoder_output, embedding=embedding,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_lengths=tgt_lengths,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_lengths=src_lengths,
            memory_key_padding_mask=src_key_padding_mask
        )
        if not self.batch_first:
            decoder_output = decoder_output.transpose(0, 1)
        if not self.share_input_output_embedding:
            decoder_output = self.out_projection(decoder_output)
        else:
            decoder_output = F.linear(decoder_output, self.src_embedding.weight)
        return decoder_output


class TransformerTorch(TransformerBase):

    def __init__(self, ntoken, d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
                 activation='relu', use_vocab_attn=False, use_pos_attn=False,
                 relative_clip=0, highway=False, device=None, max_sent_length=64,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 share_vocab_embedding=False, fix_pos_encoding=True):
        super(TransformerTorch, self).__init__()
        self.use_src_mask = use_src_mask
        self.use_tgt_mask = use_tgt_mask
        self.use_memory_mask = use_memory_mask
        self.device = device
        self.factor = math.sqrt(d_model)
        # src embedding
        self.src_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # output embedding
        self.share_input_output_embedding = share_input_output_embedding
        if not share_input_output_embedding:
            self.out_projection = nn.Linear(d_model, ntoken)
        # tgt embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        if not share_encoder_decoder_embedding:
            self.tgt_embedding = nn.Embedding(ntoken, d_model, padding_idx=0)
        # pos embedding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=False,
            max_len=max_sent_length)
        self.pos_decoder = PositionalEncoding(
            d_model, dropout, residual=True, device=device, requires_grad=False,
            max_len=max_sent_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout)

    def forward(self, src, tgt, src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embed = self.src_embedding(src)
        src_embed = self.pos_encoder(src_embed, input_lens=src_lengths)
        if not self.share_encoder_decoder_embedding:
            tgt_embed = self.tgt_embedding(tgt)
        else:
            tgt_embed = self.src_embedding(tgt)
        tgt_embed = self.pos_decoder(tgt_embed, input_lens=tgt_lengths)
        # generate mask
        src_len, tgt_len = src_embed.shape[1], tgt_embed.shape[1]
        src_mask = self._subsequent_mask(
            src_len, src_len, self.use_src_mask, self.device)
        tgt_mask = self._subsequent_mask(
            tgt_len, tgt_len, self.use_tgt_mask, self.device)
        memory_mask = self._subsequent_mask(
            src_len, tgt_len, self.use_memory_mask, self.device)
        if src_lengths is not None and src_key_padding_mask is None:
            src_key_padding_mask = ~generate_key_padding_mask(
                src.shape[1], src_lengths)
        if tgt_lengths is not None and tgt_key_padding_mask is None:
            tgt_key_padding_mask = ~generate_key_padding_mask(
                tgt.shape[1], tgt_lengths)
        # forward
        output = self.transformer(
            src_embed.permute(1, 0, 2), tgt_embed.permute(1, 0, 2),
            src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask)\
            .permute(1, 0, 2)
        if not self.share_input_output_embedding:
            output = self.out_projection(output)
        else:
            output = F.linear(output, self.src_embedding.weight)
        return output, None  # need transpose for CE Loss ! ! ! e.g. output.permute(0, 2, 1)

    def _subsequent_mask(self, src_len, tgt_len, use_mask, device=None):
        if use_mask:
            mask = generate_triu_mask(src_len, tgt_len, device=device)
            return mask.float().masked_fill(mask == 0, float(1e-7)).masked_fill(mask == 1, float(0.0))
        else:
            return None

    def encode(self, src, src_lengths=None, src_key_padding_mask=None):
        pass

    def decode(self, encoder_output, decoder_input,
               src_lengths=None, tgt_lengths=None,
               src_key_padding_mask=None, tgt_key_padding_mask=None):
        pass
