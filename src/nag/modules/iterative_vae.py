import math
import copy
import numpy as np
import torch
import random
from torch import nn
from torch.nn import functional as F

from .length_predictor import LengthPredictor
from .learned_position_embedding import LearnedPositionalEmbedding
from .vae_bottleneck import VAEBottleNeck
from .transformer_encoder import TransformerEncoder, TransformerEncoderLayer
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from ..utils import generate_key_padding_mask, summary


KL_WEIGHTS = np.array([36, 25, 16, 9, 4, 1]) * 1


class KLAnnealingScheduler(object):
    def __init__(self, init_weight, max_weight, step):
        super(KLAnnealingScheduler, self).__init__()
        self.init_weight = init_weight
        self.max_kl_weight = max_weight
        self.step_cycle = step
        self.step = 0
        self.k = (self.max_kl_weight - self.init_weight) / self.step_cycle

    def update_step(self):
        if self.step > self.step_cycle:
            self.step += 1

    @property
    def kl_weight(self):
        return self.init_weight + self.k * self.step

kl_scheduler = KLAnnealingScheduler(0.0, 1.0, 30000)

class Highway(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Highway, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, 1)

    def forward(self, before, after):
        g = torch.sigmoid(self.gate(before))
        return self.layernorm(before * g + self.dropout(after * (1. - g)))

def gumbel_softmax(logits, dim, tau=1.0, eps=1e-20):
    U = torch.rand(dim).cuda()
    G = -torch.log(-torch.log(U + eps) + eps)
    y = logits + G
    return F.softmax(y / tau, dim=-1)


def log_gumbel_softmax(logits, tau=1.0, eps=1e-20):
    dim = logits.shape[2]  # V
    U = torch.rand(dim).cuda()
    G = -torch.log(-torch.log(U + eps) + eps)
    y = logits + G
    return F.log_softmax(y / tau, dim=-1)


def kl_div(input_logits, target_logits):
    return (input_logits.exp() * (input_logits - target_logits)).sum()


class TransformerDecoderLayerVAE(nn.Module):
    def __init__(self, d_model, vocab_size, nhead, dim_feedforward,
                 attn_dropout=0.1, ff_dropout=0.3,
                 relative_clip=4, device=None,
                 use_vocab_attn=False, highway=False, postnorm=True,
                 batch_first=True, padding_idx=0, max_sent_length=64):
        super(TransformerDecoderLayerVAE, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(
            d_tar=d_model,
            d_src=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            ff_dropout=ff_dropout,
            activation='gelu',
            gumbels=False,
            relative_clip=relative_clip,
            use_wo=True,
            no_norm=False,
            device=device,
            use_vocab_attn=use_vocab_attn,
            use_pos_attn=False,
            position_encoding_layer=None,
            highway=highway,
            postnorm=postnorm,
            batch_first=batch_first
        )
        self.batch_first = batch_first
        self.alpha_linear = nn.Linear(d_model, 1)
        self.beta_linear = nn.Linear(d_model, 1)
        self.residual = Highway(d_model, dropout=0.1)
        # temporary no trainable
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(1.0)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta.data.fill_(1.0)
        self.padding_idx = padding_idx

    def forward(self, prev_tgt_embed, src_embed, tgt_seq,
                input_embedding, output_embedding, tgt_lengths=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                vocab_attn_emb=None, use_tgt=False, use_vae=True):
        recon_loss, kl_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        if use_vae:
            decoder_layer_prev_input = output_embedding(prev_tgt_embed)

            # add softplus to be positive
            alpha = F.softplus(self.alpha_linear(prev_tgt_embed))
            beta = F.softplus(self.beta_linear(prev_tgt_embed)) 

            decoder_layer_logprob = F.log_softmax(decoder_layer_prev_input, dim=-1)
            recon_loss = F.nll_loss(
                decoder_layer_logprob.permute(1, 2, 0), tgt_seq,
                ignore_index=self.padding_idx, reduction='sum'
            )
            if self.training and use_tgt:
                tgt_onehot_probs = F.one_hot(tgt_seq, decoder_layer_prev_input.shape[2]).transpose(0, 1)
                #decoder_layer_joined = (self.alpha * decoder_layer_prev_input + self.beta * tgt_onehot_probs)
                decoder_layer_joined = alpha * decoder_layer_logprob + beta * tgt_onehot_probs
            else:
                decoder_layer_joined = alpha * decoder_layer_logprob
                #decoder_layer_joined = self.alpha * decoder_layer_prev_input

            decoder_layer_joined_logprob = F.log_softmax(decoder_layer_joined, dim=-1)
            decoder_layer_joined_logprob_sampled = log_gumbel_softmax(decoder_layer_joined, tau=1.0)

            kl_loss = kl_div(decoder_layer_joined_logprob, decoder_layer_logprob) # kl(logits, alpha*logits+beta*target)
            #kl_loss = kl_div(decoder_layer_joined_logprob, F.log_softmax(alpha * decoder_layer_logprob, dim=-1)) # kl(alpha*logits, alpha*logits+beta*target)
            decoder_layer_input_embed = input_embedding(
                decoder_layer_joined_logprob_sampled.exp())
            decoder_layer_input_embed = self.residual(prev_tgt_embed, decoder_layer_input_embed)  # highway residual block
        else:
            decoder_layer_input_embed = prev_tgt_embed
        # decoder_layer_joined_logprob = F.one_hot(tgt_seq, decoder_layer_logits.shape[2]).transpose(0, 1).float()
        # decoder_layer_input_embed = input_embedding(decoder_layer_joined_logprob)

        decoder_layer_output = self.decoder_layer(
            tgt=decoder_layer_input_embed,
            src=src_embed,
            embedding=vocab_attn_emb,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        if use_vae:
            beta = torch.mean(beta).detach()
            alpha = torch.mean(alpha).detach()
        else:
            beta = self.beta
            alpha = self.alpha
        return decoder_layer_output, recon_loss, kl_loss, alpha, beta


class TransformerDecoderVAE(nn.Module):
    def __init__(self, nlayers, d_model, vocab_size, nhead, dim_feedforward,
                 attn_dropout=0.1, ff_dropout=0.3,
                 relative_clip=4, device=None,
                 use_vocab_attn=False, highway=False, postnorm=True,
                 batch_first=True, padding_idx=0, max_sent_length=64):
        super(TransformerDecoderVAE, self).__init__()
        decoder_layer = TransformerDecoderLayerVAE(
            d_model, vocab_size, nhead, dim_feedforward,
            attn_dropout=attn_dropout, ff_dropout=ff_dropout,
            relative_clip=relative_clip, device=device,
            use_vocab_attn=use_vocab_attn, highway=highway, postnorm=postnorm,
            batch_first=batch_first, padding_idx=padding_idx)
        self.padding_idx = padding_idx
        self.decoder = nn.ModuleList([
            copy.deepcopy(decoder_layer) for i in range(nlayers)])

    def forward(self, prev_decoder_input, memory, tgt_seq,
                input_embedding, output_embedding, layer_output_embedding,
                embedding=None,
                tgt_lengths=None, tgt_key_padding_mask=None,
                src_lengths=None, memory_key_padding_mask=None):
        src_len = memory.shape[0]
        tgt_len = prev_decoder_input.shape[0]
        if src_lengths is not None and memory_key_padding_mask is None:
            memory_key_padding_mask = generate_key_padding_mask(
                src_len, src_lengths)
        if tgt_lengths is not None and tgt_key_padding_mask is None:
            tgt_key_padding_mask = generate_key_padding_mask(
                tgt_len, tgt_lengths)
        output = prev_decoder_input
        re_losses, kl_losses = [], []
        alphas, betas = [], []
        # use_tgt = (random.randint(0, 2) == 0)
        use_tgt = True
        for (index, layer) in enumerate(self.decoder):
            output, re_loss, kl_loss, alpha, beta = layer(prev_tgt_embed=output,
                                             src_embed=memory, tgt_seq=tgt_seq,
                                             input_embedding=input_embedding,
                                             output_embedding=layer_output_embedding,
                                             vocab_attn_emb=embedding, tgt_lengths=tgt_lengths,
                                             tgt_key_padding_mask=tgt_key_padding_mask,
                                             memory_key_padding_mask=memory_key_padding_mask,
                                             use_tgt=use_tgt, use_vae=index > 0)
            re_losses.append(re_loss)
            kl_losses.append(kl_loss)
            alphas.append(alpha.item())
            betas.append(beta.item())
        # change it ???
        output = layer_output_embedding(output)
        logprobs = F.log_softmax(output, dim=-1)
        final_loss = F.nll_loss(
            logprobs.permute(1, 2, 0), tgt_seq,
            ignore_index=self.padding_idx, reduction='sum')
        re_losses = re_losses[1:] + [final_loss]
        return logprobs, re_losses, kl_losses, alphas, betas


class IterativeVAE(nn.Module):
    def __init__(self, ntoken, d_model,
                 attn_dropout=0.1, ff_dropout=0.3,
                 num_encoder_layers=6, num_decoder_layer=6,
                 post_norm=True, share_encoder_decoder_embedding=False,
                 share_decoder_layer_ouput=True, vocab_attn=False,
                 relative_clip=4, highway=False, max_sent_length=64,
                 postnorm=True, padding_idx=0, device=None):
        super(IterativeVAE, self).__init__()
        print("model kl_weight: ", KL_WEIGHTS)
        self.dim_feedforward = 4 * d_model
        self.nhead = d_model // 64
        self.ntoken = ntoken
        self.device = device
        self.padding_idx = padding_idx
        # build model
        self.input_embedding = nn.Linear(ntoken, d_model, bias=True)
        self.output_embedding = nn.Linear(d_model, ntoken, bias=True)
        self.layer_output_embedding = nn.Linear(d_model, ntoken, bias=True)
        # vocab attn
        if vocab_attn:
            self.vocab_embed = nn.Parameter(torch.Tensor(ntoken, d_model))
            nn.init.xavier_uniform_(self.vocab_embed)
        else:
            self.vocab_embed = None
        # embedding
        self.share_encoder_decoder_embedding = share_encoder_decoder_embedding
        self.src_embedding = nn.Embedding(ntoken, d_model, padding_idx=padding_idx)
        self.pos_encoder = LearnedPositionalEmbedding(
            d_model, dropout=0.1, max_len=max_sent_length, residual=True)
        self.pos_decoder = LearnedPositionalEmbedding(
            d_model, dropout=0.1, max_len=max_sent_length, residual=True)
        self.length_predictor = LengthPredictor(
            d_model, min_value=-20, max_value=20,
            output_transform=False, device=device)
        encoder_layer = TransformerEncoderLayer(
            d_src=d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=attn_dropout,
            ff_dropout=ff_dropout,
            activation='gelu',
            gumbels=False,
            relative_clip=relative_clip,
            use_wo=True,
            no_norm=False,
            device=device,
            highway=highway,
            postnorm=postnorm,
            batch_first=False
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoderVAE(
            nlayers=num_decoder_layer,
            d_model=d_model,
            vocab_size=ntoken,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            relative_clip=relative_clip,
            device=device,
            use_vocab_attn=vocab_attn,
            highway=highway,
            postnorm=postnorm,
            batch_first=False,
            padding_idx=padding_idx,
            max_sent_length=max_sent_length
        )

    def forward(self, src, target, src_lengths=None, tgt_lengths=None):
        if self.training:
            kl_scheduler.update_step()
        src_embed = self.src_embedding(src)
        src_embed = self.pos_encoder(src_embed, input_lens=src_lengths)
        src_embed = src_embed.transpose(0, 1)

        # encoder
        encoder_hidden, encoder_output = self.encoder(
            src_embed, src_lengths=src_lengths)
        # length predict
        decoder_input, delta_length_probs, tgt_lengths = self.length_predictor(
            encoder_output.transpose(0, 1), src_lengths, tgt_lengths)  # B x L x E

        decoder_input = self.pos_decoder(
            decoder_input,
            input_lens=tgt_lengths).transpose(0, 1)  # B x E x L
        # decoder
        decoder_output_logprobs, re_losses, kl_losses, alphas, betas = self.decoder(
            prev_decoder_input=decoder_input,
            memory=encoder_output,
            tgt_seq=target,
            input_embedding=self.input_embedding,
            output_embedding=self.output_embedding,
            layer_output_embedding=self.layer_output_embedding,
            embedding=self.vocab_embed,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths
        )
        decoder_output_probs = decoder_output_logprobs.transpose(0, 1).exp()

        delta_length_gold = torch.clamp(
            tgt_lengths - src_lengths + 20, min=1, max=40)
        length_lprobs = F.log_softmax(delta_length_probs, dim=-1)
        length_loss = -length_lprobs.gather(dim=-1, index=delta_length_gold.unsqueeze(-1))

        loss = sum(re_losses) + sum([weight * kl for (weight, kl) in zip(KL_WEIGHTS, kl_losses)]) * kl_scheduler.kl_weight + length_loss.sum()
        # loss = re_losses[-1] + KL_WEIGHTS * sum(kl_losses) + length_loss.sum()
        ntokens = target.ne(self.padding_idx).sum().data.item()

        decoder_output_probs[:, :, 1] = 0.  # ignore <unk>
        out_seqs = torch.argmax(decoder_output_probs, dim=2)  # N x L x V -> N x L
        out_lens = torch.argmax(delta_length_probs, dim=-1) + src_lengths - 20

        return out_seqs, loss, re_losses, kl_losses, out_lens, ntokens, alphas, betas, kl_scheduler.kl_weight

    def show_graph(self):
        from torchviz import make_dot
        summary(self, type_size=4)
        dummy_src_lens = (20 * torch.rand(12)).long().to(self.device)
        dummy_src = torch.rand(12, max(dummy_src_lens)).long().to(self.device)
        dummy_tgt_lens = (16 * torch.rand(12)).long().to(self.device)
        dummy_tgt = torch.rand(12, max(dummy_tgt_lens)).long().to(self.device)
        _, loss, _, _, _, _, _, _, _ = self.forward(
            dummy_src, dummy_tgt, src_lengths=dummy_src_lens, tgt_lengths=dummy_tgt_lens)
        g = make_dot(
            (loss,),
            params=dict(self.named_parameters()))
        # g.render('iterative_vae', view=False)
