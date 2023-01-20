import math
import torch
from torch import nn
from torch.nn import functional as F

from .bert_layernorm import BertLayerNorm
from .transformer import Transformer
from ..utils import summary, generate_key_padding_mask


def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam[beam < 2] = 2
    return beam


def duplicate_encoder_out(encoder_out, bsz, beam_size):
    encoder_out['encoder_out'] = encoder_out['encoder_out'].unsqueeze(2).repeat(1, 1, beam_size, 1).view(-1, bsz * beam_size, encoder_out['encoder_out'].size(-1))
    if encoder_out['encoder_padding_mask'] is not None:
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].unsqueeze(1).repeat(1, beam_size, 1).view(bsz * beam_size, -1)


class TransformerConditionalMasked(nn.Module):

    def __init__(self, ntoken, d_model, nhead=8, max_sent_length=64,
                 num_encoder_layers=6, num_decoder_layers=6, ff_dropout=0.1,
                 dim_feedforward=2048, postnorm=True, dropout=0.1, gumbels=False,
                 activation='relu', relative_clip=0, highway=False, device=None,
                 share_input_output_embedding=False, share_encoder_decoder_embedding=False,
                 other_model_embedding=None, mask_id=2, cls_id=3,
                 learned_pos_embedding=False, batch_first=False):
        super(TransformerConditionalMasked, self).__init__()
        self.cls_token_id = cls_id
        self.mask_token_id = mask_id
        self.device = device
        self.max_sent_length = max_sent_length
        self.batch_first = batch_first
        self.transformer = Transformer(
            ntoken, d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, postnorm=postnorm, dropout=dropout, ff_dropout=ff_dropout, gumbels=gumbels,
            use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
            activation=activation, use_vocab_attn=False, use_pos_attn=False,
            relative_clip=0, highway=False, device=device, max_sent_length=max_sent_length,
            share_input_output_embedding=share_input_output_embedding,
            share_encoder_decoder_embedding=share_encoder_decoder_embedding,
            share_vocab_embedding=True, learned_pos_embedding=learned_pos_embedding,
            need_tgt_embed=True, batch_first=batch_first)
        self.embed_length = nn.Embedding(max_sent_length, d_model)
        if other_model_embedding is not None:
            self.transformer.src_embedding = other_model_embedding
        self.dropout = nn.Dropout(0.1)
        self.factor = math.sqrt(d_model)
        self.apply(self.init_parameters)

    def init_parameters(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src, tgt, src_lengths, tgt_lengths):
        '''
        use [CLS] to predict target length
        '''
        encoder_output = self.encode(src, src_lengths)

        pred_lengths_probs = torch.matmul(
            encoder_output[:, 0, :], self.embed_length.weight.transpose(0, 1)).float() # B x max_sent_length
        pred_lengths_probs[:, 0] += float('-inf')

        encoder_output = encoder_output[:, 1:, :]

        decoder_output = self.transformer.decode(
            encoder_output, tgt, src_lengths, tgt_lengths)
        return decoder_output, pred_lengths_probs

    def encode(self, src, src_lengths):
        cls_embed = self.embed_length(src.new(src.size(0), 1).fill_(0))
        src_embed = self.transformer.src_embedding(src)
        src_embed = self.transformer.pos_encoder(
            src_embed, input_lens=src_lengths, use_dropout=False)
        src_embed = torch.cat([cls_embed, src_embed], dim=1)
        src_embed = self.dropout(src_embed)
        if not self.batch_first:
            src_embed = src_embed.transpose(0, 1)
        encoder_hidden, encoder_output = self.transformer.encoder(
            src_embed, embedding=None, src_mask=None,
            src_lengths=src_lengths+1)
        if not self.batch_first:
            encoder_output = encoder_output.transpose(0, 1)
        return encoder_output

    def show_graph(self):
        from torchviz import make_dot
        summary(self, type_size=4)
        dummy_src = torch.rand(12, 20).long().to(self.device)
        dummy_tgt = torch.rand(12, 16).long().to(self.device)
        dummy_src_lens = (20 * torch.rand(12)).long().to(self.device)
        dummy_tgt_lens = (16 * torch.rand(12)).long().to(self.device)
        decoder_output_probs, pred_lengths_probs = self.forward(
            dummy_src, tgt=dummy_tgt, src_lengths=dummy_src_lens, tgt_lengths=dummy_tgt_lens)
        g = make_dot((decoder_output_probs, pred_lengths_probs,), params=dict(self.named_parameters()))
        g.render('transformer_arch', view=False)

    def _generate_worst_mask(self, token_probs, num_mask, pred_lengths):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(
            max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        return torch.stack(
            [torch.ones(seq_len, dtype=torch.bool, device=token_probs.device)
                .index_fill(dim=0, index=mask, value=0) for mask in masks],
            dim=0)
        # print('mask: ', masks)
        # masks = [
        #     torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        # return torch.stack(masks, dim=0)

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

    def generate(self, src, src_lengths, mask_iter=5, tgt_dict=None, length_beam_size=1):
        def convert_tokens(token_ids, tgt_dict, length=None):
            if length is not None:
                token_ids = token_ids[:length]
            if tgt_dict is not None:
                return ' '.join(tgt_dict[idx] for idx in token_ids)
            else:
                return token_ids
        output_file = open('./save/cmlm_samples.txt', 'a', encoding='utf-8')
        encoder_output = self.encode(src, src_lengths)
        pred_lengths_probs = torch.matmul(
            encoder_output[:, 0, :],
            self.embed_length.weight.transpose(0, 1)).float()  # B x max_sent_length
        pred_lengths_probs[:, 0] += float('-inf')
        encoder_output = encoder_output[:, 1:, :]
        pred_lengths = torch.argmax(pred_lengths_probs, dim=-1)
        max_pred_length = max(torch.max(pred_lengths).item(), 1)

        tgt = torch.LongTensor(
            src.shape[0], max_pred_length).fill_(self.mask_token_id).to(src.device)
        pad_mask = generate_key_padding_mask(max_pred_length, pred_lengths)

        tgt_tokens = tgt.masked_fill(pad_mask == 0, value=0)
        # print("Mask Input: ", convert_tokens(tgt_tokens[0], tgt_dict, pred_lengths[0]))

        output = self.transformer.decode(
            encoder_output, tgt_tokens, src_lengths, pred_lengths)
        tgt_tokens, tgt_probs = self.generate_step_with_prob(output)
        tgt_tokens.masked_fill_(pad_mask == 0, value=0)
        tgt_probs.masked_fill_(pad_mask == 0, value=1.0)
        print("="*20+'\n', "Src: ", convert_tokens(src[0], tgt_dict, src_lengths[0]), file=output_file)
        print("Initialization: ", convert_tokens(tgt_tokens[0], tgt_dict, pred_lengths[0]), file=output_file)

        for counter in range(1, mask_iter):
            num_mask = (pred_lengths.float() * (1.0 - (counter / mask_iter))).long()

            mask = self._generate_worst_mask(tgt_probs, num_mask, pred_lengths)

            tgt_tokens.masked_fill_(mask == 0, value=self.mask_token_id)
            tgt_tokens.masked_fill_(pad_mask == 0, value=0)

            print("Step: ", counter, file=output_file)
            print("Masking Input: ", convert_tokens(tgt_tokens[0], tgt_dict, pred_lengths[0]), file=output_file)

            output = self.transformer.decode(
                encoder_output, tgt_tokens, src_lengths, pred_lengths)
            new_tgt_tokens, new_tgt_probs = self.generate_step_with_prob(output)

            tgt_tokens[mask == 0] = new_tgt_tokens[mask == 0]
            tgt_tokens.masked_fill_(pad_mask == 0, value=0)
            tgt_probs[mask == 0] = new_tgt_probs[mask == 0]
            tgt_probs.masked_fill_(pad_mask == 0, value=1.0)

            print("Prediction: ", convert_tokens(tgt_tokens[0], tgt_dict, pred_lengths[0]), file=output_file)
            # print("Probs: ", tgt_probs[0][:pred_lengths[0]])
        print("", file=output_file)
        output_file.close()
        return tgt_tokens, output, pred_lengths_probs

    def generate_step_with_prob(self, decoder_output):
        probs = F.softmax(decoder_output, dim=-1)
        tgt_probs, tgt_tokens = torch.max(probs, dim=-1)
        return tgt_tokens, tgt_probs
