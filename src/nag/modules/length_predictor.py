import torch
from torch import nn
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from ..utils import generate_key_padding_mask


class LengthPredictor(nn.Module):
    def __init__(self, embed_size, min_value=-20, max_value=20,
                 output_transform=False, device=None):
        super(LengthPredictor, self).__init__()
        self.out_size = max_value - min_value + 1
        self.min_value = min_value
        self.pooling_layer = nn.AdaptiveMaxPool1d(output_size=1)
        self.mlp = nn.Linear(embed_size, self.out_size)
        self.output_transform = output_transform
        self.device = device
        if output_transform:
            self.decoder_layer = TransformerDecoderLayer(
                d_tar=embed_size,
                d_src=embed_size,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.3,
                ff_dropout=0.1,
                activation='gelu',
                gumbels=False,
                relative_clip=0,
                use_wo=True,
                no_norm=False,
                device=device,
                use_vocab_attn=False,
                use_pos_attn=False,
                position_encoding_layer=None,
                highway=False,
                postnorm=True,
                batch_first=True
            )

    def forward(self, x, src_length, tgt_length=None):
        '''
        in: B x length x embed_size
        out: B x new_length
        '''
        out = self.pooling_layer(x.permute(0, 2, 1)).squeeze(2)  # out: B x embed_size
        len_out_prob = self.mlp(out)  # out: B x [-m, m]
        if tgt_length is None:
            tgt_length = torch.clamp(
                torch.argmax(len_out_prob.detach(), dim=1) + src_length + self.min_value,
                min=2)   # out: B
        output = self._soft_copy(
            x, src_length, tgt_length)
        if self.output_transform:
            src_len = x.shape[1]
            tgt_len = output.shape[1]
            if src_length is not None:
                memory_key_padding_mask = generate_key_padding_mask(
                    src_len, src_length)
            if tgt_length is not None:
                tgt_key_padding_mask = generate_key_padding_mask(
                    tgt_len, tgt_length)
            output = self.decoder_layer(
                output, x, embedding=None,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_lengths=tgt_length)
        return output, len_out_prob, tgt_length

    def _soft_copy(self, encoder_output, src_lengths, tgt_lengths):
        bz, L, _ = encoder_output.shape
        max_tgt_len = torch.max(tgt_lengths).item()
        decoder_input = []
        for i in range(bz):
            src_len, tgt_len = src_lengths[i].item(), tgt_lengths[i].item()
            index = [(j*src_len)//tgt_len for j in range(tgt_len)] + [min(src_len, L-1)] * (max_tgt_len-tgt_len)
            decoder_input.append(
                torch.index_select(encoder_output[i], dim=0, index=torch.cuda.LongTensor(index)))
        return torch.stack(decoder_input, dim=0)
