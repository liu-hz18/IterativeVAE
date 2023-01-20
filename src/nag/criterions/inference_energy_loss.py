import torch
from torch import nn
from torch.nn import functional as F
from .label_smooth_crossentropy import LabelSmoothedCrossEntropyLoss
from ..modules.operators import (
    StraightThroughLogits,
    StraightThroughSoftmax,
    GumbelSoftmax,
    onehot2d,
)


class InferenceEnergyLoss(nn.Module):
    """docstring for InferenceEnergyLoss"""
    operator_map = {
        'SX': nn.Softmax(dim=2),
        'STL': StraightThroughLogits(),
        'SG': GumbelSoftmax(hard=True, tau=1, dim=-1),
        'ST': StraightThroughSoftmax(dim=-1),
        'GX': GumbelSoftmax(hard=False, tau=1, dim=-1),
    }

    def __init__(self, model_E, operator_input='SX', operator_target='SX', reduction='mean',
                 pad=0, bos=3, eos=4):
        super(InferenceEnergyLoss, self).__init__()
        assert type(model_E).__name__ == 'TransformerContinuousDecoder', \
            'teacher model must be class: `TransformerContinuousDecoder` !'
        self.model_E = model_E
        self.cross_entropy = LabelSmoothedCrossEntropyLoss(
            eps=0.1, ignore_index=pad, reduction=reduction)
        self.reduction = reduction
        self.operator_input = self.operator_map[operator_input]
        self.operator_target = self.operator_map[operator_target]
        self.pad = pad
        self.bos = bos
        self.eos = eos

    def forward(self, enc_input, dec_input, dec_target,
                src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        dec_input_log_probs = F.log_softmax(dec_input, dim=-1)
        dec_input_probs = self.operator_input(dec_input_log_probs)[:, :-1, :]
        dec_output_probs = self.operator_target(dec_input_log_probs)

        dec_input_probs = dec_input_probs.masked_fill(
            dec_target.eq(self.pad), onehot2d(self.pad, dec_input_probs.shape[2]))

        bz = enc_input.shape[0]
        bos_prev = torch.LongTensor([self.bos]).to(enc_input.device).repeat(bz, 1)
        eos_prev = torch.LongTensor([self.eos]).to(enc_input.device).repeat(bz, 1)
        enc_input = torch.cat((bos_prev, enc_input, eos_prev), dim=1)

        dec_output, encoder_output = self.model_E(
            enc_input, dec_input_probs, src_lengths, tgt_lengths,
            src_key_padding_mask, tgt_key_padding_mask)
        energy_output_probs = F.log_softmax(dec_output, dim=-1)

        model_loss = self.cross_entropy(dec_input.permute(0, 2, 1), dec_target)
        teacher_loss = self.cross_entropy(dec_output[:, :-1, :].permute(0, 2, 1), dec_target)
        energy_loss = -torch.sum(dec_output_probs * energy_output_probs, dim=-1)
        if self.reduction == 'mean':
            energy_loss = energy_loss.mean()
        else:
            energy_loss = energy_loss.sum()
        return energy_loss + teacher_loss + model_loss
