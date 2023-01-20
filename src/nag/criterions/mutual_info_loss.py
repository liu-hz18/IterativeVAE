import torch
from torch import nn
from torch.nn import functional as F
from .label_smooth_crossentropy import LabelSmoothedCrossEntropyLoss
from ..modules.operators import (
    StraightThroughLogits,
    StraightThroughSoftmax,
    GumbelSoftmax,
    onehot2d
)


class MutualInformationLoss(nn.Module):
    operator_map = {
        'SX': nn.Softmax(dim=2),
        'STL': StraightThroughLogits(),
        'SG': GumbelSoftmax(hard=True, tau=1, dim=-1),
        'ST': StraightThroughSoftmax(dim=-1),
        'GX': GumbelSoftmax(hard=False, tau=1, dim=-1),
    }

    def __init__(self, model_E, operator_input='SX', reduction='mean',
                 pad=0, bos=3, eos=4):
        assert type(model_E).__name__ == 'TransformerContinuousEncoder', \
            'teacher model must be class: `TransformerContinuousEncoder` !'
        self.model_E = model_E
        self.cross_entropy = LabelSmoothedCrossEntropyLoss(
            eps=0.1, ignore_index=pad, reduction=reduction)
        self.reduction = reduction
        self.operator_input = self.operator_map[operator_input]
        self.pad = pad
        self.bos = bos
        self.eos = eos

    def forward(self, src, dec_output, dec_target,
                src_lengths=None, tgt_lengths=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        dec_output_log_probs = F.log_softmax(dec_output, dim=-1)
        dec_output_probs = self.operator_input(dec_output_log_probs)
        dec_output_probs = dec_output_probs.masked_fill(
            dec_target.eq(self.pad), onehot2d(self.pad, dec_output_probs.shape[2]))

        bz = src.shape[0]
        bos_prev = torch.LongTensor([self.bos]).to(src.device).repeat(bz, 1)
        eos_prev = torch.LongTensor([self.eos]).to(src.device).repeat(bz, 1)
        src = torch.cat((bos_prev, src, eos_prev), dim=1)

        teacher_dec_input = src[:, :-1]
        teacher_target = src[:, 1:]

        teacher_dec_output, teacher_enc_output = self.model_E(
            dec_output_probs, teacher_dec_input,
            src_lengths=tgt_lengths, tgt_lengths=src_lengths,
            src_key_padding_mask=tgt_key_padding_mask,
            tgt_key_padding_mask=src_key_padding_mask)

        mmi_loss = self.cross_entropy(
            teacher_dec_output.permute(0, 2, 1), teacher_target)
        model_loss = self.cross_entropy(
            dec_output.permute(0, 2, 1), dec_target)

        return mmi_loss + model_loss
