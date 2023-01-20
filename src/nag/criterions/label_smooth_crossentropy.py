from torch import nn
from torch.nn import functional as F


class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    For training, the loss is smoothed with parameter eps,
    while for evaluation, the smoothing is disabled.
    """
    def __init__(self, eps, ignore_index=-100, weight=None, reduction='mean'):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # [batch, c, d1, ..., dk]
        log_soft = F.log_softmax(input, dim=1)
        loss = log_soft * -1.
        # [batch, d1, ..., dk]
        nll_loss = F.nll_loss(
            log_soft, target, weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)
        if self.training:
            # [batch, c, d1, ..., dk]
            inf_mask = loss.eq(float('inf'))
            # [batch, d1, ..., dk]
            smooth_loss = loss.masked_fill(inf_mask, 0.).sum(dim=1)
            eps_i = self.eps / (1.0 - inf_mask.float()).sum(dim=1)
            return nll_loss * (1. - self.eps) + (smooth_loss * eps_i).mean()
        else:
            return nll_loss


class LabelSmoothedCrossEntropyLossWithLength(nn.Module):
    """
    Cross Entropy loss with label smoothing.
    For training, the loss is smoothed with parameter eps,
    while for evaluation, the smoothing is disabled.
    """
    def __init__(self, eps, ignore_index=-100, reduction='sum'):
        super(LabelSmoothedCrossEntropyLossWithLength, self).__init__()
        self.eps = eps
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, output, target, output_lens, target_lens):
        # [batch, d1, ..., dk, c]
        target = target.contiguous().view(-1, 1)

        log_soft = F.log_softmax(output, dim=-1)
        lprobs = log_soft.contiguous().view(-1, log_soft.size(-1))
        length_lprobs = F.log_softmax(output_lens, dim=-1)

        non_pad_mask = target.ne(self.ignore_index)
        ntokens = non_pad_mask.sum().data.item()

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        length_loss = -length_lprobs.gather(dim=-1, index=target_lens.unsqueeze(-1))

        avg_nll_loss = nll_loss.mean()

        if self.reduction == 'sum':
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
        else:
            nll_loss = avg_nll_loss
            smooth_loss = smooth_loss.mean()
            length_loss = length_loss.mean()

        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss + length_loss

        return loss, avg_nll_loss, ntokens
