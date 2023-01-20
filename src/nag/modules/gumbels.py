
from torch import nn
from torch.nn import functional as F


class GumbelSoftmax(nn.Module):
    def __init__(self, dim=None, tau=1):
        super(GumbelSoftmax, self).__init__()
        self.dim = dim
        self.tau = tau

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.tau, dim=self.dim)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)
