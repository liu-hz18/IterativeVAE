
import torch
from torch import nn
from torch.nn import functional as F


def neighbor_cosine_similarity(hidden):
    hidden_shift_left = hidden[:][1:][:]
    hidden_shift_right = hidden[:][:-1][:]
    return torch.mean(F.cosine_similarity(hidden_shift_left, hidden_shift_right, dim=2))


def similarity_regularization(hidden, out, alpha=0.1):
    L_hidden = neighbor_cosine_similarity(hidden)
    L_prob = neighbor_cosine_similarity(out)
    loss = 1 + L_hidden * (1 - L_prob)
    return alpha * loss
