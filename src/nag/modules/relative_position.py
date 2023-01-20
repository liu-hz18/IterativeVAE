import torch
from torch import nn


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position, device=None):
        '''
        :param num_units: d_a
        :param max_relative_position: k
        '''
        super(RelativePosition, self).__init__()
        self.num_units = num_units
        self.device = device
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position*2+1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        '''
        for self-att: length_q == length_k == length_x
        return: embeddings: length_q x length_k x d_a
        '''
        range_vec_q = torch.arange(start=0, end=length_q).to(self.device)
        range_vec_k = torch.arange(start=0, end=length_k).to(self.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        final_mat = torch.clamp(
            distance_mat, min=-self.max_relative_position, max=self.max_relative_position)
        # final_mat = final_mat + self.max_relative_position
        embeddings = self.embeddings_table[final_mat.long()]
        return embeddings
