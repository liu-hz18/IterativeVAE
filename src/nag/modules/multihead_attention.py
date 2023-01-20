import math
import torch
from torch import nn
from torch.nn import functional as F

from .relative_position import RelativePosition
from .operators import GumbelSoftmax


class MultiHeadAttention(nn.Module):
    '''MultiHeadAttention with relative-position-encoding'''
    def __init__(self, embed_dim, kdim=None, vdim=None, nhead=8, dropout=0.1,
                 activation=None, relative_clip=0, gumbels=False, bias=True,
                 device=None, use_wo=True, batch_first=False, add_bias_kv=False,
                 add_zero_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.relative_clip = relative_clip
        self.batch_first = batch_first
        self.smooth_temp = 1. / math.sqrt(self.head_dim)
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"
        self.activation = activation

        if relative_clip > 0:
            self.relative_position_k = RelativePosition(
                self.head_dim, relative_clip, device)
            self.relative_position_v = RelativePosition(
                self.head_dim, relative_clip, device)
        if self.relative_clip <= 0 and self.qkv_same_dim:
            self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        else:
            self.w_q = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.w_k = nn.Linear(self.kdim, embed_dim, bias=bias)
            self.w_v = nn.Linear(self.vdim, embed_dim, bias=bias)
        if gumbels:
            self.soft_max = GumbelSoftmax(dim=-1)
        else:
            self.soft_max = nn.Softmax(dim=-1)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(self.dropout_p)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn

        self.use_wo = use_wo
        if use_wo:
            self.w_o = nn.Linear(embed_dim, embed_dim, bias)
        self.reset_parameters()

    def reset_parameters(self):
        inv_sqrt2 = 1 / math.sqrt(2)
        if hasattr(self, 'in_proj'):
            nn.init.xavier_uniform_(self.in_proj.weight, gain=inv_sqrt2)
        else:
            nn.init.xavier_uniform_(self.w_q.weight, gain=inv_sqrt2)
            nn.init.xavier_uniform_(self.w_k.weight, gain=inv_sqrt2)
            nn.init.xavier_uniform_(self.w_v.weight, gain=inv_sqrt2)
        nn.init.xavier_uniform_(self.w_o.weight)
        if self.w_o.bias is not None:
            nn.init.constant_(self.w_o.bias, 0.)

    def _reshape_to_batches(self, x):
        if not self.batch_first:
            seq_len, batch_size, in_feature = x.size()
            return x.reshape(-1, batch_size*self.nhead, in_feature//self.nhead)\
                    .transpose(0, 1)
        else:
            batch_size, seq_len, in_feature = x.size()
            return x.reshape(batch_size, -1, self.nhead, in_feature//self.nhead)\
                    .permute(0, 2, 1, 3)\
                    .reshape(batch_size*self.nhead, -1, in_feature//self.nhead)

    def _reshape_from_batches(self, x):
        batch_mul_nhead, seq_len, in_feature_div_nhead = x.size()
        if not self.batch_first:
            return x.transpose(0, 1)\
                    .reshape(-1, batch_mul_nhead//self.nhead, in_feature_div_nhead*self.nhead)
        else:
            return x.reshape(batch_mul_nhead//self.nhead, self.nhead, -1, in_feature_div_nhead)\
                    .permute(0, 2, 1, 3)\
                    .reshape(batch_mul_nhead//self.nhead, -1, in_feature_div_nhead*self.nhead)

    def _scaled_dot_product_attn(self, q, k, v, mask=None):
        scores = q.bmm(k.transpose(1, 2)) * self.smooth_temp
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e8)
        attn = self.soft_max(scores)
        avg_attn_over_heads = attn.detach().reshape(attn.shape[0]//self.nhead, self.nhead, attn.shape[1], attn.shape[2]).mean(dim=1)

        attn = self.dropout(attn)
        return attn.bmm(v), avg_attn_over_heads

    def _relative_attn(self, q, k, v, mask=None):
        _, length_q, _ = q.size()
        _, length_k, _ = k.size()  # (B x L x Da)
        _, length_v, _ = v.size()  # (B x L x Da)
        r_k = self.relative_position_k(length_q, length_k)  # (L x L x Da)
        r_v = self.relative_position_v(length_q, length_v)  # (L x L x Da)
        relative = q.unsqueeze(2).matmul(r_k.transpose(1, 2)).squeeze(2)
        dot = q.bmm(k.transpose(1, 2))
        scores = (relative + dot) * self.smooth_temp
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e8)
        attn = self.soft_max(scores)  # (nhead*bz) x src_length x tgt_length

        avg_attn_over_heads = attn.detach().reshape(attn.shape[0]//self.nhead, self.nhead, attn.shape[1], attn.shape[2]).mean(dim=1)

        attn = self.dropout(attn)
        attn = attn.bmm(v) + attn.unsqueeze(3).mul(r_v).sum(dim=2)
        return attn, avg_attn_over_heads

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        '''
        :q: B x L_q x E (tgt)
        :k: B x L_k x E (src)
        :v: B x L_v x E (src)
            assert L_k == L_v
        :attn_mask: L_q x L_k, Tensor(bool), (tgt x src)
        :key_padding_mask: B x L_k, Tensor(bool)
            value `0` is masked !!!
        :output: B x L x E
        '''
        if self.relative_clip <= 0:
            return self.forward_torch(query, key, value, attn_mask, key_padding_mask)
        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        # create mask
        mask = None
        if attn_mask is not None:
            mask = attn_mask = attn_mask.repeat(q.shape[0], 1, 1)  # BxN x L_q x L_k
        if key_padding_mask is not None:
            mask = key_padding_mask = key_padding_mask.unsqueeze(1).repeat(self.nhead, q.shape[1], 1)

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask & key_padding_mask

        if self.relative_clip > 0:
            y, attn_weights = self._relative_attn(q, k, v, mask)
        else:
            y, attn_weights = self._scaled_dot_product_attn(q, k, v, mask)
        y = self._reshape_from_batches(y)
        if self.use_wo:
            y = self.w_o(y)
            if self.activation is not None:
                y = self.activation(y)

        return y

    def forward_torch(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if hasattr(self, 'in_proj') and self.relative_clip <= 0 and self.qkv_same_dim:
            in_proj_bias = self.in_proj.bias
        else:
            in_proj_bias = torch.cat((self.w_q.bias, self.w_k.bias, self.w_v.bias))
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        attn_mask = ~attn_mask if attn_mask is not None else None
        key_padding_mask = ~key_padding_mask if key_padding_mask is not None else None
        output = F.multi_head_attention_forward(
                query=query,
                key=key,
                value=value,
                embed_dim_to_check=self.embed_dim,
                num_heads=self.nhead,
                in_proj_weight=self.in_proj.weight if self.qkv_same_dim else torch.empty([0]),
                in_proj_bias=in_proj_bias,
                bias_k=self.bias_k,
                bias_v=self.bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=self.dropout_p,
                out_proj_weight=self.w_o.weight,
                out_proj_bias=self.w_o.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
                use_separate_proj_weight=not self.qkv_same_dim,
                q_proj_weight=None if self.qkv_same_dim else self.w_q.weight,
                k_proj_weight=None if self.qkv_same_dim else self.w_k.weight,
                v_proj_weight=None if self.qkv_same_dim else self.w_v.weight,
            )[0]
        if not self.batch_first:
            return output
        else:
            return output.transpose(0, 1)


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class FaireseqMultiHeadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.enable_torch_version and not self.onnx_trace and not static_kv:
            if self.qkv_same_dim:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      self.in_proj_weight,
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask)
            else:
                return F.multi_head_attention_forward(query, key, value,
                                                      self.embed_dim, self.num_heads,
                                                      torch.empty([0]),
                                                      self.in_proj_bias, self.bias_k, self.bias_v,
                                                      self.add_zero_attn, self.dropout,
                                                      self.out_proj.weight, self.out_proj.bias,
                                                      self.training, key_padding_mask, need_weights,
                                                      attn_mask, use_separate_proj_weight=True,
                                                      q_proj_weight=self.q_proj_weight,
                                                      k_proj_weight=self.k_proj_weight,
                                                      v_proj_weight=self.v_proj_weight)

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
