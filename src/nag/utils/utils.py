import sys
import os
import numpy as np
import random
from inspect import signature
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F


def init_seed(manual_seed):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    print(f'manual seed: {manual_seed}')


class PadCollate(object):
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """
    def __init__(self, dim=0, pad_id=0, device=None):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.pad_id = pad_id
        self.device = device

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, tensor)
        reutrn:
            xs - a LongTensor of all srcs in batch after padding
            ys - a LongTensor of all tgts in batch after padding
        """
        # find longest sequence
        from torch.nn.utils.rnn import pad_sequence
        src_lens = torch.LongTensor([item[0].shape[0] for item in batch]).to(self.device)
        tgt_lens = torch.LongTensor([item[1].shape[0] for item in batch]).to(self.device)
        srcs = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
        tgt_prev = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
        tgt_gold = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=self.pad_id).to(self.device)
        return srcs, tgt_prev, tgt_gold, src_lens, tgt_lens

    def __call__(self, batch):
        return self.pad_collate(batch)


def summary(model, file=sys.stderr, type_size=4):  # float32
    def repr(model):
        from functools import reduce
        from torch.nn.modules.module import _addindent
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines
        for name, p in model._parameters.items():
            if p is not None:
                total_params += reduce(lambda x, y: x * y, p.shape)
        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, _ = repr(model)
    count = sum(p.numel() for p in model.parameters())
    if file is not None:
        print(string, file=file)
        count_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('| \033[92m{:,}\033[0m params (\033[92m{:,}\033[0m trainable)'.format(
            count, count_train))
        print(f'memory use ~ {(20 * count * type_size * 2 / 1000 / 1000):4f}MiB')
    return count


def typeassert(*type_args, **type_kwargs):
    def decorate(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)
        return wrapper
    return decorate


def get_index(alist, idx):
    try:
        rank = alist.index(idx)
    except ValueError:
        rank = -1
    return rank


def generate_triu_mask(src_len, tgt_len, device=None):
    r"""Generate a square mask for the sequence. 
        masked positions are filled with bool(0).
    """
    mask = torch.BoolTensor(
        torch.triu(torch.ones(src_len, tgt_len)) == 1).to(device)
    return mask.transpose(0, 1)


def generate_key_padding_mask(max_len, lengths):
    '''
    key is padded with bool(0)
    '''
    mask = torch.BoolTensor(
        (np.expand_dims(np.arange(max_len), 1)\
         < np.expand_dims(lengths.cpu().numpy(), 0))).to(lengths.device)
    return mask.transpose(0, 1)


def load_model_state_dict(model, ckpt_name, device):
    if os.path.isfile(ckpt_name + ".ckpt"):
        print(f'loading model from {ckpt_name}...')
        #print(torch.load(ckpt_name + ".ckpt", map_location=device))
        model.load_state_dict(
            torch.load(ckpt_name + ".ckpt", map_location=device),
            strict=True)
    else:
        print('Checkpoint not found! Model remain unchanged!')
    return model


def restore_state_at_step(model, ckpt_name, device, save_dir='./save'):
    ckpt_name = os.path.join(os.path.join(save_dir, ckpt_name), ckpt_name)
    return load_model_state_dict(model, ckpt_name, device)


def restore_best_state(model, ckpt_name, device, save_dir='./save'):
    ckpt_name = os.path.join(os.path.join(save_dir, ckpt_name), ckpt_name) + '_best'
    # ckpt_name = os.path.join(save_dir, ckpt_name) + '_best'
    return load_model_state_dict(model, ckpt_name, device)


def restore_last_state(model, ckpt_name, device, save_dir='./save'):
    ckpt_name = os.path.join(os.path.join(save_dir, ckpt_name), ckpt_name) + '_last'
    # ckpt_name = os.path.join(save_dir, ckpt_name) + '_best'
    return load_model_state_dict(model, ckpt_name, device)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
