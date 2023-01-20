import os
import copy
import numpy as np
import joblib
from tqdm import tqdm
import torch


class SingleTurnDialogDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_name_list, vocab_bulider,
                 save_process=False, max_len=64, samples=None,
                 add_bos=True, add_eos=True, use_mask=False, mask_all=True,
                 inverse=False, src_suffix='.post', tgt_suffix='.response'):
        '''
        :param data_dir:   string, data dir
        :param data_files: List, [filename1, filename2, ...]
        '''
        super(SingleTurnDialogDataset, self).__init__()
        self.src_suffix = src_suffix
        self.tgt_suffix = tgt_suffix
        self.data_dir = data_dir
        self.file_name_list = file_name_list
        self.vocab_bulider = vocab_bulider
        self.vocab_bulider.ignore_unk_error = True
        self.save_process = save_process
        self.max_len = max_len
        self.add_eos, self.add_bos = add_eos, add_bos
        self.samples = samples if samples is not None else 100000
        self.posts, self.reps = [], []
        self.count = 0
        self.bos_id, self.eos_id = vocab_bulider.bos, vocab_bulider.eos
        self.pad_id, self.unk_id = vocab_bulider.pad, vocab_bulider.unk
        self.use_mask = use_mask
        if use_mask:
            self.mask_id = vocab_bulider.mask
        self.mask_all = mask_all
        self.inverse = inverse

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        dec_input, dec_target = self.mask_seq(self.reps[idx])
        return self.posts[idx], dec_input, dec_target

    def _convert_line_to_ids(self, line):
        raise NotImplementedError

    def _save_pickle(self, file_name, tensor_list):
        fname = os.path.join(self.data_dir, file_name + '.pkl')
        joblib.dump(tensor_list, fname, compress=True)

    def _read_pickle(self, file_name):
        print(f'Reading from pickle file...{file_name}')
        fname = os.path.join(self.data_dir, file_name + '.pkl')
        sentences = joblib.load(fname)
        return sentences

    def _read_data_file(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        if os.path.exists(file_path + '.pkl') and self.save_process:
            sentences = self._read_pickle(file_name)
        else:
            sentences = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f'reading: {file_name}'):
                    sentence = self._convert_line_to_ids(line)
                    if len(sentence) > self.max_len:
                        sentence = sentence[:self.max_len]
                    if self.add_bos and self.add_eos:
                        sentences.append(torch.LongTensor([self.bos_id] + sentence + [self.eos_id]))
                    elif self.add_bos:
                        sentences.append(torch.LongTensor([self.bos_id] + sentence))
                    elif self.add_eos:
                        sentences.append(torch.LongTensor(sentence + [self.eos_id]))
                    else:
                        sentences.append(torch.LongTensor(sentence))
                    self.count += 1
                    if self.count > self.samples:
                        break
            if self.save_process:
                self._save_pickle(file_name, sentences)
        return sentences

    def _prepare_dataset(self):
        for file_name in self.file_name_list:
            last_count = self.count
            self.posts.extend(self._read_data_file(file_name + self.src_suffix))
            self.count = last_count
            self.reps.extend(self._read_data_file(file_name + self.tgt_suffix))
        if self.inverse:
            self.posts, self.reps = self.reps, self.posts

    def sample(self):
        idx = np.random.randint(0, len(self.posts))
        return self.__getitem__(idx)

    def mask_seq(self, target):
        if not self.use_mask:
            return target, target
        else:
            target_len = len(target)
            if not self.mask_all:
                mask_size = np.random.randint(1, target_len+1)
                mask_ids = np.random.choice(target_len, size=mask_size, replace=False)
                dec_target = target.new([self.pad_id] * target_len)
                dec_target[mask_ids] = target[mask_ids]
                dec_input = copy.deepcopy(target)
                dec_input[mask_ids] = self.mask_id
                return dec_input, dec_target
                # return dec_input, target
            else:
                return target.new([self.mask_id]*target_len), target


class OpenSubDataset(SingleTurnDialogDataset):

    def __init__(self, data_dir, file_name_list, vocab_bulider,
                 unk_token=None, save_process=False, max_len=64, samples=None,
                 add_bos=True, add_eos=True, use_mask=False, mask_all=True,
                 inverse=False, src_suffix='.post', tgt_suffix='.response'):
        super(OpenSubDataset, self).__init__(
            data_dir, file_name_list, vocab_bulider, save_process, max_len, samples,
            add_bos, add_eos, use_mask, mask_all, inverse, src_suffix, tgt_suffix)
        self.unk_token = unk_token
        self._prepare_dataset()
        assert len(self.posts) == len(self.reps), 'length of posts DON\'T MATCH length of reps'

    def _convert_to_id(self, token):
        x = '<unk>' if token == self.unk_token else token
        return self.vocab_bulider[x]

    def _convert_line_to_ids(self, line):
        return list(map(lambda x: self._convert_to_id(x), line.strip().split()))


class IMSDBDataset(SingleTurnDialogDataset):

    def __init__(self, data_dir, file_name_list, vocab_bulider,
                 save_process=False, max_len=64, samples=None,
                 add_bos=True, add_eos=True, use_mask=False, mask_all=True,
                 inverse=False, src_suffix='.post', tgt_suffix='.response'):
        super(IMSDBDataset, self).__init__(
            data_dir, file_name_list, vocab_bulider, save_process, max_len, samples,
            add_bos, add_eos, use_mask, mask_all, inverse, src_suffix, tgt_suffix)
        self._prepare_dataset()
        assert len(self.posts) == len(self.reps), 'length of posts DON\'T MATCH length of reps'

    def _convert_line_to_ids(self, line):
        return list(map(lambda x: self.vocab_bulider[x], line.strip().split()))
