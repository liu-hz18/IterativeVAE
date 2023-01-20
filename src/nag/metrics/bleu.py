
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'


def tensor_index(atensor, value):
    b = torch.tensor([value], device=atensor.device)
    pos = torch.nonzero(torch.eq(atensor, b), as_tuple=False).squeeze(1)
    p = -1
    try:
        p = pos[0]
    except:
        pass
    return p


class BLEUMetric(object):
    bleu_weight_dict = {
        '1': [1.0, 0.0, 0.0, 0.0],
        '2': [0.5, 0.5, 0.0, 0.0],
        '3': [0.334, 0.333, 0.333, 0.0],
        '4': [0.25, 0.25, 0.25, 0.25],
    }

    def __init__(self, id2vocab, ngram=-1, ignore_smoothing_error=False):
        self.id2vocab = id2vocab
        self.vocab_len = len(id2vocab)
        self.pad_id = id2vocab.index(PAD_TOKEN)
        # self.eos_id = id2vocab.index(EOS_TOKEN)
        self.ignore_smoothing_error = ignore_smoothing_error
        self._reference = []
        self._candidate = []
        self.smooth = SmoothingFunction()
        if ngram <= 0 or ngram > 4:
            self.weights = [0.25, 0.25, 0.25, 0.25]
        else:
            self.weights = self.bleu_weight_dict[str(ngram)]

    def _batch_trim(self, batch_ref, batch_can):
        for data in batch_ref:
            self._reference.append([self._convert_to_words(data)])
        for data in batch_can:
            self._candidate.append(self._convert_to_words(data))

    def _trim_before_target(self, lists, target_id):
        lists = lists[:tensor_index(lists, target_id)]
        return lists

    def _drop_pad(self, lists):
        idx = len(lists)
        while idx > 0 and lists[idx - 1] == self.pad_id:
            idx -= 1
        ids = lists[:idx]
        return ids

    def _convert_to_words(self, id_list):
        ids = self._drop_pad(self._trim_before_target(id_list, target_id=self.pad_id))
        words = list(map(lambda word: self.id2vocab[word] if word < self.vocab_len else '<unk>',
                         ids))
        return words

    def _clip_seq(self, id_list, length):
        return list(map(lambda word: self.id2vocab[word] if word < self.vocab_len else '<unk>',
                        id_list[:length] if length > 0 else id_list))

    def _batch_clip(self, batch_ref, batch_can, can_lenths):
        for data in batch_ref:
            self._reference.append([self._convert_to_words(data)])
        for data, length in zip(batch_can, can_lenths):
            self._candidate.append(self._clip_seq(data, length))

    def _calculate(self):
        try:
            corpus_score = corpus_bleu(
                self._reference, self._candidate, weights=self.weights,
                smoothing_function=self.smooth.method3)
        except ZeroDivisionError as _:
            if not self.ignore_smoothing_error:
                raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
                    usually caused when there is only one sample and the sample length is 1.")
            corpus_score = 0.
        return corpus_score

    def forward(self, references, candidates, lengths=None):
        self._reference, self._candidate = [], []
        if lengths is None:
            self._batch_trim(references, candidates)
        else:
            self._batch_clip(references, candidates, lengths)
        return self._calculate()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
