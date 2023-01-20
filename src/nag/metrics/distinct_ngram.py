
class DistinctNGram(object):
    """docstring for DistinctNGram"""
    def __init__(self, ngram=2):
        super(DistinctNGram, self).__init__()
        self.ngram = ngram
        self.gram_dict = {}

    def _clip_seq(self, id_list, length):
        return list(map(str, id_list[:length] if length > 0 else id_list))

    def _stat_ngram_in_seq(self, tokens):
        tlen = len(tokens)
        for i in range(0, tlen - self.ngram + 1):
            ngram_token = ' '.join(tokens[i:(i + self.ngram)])
            if self.gram_dict.get(ngram_token) is not None:
                self.gram_dict[ngram_token] += 1
            else:
                self.gram_dict[ngram_token] = 1

    def _batch_stat(self, candidates, lengths):
        for seq, length in zip(candidates, lengths):
            self._stat_ngram_in_seq(self._clip_seq(seq, length))

    def forward(self, candidates, lengths):
        self.gram_dict = {}
        self._batch_stat(candidates, lengths)
        return len(self.gram_dict.keys()) / (sum(self.gram_dict.values()) + 1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
