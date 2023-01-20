import random
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    # gpu config
    parser.add_argument('--gpuid', type=int, default=0, help='id of GPU to use')
    parser.add_argument('--manualSeed', type=int, default=random.randint(1, 10000), help='manual seed')
    parser.add_argument('--half', action='store_true', help='half precision floating point')
    # basic config
    parser.add_argument('--beam', type=int, default=1, help='use beam search (size)')
    parser.add_argument('--mine', action='store_true', help='use my Transformer')
    parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
    parser.add_argument('--cotk', action='store_true', help='use \'cotk\' OpenSubDataset')
    parser.add_argument('--batchsize', type=int, default=1024, help='input batch size')
    parser.add_argument('--realbatch', type=int, default=128, help='real batch size')
    parser.add_argument('--logstep', type=int, default=50, help='log interval')
    parser.add_argument('--schedulerstep', type=int, default=10000, help='step size for scheduler')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--trainsamples', type=int, default=50000, help='samples to train')
    parser.add_argument('--validsamples', type=int, default=5000, help='samples to eval')
    parser.add_argument('--inverse', action='store_true', help='inverse tgt and src')
    # vocab builder
    parser.add_argument('--mincount', type=int, default=1, help='min count of vocab')
    parser.add_argument('--update', action='store_true', help='min count of vocab')
    # RAdam
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lengthratio', type=float, default=0.1, help='ratio of length loss')
    parser.add_argument('--beta1', type=float, default=0.9, help='hyperparameter \'beta1\' for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='hyperparameter \'beta2\' for Adam')
    parser.add_argument('--eps', type=float, default=1e-6, help='hyperparameter \'eps\' for Adam')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='hyperparameter \'weight_decay\' for Adam')
    # model hyperparameters
    parser.add_argument('--gumbels', action='store_true', help='use gumbels softmax')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--nhead', type=int, default=8, help='number of head in MultiheadAttention()')
    parser.add_argument('--embedsize', type=int, default=512, help='embedding size of nn.Embedding()')  # 1024
    parser.add_argument('--encoderlayer', type=int, default=6, help='number of encoder layers')
    parser.add_argument('--decoderlayer', type=int, default=6, help='number of decoder layers')
    parser.add_argument('--feedforward', type=int, default=2048, help='dimension of Feedforward Net')  # 4096
    parser.add_argument('--nolayernorm', action='store_true', help='disables layernorm')
    parser.add_argument('--delta', type=int, default=20, help='delta length between \'src\' and \'tgt\'')
    parser.add_argument('--posattn', action='store_true', help='use Multihead Positional attention')
    parser.add_argument('--vocabattn', action='store_true', help='use Vocabulary Attention')
    parser.add_argument('--learned', default=True, action='store_true', help='learned positional encoding layer')
    # warm up
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--warmup', action='store_true', help='whether to use warm up')
    parser.add_argument('--warmup_step', type=int, default=2000, help='warmup epochs')
    parser.add_argument('--gamma', type=float, default=1.00, help='decay coefficient of learning rate')
    # fine tune
    parser.add_argument('--ft', action='store_true', help='fine-tune')
    parser.add_argument('--ckpt', type=str, default='nag', help='fine-tune file name')
    parser.add_argument('--energy', type=str, default='transformer', help='fine-tune file name')
    return parser.parse_args()
