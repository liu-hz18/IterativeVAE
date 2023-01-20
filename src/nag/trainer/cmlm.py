import math
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..modules import TransformerConditionalMasked
from ..dataset import OpenSubDataset, IMSDBDataset
from ..metric import BLEUMetric, DistinctNGram
from ..logger import LogManager, SummaryHelper
from ..vocab_helper import VocabBulider
from ..utils import PadCollate, get_index, restore_best_state, init_seed


class CMLMTrainer(object):

    def __init__(self, args):
        super(CMLMTrainer, self).__init__()
        self.args = args
        begin_time = time.strftime("%H%M%S", time.localtime())
        self.model_name = 'cmlm' + begin_time
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(args.gpuid)
        init_seed(args.manualSeed)
        self.ACCUMULATION = args.batchsize // args.realbatch

    def build_metric(self, vocab_bulider):
        self.bleu1 = BLEUMetric(vocab_bulider.id2vocab, ngram=1, ignore_smoothing_error=True)
        self.bleu2 = BLEUMetric(vocab_bulider.id2vocab, ngram=2, ignore_smoothing_error=True)
        self.distinct_1 = DistinctNGram(ngram=1)
        self.distinct_2 = DistinctNGram(ngram=2)

    def build_vocab(self):
        self.vocab_bulider = VocabBulider.build(
            data_dir='./data/opensubtitles', vocab_file_list=['dialogue_length3_6.post'],
            min_count=self.args.mincount, rebuild=self.args.update)
        self.mylogger.log_info('vocab size: %d' % len(self.vocab_bulider))

    def build_logger(self):
        self.mylogger = LogManager(
            checkpoint_step=10,
            save_dir='./save',
            model_name=self.model_name,
            log_file_name=self.model_name + '.log',
            mode='min', device=self.device)
        self.mylogger.save_args(self.args)
        self.writer = SummaryHelper(save_dir='./save', model_name=self.model_name)

    def build_dataloader(self, add_bos, add_eos):
        if self.args.cotk:  # use dataset in paper 'Non-Autoregressive Neural Dialogue Generation'
            opensub_file_name_list = ['opensub_pair_dev', 'opensub_pair_test', 'opensub_pair_train']
            opensub_dataset = OpenSubDataset(
                data_dir='./data/opensubtitles', vocab_bulider=self.vocab_bulider,
                file_name_list=opensub_file_name_list, unk_token=None,
                save_process=False, samples=self.args.trainsamples, add_bos=add_bos, add_eos=add_eos)
        else:  # use dataset in paper 'cotk'
            opensub_file_name_list = ['dialogue_length3_6']
            opensub_dataset = OpenSubDataset(
                data_dir='./data/opensubtitles', vocab_bulider=self.vocab_bulider,
                file_name_list=opensub_file_name_list, unk_token='UNknown',
                save_process=False, samples=self.args.trainsamples, add_bos=add_bos, add_eos=add_eos)
        print(opensub_dataset.sample())
        self.train_loader = DataLoader(
            opensub_dataset, batch_size=self.args.realbatch,
            collate_fn=PadCollate(
                dim=0, pad_id=self.vocab_bulider.padid, inverse=self.args.inverse, device=self.device),
            shuffle=True, num_workers=self.args.workers, drop_last=True)

        imsdb_file_name_list = ['imsdb_lower']
        imsdb_dataset = IMSDBDataset(
            data_dir='./data/imsdb', vocab_bulider=self.vocab_bulider,
            file_name_list=imsdb_file_name_list, save_process=False,
            samples=self.args.validsamples, add_bos=add_bos, add_eos=add_eos)
        print(imsdb_dataset.sample())
        self.dev_loader = DataLoader(
            imsdb_dataset, batch_size=self.args.realbatch,
            collate_fn=PadCollate(
                dim=0, pad_id=self.vocab_bulider.padid, inverse=self.args.inverse, device=self.device),
            shuffle=True, num_workers=self.args.workers, drop_last=True)

    def build_model(self):
        opt = self.args
        self.model = TransformerConditionalMasked(
            ntoken=len(self.vocab_bulider), d_model=opt.embedsize, nhead=opt.nhead, max_sent_length=64,
            num_encoder_layers=opt.encoderlayer, num_decoder_layers=opt.decoderlayer,
            dim_feedforward=opt.feedforward, postnorm=True, dropout=opt.dropout, gumbels=opt.gumbels,
            activation='relu', relative_clip=0, highway=False, device=device,
            share_input_output_embedding=False, share_encoder_decoder_embedding=True).to(device)
        self.model.show_graph()
        if opt.half:
            self.model = self.model.half()
        if opt.ft:
            self.model = restore_best_state(
                self.model, opt.ckpt, save_dir='./save', device=self.model.device)

    def build_optimizer(self):
        pass


