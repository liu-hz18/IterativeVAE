import os
import logging
import logging.handlers
from configparser import ConfigParser

import torch
from tensorboardX import SummaryWriter


class LogManager(object):
    def __init__(self, checkpoint_step, save_dir, model_name, log_file_name, mode='min', device=None):
        super(LogManager, self).__init__()
        self.checkpoint_step = checkpoint_step
        self.save_dir = os.path.join(save_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.mode = mode
        self.device = device
        self.best_value = float("inf") if mode == "min" else float("-inf")

        self.ckpt_path_name = os.path.join(self.save_dir, model_name)
        self.log_file_name = os.path.join(self.save_dir, log_file_name)

        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s-%(name)s][%(levelname)s]%(message)s')
        self.logger = logging.getLogger(model_name)
        console_handle = logging.StreamHandler()
        console_handle.setLevel(logging.INFO)
        file_handle = logging.FileHandler(self.log_file_name, 'w', encoding='utf-8')
        file_handle.setLevel(logging.INFO)
        # self.logger.addHandler(console_handle)
        self.logger.addHandler(file_handle)

    def log(self, step, epoch, model, value=None, info='', is_train=True):
        if is_train:
            if len(info) > 1:
                self.logger.info(' train: %4d | %s' % (step, info))
            torch.save(model.state_dict(), self.ckpt_path_name + "_last.ckpt")
        else:
            # self._save_state(epoch, model, value)
            if len(info) > 1:
                self.logger.info(' eval:  %4d | %s' % (step, info))

    def save(self, epoch, model, value=None, is_train=True):
        if is_train:
            torch.save(model.state_dict(), self.ckpt_path_name + "_last.ckpt")
        else:
            self._save_state(epoch, model, value)

    def _save_state(self, epoch, model, value=None):
        if epoch % self.checkpoint_step == 0:
            torch.save(model.state_dict(), self.ckpt_path_name + "_%d.ckpt" % epoch)
        if value is None:
            return
        if self.mode == "min" and value < self.best_value:
            torch.save(model.state_dict(), self.ckpt_path_name + "_best.ckpt")
        elif self.mode == "max" and value > self.best_value:
            torch.save(model.state_dict(), self.ckpt_path_name + "_best.ckpt")

    def log_info(self, info=''):
        self.logger.info(info)

    def save_args(self, opt):
        config = ConfigParser()
        config['configs'] = vars(opt)
        self.log_info(opt)
        with open(self.ckpt_path_name + '.args', 'w', encoding='utf-8') as f:
            config.write(f)


class SummaryHelper(object):
    """docstring for SummaryHelper"""
    def __init__(self, save_dir, model_name):
        super(SummaryHelper, self).__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        save_path = os.path.join(self.save_dir, self.model_name)
        self.writer = SummaryWriter(log_dir=save_path)
        self.train_step = 0
        self.valid_step = 0

    def log_loss(self, loss, mode='train'):
        if mode == 'train':
            self.writer.add_scalar(mode + '_loss', loss, self.train_step)
            self.train_step += 1
        else:
            self.writer.add_scalar(mode + '_loss', loss, self.valid_step)
            self.valid_step += 1

    def log_scalar(self, scalar_name, value, global_step, mode='train'):
        self.writer.add_scalar(mode + '_' + scalar_name, value, global_step)

    def log_scalar_list(self, scalar_name, values, global_step, mode='train'):
        for i, value in enumerate(values):
            self.log_scalar(scalar_name + str(i), value, global_step, mode)

    def add_model(self, model, input_data):
        self.writer.add_graph(model, (input_data,))

    def add_embedding(self, global_step, embed_weight, labels):
        self.writer.add_embedding(
            embed_weight, metadata=labels, tag='word embedding', global_step=global_step)

    def add_attn(self, attn_weight):
        pass

    def add_text(self, tag, sentence, global_step):
        self.writer.add_text(tag, sentence, global_step=global_step)

    def close(self):
        self.writer.close()


def plot_attention(input_sentence, output_words, attentions, name='attn', show_label=True):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    if show_label:
        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
        ax.set_yticklabels([''] + output_words)
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(name + ".pdf", format="pdf")
