import os
import re
import math
import time
import numpy as np
from tqdm import tqdm
import itertools
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nag.modules import IterativeVAE
from nag.utils import (
    LogManager, SummaryHelper,
    PadCollate, get_index, restore_best_state, init_seed,
    restore_last_state,
)
from nag.metrics import BLEUMetric, DistinctNGram
from nag.vocab_helper import VocabBulider
from nag.dataset import OpenSubDataset, IMSDBDataset
from nag.optimizer import RAdam, OptimizerManager
from nag.options import parse_args


def train(epoch, model, dataloader, optimizer, scheduler):
    global global_train_step
    model.train()
    bleu1_score = 0.
    bleu2_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in tqdm(enumerate(dataloader, 0), desc='train', total=len(opensub_dataset)//opt.realbatch):
        out_seqs, loss, re_losses, kl_losses, out_lens, ntokens, alphas, betas, kl_weight = model(
            src, tgt_gold, src_lengths=src_lens, tgt_lengths=tgt_lens)
        # loss
        avg_loss = optim_manager.backward(loss, ntokens)
        total_norm = optim_manager.step()
        re_losses = [re_loss.item() / ntokens for re_loss in re_losses]
        kl_losses = [kl_loss.item() / ntokens for kl_loss in kl_losses]
        # calculate metrics
        bleu1_score += bleu_1(tgt_gold, out_seqs, out_lens)
        bleu2_score += bleu_2(tgt_gold, out_seqs, out_lens)
        distinct_1_score += distinct_1(out_seqs, out_lens)
        distinct_2_score += distinct_2(out_seqs, out_lens)
        # summary writer
        global_train_step += 1
        if (i+1) % ACCUMULATION == 0:
            writer.log_loss(avg_loss*ACCUMULATION, mode='train')
            writer.log_scalar_list('kl_loss_', kl_losses, global_train_step, mode='train')
            writer.log_scalar_list('recon_loss_', re_losses, global_train_step, mode='train')
            writer.log_scalar_list('alpha_', alphas, global_train_step, mode='train')
            writer.log_scalar_list('beta_', betas, global_train_step, mode='train')
        if (global_train_step+1) % (ACCUMULATION*100) == 0:
            reconstruct(epoch, model, imsdb_dataloader)
        if (i+1) % opt.logstep == 0:
            avg_bleu1 = bleu1_score / opt.logstep
            avg_bleu2 = bleu2_score / opt.logstep
            avg_length = torch.mean(out_lens.float())
            avg_distinct_1 = distinct_1_score / opt.logstep
            avg_distinct_2 = distinct_2_score / opt.logstep
            mylogger.log(
                i, epoch, model, value=avg_loss, is_train=True,
                info=f'loss: {avg_loss:.4f} | ppl: {math.exp(min(avg_loss, 60)):.4f} | BLEU1: {avg_bleu1:.4f} | BLEU2: {avg_bleu2:.4f} | avg_length: {avg_length:.1f} | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f} | gnorm: {total_norm:.4f} |\nre_loss: {str(re_losses)} |\nkl_loss: {str(kl_losses)} |\nalpha: {str(alphas)} |\nbeta: {str(betas)}|\n kl_weight: {str(kl_weight)}')
            bleu1_score = 0.
            distinct_1_score, distinct_2_score = 0., 0.
            show_gen_seq(src[:2], out_seqs[:2], out_lens[:2], tgt_gold[:2], vocab_bulider, global_train_step, mode='train')


def eval(epoch, model, dataloader):
    global global_valid_step
    model.eval()
    total_loss = 0.
    bleu1_score = 0.
    bleu2_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    re_loss_list, kl_loss_list = [], []
    fout = open(os.path.join('./save/' + model_name + '/', model_name + '_' + str(epoch)), 'w', encoding='utf-8')
    with torch.no_grad():
        for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in tqdm(enumerate(dataloader, 0), desc='eval', total=len(imsdb_dataset)//opt.realbatch):
            out_seqs, loss, re_losses, kl_losses, out_lens, ntokens, alphas, betas, kl_weight = model(
                src, tgt_gold, src_lengths=src_lens, tgt_lengths=tgt_lens)
            # loss
            total_loss += (loss / ntokens).item()
            re_losses = [re_loss.item() / ntokens for re_loss in re_losses]
            kl_losses = [kl_loss.item() / ntokens for kl_loss in kl_losses]
            re_loss_list.append(re_losses)
            kl_loss_list.append(kl_losses)
            # calculate metrics
            bleu1_score += bleu_1(tgt_gold, out_seqs, out_lens)
            bleu2_score += bleu_2(tgt_gold, out_seqs, out_lens)
            distinct_1_score += distinct_1(out_seqs, out_lens)
            distinct_2_score += distinct_2(out_seqs, out_lens)
            # show sequence
            for (src_seq, tgt_seq, out_seq) in zip(src, tgt_gold, out_seqs):
                ret = convert_ids_to_seq(src_seq, vocab_bulider)
                fout.write('S- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                ret = convert_ids_to_seq(tgt_seq, vocab_bulider)
                fout.write('T- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                ret = convert_ids_to_seq(out_seq, vocab_bulider)
                fout.write('H- ' + ' '.join(ret) + '\n')
                ret = [k for k, g in itertools.groupby(ret)]
                fout.write('P- ' + ' '.join(ret) + '\n')
                fout.write('\n')
            if (i+1) % opt.logstep == 0:
                show_gen_seq(
                    src[:2], out_seqs[:2], out_lens[:2], tgt_gold[:2], vocab_bulider, global_valid_step, mode='valid')
        # summary
        global_valid_step += 1
        avg_loss = total_loss / i
        avg_bleu1 = bleu1_score / i
        avg_bleu2 = bleu2_score / i
        avg_length = torch.mean(out_lens.float())
        avg_distinct_1 = distinct_1_score / i
        avg_distinct_2 = distinct_2_score / i
        writer.log_loss(avg_loss, mode='valid')
        kl_losses = np.mean(np.array(kl_loss_list), axis=0)
        re_losses = np.mean(np.array(re_loss_list), axis=0)
        writer.log_scalar_list('kl_loss_', kl_losses, global_valid_step, mode='valid')
        writer.log_scalar_list('recon_loss_', re_losses, global_valid_step, mode='valid')
        mylogger.log(
            i, global_valid_step, model, value=avg_bleu1, is_train=False,
            info=f'loss: {avg_loss:.4f} | ppl: {math.exp(min(avg_loss, 60)):.4f} | BLEU1: {avg_bleu1:.4f} | BLEU2: {avg_bleu2:.4f} | avg_length: {avg_length:.1f} | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f} | re_loss: {str(re_losses)} | kl_loss: {str(kl_losses)}')


def reconstruct(epoch, model, dataloader):
    global global_recon_step
    model.train()
    total_loss = 0.
    bleu1_score = 0.
    bleu2_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    re_loss_list, kl_loss_list = [], []
    fout = open(os.path.join('./save/' + model_name + '/', model_name + '_' + str(epoch) + ".recon"), 'w', encoding='utf-8')
    with torch.no_grad():
        for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in tqdm(enumerate(dataloader, 0), desc='eval', total=len(imsdb_dataset)//opt.realbatch):
            out_seqs, loss, re_losses, kl_losses, out_lens, ntokens, alphas, betas, kl_weight = model(
                src, tgt_gold, src_lengths=src_lens, tgt_lengths=tgt_lens)
            # loss
            total_loss += (loss / ntokens).item()
            re_losses = [re_loss.item() / ntokens for re_loss in re_losses]
            kl_losses = [kl_loss.item() / ntokens for kl_loss in kl_losses]
            re_loss_list.append(re_losses)
            kl_loss_list.append(kl_losses)
            # calculate metrics
            bleu1_score += bleu_1(tgt_gold, out_seqs, out_lens)
            bleu2_score += bleu_2(tgt_gold, out_seqs, out_lens)
            distinct_1_score += distinct_1(out_seqs, out_lens)
            distinct_2_score += distinct_2(out_seqs, out_lens)
            # show sequence 
            for i, (src_seq, tgt_seq, out_seq) in enumerate(zip(src, tgt_gold, out_seqs)):
                ret = convert_ids_to_seq(src_seq, vocab_bulider)
                fout.write('S- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                ret = convert_ids_to_seq(tgt_seq, vocab_bulider)
                fout.write('T- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                ret = convert_ids_to_seq(out_seq[:tgt_lens[i]], vocab_bulider)
                fout.write('H- ' + ' '.join(ret) + '\n')
                ret = [k for k, g in itertools.groupby(ret)]
                fout.write('P- ' + ' '.join(ret) + '\n')
                fout.write('\n')
            if (i+1) % opt.logstep == 0:
                show_gen_seq(
                    src[:2], out_seqs[:2], out_lens[:2], tgt_gold[:2], vocab_bulider, global_recon_step, mode='recon')
        # summary
        global_recon_step += 1
        avg_loss = total_loss / i
        avg_bleu1 = bleu1_score / i
        avg_bleu2 = bleu2_score / i
        avg_length = torch.mean(out_lens.float())
        avg_distinct_1 = distinct_1_score / i
        avg_distinct_2 = distinct_2_score / i
        writer.log_loss(avg_loss, mode='reconstruct')
        kl_losses = np.mean(np.array(kl_loss_list), axis=0)
        re_losses = np.mean(np.array(re_loss_list), axis=0)
        writer.log_scalar_list('kl_loss_', kl_losses, global_recon_step, mode='reconstruct')
        writer.log_scalar_list('recon_loss_', re_losses, global_recon_step, mode='reconstruct')
        mylogger.log(
            i, global_recon_step, model, value=avg_bleu1, is_train=False,
            info=f'loss: {avg_loss:.4f} | ppl: {math.exp(min(avg_loss, 60)):.4f} | BLEU1: {avg_bleu1:.4f} | BLEU2: {avg_bleu2:.4f} | avg_length: {avg_length:.1f} | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f} | re_loss: {str(re_losses)} | kl_loss: {str(kl_losses)}')


def run_model(model, train_loader, eval_loader, niter, optimizer, scheduler):
    mylogger.log_info('Running Model')
    for i in range(niter):
        mylogger.log_info(f'EPOCH: {i}, lr: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        train(i, model, train_loader, optimizer, scheduler)
        eval(i, model, eval_loader)
        # reconstruct(i, model, eval_loader)


def convert_ids_to_seq(id_seq, vocab_bulider):
    word_list = [vocab_bulider.id_to_word(idx) for idx in id_seq]
    return re.sub(r'(<unk> )+', r'<unk> ', ' '.join(word_list)).replace('<eos>', '').split()


def show_gen_seq(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth, vocab_bulider, step, mode='train'):
    for in_id, out_id, out_len, gold_id in zip(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth):
        in_seq = convert_ids_to_seq(in_id, vocab_bulider)
        out_seq = convert_ids_to_seq(out_id[:out_len] if out_len > 0 else out_id, vocab_bulider)
        out_seq_processed = [k for k, g in itertools.groupby(out_seq)]
        gold_seq = convert_ids_to_seq(gold_id, vocab_bulider)
        writer.add_text(tag=mode + '_post', sentence=' '.join(in_seq[:get_index(in_seq, '<pad>')]), global_step=step)
        writer.add_text(tag=mode + '_pred', sentence=' '.join(out_seq), global_step=step)
        writer.add_text(tag=mode + '_reps', sentence=' '.join(gold_seq[:get_index(in_seq, '<pad>')]), global_step=step)
        writer.add_text(tag=mode + '_pred_processed', sentence=' '.join(out_seq_processed), global_step=step)


if __name__ == '__main__':
    begin_time = time.strftime("%H%M%S", time.localtime())
    model_name = 'vae' + begin_time

    src_suffix = '.post'
    tgt_suffix = '.response'
    data_dir = './data/opensubtitles'
    dev_data_dir = './data/imsdb'
    opensub_file_name_list = ['dialogue_length3_6']
    imsdb_file_name_list = ['imsdb_lower']
    vocab_file_list = ['dialogue_length3_6.post', 'dialogue_length3_6.response']
    """
    src_suffix = '.en'
    tgt_suffix = '.ro'
    data_dir = './data/wmt16en-ro'
    dev_data_dir = './data/wmt16en-ro'
    opensub_file_name_list = ['corpus.bpe']
    imsdb_file_name_list = ['dev.bpe']
    vocab_file_list = ['corpus.bpe.en', 'corpus.bpe.ro']
    """

    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(opt.gpuid)
    print(device)
    init_seed(opt.manualSeed)
    ACCUMULATION = opt.batchsize // opt.realbatch
    opt.logstep = max(opt.logstep, ACCUMULATION)

    mylogger = LogManager(checkpoint_step=10,
                          save_dir='./save',
                          model_name=model_name,
                          log_file_name=model_name + '.log',
                          mode='min', device=device)
    mylogger.save_args(opt)
    writer = SummaryHelper(save_dir='./save', model_name=model_name)

    vocab_bulider = VocabBulider(
        data_dir, src_files=vocab_file_list, ignore_unk_error=True,
        vocab_file='vocab.txt', min_count=opt.mincount, update=opt.update)
    print('most common 50:', vocab_bulider.most_common(50))
    mylogger.log_info('vocab size: %d' % len(vocab_bulider))

    # metircs
    bleu_1 = BLEUMetric(vocab_bulider.id2vocab, ngram=1, ignore_smoothing_error=True)
    bleu_2 = BLEUMetric(vocab_bulider.id2vocab, ngram=2, ignore_smoothing_error=True)
    distinct_1 = DistinctNGram(ngram=1)
    distinct_2 = DistinctNGram(ngram=2)

    # dataset and dataloader
    if opt.cotk:  # use dataset in paper 'cotk'
        opensub_file_name_list = ['opensub_pair_dev', 'opensub_pair_test', 'opensub_pair_train']
        opensub_dataset = OpenSubDataset(
            data_dir=data_dir, vocab_bulider=vocab_bulider,
            file_name_list=opensub_file_name_list, unk_token=None,
            save_process=False, samples=opt.trainsamples, add_bos=False, add_eos=True,
            inverse=opt.inverse, use_mask=False, train=True, src_suffix=src_suffix, tgt_suffix=tgt_suffix)
    else:  # use dataset in paper 'Non-Autoregressive Neural Dialogue Generation'
        opensub_dataset = OpenSubDataset(
            data_dir=data_dir, vocab_bulider=vocab_bulider,
            file_name_list=opensub_file_name_list, unk_token='UNknown',
            save_process=False, samples=opt.trainsamples, add_bos=False, add_eos=True,
            inverse=opt.inverse, use_mask=False, src_suffix=src_suffix, tgt_suffix=tgt_suffix)
    print(opensub_dataset.sample())
    opensub_dataloader = DataLoader(
        opensub_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(
            dim=0, pad_id=vocab_bulider.padid, device=device),
        shuffle=True, num_workers=opt.workers, drop_last=True)

    imsdb_dataset = IMSDBDataset(
        data_dir=dev_data_dir, vocab_bulider=vocab_bulider,
        file_name_list=imsdb_file_name_list, save_process=False,
        samples=opt.validsamples, add_bos=False, add_eos=False,
        inverse=opt.inverse, use_mask=False, src_suffix=src_suffix, tgt_suffix=tgt_suffix)
    print(imsdb_dataset.sample())
    imsdb_dataloader = DataLoader(
        imsdb_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(
            dim=0, pad_id=vocab_bulider.padid, device=device),
        shuffle=True, num_workers=opt.workers, drop_last=True)

    # model definition
    model = IterativeVAE(
        ntoken=len(vocab_bulider), d_model=opt.embedsize,
        attn_dropout=0.1, ff_dropout=0.3,
        num_encoder_layers=opt.encoderlayer, num_decoder_layer=opt.decoderlayer,
        post_norm=True, share_encoder_decoder_embedding=False,
        share_decoder_layer_ouput=True, vocab_attn=opt.vocabattn,
        relative_clip=4, highway=False,
        postnorm=True, padding_idx=vocab_bulider.padid, device=device
    ).to(device)
    model.show_graph()

    if opt.half:
        model = model.half()
    if opt.ft:
        model = restore_last_state(model, opt.ckpt, save_dir='./save', device=model.device)

    # optimizer and scheduler
    if opt.warmup:
        # optim.Adam()
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1., betas=(opt.beta1, opt.beta2), eps=opt.eps)
        rate_ratio = 1. / math.sqrt(opt.embedsize)
        # top_lr = 1 / sqrt(d_model * warmup_step) at step == warmup_step
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: rate_ratio * min(1. / math.sqrt(step+1), step*(opt.warmup_step**(-1.5))))
    else:
        # optim.Adam()
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.schedulerstep, gamma=opt.gamma)
    # loss function
    optim_manager = OptimizerManager(
        model, optimizer, scheduler, update_freq=ACCUMULATION, max_norm=25.0)

    # run model
    global_train_step, global_valid_step, global_recon_step = 0, 0, 0
    run_model(
        model, opensub_dataloader, imsdb_dataloader,
        opt.niter, optimizer, scheduler)
    writer.close()
