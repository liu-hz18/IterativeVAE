import os
import math
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from nag.modules import TransformerConditionalMasked
from nag.utils import (
    LogManager, SummaryHelper,
    PadCollate, get_index, init_seed,
    restore_best_state,
    restore_last_state,
)
from nag.metrics import BLEUMetric, DistinctNGram
from nag.vocab_helper import VocabBulider
from nag.dataset import OpenSubDataset, IMSDBDataset
from nag.optimizer import RAdam, Adam, AdamW, OptimizerManager
from nag.options import parse_args
from nag.criterions import (
    LabelSmoothedCrossEntropyLoss,
    LabelSmoothedCrossEntropyLossWithLength,
)


def train(epoch, model, dataloader, criterion, optimizer, scheduler):
    global global_train_step
    model.train()
    criterion.train()
    model.zero_grad()
    total_token_loss = 0.
    bleu1_score = 0.
    bleu2_score = 0.
    total_norm = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    info_dict = {
        'loss': .0, 'nll_loss': .0, 'ppl': .0, 'gnorm': .0, 'avg_length': .0,
        'bleu1': .0, 'bleu2': .0, 'd1': .0, 'd2': .0}
    pbar = tqdm(
            enumerate(dataloader, 0),
            desc='train',
            total=len(opensub_dataset)//opt.realbatch,
            postfix=info_dict)
    for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in pbar:
        decoder_output_probs, pred_lengths_probs = model(
            src, tgt=tgt_prev, src_lengths=src_lens, tgt_lengths=tgt_lens)
        # loss
        loss, nll_loss, ntokens = criterion(
            output=decoder_output_probs,
            target=tgt_gold,
            output_lens=pred_lengths_probs,
            target_lens=tgt_lens
        )
        # print(loss.item(), ntokens)
        avg_loss = optim_manager.backward(loss, ntokens)
        total_norm = optim_manager.step()

        total_token_loss += nll_loss.item()

        out_seqs = torch.argmax(decoder_output_probs, dim=2)
        pred_lengths = torch.argmax(pred_lengths_probs, dim=1)
        # calculate metrics
        # bleu1_score += bleu_1(tgt_gold, out_seqs, tgt_lens)
        # bleu2_score += bleu_2(tgt_gold, out_seqs, tgt_lens)
        # distinct_1_score += distinct_1(out_seqs, tgt_lens)
        # distinct_2_score += distinct_2(out_seqs, tgt_lens)
        # summary writer
        global_train_step += 1
        writer.log_loss(loss.item()*ACCUMULATION, mode='train')

        if (i+1) % opt.logstep == 0:
            info_dict['loss'] = avg_loss
            info_dict['nll_loss'] = total_token_loss / opt.logstep
            info_dict['ppl'] = math.exp(info_dict['nll_loss'])
            info_dict['bleu1'] = bleu1_score / opt.logstep
            info_dict['bleu2'] = bleu2_score / opt.logstep
            info_dict['d1'] = distinct_1_score / opt.logstep
            info_dict['d2'] = distinct_2_score / opt.logstep
            info_dict['avg_length'] = torch.mean(pred_lengths.float()).item()
            info_dict['gnorm'] = total_norm
            pbar.set_postfix(info_dict)
            mylogger.log(i, epoch, model, value=avg_loss, is_train=True)
            print('mask input:', ' '.join(convert_ids_to_seq(tgt_prev[0], vocab_bulider)))
            print('pred seq:', ' '.join(convert_ids_to_seq(out_seqs[0], vocab_bulider)))
            print('groud truth:', ' '.join(convert_ids_to_seq(tgt_gold[0], vocab_bulider)))
            total_token_loss = 0.
            bleu1_score = bleu2_score = 0.
            distinct_1_score, distinct_2_score = 0., 0.
            show_gen_seq(
                src[:2], out_seqs[:2], tgt_lens[:2], tgt_gold[:2], vocab_bulider,
                global_train_step, mode='train')
    mylogger.save(epoch, model, value=avg_loss, is_train=True)
    pbar.close()


def eval(epoch, model, dataloader, criterion):
    global global_valid_step
    model.eval()
    criterion.eval()
    model.zero_grad()
    total_token_loss = 0.
    bleu1_score = 0.
    bleu2_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    fout = open(
        os.path.join('./save/' + model_name + '/', model_name + '_' + str(epoch)),
        'w', encoding='utf-8')
    with torch.no_grad():
        for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in tqdm(
                enumerate(dataloader, 0), desc='eval',
                total=len(imsdb_dataset)//opt.realbatch):
            # decoder_output_probs, pred_lengths_probs = model(
            #     src, tgt=tgt_prev, src_lengths=src_lens, tgt_lengths=tgt_lens)
            # pred_tokens = torch.argmax(decoder_output_probs, dim=-1)
            pred_tokens, decoder_output_probs, pred_lengths_probs = model.generate(
                src, src_lengths=src_lens, mask_iter=10, tgt_dict=vocab_bulider.id2vocab)
            min_len = min(decoder_output_probs.shape[1], tgt_gold.shape[1])
            pred_lengths = torch.argmax(pred_lengths_probs, dim=1)
            loss, nll_loss, ntokens = criterion(
                output=decoder_output_probs[:, :min_len, :],
                target=tgt_gold[:, :min_len],
                output_lens=pred_lengths_probs,
                target_lens=tgt_lens
            )
            total_token_loss += nll_loss.item()
            # calculate metrics
            bleu1_score += bleu_1(tgt_gold, pred_tokens, tgt_lens)
            bleu2_score += bleu_2(tgt_gold, pred_tokens, tgt_lens)
            distinct_1_score += distinct_1(pred_tokens, tgt_lens)
            distinct_2_score += distinct_2(pred_tokens, tgt_lens)
            # show sequence
            global_valid_step += 1
            for (src_seq, tgt_seq, out_seq) in zip(src, tgt_gold, pred_tokens):
                ret = convert_ids_to_seq(src_seq, vocab_bulider)
                fout.write('S- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                ret = convert_ids_to_seq(tgt_seq, vocab_bulider)
                fout.write('T- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                ret = convert_ids_to_seq(out_seq, vocab_bulider)
                fout.write('H- ' + ' '.join(ret[:get_index(ret, '<pad>')]) + '\n')
                fout.write('\n')
        # summary
        show_gen_seq(
            src[:2], pred_tokens[:2], tgt_lens[:2], tgt_gold[:2],
            vocab_bulider, global_valid_step, mode='valid')
        avg_loss = total_token_loss / i
        avg_bleu1 = bleu1_score / i
        avg_bleu2 = bleu2_score / i
        avg_distinct_1 = distinct_1_score / i
        avg_distinct_2 = distinct_2_score / i
        avg_length = torch.mean(pred_lengths.float())
        writer.log_loss(avg_loss, mode='valid')
        mylogger.log(
            i, epoch, model, value=avg_loss, is_train=False,
            info=f'loss: {avg_loss:.4f} | ppl: {math.exp(avg_loss):.4f} | BLEU1: {avg_bleu1:.4f} \
            | BLEU2: {avg_bleu2:.4f} | avg_length: {avg_length:.1f} \
            | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f}')
    mylogger.save(epoch, model, value=avg_loss, is_train=False)
    fout.close()


def run_model(model, train_loader, eval_loader, niter, criterion, optimizer, scheduler):
    mylogger.log_info('Running Model')
    for i in range(niter):
        mylogger.log_info(f'EPOCH: {i}, lr: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        #train(i, model, train_loader, criterion, optimizer, scheduler)
        eval(i, model, eval_loader, criterion)


def convert_ids_to_seq(id_seq, vocab_bulider):
    return [vocab_bulider.id_to_word(idx) for idx in id_seq]


def show_gen_seq(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth, vocab_bulider, step, mode='train'):
    for in_id, out_id, out_len, gold_id in zip(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth):
        in_seq = convert_ids_to_seq(in_id, vocab_bulider)
        out_seq = convert_ids_to_seq(out_id[:out_len] if out_len > 0 else out_id, vocab_bulider)
        gold_seq = convert_ids_to_seq(gold_id, vocab_bulider)
        writer.add_text(tag=mode + '_post', sentence=' '.join(in_seq[:get_index(in_seq, '<pad>')]), global_step=step)
        writer.add_text(tag=mode + '_pred', sentence=' '.join(out_seq), global_step=step)
        writer.add_text(tag=mode + '_reps', sentence=' '.join(gold_seq[:get_index(in_seq, '<pad>')]), global_step=step)


if __name__ == '__main__':
    begin_time = time.strftime("%H%M%S", time.localtime())
    model_name = 'cmlm' + begin_time
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(opt.gpuid)
    init_seed(opt.manualSeed)
    ACCUMULATION = opt.batchsize // opt.realbatch

    mylogger = LogManager(checkpoint_step=1,
                          save_dir='./save',
                          model_name=model_name,
                          log_file_name=model_name + '.log',
                          mode='min', device=device)
    mylogger.save_args(opt)
    writer = SummaryHelper(save_dir='./save', model_name=model_name)

    vocab_bulider = VocabBulider.build(
        data_dir='./data/opensubtitles', vocab_file_list=['dialogue_length3_6.post'],
        min_count=opt.mincount, rebuild=opt.update)
    mylogger.log_info('vocab size: %d' % len(vocab_bulider))

    # metircs
    bleu_1 = BLEUMetric(vocab_bulider.id2vocab, ngram=1, ignore_smoothing_error=True)
    bleu_2 = BLEUMetric(vocab_bulider.id2vocab, ngram=2, ignore_smoothing_error=True)
    distinct_1 = DistinctNGram(ngram=1)
    distinct_2 = DistinctNGram(ngram=2)

    # dataset and dataloader
    if opt.cotk:  # use dataset in paper 'Non-Autoregressive Neural Dialogue Generation'
        opensub_file_name_list = ['opensub_pair_dev', 'opensub_pair_test', 'opensub_pair_train']
        unk_token = None
    else:  # use dataset in paper 'cotk'
        opensub_file_name_list = ['dialogue_length3_6']
        unk_token = 'UNknown'
    opensub_dataset = OpenSubDataset(
        data_dir='./data/opensubtitles', vocab_bulider=vocab_bulider,
        file_name_list=opensub_file_name_list, unk_token=unk_token,
        save_process=False, samples=opt.trainsamples, add_bos=False, add_eos=True,
        use_mask=True, mask_all=False, inverse=opt.inverse)
    print(opensub_dataset.sample())
    opensub_dataloader = DataLoader(
        opensub_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(
            dim=0, pad_id=vocab_bulider.pad, device=device),
        shuffle=True, num_workers=opt.workers, drop_last=True)

    imsdb_file_name_list = ['imsdb_lower']
    imsdb_dataset = IMSDBDataset(
        data_dir='./data/imsdb', vocab_bulider=vocab_bulider,
        file_name_list=imsdb_file_name_list, save_process=False,
        samples=opt.validsamples, add_bos=False, add_eos=True,
        use_mask=True, mask_all=False, inverse=opt.inverse)
    print(imsdb_dataset.sample())
    imsdb_dataloader = DataLoader(
        imsdb_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(
            dim=0, pad_id=vocab_bulider.pad, device=device),
        shuffle=False, num_workers=opt.workers, drop_last=True)

    # model definition
    model = TransformerConditionalMasked(
        ntoken=len(vocab_bulider), d_model=opt.embedsize, nhead=opt.nhead, max_sent_length=64,
        num_encoder_layers=opt.encoderlayer, num_decoder_layers=opt.decoderlayer,
        dim_feedforward=opt.feedforward, postnorm=True, dropout=opt.dropout,
        ff_dropout=0.0,
        gumbels=opt.gumbels,
        activation='gelu', relative_clip=0, highway=False, device=device,
        share_input_output_embedding=False, share_encoder_decoder_embedding=True,
        mask_id=vocab_bulider.mask, cls_id=vocab_bulider.bos,
        learned_pos_embedding=opt.learned,
        batch_first=False).to(device)
    model.show_graph()
    if opt.half:
        model = model.half()
    if opt.ft:
        model = restore_last_state(model, opt.ckpt, save_dir='./save', device=model.device)

    # optimizer and scheduler
    optimizers = {'adam': Adam, 'radam': RAdam, 'adamw': AdamW, 'sgd': torch.optim.SGD}
    if opt.optimizer == 'adam':
        opt.warmup = True
    if opt.warmup:
        optimizer = optimizers[opt.optimizer](
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1., betas=(opt.beta1, opt.beta2), eps=opt.eps,
            weight_decay=opt.weight_decay)
        rate_ratio = 1. / math.sqrt(opt.embedsize)
        warmup_step = int(1. / (opt.lr * opt.lr * opt.embedsize))
        warmup_step_increase_rate = warmup_step**(-1.5)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: rate_ratio * min(1. / math.sqrt(step+1), step*warmup_step_increase_rate+1e-5))
    else:
        optimizer = optimizers[opt.optimizer](
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps,
            weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.schedulerstep, gamma=opt.gamma)
    # loss function
    criterion = LabelSmoothedCrossEntropyLossWithLength(
        eps=0.1, ignore_index=vocab_bulider.pad, reduction='sum')
    optim_manager = OptimizerManager(
        model, optimizer, scheduler, update_freq=ACCUMULATION, max_norm=25.0)
    # run model
    global_train_step, global_valid_step = 0, 0
    run_model(
        model, opensub_dataloader, imsdb_dataloader,
        opt.niter, criterion, optimizer, scheduler)
    writer.close()
