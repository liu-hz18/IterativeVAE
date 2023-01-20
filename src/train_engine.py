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

from nag.modules import (
    TransformerConditionalMasked,
    TransformerContinuousDecoder,
)
from nag.utils import (
    LogManager, SummaryHelper,
    PadCollate, get_index, restore_best_state, init_seed,
)
from nag.metrics import BLEUMetric, DistinctNGram
from nag.vocab_helper import VocabBulider
from nag.dataset import OpenSubDataset, IMSDBDataset
from nag.optimizer import RAdam
from nag.options import parse_args
from nag.criterions import (
    LabelSmoothedCrossEntropyLoss,
    InferenceEnergyLoss,
)
from nag.modules.operators import onehot3d


def train(epoch, model, dataloader, criterion_energy, criterionL, optimizer, scheduler):
    global global_train_step
    model.train()
    total_loss = 0.
    total_loss_model = 0.
    bleu1_score = bleu2_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in tqdm(
            enumerate(dataloader, 0), desc='train',
            total=len(opensub_dataset)//opt.realbatch):
        decoder_output_probs, pred_lengths_probs = model(
            src, tgt=tgt_prev, src_lengths=src_lens, tgt_lengths=tgt_lens)
        decoder_output_probs_T = decoder_output_probs.permute(0, 2, 1)
        out_seqs = torch.argmax(decoder_output_probs, dim=2)
        pred_lengths = torch.argmax(pred_lengths_probs, dim=1)
        # loss
        loss_sentence = criterion_energy(
            enc_input=src, dec_input=decoder_output_probs,
            dec_target=tgt_gold, src_lengths=src_lens, tgt_lengths=tgt_lens)
        loss_model = criterionM(decoder_output_probs_T, tgt_gold)
        loss_length = criterionL(pred_lengths_probs, tgt_lens)
        loss = (loss_sentence + loss_length * opt.lengthratio) / ACCUMULATION
        loss.backward()
        total_loss += loss.item()
        total_loss_model += loss_model.item()
        # calculate metrics
        bleu1_score += bleu_1(tgt_gold, out_seqs, tgt_lens)
        bleu2_score += bleu_2(tgt_gold, out_seqs, tgt_lens)
        distinct_1_score += distinct_1(out_seqs, tgt_lens)
        distinct_2_score += distinct_2(out_seqs, tgt_lens)
        # summary writer
        global_train_step += 1
        writer.log_loss(loss.item()*ACCUMULATION, mode='train')
        if (i+1) % ACCUMULATION == 0:
            # clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if (i+1) % opt.logstep == 0:
            avg_loss = (total_loss / opt.logstep) * ACCUMULATION
            avg_bleu1 = bleu1_score / opt.logstep
            avg_bleu2 = bleu2_score / opt.logstep
            avg_distinct_1 = distinct_1_score / opt.logstep
            avg_distinct_2 = distinct_2_score / opt.logstep
            avg_length = torch.mean(pred_lengths.float())
            mylogger.log(
                i, epoch, model, value=avg_loss, is_train=True,
                info=f'energy: {avg_loss:.4f} \
                | ppl: {math.exp(total_loss_model*ACCUMULATION/opt.logstep):.4f} \
                | BLEU1: {avg_bleu1:.4f} \
                | BLEU2: {avg_bleu2:.4f} \
                | avg_length: {avg_length:.1f} \
                | d1: {avg_distinct_1:.3f} \
                | d2: {avg_distinct_2:.3f}')
            total_loss = 0.
            total_loss_model = 0.
            bleu1_score = bleu2_score = 0.
            distinct_1_score, distinct_2_score = 0., 0.
            show_gen_seq(
                src[:2], out_seqs[:2], tgt_lens[:2], tgt_gold[:2], vocab_bulider,
                global_train_step, mode='train')


def eval(epoch, model, dataloader, criterion_energy, criterionL):
    global global_valid_step
    total_loss = 0.
    total_loss_model = 0.
    bleu1_score, bleu2_score = 0., 0.
    distinct_1_score, distinct_2_score = 0., 0.
    fout = open(
        os.path.join('./save/' + model_name + '/', model_name + '_' + str(epoch)),
        'w', encoding='utf-8')
    with torch.no_grad():
        for i, (src, tgt_prev, tgt_gold, src_lens, tgt_lens) in tqdm(
                enumerate(dataloader, 0), desc='eval', total=len(imsdb_dataset)//opt.realbatch):
            decoder_output_probs, pred_lengths_probs = model.generate(
                src, src_lengths=src_lens, mask_iter=5)
            min_len = min(decoder_output_probs.shape[1], tgt_gold.shape[1])
            decoder_output_probs = decoder_output_probs[:, :min_len, :]
            tgt_gold = tgt_gold[:, :min_len]
            output_seqs = torch.argmax(decoder_output_probs, dim=2)
            decoder_output_probs_T = decoder_output_probs.permute(0, 2, 1)
            # loss
            loss_sentence = criterion_energy(
                enc_input=src, dec_input=decoder_output_probs,
                dec_target=tgt_gold, src_lengths=src_lens, tgt_lengths=tgt_lens)
            loss_length = criterionL(pred_lengths_probs, tgt_lens)
            loss_model = criterionM(decoder_output_probs_T, tgt_gold)
            pred_lengths = torch.argmax(pred_lengths_probs, dim=1)
            loss = (loss_sentence + loss_length * opt.lengthratio)
            total_loss += loss.item()
            total_loss_model += loss_model.item()
            # calculate metrics
            bleu1_score += bleu_1(tgt_gold, output_seqs, tgt_lens)
            bleu2_score += bleu_2(tgt_gold, output_seqs, tgt_lens)
            distinct_1_score += distinct_1(output_seqs, tgt_lens)
            distinct_2_score += distinct_2(output_seqs, tgt_lens)
            # show sequence
            global_valid_step += 1
            for out_seq in output_seqs:
                ret = convert_ids_to_seq(out_seq, vocab_bulider)
                fout.write(' '.join(ret[:get_index(ret, '<eos>')]) + '\n')
        # summary
        show_gen_seq(
            src[:2], output_seqs[:2], tgt_lens[:2], tgt_gold[:2],
            vocab_bulider, global_valid_step, mode='valid')
        avg_loss = total_loss / i
        avg_bleu1 = bleu1_score / i
        avg_bleu2 = bleu2_score / i
        avg_distinct_1 = distinct_1_score / i
        avg_distinct_2 = distinct_2_score / i
        avg_length = torch.mean(pred_lengths.float())
        writer.log_loss(avg_loss, mode='valid')
        mylogger.log(
            i, epoch, model, value=avg_loss, is_train=False,
            info=f'energy: {avg_loss:.4f} | ppl: {math.exp(total_loss_model/i):.4f} \
            | BLEU1: {avg_bleu1:.4f} \
            | BLEU2: {avg_bleu2:.4f} | avg_length: {avg_length:.1f} \
            | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f}')
    fout.close()


def run_model(model, train_loader, eval_loader, niter,
              criterion_energy, criterionL, optimizer, scheduler):
    mylogger.log_info('Running Model')
    for i in range(niter):
        mylogger.log_info(f'EPOCH: {i}, lr: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        train(
            i, model, train_loader, criterion_energy, criterionL, optimizer, scheduler)
        eval(i, model, eval_loader, criterion_energy, criterionL)


def convert_ids_to_seq(id_seq, vocab_bulider):
    return [vocab_bulider.id_to_word(idx) for idx in id_seq]


def show_gen_seq(batch_in_seqs, batch_out_seqs, e_pred_seqs, batch_out_lens, groud_truth, vocab_bulider, step, mode='train'):
    for in_id, out_id, out_len, gold_id, e_id in zip(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth, e_pred_seqs):
        in_seq = convert_ids_to_seq(in_id, vocab_bulider)
        out_seq = convert_ids_to_seq(out_id[:out_len] if out_len > 0 else out_id, vocab_bulider)
        e_pred_seq = convert_ids_to_seq(e_id[:out_len] if out_len > 0 else e_id, vocab_bulider)
        gold_seq = convert_ids_to_seq(gold_id, vocab_bulider)
        writer.add_text(tag=mode + '_post', sentence=' '.join(in_seq[:get_index(in_seq, '<pad>')]), global_step=step)
        writer.add_text(tag=mode + '_pred', sentence=' '.join(out_seq), global_step=step)
        writer.add_text(tag=mode + '_e_pred', sentence=' '.join(e_pred_seq), global_step=step)
        writer.add_text(tag=mode + '_reps', sentence=' '.join(gold_seq[:get_index(in_seq, '<pad>')]), global_step=step)


if __name__ == '__main__':
    begin_time = time.strftime("%H%M%S", time.localtime())
    model_name = 'engine' + begin_time
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(opt.gpuid)
    init_seed(opt.manualSeed)
    ACCUMULATION = opt.batchsize // opt.realbatch

    mylogger = LogManager(checkpoint_step=10,
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
        save_process=False, samples=opt.trainsamples, add_bos=False, add_eos=False,
        use_mask=True, train=True)
    print(opensub_dataset.sample())
    opensub_dataloader = DataLoader(
        opensub_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(dim=0, pad_id=vocab_bulider.pad, device=device),
        shuffle=True, num_workers=opt.workers, drop_last=True)

    imsdb_file_name_list = ['imsdb_lower']
    imsdb_dataset = IMSDBDataset(
        data_dir='./data/imsdb', vocab_bulider=vocab_bulider,
        file_name_list=imsdb_file_name_list, save_process=False,
        samples=opt.validsamples, add_bos=False, add_eos=False,
        use_mask=True, train=False)
    print(imsdb_dataset.sample())
    imsdb_dataloader = DataLoader(
        imsdb_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(dim=0, pad_id=vocab_bulider.pad, device=device),
        shuffle=False, num_workers=opt.workers, drop_last=True)

    # model definition
    saved_model = opt.energy
    energy_model = TransformerContinuousDecoder(
        ntoken=len(vocab_bulider), d_model=opt.embedsize, nhead=opt.nhead,
        num_encoder_layers=opt.encoderlayer, num_decoder_layers=opt.decoderlayer,
        dim_feedforward=opt.feedforward, postnorm=True, dropout=.0, gumbels=opt.gumbels,
        use_src_mask=False, use_tgt_mask=True, use_memory_mask=False,
        activation='relu', use_vocab_attn=False, use_pos_attn=False,
        relative_clip=0, highway=False, device=device, max_sent_length=32,
        share_input_output_embedding=False, share_encoder_decoder_embedding=True,
        share_vocab_embedding=True, fix_pos_encoding=opt.fix,
        bos_token=vocab_bulider.bos).to(device)
    energy_model.load_state_dict(torch.load("./save/" + saved_model + "/" + saved_model + "_last.ckpt"))
    energy_model.show_graph()

    model = TransformerConditionalMasked(
        ntoken=len(vocab_bulider), d_model=opt.embedsize, nhead=opt.nhead, max_sent_length=64,
        num_encoder_layers=opt.encoderlayer, num_decoder_layers=opt.decoderlayer,
        dim_feedforward=opt.feedforward, postnorm=True, dropout=opt.dropout, gumbels=opt.gumbels,
        activation='relu', relative_clip=0, highway=False, device=device,
        share_input_output_embedding=False, share_encoder_decoder_embedding=True,
        mask_id=vocab_bulider.mask, cls_id=vocab_bulider.bos).to(device)
    model.show_graph()
    if opt.half:
        model = model.half()
    if opt.ft:
        model = restore_best_state(model, opt.ckpt, save_dir='./save', device=model.device)

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
    criterionM = LabelSmoothedCrossEntropyLoss(eps=0.1, ignore_index=vocab_bulider.pad)  # for Transformer
    criterion_energy = InferenceEnergyLoss(
        energy_model, operator_input='SX', operator_target='SX',
        pad=vocab_bulider.pad,
        bos=vocab_bulider.bos,
        eos=vocab_bulider.eos)
    criterionL = nn.CrossEntropyLoss()  # for length-predictor

    # run model
    global_train_step, global_valid_step = 0, 0
    run_model(
        model, opensub_dataloader, imsdb_dataloader,
        opt.niter, criterion_energy, criterionL, optimizer, scheduler)
    writer.close()
