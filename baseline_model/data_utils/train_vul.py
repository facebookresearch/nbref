# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import pairwise

import random
import math
import os
import time
import pdb
import numpy as np

from random import randint
from config import *
import argparse
import pdb
import json
import random
import data_utils
import tqdm

def train_eval(data, labels, batch_graph, model, device, optimizer, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        if isinstance(data, tuple):
            data = tuple(c.to(device) for c in data)
        elif not isinstance(data, list):  # if we use dataparallel this is done later
            data = data.to(device)

        
        labels = smooth_one_hot(labels, 2, 0.05)
        labels = labels.to(device)

        output = model(data, batch_graph=batch_graph)
        loss = criterion(output, labels.float())# if ONE_HOT_LABELS else labels.float())

        if train:
            optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.detach().cpu(), output.detach().cpu()

def smooth_one_hot(true_labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        # loss = F.cross_entropy(pred, gold, ignore_index = trg_pad_idx)
        # pdb.set_trace()
        # loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def print_performances(header, loss, accu, start_time):
    print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
          'elapse: {elapse:3.3f} min'.format(
              header=f"({header})", ppl=math.exp(min(loss, 100)),
              accu=100*accu, elapse=(time.time()-start_time)/60))

def pk_sim(h, p, k):
    h: torch.Tensor  # (p * k, dim)

    X = h.repeat_interleave(p * k, dim=0)
    Y = h.repeat(p * k, 1)
    sim = torch.cosine_similarity(X, Y)
    sim = sim.view(p * k, p * k)

    class_ids = torch.arange(p).repeat_interleave(k)
    inds = torch.triu_indices(p * k, p * k, offset=1)
    sim = sim[inds[0], inds[1]]
    positive = (class_ids[inds[0]] == class_ids[inds[1]]).to(device=h.device)
    s_p = sim[positive]
    s_n = sim[~positive]

    return s_p, s_n


def iterations(args, epoch, model, criterion, optimizer, data_iter, num_iters, training, device):
    model.train(training)

    total_loss = 0

    with tqdm.trange(num_iters) as progress:
        for _ in progress:
            input, edges = next(data_iter)
            # input = [x.to(device=args.device) for x in input]
            input = input.to(device)
            code_vecs = model(input, edges)  # (p * k, dim)
            s_p, s_n = pk_sim(code_vecs, args.p, args.k)
            loss = criterion(s_p, s_n)

            total_loss += loss.item()

            if training:
                model.zero_grad()
                loss.backward()
                optimizer.step()

                args.summary.step()
                args.summary.add_scalar('train/loss', loss.item())

            progress.set_description(f'Epoch {epoch} loss: {loss.item():.8f}')

    avg_loss = total_loss / num_iters

    if training:
        print(f'- training avg loss: {avg_loss:.8f}')
    else:
        print(f'- validation avg loss: {avg_loss:.8f}')

    return avg_loss

# def train(args, model, dataset, train_split, valid_split, test_split):
#     criterion = CircleLoss(gamma=args.gamma, m=args.margin)
#     train_gen_fun = dataset.get_pk_sample_generator_function(
#         train_split, args.p, args.k)
#     valid_gen_fun = dataset.get_pk_sample_generator_function(
#         valid_split, args.p, args.k)
#     train_num_iters = args.train_epoch_size
#     valid_num_iters = args.valid_epoch_size

#     criterion.to(args.device)

#     optimizer = optim.AdamW(model.parameters(), lr=args.lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_num+1)

#     args.summary = TrainingSummaryWriter(args.log_dir)

#     best_val = None
#     best_epoch = 0

#     for epoch in range(1, args.epoch_num + 1):
#         iterations(args, epoch, model, criterion, optimizer,
#                    train_gen_fun(), train_num_iters, True)

#         best_val, best_epoch = validate(args, model, dataset, valid_split, criterion,
#                                         epoch, best_val, best_epoch)

#         print(f'Epoch {epoch}: lr: {scheduler.get_last_lr()[0]*1000}')
#         scheduler.step()
#         # output_path = os.path.join(args.save, f'model.ep{epoch}.pt')
#         # torch.save(model.state_dict(), output_path)

#         if epoch == best_epoch and (args.save is not None):
#             output_path = os.path.join(args.save, f'model.pt')
#             torch.save(model.state_dict(), output_path)
#     test(args, model, dataset, test_split)


def validate(args, model, dataset, test_split, criterion, epoch, best_val, best_epoch, device):
    code_vecs, pids = run_test(args, model, dataset, test_split, device)

    code_vecs = code_vecs.numpy()
    pids = pids.numpy()
    sim = pairwise.cosine_similarity(code_vecs)

    map_r = map_at_r(sim, pids)
    if best_val is None or map_r > best_val:
        best_val = map_r
        best_epoch = epoch
    print(f'* validation MAP@R: {map_r}, best epoch: {best_epoch}, best MAP@R: {best_val}')

    args.summary.add_scalar('train/map_r', map_r)

    return best_val, best_epoch


def run_test(args, model, dataset, test_split, device):
    model.eval()

    test_gen_fun, num_iters = dataset.get_data_generator_function(
        test_split, args.batch_size, shuffle=False)

    code_vecs = []
    pids = []
    with tqdm.tqdm(test_gen_fun(), total=num_iters) as progress:
        for input, pids_batch in progress:
            input, edges = input
            input = input.to(device)
            # input = [x.to(device=args.device) for x in input]
            with torch.no_grad():
                v = model(input, edges)
            code_vecs.append(v.detach().cpu())
            pids.append(pids_batch)
    code_vecs = torch.cat(code_vecs, dim=0)
    pids = torch.cat(pids)

    return code_vecs, pids


def get_pairwise_scores_and_labels(sim, pids):
    inds = np.tril_indices(len(pids))
    scores = sim[inds]
    labels = pids[inds[0]] == pids[inds[1]]
    return scores, labels


def area_under_prg(labels, scores):
    prg_curve = prg.create_prg_curve(labels, scores)
    auprg = prg.calc_auprg(prg_curve)
    return auprg


def map_at_r(sim, pids):
    r = np.bincount(pids) - 1
    max_r = r.max()

    mask = np.arange(max_r)[None, :] < r[pids][:, None]

    sim = np.copy(sim)
    np.fill_diagonal(sim, -np.inf)
    result = np.argsort(sim, axis=1)[:, :-max_r-1:-1]
    tp = (pids[result] == pids[:, None])
    tp[~mask] = False

    p = np.cumsum(tp, axis=1) / np.arange(1, max_r+1)[None, :]
    ap = (p * tp).sum(axis=1) / r[pids]

    return ap.mean()


def test(args, model, dataset, test_split, device):
    code_vecs, pids = run_test(args, model, dataset, test_split, device)
    code_vecs = code_vecs.numpy()
    pids = pids.numpy()
    sim = pairwise.cosine_similarity(code_vecs)
    compute_metrics(sim, pids)


def compute_metrics(sim, pids):
    scores, labels = get_pairwise_scores_and_labels(sim, pids)

    ap = average_precision_score(labels, scores)
    auprg = area_under_prg(labels, scores)
    map_r = map_at_r(sim, pids)

    print(f'MAP@R: {map_r}, AP: {ap}, AUPRG: {auprg}')

# def train(model, iterator, optimizer, trg_pad_idx, device, smoothing, criterion, clip):

#     model.train()
#     total_loss, n_word_total, n_word_correct = 0, 0, 0
#     epoch_loss = 0

#     for i, batch in enumerate(iterator):

#         # print(i)
#         src = batch.src
#         trg = batch.trg


#         src = src.permute(1,0)
#         trg = trg.permute(1,0)
#         # pdb.set_trace()
#         optimizer.zero_grad()
#         output, _ = model(src, trg[:,:-1])

#         #output = [batch size, trg len - 1, output dim]
#         #trg = [batch size, trg len]

#         output_dim = output.shape[-1]

#         output = output.contiguous().view(-1, output_dim)
#         trg = trg[:,1:].contiguous().view(-1)
#         #trg = [(trg sent len - 1) * batch size]
#         #output = [(trg sent len - 1) * batch size, output dim]
#         # trg = trg.permute(1,0)
#         trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, trg_pad_idx))

#         # pdb.set_trace()
#         loss, n_correct, n_word = cal_performance(
#             output, gold, trg_pad_idx, smoothing=smoothing)
#         # loss = criterion(output, trg)
#         loss.backward()

#         n_word_total += n_word
#         n_word_correct += n_correct
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         # optimizer.step_and_update_lr()
#         epoch_loss += loss.item()

#     loss_per_word = epoch_loss/n_word_total
#     accuracy = n_word_correct/n_word_total
#     return loss_per_word, accuracy
#     # return epoch_loss / len(iterator)

# def evaluate(model, iterator, device, trg_pad_idx, criterion):

#     model.eval()

#     total_loss, n_word_total, n_word_correct = 0, 0, 0
#     epoch_loss = 0

#     with torch.no_grad():

#         for i, batch in enumerate(iterator):

#             src = batch.src
#             trg = batch.trg

#             src = src.permute(1,0)
#             trg = trg.permute(1,0)

#             output, _ = model(src, trg[:,:-1])
#             trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, trg_pad_idx))
#             output_dim = output.shape[-1]

#             output = output.contiguous().view(-1, output_dim)
#             trg = trg[:,1:].contiguous().view(-1)

#             loss, n_correct, n_word = cal_performance(
#                 output, gold, trg_pad_idx, smoothing=False)
#             # loss = criterion(output, trg)

#             n_word_total += n_word
#             n_word_correct += n_correct
#             epoch_loss += loss.item()

#     loss_per_word = epoch_loss/n_word_total
#     accuracy = n_word_correct/n_word_total

#     return loss_per_word, accuracy

# def evaluate_att(model, iterator, criterion,dump_file=False):

#     model.eval()

#     epoch_loss = 0

#     trg_all = []
# def train(model, iterator, optimizer, trg_pad_idx, device, smoothing, criterion, clip):

#     model.train()
#     total_loss, n_word_total, n_word_correct = 0, 0, 0
#     epoch_loss = 0

#     for i, batch in enumerate(iterator):

#         # print(i)
#         src = batch.src
#         trg = batch.trg


#         src = src.permute(1,0)
#         trg = trg.permute(1,0)
#         # pdb.set_trace()
#         optimizer.zero_grad()
#         output, _ = model(src, trg[:,:-1])

#         #output = [batch size, trg len - 1, output dim]
#         #trg = [batch size, trg len]

#         output_dim = output.shape[-1]

#         output = output.contiguous().view(-1, output_dim)
#         trg = trg[:,1:].contiguous().view(-1)
#         #trg = [(trg sent len - 1) * batch size]
#         #output = [(trg sent len - 1) * batch size, output dim]
#         # trg = trg.permute(1,0)
#         trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, trg_pad_idx))

#         # pdb.set_trace()
#         loss, n_correct, n_word = cal_performance(
#             output, gold, trg_pad_idx, smoothing=smoothing)
#         # loss = criterion(output, trg)
#         loss.backward()

#         n_word_total += n_word
#         n_word_correct += n_correct
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         # optimizer.step_and_update_lr()
#         epoch_loss += loss.item()

#     loss_per_word = epoch_loss/n_word_total
#     accuracy = n_word_correct/n_word_total
#     return loss_per_word, accuracy
#     # return epoch_loss / len(iterator)

# def evaluate(model, iterator, device, trg_pad_idx, criterion):

#     model.eval()

#     total_loss, n_word_total, n_word_correct = 0, 0, 0
#     epoch_loss = 0

#     with torch.no_grad():

#         for i, batch in enumerate(iterator):

#             src = batch.src
#             trg = batch.trg

#             src = src.permute(1,0)
#             trg = trg.permute(1,0)

#             output, _ = model(src, trg[:,:-1])
#             trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, trg_pad_idx))
#             output_dim = output.shape[-1]

#             output = output.contiguous().view(-1, output_dim)
#             trg = trg[:,1:].contiguous().view(-1)

#             loss, n_correct, n_word = cal_performance(
#                 output, gold, trg_pad_idx, smoothing=False)
#             # loss = criterion(output, trg)

#             n_word_total += n_word
#             n_word_correct += n_correct
#             epoch_loss += loss.item()

#     loss_per_word = epoch_loss/n_word_total
#     accuracy = n_word_correct/n_word_total

#     return loss_per_word, accuracy

# def evaluate_att(model, iterator, criterion,dump_file=False):

#     model.eval()

#     epoch_loss = 0

#     trg_all = []
#     output_all = []
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):

#             src = batch.src.transpose(1,0)
#             trg_raw = batch.trg.transpose(1,0)
#             output_raw = model(src, trg_raw[:,:-1])

#             #output = [batch size, trg sent len - 1, output dim]
#             #trg = [batch size, trg sent len]

#             output = output_raw.contiguous().view(-1, output_raw.shape[-1])
#             trg = trg_raw[:,1:].contiguous().view(-1)

#             #output = [batch size * trg sent len - 1, output dim]
#             #trg = [batch size * trg sent len - 1]

#             loss = criterion(output, trg)
#             epoch_loss += loss.item()

#             if(dump_file is True):
#                 trg_all.append(batch.trg)
#                 output_all.append(torch.max(output_raw,2)[1])

#     return trg_all, output_all, epoch_loss / len(iterator)

# def train_att(model, iterator, optimizer, criterion, clip):

#     model.train()

#     epoch_loss = 0

#     for i, batch in enumerate(iterator):

#         src = batch.src.transpose(1,0)
#         trg = batch.trg.transpose(1,0)
#         # src = src.transpose(0,1)
#         # src = src.transpose(0,1)

#         optimizer.optimizer.zero_grad()

#         # print(i)
#         # if i == 40 :
#         #     pdb.set_trace()
#         output = model(src, trg[:,:-1])

#         #output = [batch size, trg sent len - 1, output dim]
#         #trg = [batch size, trg sent len]

#         output = output.contiguous().view(-1, output.shape[-1])
#         trg = trg[:,1:].contiguous().view(-1)

#         #output = [batch size * trg sent len - 1, output dim]
#         #trg = [batch size * trg sent len - 1]

#         loss = criterion(output, trg)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         epoch_loss += loss.item()

#     return epoch_loss / len(iterator)

#             src = batch.src.transpose(1,0)
#             trg_raw = batch.trg.transpose(1,0)
#             output_raw = model(src, trg_raw[:,:-1])

#             #output = [batch size, trg sent len - 1, output dim]
#             #trg = [batch size, trg sent len]

#             output = output_raw.contiguous().view(-1, output_raw.shape[-1])
#             trg = trg_raw[:,1:].contiguous().view(-1)

#             #output = [batch size * trg sent len - 1, output dim]
#             #trg = [batch size * trg sent len - 1]

#             loss = criterion(output, trg)
#             epoch_loss += loss.item()

#             if(dump_file is True):
#                 trg_all.append(batch.trg)
#                 output_all.append(torch.max(output_raw,2)[1])

#     return trg_all, output_all, epoch_loss / len(iterator)

# def train_att(model, iterator, optimizer, criterion, clip):

#     model.train()

#     epoch_loss = 0

#     for i, batch in enumerate(iterator):

#         src = batch.src.transpose(1,0)
#         trg = batch.trg.transpose(1,0)
#         # src = src.transpose(0,1)
#         # src = src.transpose(0,1)

#         optimizer.optimizer.zero_grad()

#         # print(i)
#         # if i == 40 :
#         #     pdb.set_trace()
#         output = model(src, trg[:,:-1])

#         #output = [batch size, trg sent len - 1, output dim]
#         #trg = [batch size, trg sent len]

#         output = output.contiguous().view(-1, output.shape[-1])
#         trg = trg[:,1:].contiguous().view(-1)

#         #output = [batch size * trg sent len - 1, output dim]
#         #trg = [batch size * trg sent len - 1]

#         loss = criterion(output, trg)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         epoch_loss += loss.item()

#     return epoch_loss / len(iterator)
