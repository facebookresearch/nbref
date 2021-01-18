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

import spacy

import random
import math
import os
import time
import pdb
import numpy as np

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

from random import randint
from config import *
import argparse
import pdb
import json
import random
import data_utils

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

def train(model, iterator, optimizer, trg_pad_idx, device, smoothing, criterion, clip):

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    epoch_loss = 0

    for i, batch in enumerate(iterator):

        # print(i)
        src = batch.src
        trg = batch.trg


        src = src.permute(1,0)
        trg = trg.permute(1,0)
        # pdb.set_trace()
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        # trg = trg.permute(1,0)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, trg_pad_idx))

        # pdb.set_trace()
        loss, n_correct, n_word = cal_performance(
            output, gold, trg_pad_idx, smoothing=smoothing)
        # loss = criterion(output, trg)
        loss.backward()

        n_word_total += n_word
        n_word_correct += n_correct
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        # optimizer.step_and_update_lr()
        epoch_loss += loss.item()

    loss_per_word = epoch_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    # return epoch_loss / len(iterator)

def evaluate(model, iterator, device, trg_pad_idx, criterion):

    model.eval()

    total_loss, n_word_total, n_word_correct = 0, 0, 0
    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            src = src.permute(1,0)
            trg = trg.permute(1,0)

            output, _ = model(src, trg[:,:-1])
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch.trg, trg_pad_idx))
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            loss, n_correct, n_word = cal_performance(
                output, gold, trg_pad_idx, smoothing=False)
            # loss = criterion(output, trg)

            n_word_total += n_word
            n_word_correct += n_correct
            epoch_loss += loss.item()

    loss_per_word = epoch_loss/n_word_total
    accuracy = n_word_correct/n_word_total

    return loss_per_word, accuracy
    # return epoch_loss / len(iterator)

# def evaluate(model, iterator, criterion,dump_file=False):
#
#     model.eval()
#     epoch_loss = 0
#     trg_all = []
#     output_all = []
#     collect_hidden_all = []
#     with torch.no_grad():
#         for i, batch in enumerate(iterator):
#
#             src = batch.src
#             trg_raw = batch.trg
#             output_raw, collect_hidden,_ = model(src, trg_raw, 0) #turn off teacher forcing
#             # output_raw = model(src, trg_raw, 0) #turn off teacher forcing
#
#             #trg = [trg sent len, batch size]
#             #output = [trg sent len, batch size, output dim]
#
#             output = output_raw[1:].view(-1, output_raw.shape[-1])
#             trg = trg_raw[1:].view(-1)
#
#             #trg = [(trg sent len - 1) * batch size]
#             #output = [(trg sent len - 1) * batch size, output dim]
#
#             loss = criterion(output, trg)
#
#             epoch_loss += loss.item()
#             if(dump_file is True):
#                 collect_hidden_all.append(collect_hidden)
#                 trg_all.append(trg_raw)
#                 output_all.append(torch.max(output_raw,2)[1])
#     return trg_all, output_all, collect_hidden_all, epoch_loss / len(iterator)

def evaluate_att(model, iterator, criterion,dump_file=False):

    model.eval()

    epoch_loss = 0

    trg_all = []
    output_all = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch.src.transpose(1,0)
            trg_raw = batch.trg.transpose(1,0)
            output_raw = model(src, trg_raw[:,:-1])

            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]

            output = output_raw.contiguous().view(-1, output_raw.shape[-1])
            trg = trg_raw[:,1:].contiguous().view(-1)

            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            if(dump_file is True):
                trg_all.append(batch.trg)
                output_all.append(torch.max(output_raw,2)[1])

    return trg_all, output_all, epoch_loss / len(iterator)

def train_att(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src.transpose(1,0)
        trg = batch.trg.transpose(1,0)
        # src = src.transpose(0,1)
        # src = src.transpose(0,1)

        optimizer.optimizer.zero_grad()

        # print(i)
        # if i == 40 :
        #     pdb.set_trace()
        output = model(src, trg[:,:-1])

        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)

        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
