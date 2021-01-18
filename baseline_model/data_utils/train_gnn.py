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

import random
import math
import os
import time
import pdb
import numpy as np

import dgl
from random import randint
from config import *
import argparse
import pdb
import json
import random
import data_utils
# import gnn_utils

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

# def tokenize_de(text):
#     """
#     Tokenizes German text from a string into a list of strings (tokens) and reverses it
#     """
#     return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

# def tokenize_en(text):
#     """
#     Tokenizes English text from a string into a list of strings (tokens)
#     """
#     return [tok.text for tok in spacy_en.tokenizer(text)]

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def print_performances(header, loss, accu, start_time):
    print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
          'elapse: {elapse:3.3f} min'.format(
              header=f"({header})", ppl=math.exp(min(loss, 100)),
              accu=100*accu, elapse=(time.time()-start_time)/60))


def preprocessing_edge(edges, nodes):
    reverse_edge = {0: 1, 1: 0}
    # edges = (edges).tolist()
    g = dgl.DGLGraph()
    edge_types = []
    g.add_nodes(nodes)
    for s, e, t in edges:
        if s == 0 and e ==0 and t ==0:
            break
        g.add_edge(s, t)
        edge_types.append(e)
        if e in reverse_edge:
            g.add_edge(t, s)
            edge_types.append(reverse_edge[e])
    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
    return g

def graph_src_scatter(src, g, hid_dim_in):
    graphs = []
    for ii in range(0,src.shape[0]):
        idmap = range(0,src.shape[1])
        g[ii].add_nodes(len(idmap) - len(g[ii].nodes))
        g[ii].ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
        annotation = torch.zeros([src.shape[1], hid_dim_in], dtype=torch.long).cuda()
        annotation.scatter_(1,src[ii].view(-1,1), value=torch.tensor(1))
        g[ii].ndata['annotation'] = annotation 
        graphs.append(g[ii])

    src_ = dgl.batch(graphs)
    return src_

def preprocessing_graph(src,edge, hid_dim_in):
    reverse_edge = {0: 1, 1: 0}
    g = dgl.DGLGraph()
    src_len = len(src)
    g.add_nodes(src_len)
    idmap = range(0,src_len)
    g.ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
    edge_types = []
    annotation = torch.zeros([ src_len, hid_dim_in], dtype=torch.long)

    for idx in range(0,src_len):
        annotation[idx][src[idx]-1] = 1

    for s, e, t in edge:
        if s == 0 and e ==0 and t ==0:
            break

        g.add_edge(s, t)
        edge_types.append(e)
        if e in reverse_edge:
            g.add_edge(t, s)
            edge_types.append(reverse_edge[e])
    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
    g.ndata['annotation'] = annotation 
    return g

def graph_src(src,edges,hid_dim_in):
    graphs = []

    reverse_edge = {0: 1, 1: 0}
    # edges = (edges).tolist()
    for ii in range(0,src.shape[0]):
        g = dgl.DGLGraph()

        g.add_nodes(src.shape[1])
        idmap = range(0,src.shape[1])
        g.ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
        # torch.tensor(src[ii,:], dtype=torch.long)
        edge_types = []
        annotation = torch.zeros([src.shape[1], hid_dim_in], dtype=torch.long)

        for idx in range(0,src.shape[1]):
            annotation[idx][src[ii,:][idx]] = 1

        for s, e, t in edges[ii]:
            if s == 0 and e ==0 and t ==0:
                break

            g.add_edge(s, t)
            edge_types.append(e)
            if e in reverse_edge:
                g.add_edge(t, s)
                edge_types.append(reverse_edge[e])
        g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
        g.ndata['annotation'] = annotation #src[ii,:].clone().detach().long()
        graphs.append(g)

    src_ = dgl.batch(graphs)
    return src_

# def preprocessing_batch(iterator, hid_dim_in):
#     src_list = []
#     trg_list = []
#     print("preprocessing....")
#     for batch  in (iterator):
#         src = batch.src
#         trg = batch.trg
#         trg = trg.permute(1,0)
#         src = src.permute(1,0)
#         src_ = graph_src(src, batch.edge, hid_dim_in)
#         src_list.append(src_)
#         trg_list.append(trg)
#     return src_list,trg_list
    

def train(model, iterator, optimizer, trg_pad_idx, device,\
             smoothing, criterion, clip, hid_dim_in):#, scheduler):

    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    epoch_loss = 0

    i = 0
    for batch  in (iterator):
        src = batch.src
        trg = batch.trg
        trg = trg.permute(1,0)
        src = src.permute(1,0)

        # src_ = graph_src(src, batch.edge, hid_dim_in)
        # src_ = graph_src_scatter(src, batch.edge_proc, hid_dim_in)
        src_ = dgl.batch(batch.graph_proc)
        # optimizer.zero_grad()
        optimizer.optimizer.zero_grad()
        output, _ = model(src, src_, trg[:,:-1], batch.edge)

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
        n_word_total += n_word
        n_word_correct += n_correct
        epoch_loss += loss.item()
        # scheduler.step()

    loss_per_word = epoch_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy
    # return epoch_loss / len(iterator)

def evaluate(model, iterator,  device, trg_pad_idx, criterion, hid_dim_in):

    model.eval()

    total_loss, n_word_total, n_word_correct = 0, 0, 0
    epoch_loss = 0

    with torch.no_grad():

        for batch in (iterator ):

            src = batch.src
            trg = batch.trg
            trg = trg.permute(1,0)
            src = src.permute(1,0)
            # src = batch.src
            # trg = batch.trg

            # src = graph_src(src, batch.edge)
            # src_ = graph_src(src, batch.edge, hid_dim_in)
            src_ = dgl.batch(batch.graph_proc)
            output, _ = model(src, src_, trg[:,:-1], batch.edge)
            # output, _ = model(src, trg[:,:-1])
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

            # pdb.set_trace()
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
