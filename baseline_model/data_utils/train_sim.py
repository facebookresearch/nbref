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
import dgl
from prg import prg

def preprocessing_batch_tmp(src_len, graph, device):
    batch_g = []
    max_len = max(src_len)
    for i in range(len(src_len)):
        g = dgl.graph(graph[i].edges())
        g.edata['type'] = graph[i].edata['type']
        g.add_nodes(max_len - src_len[i])
        batch_g.append(g)

    batch_graph = dgl.batch(batch_g).to(device)
    return batch_graph

def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

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
            input, input_len = next(data_iter)
            batch_graph_tmp = None #preprocessing_batch_tmp(input_len, input, device)
            input = dgl.batch(input)
            input = input.to(device)
            code_vecs = model(input, batch_graph=batch_graph_tmp)  # (p * k, dim)
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
            input, input_len = input
            input = dgl.batch(input)
            input = input.to(device)
            with torch.no_grad():
                v = model(input, batch_graph=None)
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
