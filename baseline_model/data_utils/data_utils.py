# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
# import torch.optim as optim

# from torchtext.datasets import TranslationDataset, Multi30k
# from torchtext.data import Field, BucketIterator, metrics
# from torchtext.data.metrics import bleu_score
# import spacy

import random, warnings
import math
import os
import time
import pdb
import numpy as np

from torch.autograd import Variable
from random import randint
import argparse
import pdb
import json
import random
from tqdm import tqdm
from sklearn.metrics import *
import dgl, functools

def split_dataset(src_list, p = 0):
    split_src_list = []
    if p == 0 :
        src_list = [src_list]
    for elem in src_list:
        src_ = elem.split([0.5,0.5])
        split_src_list += [p for p in src_]
    return split_src_list

def gen_args(args, src_f, src_g):
    for i in range(args.gen_num):
        yield os.path.join(args.cache_asm_path, str(i)), src_f[i], src_g[i], args.hid_dim, args.device, i

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    trgs = []
    pred_trgs = []

    for datum in data:

        src = vars(datum)['src']
        trg = vars(datum)['trg']

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        #cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        # tokens = [token.lower() for token in sentence]
        tokens = [token for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        # with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def init_tqdm(it, desc, log=False):
    if log:
        return tqdm(it, desc=desc)
    return it

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def np_to_tensor(inp, output_type, cuda_flag, volatile_flag=False):
	if output_type == 'float':
		inp_tensor = Variable(torch.FloatTensor(inp), volatile=volatile_flag)
	elif output_type == 'int':
		inp_tensor = Variable(torch.LongTensor(inp), volatile=volatile_flag)
	else:
		print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor.cuda()
	return inp_tensor

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def get_logger(log_path, **kwargs):
    with open(log_path, 'w+') as f_log:
        f_log.write("create file\n")
    return functools.partial(logging, log_path=log_path, **kwargs)

def word_ids_to_sentence(id_tensor, vocab, join=None):
    """Converts a sequence of word ids to a sentence"""
    if isinstance(id_tensor, torch.LongTensor):
        ids = id_tensor.transpose(0, 1).contiguous().view(-1)
    elif isinstance(id_tensor, np.ndarray):
        ids = id_tensor.transpose().reshape(-1)
    batch = [vocab.itos[ind] for ind in ids] # denumericalize
    if join is None:
        return batch
    else:
        return join.join(batch)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def is_torch_geometric_model(args):
#     if args.model in [M_GGNN, M_GGNN_SIMPLE, M_DEVIGN, M_GAT, M_CGCN, M_3GNN]:
#         return True
#     return False


# def is_text_model(args):
#     if args.model in [M_CNN, M_CNN_RF, M_LSTM, M_BILSTM, M_BILSTM_ATTN]:
#         return True
#     return False


def report(labels, predictions):

    # print(labels, predictions)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    # print(predictions)
    if len(predictions.shape) > 1:  # one-hot
        predictions = np.argmax(predictions, axis=1)
        # labels = np.argmax(labels, axis=1)
    else:
        predictions = predictions.round()
    # print(predictions)
    # print(labels)

    # print(sum(labels), sum(predictions))
    pres, recs, thresholds = precision_recall_curve(labels, predictions)
    f1, pre, rec = (0, 0, 0) if sum(predictions) == 0 else \
        (f1_score(labels, predictions),
         precision_score(labels, predictions),
         recall_score(labels, predictions))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcc = matthews_corrcoef(labels, predictions)

    return {"acc": accuracy_score(labels, predictions),
            "f1": f1, "pre": pre, "rec": rec,
            "mcc": mcc,
            "roc_auc": roc_auc_score(labels, predictions),  # Area Under the Receiver Operating Characteristic Curve
            "pr_auc": auc(recs, pres),  # area under precision-recall curve
            "avg_pre": average_precision_score(labels, predictions),
            }


def report_to_str(metrics, keys=True):
    if keys:
        return "\t".join([k.upper() + (': {:.2f}'.format(v*100) if k != "loss" else ': {:.4f}'.format(v)) for k, v in metrics.items()])

    return ",".join(['{:.2f}'.format(v*100) if k != "loss" else '{:.4f}'.format(v) for k, v in metrics.items()])


def report_to_str_report(metrics):
    return ['{:.2f}'.format(v*100) if k != "loss" else '{:.4f}'.format(v) for k, v in metrics.items()]


def summary_report(metrics):
    result = {}
    for k, v in metrics.items():
        result[k] = sum(v)/len(v)

    return result