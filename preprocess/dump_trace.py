# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import pdb
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, RawField, BucketIterator,TabularDataset,Dataset,Example
from torchtext.data.metrics import bleu_score

import torch.nn.functional as F
import random
import math
import os
import time,json
import pdb, sys

from baseline_model.data_utils.train_tree_encoder import processing_data
import baseline_model.data_utils.Tree as Tree 

import baseline_model.data_utils.Constants as Constants
import baseline_model.config as config
import argparse, pickle
import glob, multiprocessing
import numpy as np

from torch.utils.data import DataLoader
from collections import defaultdict 

import resource
from torch.nn.parallel import DistributedDataParallel

def main():
    title='dump-trace'
    argParser = config.get_arg_parser(title)
    args = argParser.parse_args()
    max_len_trg = 0
    max_len_src = 0
    sys.modules['Tree'] = Tree

    with open(args.golden_c_path,'rb') as file_c:
        trg = pickle.load(file_c)


    SEED=1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    exp_list = []
    SRC = Field(
                init_token = '<sos>',
                eos_token = '<eos>')
    TRG = RawField()
    ID  = RawField()
    DICT_INFO = RawField()

    cache_dir = args.cache_path
    src_g = np.load(args.input_g_path, allow_pickle=True)
    src_f = np.load(args.input_f_path, allow_pickle=True)

    for i in range(0,args.gen_num):
        src_elem = src_f[i]
        dict_info = 0
        trg_elem = trg[i]['tree']
        exp = Example.fromlist([src_elem,trg_elem,i, dict_info],fields =[('src',SRC),('trg',TRG), ('id', ID), ('dict_info',DICT_INFO)] )
        exp_list.append(exp)

        len_elem_src = len(src_elem)
        len_elem_trg = trg[i]['treelen']

        if len_elem_src + 2 >= max_len_src:
            max_len_src = len_elem_src  + 2
        if len_elem_trg >= max_len_trg:
            max_len_trg = len_elem_trg + 2
    data_sets = Dataset(exp_list,fields = [('src',SRC),('trg',TRG), ('id', ID), ('dict_info', DICT_INFO)])
    trn, vld = data_sets.split([0.8,0.2,0.0])
    SRC.build_vocab(trn, min_freq = 2)

    print("Number of training examples: %d" % (len(trn.examples)))
    print("Number of validation examples: %d" % (len(vld.examples)))
    print("Unique tokens in source assembly vocabulary: %d "%(len(SRC.vocab)))
    print("Max input length : %d" % (max_len_src))
    print("Max output length : %d" % (max_len_trg))
    del trg, src_f, src_g

    BATCH_SIZE = 1

    train_iterator, valid_iterator = BucketIterator.splits(
        (trn, vld),
        batch_size = BATCH_SIZE,
        sort_key= lambda x :len(x.trg),
        sort_within_batch=False,
        sort=False)

    processing_data(cache_dir, [train_iterator, valid_iterator])

if __name__ == "__main__":
    main()
