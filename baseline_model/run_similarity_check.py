# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field, RawField, BucketIterator,TabularDataset,Dataset,Example
from torchtext.data.metrics import bleu_score
from torch.optim.lr_scheduler import LambdaLR

import torch.nn.functional as F
import spacy

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import pairwise
from prg import prg

import random
import math
import os
import time,json
import pdb, pickle

from modules.transformer_tree_model import *
from data_utils.train_gnn import *
from data_utils.Optim import *
from data_utils.data_utils import *
from data_utils.ggnn_utils import *

from data_utils.gnn_dataset import *
from data_utils.train_sim import *

import data_utils.Constants as Constants
import config
import argparse

def main():
    title='similarity-check'
    argParser = config.get_arg_parser(title)
    args = argParser.parse_args()
    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset = GNNDataset(args.dataset_dir, asm=args.asm, max_len=args.max_tolerate_len)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    SEED=1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    max_len_src = args.max_tolerate_len

    gnn = Graph_NN( annotation_size = len(dataset.vocab_asm) ,
                        out_feats = args.hid_dim,
                        n_steps = args.n_gnn_layers,
                        device = device,
                        tok_embedding=2,
                        residual=False
                        )

    enc = Encoder(
                  len(dataset.vocab_asm) ,
                  args.hid_dim,
                  args.n_layers,
                  args.n_heads,
                  args.pf_dim,
                  args.dropout,
                  device,
                  embedding_flag = args.embedding_flag,
                  max_length = max_len_src,
                  mem_dim = args.mem_dim) 

    SRC_PAD_IDX = 0
    
    model = CODE_SIM_ASM_Model(gnn, enc, args.hid_dim, SRC_PAD_IDX, device).to(device)

    model.apply(initialize_weights)

    optimizer = NoamOpt(args.hid_dim, args.lr_ratio, args.warmup, \
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    criterion = CircleLoss(gamma=args.gamma, m=args.margin)
    train_gen_fun = dataset.get_pk_sample_generator_function(
        split[0], args.p, args.k)
    valid_gen_fun = dataset.get_pk_sample_generator_function(
        split[1], args.p, args.k)
    train_num_iters = args.train_epoch_size
    valid_num_iters = args.valid_epoch_size

    criterion.to(device)

    args.summary = TrainingSummaryWriter(args.log_dir)

    best_val = None
    best_epoch = 0

    print("start training")
    for epoch in range(1, args.epoch_num + 1):
        iterations(args, epoch, model, criterion, optimizer,
                   train_gen_fun(), train_num_iters, True , device)

        best_val, best_epoch = validate(args, model, dataset, split[1], criterion,
                                        epoch, best_val, best_epoch, device)

        print(f'Epoch {epoch}')

        if epoch == best_epoch and (args.checkpoint_path is not None):
            output_path = os.path.join(args.checkpoint_path, f'model_sim_check.pt')
            torch.save(model.state_dict(), output_path)

    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'model_sim_check.pt')))
    test(args, model, dataset, split[2], device)

if __name__ == "__main__":
	main()
