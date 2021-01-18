# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset
from torchtext.data import Field, RawField, BucketIterator,TabularDataset,Dataset,Example

import torch.nn.functional as F
import spacy

import random
import math
import os
import time,json
import pdb, sys

from modules.transformer_tree_model import *
from data_utils.train_tree_encoder import *
from data_utils.Optim import *
from data_utils.data_utils import *
import data_utils.Constants as Constants
import config
import argparse, pickle
import data_utils.Tree as Tree 
import glob, multiprocessing

import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict 

import resource
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '1111'     

def gen_args_multi_gpu(args, src_f, src_g):
    for i in range(args.gen_num):
        yield os.path.join(args.cache_asm_path, str(i)), src_f[i], src_g[i], args.hid_dim, i

def text_data_collator(dataset: Dataset):
    def collate(data):
        batch = defaultdict(list)
        for datum in data:
            for name, field in dataset.fields.items():
                batch[name].append(field.preprocess(getattr(datum, name)))

        batch = {name: field.process(batch[name]) for name, field in dataset.fields.items()}
        return batch

    return collate

def load_graph(argument):
    path_pid, src_elem, edge_elem, hid_dim, pid = argument
    if os.path.exists(path_pid):
        with open(path_pid, 'rb') as f:
            graph_proc = pickle.load(f)
    else:
        graph_proc = preprocessing_graph(src_elem, edge_elem, hid_dim)
        with open(path_pid, 'wb') as f:
            pickle.dump(graph_proc, f)
    return graph_proc, pid

def load_graphs(args, src_f, src_g):
    print("processing asm graphs ...")
    if not os.path.exists(args.cache_asm_path):
        os.makedirs(args.cache_asm_path)
    graphs_proc = [None] * args.gen_num
    for argument in gen_args_multi_gpu(args, src_f, src_g):
        graph_proc, pid = load_graph(argument)
        graphs_proc[pid] = graph_proc
    return graphs_proc

def main():
    title='trf-tree'
    sys.modules['Tree'] = Tree
    argParser = config.get_arg_parser(title)
    args = argParser.parse_args()
    args.summary = TrainingSummaryWriter(args.log_dir)
    logging = get_logger(log_path=os.path.join(args.log_dir, "log" + time.strftime('%Y%m%d-%H%M%S') + '.txt'), print_=True, log_=True)

    max_len_trg, max_len_src = 0, 0
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    with open(args.golden_c_path,'rb') as file_c:
        trg = pickle.load(file_c)

    src_g = np.load(args.input_g_path, allow_pickle=True)
    src_f = np.load(args.input_f_path, allow_pickle=True)

    graphs_asm = load_graphs(args, src_f, src_g) 

    SEED=1234
    torch.manual_seed(SEED)

    exp_list = []
    SRC        = Field(init_token = '<sos>', eos_token = '<eos>')
    TRG        = RawField()
    ID         = RawField()
    DICT_INFO  = RawField()
    GRAPHS_ASM = RawField()
    NODE_NUM   = RawField()

    for i in range(0,args.gen_num):
        src_elem = src_f[i]
        broken_file_flag = 0
        if args.dump_trace:
            dict_info={}
            for path in glob.glob(os.path.join(args.cache_path,str(i)+'/*')):
                if os.path.getsize(path) > 0:
                    with open(path, 'rb') as f:
                        dict_info[path] = pickle.load(f)
                else:
                    print("broken file!" + path)
                    broken_file_flag = 1
                    break

        if broken_file_flag == 1:
            continue

        if dict_info == {}:
            continue
        trg_elem = trg[i]['tree']
        len_elem_src = graphs_asm[i].number_of_nodes()
        exp = Example.fromlist([src_elem,trg_elem,i, dict_info, graphs_asm[i], len_elem_src], \
            fields =[('src', SRC), ('trg', TRG), ('id', ID), ('dict_info', DICT_INFO), ('graphs_asm', GRAPHS_ASM), ('src_len', NODE_NUM)] )
        exp_list.append(exp)
        len_elem_trg = trg[i]['treelen']

        if len_elem_src  >= max_len_src:
            max_len_src = len_elem_src  + 2 
        if len_elem_trg >= max_len_trg:
            max_len_trg = len_elem_trg + 2

    data_sets = Dataset(exp_list,fields = [('src',SRC),('trg',TRG), ('id', ID), ('dict_info', DICT_INFO), ('graphs_asm', GRAPHS_ASM), ('src_len', NODE_NUM)])
    trn, tst, vld = data_sets.split([0.8,0.15,0.05])
    SRC.build_vocab(trn, min_freq = 2)
    #
    print("Number of training examples: %d" % (len(trn.examples)))
    print("Number of validation examples: %d" % (len(vld.examples)))
    print("Number of testing examples: %d" % (len(tst.examples)))
    print("Unique tokens in source assembly vocabulary: %d "%(len(SRC.vocab)))
    print("Max input length : %d" % (max_len_src))
    print("Max output length : %d" % (max_len_trg))


    if args.dist_gpu:
        for p in range(int(math.log(args.n_dist_gpu, 2))):
            trn = split_dataset(trn, p)

        procs = []
        for proc_id in range(args.n_dist_gpu):
            p = mp.Process(target=run_spawn, args=(proc_id, args, trn[proc_id], vld, tst, SRC, max_len_src, max_len_trg, logging))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        run_spawn(-1, args, trn, vld, tst, SRC, max_len_src, max_len_trg, logging)


def run_spawn(gpu, args, trn, vld, tst, SRC, max_len_src, max_len_trg, logging):
    print(gpu)
    if args.dist_gpu:
        torch.cuda.set_device(gpu)
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='env://',
            world_size=args.n_dist_gpu,                              
            rank=gpu
        )      
    device = torch.device(gpu)
    best_valid_loss = float('inf')
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(gpu)

    collate = text_data_collator(trn)
    train_iterator = DataLoader(trn, batch_size=args.bsz, collate_fn=collate, num_workers=args.num_workers, shuffle=False)
    collate = text_data_collator(vld)
    valid_iterator = DataLoader(vld, batch_size=args.bsz, collate_fn=collate, num_workers=args.num_workers, shuffle=False)


    INPUT_DIM = len(SRC.vocab)

    gnn_asm = Graph_NN( annotation_size = len(SRC.vocab),
                        out_feats = args.hid_dim,
                        n_steps = args.n_gnn_layers,
                        device = device
                        )

    gnn_ast = Graph_NN( annotation_size = len(SRC.vocab),
                        out_feats = args.hid_dim,
                        n_steps = args.n_gnn_layers,
                        device = device)

    enc = Encoder(INPUT_DIM,
                  args.hid_dim,
                  args.n_layers,
                  args.n_heads,
                  args.pf_dim,
                  args.dropout,
                  device,
                  args.mem_dim,
                  embedding_flag=args.embedding_flag,
                  max_length = max_len_src)

    dec = Decoder_AST(
                  args.output_dim,
                  args.hid_dim,
                  args.n_layers,
                  args.n_heads,
                  args.pf_dim,
                  args.dropout,
                  device,
                  max_length = max_len_trg)

    SRC_PAD_IDX = 0 
    TRG_PAD_IDX = 0 

    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device,  
                gnn=gnn_ast, gnn_asm=gnn_asm).to(device)
    model = model.cuda(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu) 
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX, reduction='sum').cuda(gpu)
    optimizer = NoamOpt(args.hid_dim, args.lr_ratio, args.warmup, \
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if args.training and not args.eval:
        for epoch in range(args.n_epoch):
            start_time = time.time()
            train_loss, train_acc = train_eval_tree(args, model, train_iterator, optimizer\
                                , device, criterion, max_len_trg, train_flag=True)
            torch.distributed.barrier()

            if gpu == 0:
                valid_loss, valid_acc = train_eval_tree(args, model, valid_iterator, None\
                                    , device, criterion, max_len_trg, train_flag=False)
            torch.distributed.barrier()
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss and (args.checkpoint_path is not None) and gpu == 0:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, 'model_multi_gpu.pt'))

            if gpu==0:
                logging('Epoch: %d | Time: %dm %ds | learning rate %.3f' %(epoch,epoch_mins,epoch_secs, optimizer._rate*10000))
                print_performances('Training', train_loss, train_acc, start_time, logging=logging)
                print_performances('Validation', valid_loss, valid_acc, start_time, logging=logging)

            torch.distributed.barrier()

    if gpu == 0:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'model_multi_gpu.pt')))
        start_time = time.time()
        test_loss, test_acc = test_tree(args, model, valid_iterator, TRG_PAD_IDX, device, args.label_smoothing, criterion, args.clip)
        print_performances('Test', test_loss, test_acc, start_time, logging=logging)

if __name__ == "__main__":
    main()
