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
from random import randint
from config import *
import argparse
import pdb
import json

from .Tree import Tree
StmtID = 10
Dummy = 0
def eval_training(opt, iterator, encoder, decoder_l,decoder_r, attention_decoder, encoder_optimizer, decoder_optimizer_l, decoder_optimizer_r, attention_decoder_optimizer, criterion, using_gpu):
    epoch_loss = 0
    encoder.train()
    decoder_r.train()
    decoder_l.train()
    attention_decoder.train()

    for it, batch in enumerate(iterator):
        # print(it)
        encoder_optimizer.zero_grad()
        decoder_optimizer_l.zero_grad()
        decoder_optimizer_r.zero_grad()
        attention_decoder_optimizer.zero_grad()
        enc_batch = batch.src #.transpose(1,0)
        dec_tree_batch  = batch.trg
        enclen = batch.enclen
        # enc_max_len  = enc_batch.size(1) #batch.src.shape[1]
        enc_max_len  = opt.enc_seq_length #enc_batch.size(1)
        enc_outputs = torch.zeros((len(enc_batch), enc_max_len, encoder.hidden_size), requires_grad=True)

        if using_gpu:
            enc_outputs = enc_outputs.cuda()
        enc_s = {}
        for j in range(opt.enc_seq_length + 1):
            enc_s[j] = {}

        dec_s = {}
        for i in range(opt.dec_seq_length + 1):
            dec_s[i] = {}
            for j in range(3):
                dec_s[i][j] = {}

        for i in range(1, 3):
            enc_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                enc_s[0][i] = enc_s[0][i].cuda()

        # pdb.set_trace()
        # TODO:change this part
        # import time
        # start = time.time()
        for i in range(enc_max_len):
            enc_s[i+1][1], enc_s[i+1][2] = encoder(enc_batch, i, enc_s[i][1], enc_s[i][2])
            enc_outputs[:, i, :] = enc_s[i+1][2]

        # end = time.time()
        # print("time encoding: " + str(end-start) )
        # tree decode
        queue_tree = {}
        for i in range(1, opt.batch_size+1):
            queue_tree[i] = []
            queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})
        loss = 0
        cur_index, max_index = 1,1
        dec_batch = {}
        dec_batch_trg = {}
        #print(queue_tree[1][0]["tree"].to_string());exit()
        while (cur_index <= max_index):
            #print(cur_index)
            # build dec_batch for cur_index
            max_w_len = -1
            batch_w_list = []
            batch_w_list_trg = []
            # pdb.set_trace()
            for i in range(1, opt.batch_size+1):
                w_list = []
                w_list_trg = []
                if (cur_index <= len(queue_tree[i])):
                    t = queue_tree[i][cur_index - 1]["tree"]
                    for ic in range (len(t.children)):
                        w_list.append(t.value)
                        w_list_trg.append(t.children[ic].value)
                        if(t.children[ic].value != 0):
                            queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic })
                        # else:
                        #     w_list.append(t.children[ic])
                    if len(queue_tree[i]) > max_index:
                        max_index = len(queue_tree[i])
                if len(w_list) > max_w_len:
                    max_w_len = len(w_list)
                batch_w_list.append(w_list)
                batch_w_list_trg.append(w_list_trg)
            # if(cur_index == 146):
            #     pdb.set_trace()
            # dec_batch[cur_index] = torch.zeros((opt.batch_size, max_w_len), dtype=torch.long)
            dec_batch[cur_index] = torch.zeros((opt.batch_size,2), dtype=torch.long)
            dec_batch_trg[cur_index] = torch.zeros((opt.batch_size,2), dtype=torch.long)
            for i in range(opt.batch_size):
                w_list = batch_w_list[i]
                w_list_trg = batch_w_list_trg[i]
                if len(w_list) > 0:
                    for j in range(len(w_list)):
                        dec_batch[cur_index][i][j] = w_list[j]
                        dec_batch_trg[cur_index][i][j] = w_list_trg[j]
                    # if cur_index == 1:
                    #     dec_batch[cur_index][i][0] = 0
                    # dec_batch[cur_index][i][len(w_list) ] = 1
            # print(dec_batch[cur_index])
            # initialize first decoder unit hidden state (zeros)
            # try:
            # if cur_index == 2:
            # pdb.set_trace()
            # print(dec_batch)
            if using_gpu:
                dec_batch[cur_index] = dec_batch[cur_index].cuda()
                dec_batch_trg[cur_index] = dec_batch_trg[cur_index].cuda()
            # except:
            # initialize using encoding results
            # print(cur_index)
            for j in range(1, 3):
                dec_s[cur_index][0][j] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
                if using_gpu:
                    dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

            #dec_s  1: cur_index 2: child index 3. h (1) or s (2)
            if cur_index == 1:
                for i in range(opt.batch_size):
                    # dec_s[1][0][1][i, :] = enc_s[enc_max_len][1][i, :]
                    # dec_s[1][0][2][i, :] = enc_s[enc_max_len][2][i, :]
                    try:
                        dec_s[1][0][1][i, :] = enc_s[enclen[i]][1][i, :]
                        dec_s[1][0][2][i, :] = enc_s[enclen[i]][2][i, :]
                    except:
                        pdb.set_trace()
            else:
                # pdb.set_trace()
                for i in range(1, opt.batch_size+1):
                    if (cur_index <= len(queue_tree[i])):
                        par_index = queue_tree[i][cur_index - 1]["parent"]
                        child_index = queue_tree[i][cur_index - 1]["child_index"]
                        #print("parent child")
                        #print(par_index)
                        #print(child_index)
                        # if i == 1:
                        # pdb.set_trace()
                        dec_s[cur_index][0][1][i-1,:] = dec_s[par_index][child_index][1][i-1,:]
                        dec_s[cur_index][0][2][i-1,:] = dec_s[par_index][child_index][2][i-1,:]
            #loss = 0
            #prev_c, prev_h = dec_s[cur_index, 0, 0,:,:], dec_s[cur_index, 0, 1,:,:]
            #pred_matrix = np.ndarray((20, dec_batch[cur_index].size(1)-1), dtype=object)

            parent_h = dec_s[cur_index][0][2]

            # pdb.set_trace()
            try:
                dec_s[cur_index][1][1], dec_s[cur_index][1][2] =decoder_l(dec_batch[cur_index][:,0], dec_s[cur_index][0][1], dec_s[cur_index][0][2], parent_h)
                # pdb.set_trace()
            except:
                pdb.set_trace()
            pred_l = attention_decoder(enc_outputs,dec_s[cur_index][1][2])
            # pdb.set_trace()
            loss += criterion(pred_l, dec_batch_trg[cur_index][:,0])

            try:
                dec_s[cur_index][2][1],dec_s[cur_index][2][2] = decoder_r(dec_batch[cur_index][:,1], dec_s[cur_index][0][1], dec_s[cur_index][0][2], parent_h)
            except:
                pdb.set_trace()
            pred_r = attention_decoder(enc_outputs,dec_s[cur_index][2][2])
            loss += criterion(pred_r, dec_batch_trg[cur_index][:,1])

            # pdb.set_trace()
            # for i in range(dec_batch[cur_index].size(1) - 1):
            #     #print(i)
            #     # pdb.set_trace()
            #     dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = decoder(dec_batch[cur_index][:,i], dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h)
            #     pred = attention_decoder(enc_outputs, dec_s[cur_index][i+1][2])
            #     loss += criterion(pred, dec_batch[cur_index][:,i+1])

            cur_index = cur_index + 1
        #input_string = [form_manager.get_idx_symbol(int(p)) for p in enc_batch[0,:]]
        #print("===========\n")
        #print("input string: {}\n".format(input_string))
        #print("predicted string: {}\n".format(pred_matrix[0,:]))
        #print("===========\n")

        # pdb.set_trace()
        loss = loss / opt.batch_size
        loss.backward()
        torch.nn.utils.clip_grad_value_(encoder.parameters(),opt.grad_clip)
        torch.nn.utils.clip_grad_value_(decoder_l.parameters(),opt.grad_clip)
        torch.nn.utils.clip_grad_value_(decoder_r.parameters(),opt.grad_clip)
        torch.nn.utils.clip_grad_value_(attention_decoder.parameters(),opt.grad_clip)
        encoder_optimizer.step()
        decoder_optimizer_l.step()
        decoder_optimizer_r.step()
        attention_decoder_optimizer.step()
            #print("end eval training \n ")
            #print("=====================\n")

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluation(opt, iterator, encoder, decoder_l,decoder_r, attention_decoder, criterion, using_gpu, tree_node_gen, teaching_force =True):
    epoch_loss = 0
    encoder.eval()
    decoder_r.eval()
    decoder_l.eval()
    attention_decoder.eval()
    TreeRoot = []
    TrgRoot  = []
    for it, batch in enumerate(iterator):
        enc_batch = batch.src # .transpose(1,0)
        dec_tree_batch  = batch.trg
        enclen = batch.enclen
        enc_max_len  = opt.enc_seq_length #enc_batch.size(1)
        # enc_outputs = torch.zeros((enc_batch.size(0), enc_max_len, encoder.hidden_size), requires_grad=True)
        enc_outputs = torch.zeros((len(enc_batch), enc_max_len, encoder.hidden_size), requires_grad=True)
        if using_gpu:
            enc_outputs = enc_outputs.cuda()
        enc_s = {}
        for j in range(opt.enc_seq_length + 1):
            enc_s[j] = {}

        dec_s = {}
        for i in range(opt.dec_seq_length + 1):
            dec_s[i] = {}
            for j in range(3):
                dec_s[i][j] = {}

        for i in range(1, 3):
            enc_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
            if using_gpu:
                enc_s[0][i] = enc_s[0][i].cuda()

        # TODO:change this part
        for i in range(enc_max_len):
            enc_s[i+1][1], enc_s[i+1][2] = encoder(enc_batch, i, enc_s[i][1], enc_s[i][2])
            enc_outputs[:, i, :] = enc_s[i+1][2]
        # tree decode
        queue_tree = {}
        TreeRootGen = {}
        TreeNodeCurrent = {}
        for i in range(1, opt.batch_size+1):
            if tree_node_gen:
                TreeRootGen[i] = Tree(StmtID)
                TreeNodeCurrent[i] = []
                TreeNodeCurrent[i].append(TreeRootGen[i])
            # pdb.set_trace()
            queue_tree[i]  = []
            queue_tree[i].append({"tree" : dec_tree_batch[i-1], "parent": 0, "child_index": 1})

        loss = 0
        cur_index, max_index = 1,1
        dec_batch = {}
        dec_batch_trg = {}

        while (cur_index <= max_index):
            # build dec_batch for cur_index
            max_w_len = -1
            batch_w_list = []
            batch_w_list_trg = []
            for i in range(1, opt.batch_size+1):
                w_list = []
                w_list_trg = []
                if (cur_index <= len(queue_tree[i])):
                    t = queue_tree[i][cur_index - 1]["tree"]
                    # for ic in range (t.num_children):
                    for ic in range (len(t.children)):
                        w_list.append(t.value)
                        w_list_trg.append(t.children[ic].value)
                        if(tree_node_gen):
                            NewTreeNode = Tree(Dummy)
                            NewTreeNode.parent = TreeNodeCurrent[i][cur_index - 1]
                            TreeNodeCurrent[i][cur_index - 1].children.append(NewTreeNode)
                        if(t.children[ic].value != 0):
                            if(tree_node_gen):
                                TreeNodeCurrent[i].append(NewTreeNode)
                            queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index, "child_index": ic })
                    if len(queue_tree[i]) > max_index:
                        max_index = len(queue_tree[i])
                if len(w_list) > max_w_len:
                    max_w_len = len(w_list)
                batch_w_list.append(w_list)
                batch_w_list_trg.append(w_list_trg)

            dec_batch[cur_index] = torch.zeros((opt.batch_size, 2), dtype=torch.long)
            dec_batch_trg[cur_index] = torch.zeros((opt.batch_size, 2), dtype=torch.long)
            for i in range(opt.batch_size):
                w_list = batch_w_list[i]
                w_list_trg = batch_w_list_trg[i]
                if len(w_list) > 0:
                    for j in range(len(w_list)):
                        dec_batch[cur_index][i][j] = w_list[j]
                        dec_batch_trg[cur_index][i][j] = w_list_trg[j]

            # initialize first decoder unit hidden state (zeros)
            if using_gpu:
                dec_batch[cur_index] = dec_batch[cur_index].cuda()
                dec_batch_trg[cur_index] = dec_batch_trg[cur_index].cuda()

            # initialize using encoding results
            for j in range(1, 3):
                dec_s[cur_index][0][j] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=True)
                if using_gpu:
                    dec_s[cur_index][0][j] = dec_s[cur_index][0][j].cuda()

            #dec_s  1: cur_index 2: child index 3. h (1) or s (2)
            if cur_index == 1:
                for i in range(opt.batch_size):
                    dec_s[1][0][1][i, :] = enc_s[enclen[i]][1][i, :]
                    dec_s[1][0][2][i, :] = enc_s[enclen[i]][2][i, :]
            else:
                for i in range(1, opt.batch_size+1):
                    if (cur_index <= len(queue_tree[i])):
                        par_index = queue_tree[i][cur_index - 1]["parent"]
                        child_index = queue_tree[i][cur_index - 1]["child_index"]
                        dec_s[cur_index][0][1][i-1,:] = dec_s[par_index][child_index][1][i-1,:]
                        dec_s[cur_index][0][2][i-1,:] = dec_s[par_index][child_index][2][i-1,:]
            #loss = 0
            #prev_c, prev_h = dec_s[cur_index, 0, 0,:,:], dec_s[cur_index, 0, 1,:,:]
            #pred_matrix = np.ndarray((20, dec_batch[cur_index].size(1)-1), dtype=object)
            parent_h = dec_s[cur_index][0][2]

            # left-right decoder style
            dec_s[cur_index][1][1], dec_s[cur_index][1][2] =decoder_l(dec_batch[cur_index][:,0], dec_s[cur_index][0][1], dec_s[cur_index][0][2], parent_h)
            pred_l = attention_decoder(enc_outputs,dec_s[cur_index][1][2])
            loss += criterion(pred_l, dec_batch_trg[cur_index][:,0])

            dec_s[cur_index][2][1],dec_s[cur_index][2][2] = decoder_r(dec_batch[cur_index][:,1], dec_s[cur_index][0][1], dec_s[cur_index][0][2], parent_h)
            pred_r = attention_decoder(enc_outputs,dec_s[cur_index][2][2])
            loss += criterion(pred_r, dec_batch_trg[cur_index][:,1])

            # pdb.set_trace()
            max_pred_l = torch.max(pred_l,1)[1]
            max_pred_r = torch.max(pred_r,1)[1]
            # pdb.set_trace()
            if(tree_node_gen):
                for i in range(1,opt.batch_size + 1):
                    try:
                        if(cur_index <= len(TreeNodeCurrent[i]) and len(TreeNodeCurrent[i][cur_index - 1 ].children) != 0 ):
                            TreeNodeCurrent[i][cur_index - 1 ].children[0].value = max_pred_l[i-1].item()
                            if len(TreeNodeCurrent[i][cur_index - 1 ].children) > 1:
                                TreeNodeCurrent[i][cur_index - 1 ].children[1].value = max_pred_r[i-1].item()
                    except:
                        pdb.set_trace()
                # pdb.set_trace()
        # for i in range(dec_batch[cur_index].size(1)):
            # pdb.set_trace()
            #     #print(i)
            #     # pdb.set_trace()
            #     dec_s[cur_index][i+1][1], dec_s[cur_index][i+1][2] = decoder(dec_batch[cur_index][:,i], dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h)
            #     pred = attention_decoder(enc_outputs, dec_s[cur_index][i+1][2])
            #     loss += criterion(pred, dec_batch[cur_index][:,i+1])

            cur_index = cur_index + 1
        # pdb.set_trace()
        TreeRoot.append(TreeRootGen)
        TrgRoot.append(batch.trg)
        loss = loss / opt.batch_size
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), TrgRoot , TreeRoot


# def testing(opt, iterator, encoder, decoder_l,decoder_r, attention_decoder, criterion, using_gpu,teaching_force):
