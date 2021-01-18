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
from random import randint, uniform
import argparse
import pdb
import json
import dgl
import pickle
from .data_utils import init_tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

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
        loss = loss.masked_select(non_pad_mask).sum() 
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')

    return loss

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_performances(header, loss, accu, start_time, logging=None):
    logging('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
          'elapse: {elapse:3.3f} min'.format(
              header=f"({header})", ppl=math.exp(min(loss, 100)),
              accu=100*accu, elapse=(time.time()-start_time)/60))

def get_novel_positional_encoding(node, branch, parent):
    if parent:
        _positional_encoding = [
            0.0 for _ in range(parent['child_num'])]
        _positional_encoding[branch] = 1.0
        _positional_encoding += parent['encoding']
    else:
        _positional_encoding = []
    return _positional_encoding

def preprocessing_graph(src, edge, hid_dim_in):
    g = dgl.DGLGraph()
    src_len = len(src)
    g.add_nodes(src_len)
    idmap = range(0,src_len)
    g.ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
    edge_types = []
    annotation = torch.zeros([src_len, hid_dim_in], dtype=torch.long)

    for idx in range(0,src_len):
        annotation[idx][src[idx]-1] = 1
    for s, e, t in edge:
        if s == 0 and e ==0 and t ==0:
            break
        g.add_edge(s, t)
        edge_types.append(e)
        g.add_edge(t, s)
        edge_types.append(e)

    g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)
    g.ndata['annotation'] = annotation 
    return g

def processing_data(cache_dir, iterators):
    print("preprocessing data...")
    for iterator in iterators:
        for i, batch in init_tqdm(enumerate(iterator), 'preprocess'): 
            trg = batch.trg
            id_elem  = batch.id

            path = os.path.join(cache_dir, str(id_elem[0]))
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                continue
            batch_size = len(trg)
            queue_tree = {}
            graphs = []
            graphs_data = []
            graphs_data_depth = []
            graphs_data_encoding = []

            total_tree_num = [1 for i in range(0,batch_size)]
            for i in range(1, batch_size+1):
                queue_tree[i] = []
                queue_tree[i].append({"tree" : trg[i-1], "parent": 0, "child_index": 1 , "tree_path":[], "depth": 0, "child_num":len(trg[i-1].children), "encoding":[]})
                total_tree_num[i-1]+= len(trg[i-1].children)
                g = dgl.DGLGraph()
                graphs.append(g)
                graphs_data.append([])
                graphs_data_depth.append([])
                graphs_data_encoding.append([])

            cur_index, max_index = 1,1
            ic = 0
            dict_info = {}
            last_append = [None] * batch_size

            while (cur_index <= max_index):
                max_w_len = -1
                max_w_len_path = -1
                batch_w_list_trg = []
                batch_w_list = []
                flag = 1
                for i in range(1, batch_size+1):
                    w_list_trg = []
                    if (cur_index <= len(queue_tree[i])):
                        t_node = queue_tree[i][cur_index - 1]
                        t_encode = t_node["encoding"]
                        t_depth = t_node["depth"]
                        t = t_node["tree"]
                        if ic == 0:
                            queue_tree[i][cur_index - 1]["tree_path"].append(t.value)
                        t_path = queue_tree[i][cur_index - 1]["tree_path"].copy()

                        if ic == 0 and cur_index == 1:
                            graphs[i-1].add_nodes(1)
                            graphs[i-1].add_edges(t_node["parent"], cur_index - 1)
                            graphs_data[i-1].append(t.value)
                            graphs_data_depth[i-1].append(t_depth)
                            graphs_data_encoding[i-1].append(t_encode)
    
                        elif (ic <= t_node['child_num'] - 1):
                            t_node_child = last_append[i-1]
                            graphs[i-1].add_nodes(1)
                            graphs[i-1].add_edges(t_node_child["parent"],len(queue_tree[i])-1)
                            graphs_data[i-1].append(t_node_child["tree"].value)
                            graphs_data_depth[i-1].append(t_node_child["depth"])
                            graphs_data_encoding[i-1].append(t_node_child["encoding"])

                        # if it is not expanding all the children, add children into the queue
                        if ic <= t_node['child_num'] - 1:
                            w_list_trg.append(t.children[ic].value)
                            encoding = get_novel_positional_encoding(t.children[ic], ic, t_node)
                            if(t.children[ic].value != 0): 
                                last_append[i-1] = {"tree" : t.children[ic], "parent" : cur_index - 1, "child_index": ic, "tree_path" : t_path, 
                                                     "depth" : t_depth + 1, "child_num": len(t.children[ic].children), "encoding" : encoding}
                                if len(t.children[ic].children) > 0:
                                    queue_tree[i].append({"tree" : t.children[ic], "parent" : cur_index - 1, "child_index": ic, "tree_path":t_path, \
                                                        "depth" : t_depth + 1, "child_num": len(t.children[ic].children), "encoding":encoding})

                        if(ic + 1 < t_node['child_num']):
                            flag = 0

                        if len(queue_tree[i]) > max_index:
                            max_index = len(queue_tree[i])
                    if len(t_path) > max_w_len_path:
                        max_w_len_path = len(t_path)
                    if len(graphs_data[i-1]) > max_w_len:
                        max_w_len = len(graphs_data[i-1])
                    batch_w_list_trg.append(w_list_trg)
                    batch_w_list.append(t_path)

                dict_info = {
                    'batch_w_list' : batch_w_list[0],
                    'batch_w_list_trg' : batch_w_list_trg[0],
                    'graphs': graphs[0],
                    "graph_data":torch.tensor(graphs_data[0]),
                    "graph_depth":torch.tensor(graphs_data_depth[0]),
                    "graph_data_encoding":graphs_data_encoding[0]

                }

                if batch_w_list_trg[0] == [] and ic == 0:
                    print(ic)

                with open(os.path.join(path, str(cur_index)+'_'+str(ic)), 'wb') as f:
                    if dict_info =={}:
                        print(dict_info)
                    pickle.dump(dict_info, f)

                cur_index = cur_index + flag
                ic = 0 if flag == 1 else ic + 1

def preprocessing_batch_tmp(src_len, graph, device, type_require=False):
    batch_g = []
    max_len = max(src_len)
    for i in range(len(src_len)):
        g = dgl.graph(graph[i].edges())
        if type_require:
            g.edata['type'] = graph[i].edata['type']
        g.add_nodes(max_len - g.num_nodes())
        batch_g.append(g)

    batch_graph = dgl.batch(batch_g).to(device)
    return batch_graph

def train_eval_tree(args, model, iterator, optimizer, device, \
        criterion, dec_seq_length, train_flag=True):

    if train_flag:
        mode = 'train'
        model.train()
    else:
        mode = 'valid'
        model.eval()
    n_word_total, n_word_correct = 0, 0
    epoch_loss = 0

    sample_len = args.sample_len
    batch_graph_tmp = None
    batch_size = args.bsz 
    if args.dist_gpu == True:
        model = model.module
    with torch.set_grad_enabled(train_flag):
        for i, batch in enumerate(iterator):
            dict_info = batch['dict_info']
            batch_size = len(batch['trg'])
            id_elem = batch['id']
            graphs_asm = batch['graphs_asm']
            src_len = batch['src_len']

            batch_asm = dgl.batch(graphs_asm).to(device)
            enc_src = model.gnn_asm(batch_asm)
            src_mask = model.make_src_mask(enc_src.max(2)[0])            
            if args.graph_aug:
                batch_graph_tmp = preprocessing_batch_tmp(src_len, graphs_asm, device).to(device)

            enc_src = model.encoder(enc_src, src_mask, batch_graph_tmp)
            cur_index, max_index = 1, 1
            loss = 0
            ic = 0
            cur_index_batch = [1] * batch_size

            batch_nodes_num = [None] * batch_size
            for aa in range(0, batch_size):
                batch_nodes_num[aa] = len([i for i in dict_info[aa].keys()  if '_0' in i])
                if batch_nodes_num[aa] > sample_len:
                    rand_int = np.random.randint(-sample_len*2+1, batch_nodes_num[aa])
                    if rand_int < 1:
                        cur_index_batch[aa] = 1
                    elif rand_int > batch_nodes_num[aa] - sample_len:
                        cur_index_batch[aa] = batch_nodes_num[aa] - sample_len
                    else:
                        cur_index_batch[aa] = rand_int

            max_index = max(batch_nodes_num)
            graphs = [None] * batch_size
            graphs_data = [None] * batch_size
            graphs_data_depth = [None] * batch_size
            graphs_data_encoding = [None] * batch_size

            if max_index > sample_len:
                max_index = sample_len

            while (cur_index <= max_index):
                flag             =  1
                max_w_len_path   = -1
                batch_w_list_trg = [None] * batch_size 
                batch_w_list     = [None] * batch_size
                batch_len_trg    = [0] * batch_size
                batch_graph_len_list = [0] * batch_size

                for aa in range(0, batch_size):
                    path = os.path.join(args.cache_path, str(id_elem[aa]), str(cur_index_batch[aa])+'_'+ str(ic))
                    path_next = os.path.join(args.cache_path, str(id_elem[aa]), str(cur_index_batch[aa])+'_'+ str(ic+1))
                    if path in dict_info[aa].keys():

                        batch_w_list[aa]      = dict_info[aa][path]['batch_w_list']
                        batch_len_trg[aa]     = len(dict_info[aa][path]['batch_w_list'])
                        batch_w_list_trg[aa]  = dict_info[aa][path]['batch_w_list_trg']
                        graphs[aa]            = dict_info[aa][path]['graphs'].to(device)
                        graphs_data[aa]       = dict_info[aa][path]['graph_data'].to(device, non_blocking=True)
                        batch_graph_len_list[aa]  = len(graphs_data[aa])
                        graphs_data_depth[aa] = dict_info[aa][path]['graph_depth'].to(device, non_blocking=True)
                    else:
                        graphs_data[aa] = None
                        graphs[aa] = dgl.DGLGraph().to(device)
                        batch_graph_len_list[aa] = 0
    
                    if path_next in dict_info[aa].keys(): 
                        flag = 0

                max_w_len_path = max(batch_len_trg)
                
                in_ = torch.zeros((batch_size, args.output_dim, max_w_len_path), dtype=torch.long)
                trg_list = [model.trg_pad_idx for i in range(0, batch_size)]
                w_list_len = []

                for i in range(batch_size):
                    annotation = torch.zeros([graphs[i].number_of_nodes(), args.hid_dim - args.depth_dim], dtype=torch.long).cuda()
                    depth_annotation = torch.zeros([graphs[i].number_of_nodes(), args.depth_dim], dtype=torch.long).cuda()

                    if (batch_w_list_trg[i] is not None) and len(batch_w_list_trg[i]) > 0 :
                        annotation.scatter_(1, graphs_data[i].view(-1,1), value=torch.tensor(1))
                        depth_annotation.scatter_(1, graphs_data_depth[i].view(-1,1), value=torch.tensor(1))
                        depth_annotation[cur_index_batch[i]-1][-1] = 1
                        graphs[i].ndata['annotation'] = torch.cat([annotation,depth_annotation],dim=1).float()

                    w_list_trg = batch_w_list_trg[i]
                    t_path = batch_w_list[i]
                    if t_path is not None:
                        w_list_len.append(len(t_path)-1)
                    else:
                        w_list_len.append(0)
                    if (w_list_trg is not None) and len(w_list_trg) > 0 :
                        trg_list[i] = w_list_trg[0]
                        for j in range(len(t_path)):
                            in_[i][t_path[j]][j] = 1
                        in_[i][-ic-1][len(t_path)-1] = 1
                        
                in_ = in_.float().permute(0,2,1).cuda()
                batch_graph = dgl.batch(graphs).to(device)
                trg_in = model.gnn(batch_graph) 
                if args.graph_aug:
                    batch_graph_tmp = preprocessing_batch_tmp(batch_graph_len_list, graphs, device).to(device)
                    assert batch_graph_tmp.num_nodes() == trg_in.view(-1, args.hid_dim).shape[0], 'not match ast graph'

                output = model.decoder(trg_in, in_, enc_src, src_mask, batch_graph=batch_graph_tmp)

                output_list = []
                for p in range(len(w_list_len)):
                    output_list.append(output[p][w_list_len[p]].view(1,-1))

                output = torch.cat(output_list,dim=0).view(batch_size,-1)
                output = torch.cat([output], dim=0)
                trg_ = torch.tensor(trg_list).cuda()

                loss_itr, n_correct, n_word = cal_performance(
                    output, trg_, model.trg_pad_idx, smoothing=args.label_smoothing)
                loss += loss_itr
                n_word_total += n_word
                n_word_correct += n_correct
                cur_index = cur_index + flag
                cur_index_batch = [x + flag for x in cur_index_batch] 
                ic = 0 if flag == 1 else ic + 1

            if train_flag:
                optimizer.optimizer.zero_grad()
                loss.backward()
                if args.dist_gpu:
                    for param in model.parameters():
                        if param.requires_grad and param.grad is not None:
                            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                            param.grad.data /= args.n_dist_gpu
                else:
                    args.summary.add_scalar(mode + '/loss', loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            epoch_loss += loss.item()

    loss_per_word = epoch_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def test_tree(args, model, iterator, trg_pad_idx, device, smoothing, criterion, clip):
    n_word_total, n_word_correct = 0, 0
    epoch_loss = 0
    batch_graph_tmp = None
    model.eval()
    model = model.module
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(iterator):

            trg = batch['trg']
            graphs_asm = batch['graphs_asm']
            src_len = batch['src_len']

            batch_asm = dgl.batch(graphs_asm).to(device)
            enc_src = model.gnn_asm(batch_asm)
            src_mask = model.make_src_mask(enc_src.max(2)[0])            
            if args.graph_aug:
                batch_graph_tmp = preprocessing_batch_tmp(src_len, graphs_asm, device).to(device)

            enc_src = model.encoder(enc_src, src_mask, batch_graph_tmp)
            batch_size = len(trg)

            queue_tree = {}
            graphs = []
            graphs_data = []
            graphs_data_depth = []
            total_tree_num = [1 for i in range(0,batch_size)]
            for i in range(1, batch_size+1):
                queue_tree[i] = []
                queue_tree[i].append({"tree" : trg[i-1], "parent": 0, "child_index": 1 , "tree_path":[], "depth": 0, "child_num":len(trg[i-1].children), "encoding":[], "predict":trg[i-1].value})
                total_tree_num[i-1]+= len(trg[i-1].children)
                trg[i-1].predict = trg[i-1].value
                g = dgl.DGLGraph()
                graphs.append(g)
                graphs_data.append([])
                graphs_data_depth.append([])

            cur_index, max_index = 1,1
            loss, ic = 0, 0
            last_append = [None] * batch_size

            while (cur_index <= max_index):
                max_w_len = -1
                max_w_len_path = -1
                batch_w_list_trg = []
                batch_w_list = []
                flag = 1
                t = [None] * batch_size
                batch_graph_len_list = [0] * batch_size
                graphs_tmp = [dgl.DGLGraph()] * batch_size

                for i in range(1, batch_size+1):
                    w_list_trg = []
                    if (cur_index <= len(queue_tree[i])):
                        t_node = queue_tree[i][cur_index - 1]
                        t_encode = t_node["encoding"]
                        t_depth = t_node["depth"]
                        t[i-1] = t_node["tree"]
                        if ic == 0:
                            queue_tree[i][cur_index - 1]["tree_path"].append(t[i-1].predict)
                        t_path = queue_tree[i][cur_index - 1]["tree_path"].copy()

                        if ic == 0 and cur_index == 1:
                            graphs[i-1].add_nodes(1)
                            graphs[i-1].add_edges(t_node["parent"], cur_index - 1)
                            graphs_data[i-1].append(t[i-1].predict)
                            graphs_data_depth[i-1].append(t_depth)
                        elif (ic <= t_node['child_num'] - 1):
                            t_node_child = last_append[i-1]
                            graphs[i-1].add_nodes(1)
                            graphs[i-1].add_edges(t_node_child["parent"],len(queue_tree[i])-1)
                            graphs_data[i-1].append(t_node_child["tree"].predict)
                            graphs_data_depth[i-1].append(t_node_child["depth"])

                        # if it is not expanding all the children, add children into the queue
                        if ic <= t_node['child_num'] - 1:
                            w_list_trg.append(t[i-1].children[ic].value)
                            encoding = get_novel_positional_encoding(t[i-1].children[ic], ic, t_node)
                            if(t[i-1].children[ic].value != 0):
                                last_append[i-1] = {"tree" : t[i-1].children[ic], "parent" : cur_index - 1, "child_index": ic, "tree_path" : t_path, 
                                                     "depth" : t_depth + 1, "child_num": len(t[i-1].children[ic].children), "encoding" : encoding}
                                if len(t[i-1].children[ic].children) > 0:
                                    queue_tree[i].append({"tree" : t[i-1].children[ic], "parent" : cur_index - 1, "child_index": ic, "tree_path":t_path, \
                                                        "depth" : t_depth + 1, "child_num": len(t[i-1].children[ic].children), "encoding":encoding})
                            batch_graph_len_list[i-1] = len(graphs_data[i-1])
                            graphs_tmp[i-1] = graphs[i-1]
                        else:
                            batch_graph_len_list[i-1] = 0
                            graphs_tmp[i-1] = dgl.DGLGraph()

                        if(ic + 1 < t_node['child_num']):
                            flag = 0

                        if len(queue_tree[i]) > max_index:
                            max_index = len(queue_tree[i])
                    if len(t_path) > max_w_len_path:
                        max_w_len_path = len(t_path)
                    if len(graphs_data[i-1]) > max_w_len:
                        max_w_len = len(graphs_data[i-1])
                    batch_w_list_trg.append(w_list_trg)
                    batch_w_list.append(t_path)

                trg_l = [trg_pad_idx for i in range(0, batch_size)]
                w_list_len = []

                in_ = torch.zeros((batch_size, args.output_dim, max_w_len_path), dtype=torch.long)
                for i in range(batch_size):
                    annotation = torch.zeros([graphs[i].number_of_nodes(), args.hid_dim - args.depth_dim], dtype=torch.long)
                    depth_annotation = torch.zeros([graphs[i].number_of_nodes(), args.depth_dim], dtype=torch.long)
                    for idx in range(0,len(graphs_data[i])):
                        annotation[idx][graphs_data[i][idx]] = 1
                        depth_annotation[idx][graphs_data_depth[i][idx]] = 1

                    if len(batch_w_list_trg[i]) > 0 :
                        depth_annotation[cur_index-1][-1] = 1
                    graphs[i].ndata['annotation'] = torch.cat([annotation,depth_annotation],dim=1)

                    w_list_trg = batch_w_list_trg[i]
                    t_path = batch_w_list[i]
                    w_list_len.append(len(t_path)-1)
                    if len(w_list_trg) > 0 :
                        trg_l[i] = w_list_trg[0]
                        for j in range(len(t_path)):
                            in_[i][t_path[j]][j] = 1
                        in_[i][-ic-1][len(t_path)-1] = 1

                in_ = in_.float().permute(0,2,1).cuda()
                if args.graph_aug:
                    batch_graph_tmp = preprocessing_batch_tmp(batch_graph_len_list, graphs_tmp, device).to(device)
                batch_graph = dgl.batch(graphs_tmp).to(device)
                trg_in = model.gnn(batch_graph)
                assert batch_graph_tmp.num_nodes() == trg_in.view(-1, args.hid_dim).shape[0], 'not match ast graph'

                output_l = model.decoder(trg_in, in_, enc_src, src_mask, batch_graph=batch_graph_tmp)

                output_l_list = []
                for p in range(len(w_list_len)):
                    output_l_list.append(output_l[p][w_list_len[p]].view(1,-1))

                output_l = torch.cat(output_l_list,dim=0).view(batch_size,-1)
                output = torch.cat([output_l], dim=0)
                trg_ = torch.tensor(trg_l).cuda()
                output_predict_list = output.argmax(1).tolist()
                for p, elem in enumerate(output_predict_list):
                    if t[p] is not None and (len(t[p].children) > ic ):
                        # The 1st node is root node.
                        if cur_index < 2:
                            t[p].children[ic].predict = trg_[p]
                        else:
                            t[p].children[ic].predict = elem
                loss_itr, n_correct, n_word = cal_performance(
                    output, trg_, trg_pad_idx, smoothing=smoothing)

                loss += loss_itr
                n_word_total += n_word
                n_word_correct += n_correct
                cur_index = cur_index + flag
                ic = 0 if flag == 1 else ic + 1

            epoch_loss += loss.item()
        loss_per_word = epoch_loss/n_word_total
        accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy