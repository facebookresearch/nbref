# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
"""
Data utils for processing bAbI datasets
"""

import os, pdb

from torch.utils.data import DataLoader
import dgl
import torch
import string
import numpy as np
from dgl.data.utils import download, get_download_dir, _get_dgl_url, extract_archive
from torch import nn


def _get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def get_dataloaders(batch_size, in_f, in_g, in_a, train_size, valid_size, hid_dim):
    reverse_edge = {0: 1, 1: 0}
    global hid_dim_in
    hid_dim_in = hid_dim

    return _gc_dataloader(train_size, valid_size,
    in_f, in_g, in_a, batch_size, reverse_edge, hid_dim_in)

def int2bits(x):
  return [int(b) for b in "{:064b}".format(np.abs(x))]

def _gc_dataloader(train_size, valid_size, in_f, in_g, in_a, batch_size, reverse_edge, hid_dim):
    src_token_len = 0
    # global pos_table
    # pos_table = _get_sinusoid_encoding_table(682, 416)
    def _collate_fn(batch):
        graphs = []
        labels = []
        feats = []
        for d in batch:
            edges = d['g']
            node_ids = []
            for s, e, t in edges:
                if s not in node_ids:
                    node_ids.append(s)
                if t not in node_ids:
                    node_ids.append(t)
            feat_dict = d['f']
            g = dgl.DGLGraph()
            # g.add_nodes(len(feat_dict))
            # idmap = range(0,len(feat_dict))
            # g.ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
            g.add_nodes(d['a'].shape[0])
            idmap = range(0,(d['a'].shape[0]))
            g.ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
            # nid2idx = dict(zip(node_ids, list(range(len(node_ids)))))
            # labels.append(d['eval'][-1])

            edge_types = []
            for s, e, t in edges:
                g.add_edge(s, t)
                edge_types.append(e)
                if e in reverse_edge:
                    g.add_edge(t, s)
                    edge_types.append(reverse_edge[e])
            g.edata['type'] = torch.tensor(edge_types, dtype=torch.long)

            # features
            # feats.append(feat_dict)
            # annotation = torch.zeros([len(feat_dict), hid_dim_in], dtype=torch.long)
            # pdb.set_trace()
            # node_id = torch.zeros([len(feat_dict), 12],dtype=torch.long)
            # for idx in range(0,len(feat_dict)):
            #     annotation[idx][feat_dict[idx]] = 1
            #     node_id[idx] = torch.tensor([int(b) for b in "{:012b}".format(np.abs(0))],dtype=torch.long)

            # annotation = torch.cat([annotation,node_id],dim=1) #+ pos_table[:, :annotation.size(0)]*0.25
            # g.ndata['annotation'] = torch.tensor(d['a'],dtype=torch.long)
            g.ndata['annotation'] = torch.tensor(feat_dict, dtype=torch.long)
            graphs.append(g)
        batch_graph = dgl.batch(graphs)
        # pdb.set_trace()
        return batch_graph

    def _get_dataloader(in_f, in_g, in_a, train_size, valid_size, shuffle):
        # in_f_list = np.int_(in_f).tolist()
        in_f_list = in_f.tolist()
        in_g_list = np.int_(in_g).tolist()
        # in_f_list = in_f.tolist()
        # in_g_list = in_g.tolist()
        train_dict = []
        valid_dict = []
        src_token_len = 0
        for i in range(0,train_size):
            cur_max = max(in_f_list[i])
            if max(in_f_list[i])> src_token_len:
                src_token_len = max(in_f_list[i])
            try:
                ind_f = in_f_list[i].index(0)
                in_f_tmp = in_f_list[i][:ind_f]
            except:
                in_f_tmp = in_f_list[i]
            try:
                ind_g = in_g_list[i].index([0,0,0])
                in_g_tmp = in_g_list[i][:ind_g]
            except:
                in_g_tmp = in_g_list[i]
            train_dict.append({'f':in_f_tmp,'g':in_g_tmp, 'a':in_a[i]})

        for i in range(train_size,train_size+valid_size):
            cur_max = max(in_f_list[i])
            if max(in_f_list[i])> src_token_len:
                src_token_len = max(in_f_list[i])
            try:
                ind_f = in_f_list[i].index(0)
                in_f_tmp = in_f_list[i][:ind_f]
            except:
                in_f_tmp = in_f_list[i]
            try:
                ind_g = in_g_list[i].index([0,0,0])
                in_g_tmp = in_g_list[i][:ind_g]
            except:
                in_g_tmp = in_g_list[i]
            valid_dict.append({'f':in_f_tmp,'g':in_g_tmp, 'a':in_a[i]})
        train_dataloader = DataLoader(dataset=train_dict, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
        dev_dataloader = DataLoader(dataset=valid_dict, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate_fn)
        return train_dataloader, dev_dataloader, src_token_len

    train_dataloader, dev_dataloader, src_token_len = _get_dataloader(in_f, in_g, in_a, train_size, valid_size, False)
    max_len_src = in_f.shape[1]

    return train_dataloader, dev_dataloader, max_len_src, src_token_len


def _convert_gc_dataset(train_size, node_dict, edge_dict, label_dict, path, q_type):
    total_num = 11000


    def convert(file):
        dataset = []
        d = dict()
        with open(file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip().split()
                if line[0] == '1' and len(d) > 0:
                    d = dict()
                if line[1] == 'eval':
                    # (src, edge, label)
                    if 'eval' not in d:
                        d['eval'] = (node_dict[line[2]], edge_dict[line[3]], node_dict[line[4]], label_dict[line[5]])
                        if d['eval'][1] == q_type:
                            dataset.append(d)
                            if len(dataset) >= total_num:
                                break
                else:
                    if 'edges' not in d:
                        d['edges'] = []
                    d['edges'].append((node_dict[line[1]], edge_dict[line[2]], node_dict[line[3]]))
        return dataset

    download_dir = get_download_dir()
    filename = os.path.join(download_dir, 'babi_data', path, 'data.txt')
    data = convert(filename)

    assert len(data) == total_num

    train_set = data[:train_size]
    dev_set = data[950:1000]
    test_sets = []
    for i in range(10):
        test = data[1000 * (i + 1): 1000 * (i + 2)]
        test_sets.append(test)

    return train_set, dev_set, test_sets
