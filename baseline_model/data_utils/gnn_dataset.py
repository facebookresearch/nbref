# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from typing import *
import numpy as np
import torch
import os
import pickle
from torch.nn.utils.rnn import pack_sequence

from .dataset import Dataset
import dgl

class GNNDataset(Dataset):    
    def __init__(self, dataset_dir, asm=False, max_len=None):
        with open(os.path.join(dataset_dir, 'vocab_asm.pkl'), 'rb') as f:
            self.vocab_asm = pickle.load(f)
        with open(os.path.join(dataset_dir, 'dataset_asm.pkl'), 'rb') as f:
            self.dataset_asm = pickle.load(f)
        self.max_len = max_len

    def get_dataset(self):
        return  self.dataset_asm

    def collate(self, batch):
        graphs = []
        edges  = []
        graphs_len = []

        for i, elem in enumerate(batch):
            nodes_asm, edges_asm = elem

            src_edge = np.concatenate((edges_asm[0],edges_asm[2]),0)
            des_edge = np.concatenate((edges_asm[2],edges_asm[0]),0)
            type_edge = np.concatenate((edges_asm[1],edges_asm[1]),0)

            g = dgl.DGLGraph((src_edge, des_edge))
            src_len = len(nodes_asm)
            idmap = range(0, src_len)
            g.ndata['node_id'] = torch.tensor(idmap, dtype=torch.long)
            g.edata['type'] = torch.tensor(type_edge, dtype=torch.long) 
            g.ndata['annotation'] = torch.tensor(nodes_asm, dtype=torch.long)
            graphs.append(g)
            graphs_len.append(g.num_nodes())
            edges.append(edges_asm)
        return graphs, graphs_len
