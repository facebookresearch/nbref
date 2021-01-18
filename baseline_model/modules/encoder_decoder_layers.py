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
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling
import dgl

import random
import math
import os
import time
import pdb
import numpy as np
import dgl.backend as Fdgl

READOUT_ON_ATTRS = {
    'nodes': ('ndata', 'batch_num_nodes', 'number_of_nodes'),
    'edges': ('edata', 'batch_num_edges', 'number_of_edges'),
}

def _merge_on(graph, typestr, feat, ntype_or_etype=None):
    data_attr, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, data_attr)
    feat = data[feat]

    batch_num_objs = getattr(graph, batch_num_objs_attr)(ntype_or_etype)
    feat = Fdgl.pad_packed_tensor(feat, batch_num_objs, 0 )
    return feat

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        Q = self.fc_q(query) #(bsz, q_len, h_dim)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) #(bsz, n_head, q_len, head_dim)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.cuda()
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10) # (bsz, n_head, q_len, k_len)
        x = torch.matmul(self.dropout(torch.softmax(energy, dim = -1)), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim) #(bsz, q_len, n_heads, head_dim)

        x = self.fc_o(x) #(bsz, q_len, h_dim)

        return x

class MultiHeadAttentionLayer_wMem(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device, m = 64):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.m_k = nn.Parameter(torch.FloatTensor(1, m, hid_dim))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, hid_dim)) 
        self.m = m
        self.init_weights()

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def init_weights(self):
        nn.init.normal_(self.m_k, 0, 1 / self.head_dim)
        nn.init.normal_(self.m_v, 0, 1 / self.m)

    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]
        nk = key.shape[1]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)


        m_k = np.sqrt(self.head_dim) * self.m_k.expand(batch_size, self.m, self.hid_dim)
        m_v = np.sqrt(self.m) * self.m_v.expand(batch_size, self.m, self.hid_dim)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = torch.cat([K, m_k], 1).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = torch.cat([V, m_v], 1).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.cuda()

        if mask is not None:
            energy[:,:,:,:nk] = energy[:,:,:,:nk].masked_fill(mask == 0, -1e20)

        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 mem_dim,
                 gnn_layer_num=2):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        if mem_dim != 0:
            self.self_attention = MultiHeadAttentionLayer_wMem(hid_dim, n_heads, dropout, device, m = mem_dim)
        else:
            self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.layers.append(SAGEConv(hid_dim, hid_dim, "gcn", feat_drop=0, activation=F.relu))
        for i in range(gnn_layer_num - 1):
            self.layers.append(SAGEConv(hid_dim, hid_dim, "gcn", feat_drop=0, activation=F.relu))

        self.hid_dim = hid_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask, batch_graph=None):
        _src = self.self_attention(src, src, src, src_mask) # src_mask: [bsz, src_len]
        src = self.layer_norm(src + self.dropout(_src)) #[bsz, src_len, hid_dim]
        _src = self.positionwise_feedforward(src)
        src = self.layer_norm(src + self.dropout(_src))

        if batch_graph is None:
            return src # [bsz, src_len, hid_dim]
        else:
            batch_graph.ndata['enc'] = src.view(-1,self.hid_dim)
            h1 = batch_graph.ndata.pop('enc')
            for layer in self.layers:
                h1 = layer(batch_graph, h1)
            batch_graph.ndata['enc'] = h1
            _src = _merge_on(batch_graph, 'nodes', 'enc')
            src = self.layer_norm(src + self.dropout(_src))
            return src

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 mem_dim = 0, 
                 gnn_layer_num = 2):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        if mem_dim != 0:
            self.self_attention = MultiHeadAttentionLayer_wMem(hid_dim, n_heads, dropout, device, m = mem_dim)
        else:
            self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask, batch_graph=None):
        #self attention
        _trg = self.self_attention(trg, trg, trg, trg_mask) 
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg= self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))
        if batch_graph is None:
            return trg # [bsz, src_len, hid_dim]
        else:
            batch_graph.ndata['dec'] = trg.view(-1,self.hid_dim)
            h1 = batch_graph.ndata.pop('dec')
            for layer in self.layers:
                h1 = layer(batch_graph, h1)
            batch_graph.ndata['dec'] = h1
            _trg = _merge_on(batch_graph, 'nodes', 'dec')
            trg = self.layer_norm(trg + self.dropout(_trg))
            return trg

class DecoderLayer_last(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.ast_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, ast, enc_src, trg_mask, src_mask, ast_mask):
        #self attention
        _trg = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))

        #ast attention 
        _trg= self.ast_attention(trg, ast, ast, ast_mask)
        trg = self.layer_norm(trg + self.dropout(_trg))

        #encoder attention
        _trg= self.encoder_attention(trg, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        trg = self.layer_norm(trg + self.dropout(_trg)) 

        return trg
