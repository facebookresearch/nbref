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
import numpy as np
from modules.encoder_decoder_layers import *
from modules.encoder_decoder_layers  import _merge_on

import dgl.backend as Fdgl
from torch.utils.tensorboard import SummaryWriter

def make_src_mask(src, src_pad_idx):
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_trg_mask(trg, trg_pad_idx, device):
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(3)
    trg_len = trg.shape[1]
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask

class CODE_SIM_ASM_Model(nn.Module):
    def __init__(self, gnn, enc, hid_dim, src_pad_idx, device):
        super().__init__()
        self.encoder = enc
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.gnn = gnn       
        self.dense0 = nn.Linear(hid_dim, 32)

    def forward(self, src_, batch_graph=None):
        src_ = self.gnn(src_)
        src_mask = make_src_mask(src_.max(2)[0], self.src_pad_idx)
        output = self.encoder(src_, src_mask, batch_graph) #[bsz, src_len, hid_dim]
        output = self.dense0(output)
        output = torch.cat((output.mean(1), output.max(1)[0]), 1)
        return output 

class VUL_DETECT_ASM_Model(nn.Module):
    def __init__(self, gnn, enc, src_pad_idx, device, hid_dim, output_dim):
        super().__init__()
        self.encoder = enc
        self.src_pad_idx = src_pad_idx
        self.device = device
        self.gnn = gnn       

        self.dense  = nn.Linear(hid_dim, hid_dim)
        self.dense0 = nn.Linear(hid_dim*2, 32)
        self.dense1 = nn.Linear(32, output_dim)  
        self.activation = nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=-1)

    def forward(self, src_, batch_graph=None):
        src_ = self.gnn(src_)
        src_mask = make_src_mask(src_.max(2)[0], self.src_pad_idx)
        enc_src = self.encoder(src_, src_mask, batch_graph=batch_graph) # [bsz, src_len, hid_dim]
        enc_src = self.dense(enc_src)
        output = torch.cat((enc_src.mean(1), enc_src.max(1)[0]), 1)
        output = self.dense1(self.dense0(output))

        return self.activation(output).squeeze(-1) 

class Graph_NN(nn.Module):
    def __init__(self,
                 annotation_size,
                 out_feats,
                 n_steps,
                 device,
                 n_etypes=12,
                 gnn_type='sage',
                 sage_type='gcn',
                 tok_embedding=0,
                 residual=False):

        super(Graph_NN, self).__init__()

        self.annotation_size = annotation_size
        self.out_feats = out_feats
        self.layers = nn.ModuleList()
        # control signals
        self.tok_embedding_flag = tok_embedding
        self.residual=residual
        self.gnn_type = gnn_type

        if tok_embedding==1:
            self.tok_embedding = nn.Linear(out_feats, out_feats)
        elif tok_embedding == 2:
            self.tok_embedding = nn.Sequential(
                nn.Embedding(annotation_size, out_feats),
            )
        #ggnn
        if gnn_type =='ggnn':
            self.ggnn = GatedGraphConv(in_feats=out_feats,
                                    out_feats=out_feats,
                                    n_steps=n_steps,
                                    n_etypes=n_etypes)
        #graphsage
        if gnn_type == 'sage':
            self.layers.append(SAGEConv(out_feats, out_feats, sage_type, feat_drop=0.1, activation=F.relu))
            for i in range(n_steps - 1):
                self.layers.append(SAGEConv(out_feats, out_feats, sage_type, feat_drop=0.1, activation=F.relu))

        self.device = device

    def forward(self, graphs):
        if self.tok_embedding_flag != 2:
            annotation = graphs.ndata['annotation'].float()
        else:
            annotation = graphs.ndata['annotation']

        h1 = annotation.to(self.device)

        if self.tok_embedding_flag != 0:
            h1 = self.tok_embedding(h1)
            h1_tok = h1

        if self.gnn_type == 'ggnn':
            etypes = graphs.edata.pop('type').to(self.device)
            h1 = self.ggnn(graphs, h1, etypes)

        for layer in self.layers:
            h1 = layer(graphs, h1)

        if self.residual and self.tok_embedding_flag:
            graphs.ndata['h'] =  h1 + h1_tok

        graphs.ndata['h'] =  h1
        merged = _merge_on(graphs, 'nodes', 'h')
        return merged

class Transformer(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device,
                 gnn = None,
                 gnn_asm = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.gnn = gnn
        self.gnn_asm = gnn_asm

    def make_src_mask(self, src):
        return make_src_mask(src, self.src_pad_idx)

    def make_trg_mask(self, trg):
        return make_trg_mask(trg, self.trg_pad_idx, self.device)

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 mem_dim,
                 embedding_flag = True,
                 max_length = 100
                 ):
        super().__init__()

        self.device = device
        self.embedding_flag = embedding_flag
        if embedding_flag == 0:
            self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.position_enc = PositionalEncoding(hid_dim, n_position=max_length)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device,
                                                  mem_dim)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(self, src, src_mask, batch_graph=None):
        if self.embedding_flag == 0:
            src = self.tok_embedding(src) * self.scale.cuda()
            src = src + self.position_enc(src)
        elif self.embedding_flag == 1:
            src = self.layer_norm(src) 
            src = src + self.position_enc(src)  / self.scale.cuda()
        elif self.embedding_flag == 2:
            src = src * self.scale.cuda()
            src = src + self.position_enc(src)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask, batch_graph)

        return src

class Decoder_AST(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()
        self.device = device
        self.hid_dim = hid_dim
        self.output_dim = output_dim

        self.ast_self_layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device,
                                                  mem_dim=64
                                                  )
                                     for _ in range(n_layers)])

        self.ast_layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.layers_last = nn.ModuleList([DecoderLayer_last(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.position_enc = PositionalEncoding(hid_dim, n_position=max_length)
        self.tok_embedding_lin = nn.Linear(output_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.trg_pad_idx = 0

    def make_src_mask(self, src):
        src_mask = (src != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, trg_ast, trg_in, enc_src, src_mask, batch_graph=None):

        trg_in = self.tok_embedding_lin(trg_in) * self.scale.cuda()
        trg_in = self.dropout(trg_in)

        trg_ast_mask = self.make_src_mask(trg_ast.max(2)[0])
        trg_ast = trg_ast * self.scale.cuda()
        trg_ast = self.dropout(trg_ast)

        for layer in self.ast_self_layers:
            trg_ast = layer(trg_ast, trg_ast_mask, batch_graph=None)

        for layer in self.ast_layers:
            trg_ast = layer(trg_ast, enc_src, trg_ast_mask, src_mask)

        for layer in self.layers_last:
            trg_in = layer(trg_in, trg_ast, enc_src, None, src_mask, trg_ast_mask)

        output = self.fc_out(trg_in)
        return output

