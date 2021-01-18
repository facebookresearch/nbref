# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import argparse
import os, pdb, gc
import pickle
from collections import defaultdict
import numpy as np
import multiprocessing
from asm_obj import *
import dgl
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', type=str, required=False, default="data/dataset-vul")
    parser.add_argument('--file-list', '-txt', type=str, required=False, default="data/vul/devign_asm.txt")
    parser.add_argument('--csv-dir', '-csv', type=str, required=False, default='data/vul/')
    parser.add_argument('--ins_index', '-ins', type=str, default=None)
    parser.add_argument('--num-workers', '-p', type=int, default=os.cpu_count())
    parser.add_argument('--asm-dir', '-a', type=str, required=False, default="data/vul/assembly")
    parser.add_argument('--min-freq-asm', '-t-asm', type=int, default=10)
    return parser.parse_args()


def build_graph(node, nodes, edges):
    node_id = len(nodes)
    nodes.append(node.n)
    last_id = node_id
    for c in node.children:
        edges.append((node_id, last_id + 1))
        last_id = build_graph(c, nodes, edges)
    return last_id


def load_graph(arguments):
    pid, path, tgt = arguments
    asm_nodes, asm_edges = Graphs_build_x86(path, 'pf')
    t = 1 if len(asm_nodes) > 5000 else 0
    return pid, (asm_nodes, asm_edges), t, tgt


def load_graphs(args):
    csv_file = os.path.join(args.csv_dir, "function.csv")
    paths = []
    df = pd.read_csv(csv_file, header=0, index_col=False)
    id2sha = df.to_dict()['commit_id']
    id2tgt = df.to_dict()['target']

    with open(os.path.join(args.file_list), "r") as f:
        idx = [str(i.strip()) for i in f.readlines()]

    for i in range(0, len(idx)):
        id = idx[i]
        path = os.path.join(args.asm_dir, str(id)+ "_" + id2sha[int(id)] + ".s")
        if os.path.isfile(path):
            paths.append(
                (i, path, id2tgt[int(id)]))

    graphs_asm = {}
    large_asm  = {}
    tgt_asm = {}

    # debugging
    for i in range(0, len(paths)):
        pid, graph_asm, t, tgt = load_graph(paths[i])
        graphs_asm[pid] = graph_asm
        large_asm[pid] = t
        tgt_asm[pid] = tgt

    # del graphs
    total = len(graphs_asm.keys())
    large_asm_num = sum(large_asm.values())
    print(f'Large assembly: {large_asm_num}/{total} = {large_asm_num/total:.2%}')
    return graphs_asm, tgt_asm


def get_freqs(arguments):
    pid, training_solutions, asm = arguments
    freqs = defaultdict(int)
    for t in training_solutions[0]:
        freqs[t] += 1
    return freqs


def build_vocab(graphs, args, asm=False):

    def gen_args():
        for pid, training_solutions in graphs.items():
            yield pid, training_solutions, asm

    freqs = defaultdict(int)
    
    for argument in gen_args():
        freqs_local = get_freqs(argument)
        for k, v in freqs_local.items():
            freqs[k] += v

    vocab_count_list = sorted(
        freqs.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(map(lambda wc: wc[1], vocab_count_list))
    in_vocab = 0
    vocab = {}
    for i, (word, count) in enumerate(vocab_count_list):
        if count < args.min_freq_asm and asm:
            break
        vocab[word] = i + 1  # Reserve 0 for UNK
        in_vocab += count

    vocab[''] = 0

    print(f'Vocab size: {len(vocab)}/{len(vocab_count_list)}')
    print(f'Vocab coverage: {in_vocab}/{total} = {in_vocab/total:.2%}')

    return vocab


def preprocess(arguments):
    pid, (nodes, edges), vocab, asm = arguments
    nodes = np.asarray([vocab.get(t, 0) for t in nodes])
    edges = np.asarray(edges).T
    data = (nodes, edges)
    return pid, data


def preprocess_dataset(graphs, vocab, args,asm=False):
    dataset = [None] * len(graphs)
    def gen_args():
        for i, g in graphs.items():
            yield i, g, vocab, asm

    for argument in gen_args():
        pid, data = preprocess(argument)
        dataset[pid] = data

    return dataset


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    graphs_asm, tgt_asm = load_graphs(args)
    vocab_asm = build_vocab(graphs_asm, args, asm=True)
    dataset_asm = preprocess_dataset(graphs_asm, vocab_asm, args, asm=True)
    with open(os.path.join(args.output_dir, 'vocab_asm.pkl'), 'wb') as f:
        pickle.dump(vocab_asm, f)
    with open(os.path.join(args.output_dir, 'dataset_asm.pkl'), 'wb') as f:
        pickle.dump(dataset_asm, f)
    with open(os.path.join(args.output_dir, 'tgt_asm.pkl'), 'wb') as f:
        pickle.dump(tgt_asm, f)

if __name__ == "__main__":
    main()
