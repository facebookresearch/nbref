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
from asm_mips import Graphs_build_mips 
from asm_obj import Graphs_build_x86
import dgl

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-o', type=str, required=False, default="data/dataset-gnn")
    parser.add_argument('--split', '-s', type=str, required=False, default="data/split.pkl")
    parser.add_argument('--type', '-t', type=str, default='x86')
    parser.add_argument('--num-workers', '-p', type=int,
                        default=os.cpu_count())
    parser.add_argument('--asm-dir', '-a', type=str, required=False, default="data/obj")
    parser.add_argument('--min-freq-asm', '-t-asm', type=int, default=5)
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
    pid, solution, asm_dir, ty = arguments
    asms = os.path.join(asm_dir, solution)
    if ty == 'mips':
        asm_nodes, asm_edges = Graphs_build_mips(asms, 'pf')
    else:
        asm_nodes, asm_edges = Graphs_build_x86(asms, 'pf')
    t = 1 if len(asm_nodes) > 5000 else 0
    return pid, solution, (asm_nodes, asm_edges), t 

def load_graphs(args):
    asm_dir = args.asm_dir

    problem_num = 0
    paths = []
    for problem in os.listdir(asm_dir):
        problem_dir = os.path.join(asm_dir, problem)
        if os.path.isdir(problem_dir):
            problem_num += 1
            pid = int(problem) - 1
            for solution in os.listdir(problem_dir):
                if solution.endswith('.s'):
                    paths.append(
                        (pid, solution, problem_dir, args.type))

    graphs_asm = [{} for _ in range(problem_num)]
    large_asm  = [{} for _ in range(problem_num)]

    for i in range(0, len(paths)):
        pid, solution, graph_asm, t = load_graph(paths[i])
        graphs_asm[pid][solution[:-2]] = graph_asm

    total = sum([len(elem.items()) for elem in graphs_asm])
    large_asm_num = sum([sum(elem.values()) for elem in large_asm])
    print(f'Large assembly: {large_asm_num}/{total} = {large_asm_num/total:.2%}')
    return graphs_asm

def get_freqs(arguments):
    problem_graphs, training_solutions, asm = arguments
    freqs = defaultdict(int)
    for train_sol in training_solutions:
        nodes, _ = problem_graphs[train_sol]
        for t in nodes:
            freqs[t] += 1
    return freqs

def build_vocab(graphs, split, args, asm=False):
    training_set = split[0]

    def gen_args():
        for pid, training_solutions in training_set.items():
            yield graphs[pid], training_solutions, asm

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
    pid, problem_graphs, vocab, asm = arguments
    data = {}
    for solution, (nodes, edges) in problem_graphs.items():
        nodes = np.asarray([vocab.get(t, 0) for t in nodes])
        edges = np.asarray(edges).T
        data[solution] = (nodes, edges)
    return pid, data

def preprocess_dataset(graphs, vocab, args,asm=False):
    dataset = [None] * len(graphs)
    def gen_args():
        for i, g in enumerate(graphs):
            yield i, g, vocab, asm

    for argument in gen_args():
        pid, data = preprocess(argument)
        dataset[pid] = data

    with multiprocessing.Pool(args.num_workers) as pool:
        for pid, data in pool.imap_unordered(preprocess, gen_args()):
            dataset[pid] = data
    return dataset

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.split, 'rb') as f:
        split = pickle.load(f)

    graphs_asm = load_graphs(args)

    vocab_asm = build_vocab(graphs_asm , split, args, asm=True)
    dataset_asm = preprocess_dataset(graphs_asm, vocab_asm, args, asm=True)
    with open(os.path.join(args.output_dir, 'vocab_asm.pkl'), 'wb') as f:
        pickle.dump(vocab_asm, f)
    with open(os.path.join(args.output_dir, 'dataset_asm.pkl'), 'wb') as f:
        pickle.dump(dataset_asm, f)

if __name__ == "__main__":
    main()
