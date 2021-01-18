# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import os
import argparse
import numpy as np
import pickle, pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asm-dir', '-i', type=str, required=True)
    parser.add_argument('--mode', '-m', type=str,
                        choices=('p', 'i'), required=True)
    parser.add_argument('--train', '-t', type=int, nargs='+', required=False, default=None)
    parser.add_argument('--output', '-o', type=str, required=True)
    return parser.parse_args()


def collect_casses(asm_dir):
    casses = [None] * len(os.listdir(asm_dir))
    for problem in os.listdir(asm_dir):
        problem_dir = os.path.join(asm_dir, problem)
        if os.path.isdir(problem_dir):
            problem_casses = []
            casses[int(problem) - 1] = problem_casses
            for solution in os.listdir(problem_dir):
                if solution.endswith('.s'):
                    if os.path.getsize(os.path.join(problem_dir, solution)) == 0:
                        continue
                    problem_casses.append(solution[:-2])
    return casses


def main():
    args = parse_args()

    casses = collect_casses(args.asm_dir)
    datasets = [{} for _ in range(3)]
    if args.mode == 'p':
        p = [0, 64, 80, 104]
        for i in range(3):
            for pid in range(p[i], p[i+1]):
                datasets[i][pid] = casses[pid]
    elif args.mode == 'i':
        assert args.train is not None
        p = [0, 64, 80, 104]
        for i in range(1, 3):
            for pid in range(p[i], p[i+1]):
                datasets[i][pid] = casses[pid]
        for raw_pid in args.train:
            pid = raw_pid - 1
            datasets[0][pid] = casses[pid]
    else:
        raise Exception

    print(f'Number of problems: {len(datasets[0])}, {len(datasets[1])}, {len(datasets[2])}')
    print(f'[train, val, test]:',
          list(map(lambda x: sum(map(len, x.values())), datasets)))

    with open(args.output, 'wb') as f:
        pickle.dump(datasets, f)


if __name__ == "__main__":
    main()
