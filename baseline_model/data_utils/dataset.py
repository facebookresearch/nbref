# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from abc import ABC, abstractmethod
import numpy as np
import torch, pdb


class Dataset(ABC):
    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def collate(self, batch):
        pass

    def _get_data_split(self, split):
        dataset_asm = self.get_dataset()
        data = [[] for _ in range(len(split))]
    
        pids = sorted(split.keys())
        for i, pid in enumerate(pids):
            solutions = split[pid]
            problem_split_data = data[i]
            for sol in solutions:
                if self.max_len is not None and len(dataset_asm[pid][sol][0]) < self.max_len:
                    problem_split_data.append((dataset_asm[pid][sol], i))
        return data

    def get_pk_sample_generator_function(self, split, p, k):
        data = self._get_data_split(split)
        def gen():
            while True:
                pids = np.random.choice(len(data), p, replace=False)
                batch = []
                for pid in pids:
                    solutions = data[pid]
                    sids = np.random.choice(len(solutions), k, replace=False)
                    for sid in sids:
                        sol_data, _ = solutions[sid]
                        batch.append(sol_data)
                yield self.collate(batch)
        return gen

    def get_vul_sample_generator_function(self, trn_vld_flag, batch_size):
        def gen():
            while True:
                if trn_vld_flag == 'train' :
                    pids = np.random.choice(len(self.train_index), batch_size, replace=False)
                elif trn_vld_flag == 'valid':
                    pids = np.random.choice(len(self.valid_index), batch_size, replace=False)
                yield self.collate(pids)
        return gen
        
    def get_data_generator_function(self, split, batch_size, shuffle=False):
        data = []
        for solutions in self._get_data_split(split):
            data += solutions

        num_batches = len(data) // batch_size
        if len(data) % batch_size > 0:
            num_batches += 1
        def gen():
            iter_data = data.copy()
            if shuffle:
                np.random.shuffle(iter_data)
            labels = []
            batch = []
            for sol_data, label_id in iter_data:
                    batch.append(sol_data)
                    labels.append(label_id)
                    if len(batch) == batch_size:
                        yield self.collate(batch), torch.tensor(labels)
                        labels = []
                        batch = []
            if len(batch) > 0:
                yield self.collate(batch), torch.tensor(labels)
        return gen, num_batches
