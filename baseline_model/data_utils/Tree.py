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
import numpy as np

# tree object from stanfordnlp/treelstm

def create_tree_from_flat_list(node_list, index=1):
	if index >= len(node_list)-1 or node_list[index-1] is None:
		return None
	# pdb.set_trace()
	d = node_list[index-1]
	l = index * 2
	r = l + 1
	tree = Tree(d)
	left_child  = create_tree_from_flat_list(node_list, l)
	right_child = create_tree_from_flat_list(node_list, r)

	if(left_child is not None):
		tree.add_child(left_child)
	if(right_child is not None):
		tree.add_child(right_child)
	return tree

class Tree(object):
	def __init__(self,value):
		self.value = value
		self.parent = None
		self.state = None
		self.idx = -1
		self.visited = False
		self.num_children = 0
		self.children = list()
		# self.childr = None
	def add_child(self, child):
		self.num_children += 1
		child.parent = self
		self.children.append(child)
		# if childl is not None:
	# def add_child_r(self, childr):
	# 	self.childr = childr
	# 	if childr is not None:
	# 		childr.parent = self
	# 		self.num_children += 1

	@staticmethod
	def get_root(node):
		if node.parent is None:
			return node
		else:
			return Tree.get_root(node.parent)

	def size(self):
		if getattr(self, '_size'):
			return self._size
		count = 1
		for i in range(self.num_children):
			count += self.children[i].size()
		self._size = count
		return self._size

	def depth(self):
	#if getattr(self, '_depth'):
	#    return self._depth
		count = 0
		if self.num_children > 0:
			for i in range(self.num_children):
				child_depth = self.children[i].depth()
				if child_depth > count:
					count = child_depth
			count += 1
		self._depth = count
		return self._depth

	def __iter__(self):
		if(len(self.children)>0):
			return self.children[0]
