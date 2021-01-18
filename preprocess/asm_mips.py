# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
"""build graph on assembly code"""

import numpy as np
import argparse
import json, pdb
import operator, re, os
from torch import nn

def isInt(s):
  try:
      int(s, 16)
      return True
  except ValueError:
      return False

def int2bits(x):
  return [int(b) for b in "{:064b}".format(np.abs(x))]

class Node(object):
  '''class for node

  In our graph, there are mainly three types of nodes:
    1. instruction nodes
      each instruction node comes with PC to match with dynamic features
    2. pseudo nodes for hierarchical structure inside each instruction
    3. variable / operand nodes, including register and constants
  '''

  def __init__(self, name, node_id, node_type, ins_id, pc=None, is_var=False):
    self.id = node_id
    self.type = node_type
    self.name = name
    if isinstance(name,list):
      print(name)
    self.ins_id = ins_id
    self.pc = pc
    self.is_var = is_var
    self.edges = {}

  # node information
  def __repr__(self):
    ret = ''
    if self.pc:
      ret += str(self.pc)
    ret += ' name: ' + self.name
    ret += ' id: ' + str(self.id)
    #ret += ' is variable: ' + str(self.is_var)
    return ret

class Edge(object):
  '''class for edge

  In our graph, there are mainly 2 types of edges:
    1. control flow edges between instruction nodes
      either next instruction or jump instruction, both of type 'next-ins'
    2. others, connecting nodes of type 1 and 2, and 2 and 3
  '''
  def __init__(self, name, edge_type, src, tgt):
    self.name = name
    self.type = edge_type
    self.src = src
    self.tgt = tgt

  def output(self):
    return (self.src, self.type, self.tgt)

  # edge information, in the GGNN input format of (src, type, tgt).
  def __repr__(self):
    return '(' + str(self.src) + ', ' + str(self.type) + ', ' + str(self.tgt) + ')'


class Graph_builder(object):
  def __init__(self, asm_file, feature_file, task, raw=False):
    self.asm_file = asm_file
    self.feature_file = feature_file
    self.task = task
    self.raw = raw
    self.br_types = ['j', 'jr', 'beqz']
    self.op_types = [['nop'],
                     ['jal', 'j', 'jr', 'mfhi', 'bc1t'],
                     ['beqz', 'ldc1', 'sdc1', 'sw', 'lw','move', 'mov.d', 'cvt.d.w', 'mtc1',\
                        'lui', 'bnez', 'bltz','bne', 'bgez', 'beq', 'negu', 'blez', 'mfc1', 'bgtz', 'mult'],
                      ['add.d', 'addiu', 'addu','div.d', 'mul', 'sub.d', 'subu', 'divu', 'teq', 'ori', \
                        'and', 'slt', 'slti', 'sltiu', 'sra', 'sll', 'andi', 'xori', 'sltu', 'or', 'xor'\
                        ,'srl', 'div'],
                      ['mflo','lbu','sb', 'lb', 'lhu', 'sh','srav', 'swc1']]
    # combine subarrays of self.op_types into one and build id mapping
    self.op_set = [op for op_ls in self.op_types for op in op_ls]
    self.op2id = {}
    self.id2op = {}
    for op in self.op_set:
      self.op2id[op] = len(self.op2id)
      self.id2op[len(self.op2id)-1] = op
    self.edge_types = [['next-ins'],
                       ['ins-src', 'ins-tgt'],
                       ['non-mem-reg', 'mem-base', 'mem-index-stride', 'mem-stride', 'mem-index', 'mem-offset', 'mem-start'],
                       ['last-read', 'last-write']]
    # combine subarrays of self.edge_types into one and build id mapping
    self.edge_type_set = [et for ets in self.edge_types for et in ets]
    self.edge_type_2_id = {}
    for edge_type in self.edge_type_set:
      self.edge_type_2_id[edge_type] = len(self.edge_type_2_id)

    # self.edges is a 1d array
    self.edges = []
    # self.nodes is supposed to be a 2d array of shape (num_ins, _). Each
    # subarray contains nodes of one pc
    self.nodes = []
    self.num_ins = 0
    self.num_nodes = 0
    # for building control flow edges through jump instructions
    self.pc2node = {}
    # for selecting subgraphs
    self.id2node = {}
    # for usage edges
    self.src_vnode = {}
    self.tgt_vnode = {}

    # start processing assembly to graph
    self.read_file()
    # create nodes of type 1
    self.__ins_node__()
    # create nodes of type 2 and 3
    for line in self.lines:
      self.get_opid(line)

  def read_file(self):
    global flag
    flag = 0
    strlist = []
    with open(self.asm_file, 'r') as fin:
      for lines in fin.readlines():
        if "#" in lines:
          lines = lines.split("#")[0]
        l = lines.split()
        if len(l)==0:
          continue
        if l[0]=="main:":
          flag=1
        if len(l)>=2 and l[0]=='.end' and l[1]== 'main':
          strlist.append(lines)
          break
        if flag==1:
          strlist.append(lines)
          continue

      self.lines = strlist


  def __get_op_type__(self, line):
    # get instruction type in self.op_types
    # 0: jump, no operand
    # 1: 1 operand
    # 2: 2 operands
    for op in line.split():
      for i in range(len(self.op_types)):
        if op in self.op_types[i]:
          return i
    return -1

  def get_opid(self, line):
    # get operation id and its operands in string if any
    if '$BB0' in line.split()[0]:
        self.num_ins +=1
        return
    for i, op in enumerate(line.split()):
      if op in self.op2id:
        #op_id = self.op2id[op]
        if i+1 < len(line.split()):
          operands_str = line.split()[i+1:]
        else:
          operands_str = ''
        break

    # create operand nodes from operands_str
    # i is the number of operands attached
    for i in range(len(self.op_types)):
      if op in self.op_types[i]:
        if i == 0:
          self.get_nodes(operands_str, i+1, line)
        else:
          self.get_nodes(operands_str, i, line)
        break
    return

  def get_nodes(self, operands_str, num_op, line):
    # get nodes for operands
    # if no operand, no nothing
    # 1 operand
    if num_op == 1:
      tgt = operands_str
      # clean up string
      if len(tgt) != 0 :
        assert (len(tgt) == 1 ), 'len tgt must be 1'
        self.__get_nodes__(tgt, is_tgt=True)
    # 2 operands
    elif num_op == 2:
      operands_str = "".join(operands_str)
      operands = operands_str.split('),')
      if len(operands) == 2:
        src, tgt = operands
      else:
        operands = operands_str.split(',', 1)
        src, tgt = operands

      self.__get_nodes__(src, is_tgt=False)
      self.__get_nodes__(tgt, is_tgt=True)
    elif num_op == 3:
      operands_str = "".join(operands_str)
      operands = operands_str.split('),')
      operands = operands_str.split(',')

      tgt, src1, src2 = operands
      self.__get_nodes__(tgt, is_tgt=True)
      self.__get_nodes__(src1, is_tgt=False)
      self.__get_nodes__(src2, is_tgt=False)

    self.num_ins += 1

  def __ins_node__(self):
    def prep(lines):
      pc = -4
      self.pcmap = {}
      self.pcmap_Name={}
      self.pcmap_Rev={}
      flag = 0
      filter_lines = []
      b = -1
      for ind, line in enumerate(self.lines):
        if "main:" in line:
            flag = 1
        if ("leave" in line or "ret" in line) and flag == 1:
            break
        if flag == 1:
          lsplit = line.split()
          if '$BB' in lsplit[0]:
              pc+=4
              b +=1
              self.pcmap[lsplit[0][:-1]] = pc
              self.pcmap_Name[lsplit[0][:-1]] = str(b)
              filter_lines.append(line)
          for ii, op in enumerate(lsplit):
            if (ii == 0) and (op not in self.op2id) and ('$BB' not in op)  \
                 and ('#' not in op) and ('.' not in op) and ('main' not in op):
              print(op + ' not supported!')
            if op in self.op2id:
              filter_lines.append(line)
              pc+=4
      return  filter_lines

    pc = -4
    i = -1
    b = -1
    self.lines = prep(self.lines)
    for ind, line in enumerate(self.lines):
      flag = 0
      op_id = -1
      lsplit = line.split()

      for op in lsplit:
        if op in self.op2id:
          flag = 1
          op_id = self.op2id[op]
          pc+=4
          i +=1
          break

      if '$BB' in lsplit[0]:
          flag = 1
          pc+=4
          i +=1
          b +=1

      if not flag:
        # if not flagged, this is a operation we are not supporting, add that
        print(line)
        continue
        if not isInt(line[-3]):
          raise Exception('Instruction need to be added')
        else:
          self.lines[i] += 'nop'
          op_id = self.op2id['nop']

      # add instruction node
      if '$BB' in lsplit[0]:
        node = Node(name=lsplit[0][:-1],
                      node_id=self.num_nodes,
                      node_type='ins',
                      ins_id=i,
                      pc=str(pc))

      else:
        node = Node(name='ins_'+self.id2op[op_id],
                    node_id=self.num_nodes,
                    node_type='ins',
                    ins_id=i,
                    pc=str(pc))

      # add subarray for this pc and add node
      self.nodes.append([])
      self.nodes[i].append(node)

      # build pc to node mapping for building jump edges
      self.pc2node[str(pc)] = node

      # for selecting subgraphs
      self.id2node[self.num_nodes] = node
      self.num_nodes += 1

      # add control-flow edges for next instructions
      if i > 0:
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['next-ins'],
                               src=self.nodes[i-1][0].id,
                               tgt=self.nodes[i][0].id))

    # add control-flow edges for jump instructions in self.op_types[0]
    for i, line in enumerate(self.lines):
      op  = line.split()[0]
      if  op in self.br_types and len(line.split()) > 1 and '$BB' in line:
        trg_br = line.split()[-1].strip()

        if '$BB' in trg_br:
          pc = self.pcmap[trg_br]
          assert (self.pc2node[str(pc)].name == trg_br), 'pc indexing incorrect!'
          if str(pc) in self.pc2node.keys():
            self.__add_edge__(Edge(name='',
                                    edge_type=self.edge_type_2_id['next-ins'],
                                    src=self.nodes[i][0].id,
                                    tgt=self.pc2node[str(pc)].id))

  def __clean_name__(self, name):
    # clean string for future matching with pin trace and embedding
    # initialization
    if name.find('%') != -1:
      name = name.replace('%', '')
      if name[-1] == 'd':
        name = name[:-1]
      if name[0] == 'e':
        name = 'r' + name[1:]

    name = name.replace('$', '')
    name = name.replace('%', '')
    name = name.replace('0x', '')
    return name

  def __add_node__(self, node):
    self.nodes[self.num_ins].append(node)
    self.id2node[self.num_nodes] = node
    self.num_nodes += 1
    return node

  def __add_edge__(self, edge):
    self.edges.append(edge)

  def __get_nodes__(self, operand, is_tgt):
    # parsing operands
    name = 'tgt' if is_tgt else 'src'

    # First case: simple register value = non-memory operand
    if '('  not in operand and ')' not in operand:
      if isinstance(operand, list):
        reg = operand[0]
      else:
        reg = operand
      reg_node = self.__add_node__(Node(name=reg,
                                        node_id=self.num_nodes,
                                        node_type='reg-'+name,
                                        ins_id=self.num_ins,
                                        is_var=True))

      self.__add_edge__(Edge(name='',
                             edge_type=self.edge_type_2_id['non-mem-reg'],
                             src=self.num_ins,
                             tgt=reg_node.id))

      # add data dependency edges
      if self.raw:
        if not is_tgt and reg_node.name in self.src_vnode:
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-read'],
                                  src=reg_node.id,
                                  tgt=self.src_vnode[reg_node.name].id))
            reg_node.edges[self.edge_type_2_id['last-read']] = self.src_vnode[reg_node.name]
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-read'],
                                  src=self.src_vnode[reg_node.name].id,
                                  tgt=reg_node.id))
            self.src_vnode[reg_node.name].edges[self.edge_type_2_id['last-read']] = reg_node
            self.src_vnode[reg_node.name] = reg_node
        elif not is_tgt:
            self.src_vnode[reg_node.name] = reg_node
        if is_tgt and reg_node.name in self.tgt_vnode:
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-write'],
                                  src=reg_node.id,
                                  tgt=self.tgt_vnode[reg_node.name].id))
            reg_node.edges[self.edge_type_2_id['last-write']] = self.tgt_vnode[reg_node.name]
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-write'],
                                  src=self.tgt_vnode[reg_node.name].id,
                                  tgt=reg_node.id))
            self.tgt_vnode[reg_node.name].edges[self.edge_type_2_id['last-write']] = reg_node
            self.tgt_vnode[reg_node.name] = reg_node
        elif is_tgt:
            self.tgt_vnode[reg_node.name] = reg_node
        if not is_tgt and reg_node.name in self.src_vnode:
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-read'],
                                  src=reg_node.id,
                                  tgt=self.src_vnode[reg_node.name].id))
            reg_node.edges[self.edge_type_2_id['last-read']] = self.src_vnode[reg_node.name]
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-read'],
                                  src=self.src_vnode[reg_node.name].id,
                                  tgt=reg_node.id))
            self.src_vnode[reg_node.name].edges[self.edge_type_2_id['last-read']] = reg_node
        elif not is_tgt:
            self.src_vnode[reg_node.name] = reg_node
        if is_tgt and reg_node.name in self.tgt_vnode:
            # pdb.set_trace()
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-write'],
                                  src=reg_node.id,
                                  tgt=self.tgt_vnode[reg_node.name].id))
            reg_node.edges[self.edge_type_2_id['last-write']] = self.tgt_vnode[reg_node.name]
            self.__add_edge__(Edge(name='',
                                  edge_type=self.edge_type_2_id['last-write'],
                                  src=self.tgt_vnode[reg_node.name].id,
                                  tgt=reg_node.id))
            self.tgt_vnode[reg_node.name].edges[self.edge_type_2_id['last-write']] = reg_node
        elif is_tgt:
            self.tgt_vnode[reg_node.name] = reg_node


    # Second case: memory operation: src = load, tgt=store
    else:
      # getting offset if any
      # in the format offset(base)
      # base can be expanded to (start, index, stride)
      if operand[0] == '(':
        # no offset
        base = operand
      else:
        # with offset
        offset, base = operand.rsplit('(', 1)
        # add offset node
        offset_node = self.__add_node__(Node(name=offset,
                                             node_id=self.num_nodes,
                                             node_type='offset'+ name,
                                             ins_id=self.num_ins,
                                             is_var=True))
        # add edges between type 2 and 3 (offset)
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-offset'],
                               src=self.num_ins,
                               tgt=offset_node.id))
        base = '(' + base

      if base[-1] != ')':
        base += ')'
      base = base[1:-1].split(',')

      # single base
      if len(base) == 1:
        base = base[0]
        base_node = self.__add_node__(Node(name=base,
                                           node_id=self.num_nodes,
                                           node_type='base'+name,
                                           ins_id=self.num_ins,
                                           is_var=True))
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-base'],
                               src=self.num_ins,
                               tgt=base_node.id))
      # array indexing
      else:
        if len(base) == 2:
          base += ['1']
        assert (len(base) == 3), 'base operation incorrect!'
        new_types = ['base', 'idx-stride', 'start', 'index', 'stride']
        new_nodes = []

        # add pseudo nodes
        for i in range(2):
          new_node = self.__add_node__(Node(name='pseudo_'+new_types[i],
                                            node_id=self.num_nodes,
                                            node_type=new_types[i],
                                            ins_id=self.num_ins))
          new_nodes.append(new_node)

        # add new nodes
        for i in range(3):
          new_node = self.__add_node__(Node(name=base[i],
                                            node_id=self.num_nodes,
                                            node_type=new_types[i+2],
                                            ins_id=self.num_ins,
                                            is_var=True))
          new_nodes.append(new_node)

        # add new edges:
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-index'],
                               src=new_nodes[1].id,
                               tgt=new_nodes[3].id))
        new_nodes[1].edges[self.edge_type_2_id['mem-index']] = new_nodes[3]
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-stride'],
                               src=new_nodes[1].id,
                               tgt=new_nodes[4].id))
        new_nodes[1].edges[self.edge_type_2_id['mem-stride']] = new_nodes[4]
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-start'],
                               src=self.num_ins,
                               tgt=new_nodes[2].id))
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-index-stride'],
                               src=self.num_ins,
                               tgt=new_nodes[1].id))

def Graphs_build_mips(asm_file, task):
  gb = Graph_builder(asm_file, None, task)
  subset = True

  selected_nodes = []
  selected_nodes_dict = {}
  for nodes in gb.nodes:
      selected_nodes.append(nodes)

  selected_nodes_tuple_list = sorted(selected_nodes_dict.items(), key=lambda k: k[0], reverse=False)
  index_ins = []
  for i, (index, ins) in enumerate(selected_nodes_tuple_list):
    if 'ins' in ins:
      index_ins.append(index)
  # renaming node
  if subset:
    new_id2node = {}
    rename = {}
    for i in range(len(selected_nodes)):
      for j in range(len(selected_nodes[i])):
        if selected_nodes[i][j] not in rename:
          rename[selected_nodes[i][j].id] = len(rename)
          selected_nodes[i][j].id = rename[selected_nodes[i][j].id]
          # get new id to node mapping, for later use of getting node name
          # does not overwrite previous
          new_id2node[selected_nodes[i][j].id] = selected_nodes[i][j]
    # give it to gb to pass in dynamic function
    gb.new_id2node = new_id2node
#   print(selected_nodes)
  for nodes in gb.nodes:
      for node in nodes:
        selected_nodes_dict[int(node.id)] = node.name

  selected_nodes_tuple_list = sorted(selected_nodes_dict.items(), key=lambda k: k[0], reverse=False)

  # print('==============Edges=============')
  selected_edges = []
  for edge in gb.edges:
    src = gb.id2node[edge.src]
    tgt = gb.id2node[edge.tgt]
    selected_edges.append((src.id,edge.type,tgt.id))

  del gb, selected_nodes
  return list(zip(*selected_nodes_tuple_list))[1], selected_edges #, index_ins

def main():
  parser = argparse.ArgumentParser()
  data_assembly_path = ""
  parser.add_argument('-a', '--asm',
                      help='input assembly folder',
                      required=False, type=str, default=data_assembly_path)
  parser.add_argument('-n', '--num',
                      help='input assembly folder',
                      required=False, type=int, default=10000)
  parser.add_argument('-f', '--feature',
                      help='input feature file',
                      required=False, type=str)
  parser.add_argument('-t', '--task',
                      help='task, branch prediction or prefetching',
                      required=False, type=str, default='pf')
  parser.add_argument('-b', '--binary',
                      help='binary feature',
                      action='store_true', default=False)
  parser.add_argument('-hid', '--hidden_dim',
                      help='hidden dimension',
                      required=False, type=int, default=416)
  args = parser.parse_args()

  graphs = []
  feats  = []
  featuremap = {}
  max_edges = 0
  max_nodes = 0
  token = 1
  
  in_folder  = os.path.join(args.asm, 'rand_assembly')
  out_folder = os.path.join(args.asm, 'golden_obj')

  if not os.path.exists(out_folder):
    os.makedirs(out_folder)

  for nn in range(0,args.num):
    path_ = os.path.join(in_folder,'rd_'+str(nn)+'.s')
    gb = Graph_builder(path_, args.feature, args.task)

    subset = False

    print('==============Nodes=============')
    selected_nodes = []
    for nodes in gb.nodes:
        selected_nodes.append(nodes)

    # renaming node
    if subset:
      new_id2node = {}
      rename = {}
      for i in range(len(selected_nodes)):
        for j in range(len(selected_nodes[i])):
          if selected_nodes[i][j] not in rename:
            rename[selected_nodes[i][j].id] = len(rename)
            selected_nodes[i][j].id = rename[selected_nodes[i][j].id]
            # get new id to node mapping, for later use of getting node name
            # does not overwrite previous
            new_id2node[selected_nodes[i][j].id] = selected_nodes[i][j]
      # give it to gb to pass in dynamic function
      gb.new_id2node = new_id2node
    print(selected_nodes)

    print('==============Edges=============')
    selected_edges = []
    for edge in gb.edges:
      src = gb.id2node[edge.src]
      tgt = gb.id2node[edge.tgt]
      selected_edges.append(edge)

    # renaming edge
    if subset:
      edge_rename = {}
      for i in range(len(selected_edges)):
        selected_edges[i].src = rename[selected_edges[i].src]
        selected_edges[i].tgt = rename[selected_edges[i].tgt]
        if selected_edges[i].type not in edge_rename:
          edge_rename[selected_edges[i].type] = len(edge_rename)
        selected_edges[i].type = edge_rename[selected_edges[i].type]
    print(selected_edges)
    graph = [edge.output() for edge in gb.edges]
    graph.sort(key = operator.itemgetter(1))

    feat = []
    feat_aux = []

    id2idx = {}
    idx = 0
    for elem in gb.nodes:
      for node in elem:
          if isInt(node.name):
              node.name = abs(int(node.name))
          if node.name not in featuremap.keys():
               featuremap[node.name] = token
               token +=1
          if 'tgt' in node.type or 'src' in node.type:
            if 'tgt' in node.type:
              feat_aux.append(1)
            elif 'src' in node.type:
              feat_aux.append(2)
          else:
            feat_aux.append(0)

          feat.append(featuremap[node.name])
          id2idx[node.id] = idx
          idx+=1

    for i, elem in enumerate(graph):
      graph[i] = (id2idx[elem[0]],elem[1],id2idx[elem[2]])

    if max_edges < len(graph):
        max_edges = len(graph)
    if max_nodes < len(feat):
        max_nodes = len(feat)
    graphs.append(graph)
    feats.append(feat)

  print('graph shape in [num_graphs, num_edges, 3]: ', np.shape(graphs))
  print('feature shape in [num_graphs, num_nodes, 64]: ', np.shape(feats))

  np.save(os.path.join(out_folder, 'graphs-3'), graphs)
  np.save(os.path.join(out_folder, 'feats-3'), feats)


if __name__ == '__main__':
  main()
