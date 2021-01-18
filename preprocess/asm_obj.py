# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import pdb, os, re
import numpy as np
import argparse
import json

x86_noise = ['.cfi','.set','.file','.section','.globl','.type', '.size', '.comm', '.ident',\
             '.string','.text','.end', '.mask', '.fmask','.LC\d+:$' ,'.weak','.quad', '.hidden',\
             '.align']

def find_header(content):
    #x86-64
    BBlock = re.findall(r'^.L[^C].*:$', content, re.MULTILINE)
    Func = re.findall(r'^\w+:$', content, re.MULTILINE)
    if 'main:' in Func: Func.remove('main:')
    return BBlock, Func

def content2list(content, BBlock, Funct):
    bow = []
    for line in content.splitlines():
        if not all([re.search(x, line) is None for x in x86_noise]):
            continue
        else:
            l_split = list(filter(None,re.split(r'[\s,()]',line))) 
            #handle BBlock
            if len(l_split) == 0:
                continue
            if BBlock is not None and (l_split[0] in BBlock): 
                l_split[0] = l_split[0][:-1]
            #handle Function renaming
            if Funct is not None:
                if (l_split[0] in Funct):
                    l_split[0] = 'usr_' + str(Funct.index(l_split[0]))
                elif l_split[0] == 'call' and (l_split[1]+ ':') in Funct:
                    l_split[1] = 'usr_' + str(Funct.index(l_split[1]+':'))
            bow += l_split
    return bow

def load_asm_file(path):
    asm = []
    rename = {}
    with open(path) as f:
        content = f.read()
    assert len(content) > 0
    BBlock, Funct = find_header(content)
    if(len(BBlock) == 0 ):
        print(path)
    assert len(BBlock) > 0
    asm = content2list(content,BBlock,Funct) 
    return asm    

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
  def __init__(self, asm_file, task):
    self.asm_file = asm_file 
    self.task = task

    self.regs = ['rax', 'rbx', 'rcx', 'rdx', 'rsp', 'rbp', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
    self.broken_types = ['shlq']
    self.op_types = [['ja', 'jb', 'jbe', 'jl', 'jge', 'jmp', 'je', 'jne', 'jnp', 'jns', 'js', 'jg', 'jp', 'jle', 'jae', 'callq', 'call', 'pushq', 'popq', 'jmpq', 'repz', 'retq', 'leaveq', 'cltq', 'nop', 'hlt', 'push', 'pop', 'cltd', 'cqto', ' '],
                     ['inc', 'mul', 'incw', 'incl', 'incq', 'div', 'idiv', 'idivl', 'dec', 'decw', 'neg', 'not', 'nopl', 'nopw', 'sar', 'seta', 'setl', 'setle', 'sete', 'setp', 'setnp', 'setne', 'setge', 'shr', 'setg', 'setae'],
                     ['adc', 'movups', 'movw', 'movdqa', 'movsb', 'movsbl', 'cmovbe', 'cmovl', 'cmovs', 'cmovns', 'cmovg', 'cmovne', 'cmovle',\
                       'cmovge', 'cmove', 'movzwl', 'movsl', 'mov', 'movd', 'movabs', 'movb', 'movq', 'movss', 'movsd', 'movslq', 'movaps', \
                       'movswq', 'movswl', 'movapd', 'movzbl', 'cmpw', 'cmp', 'cmpb', 'cmpq', 'lea',  'add', 'addw', 'addl', 'addq', 'addss',\
                       'addsd', 'bt', 'sub', 'subl', 'subss', 'subsd', 'sbb', 'mulss', 'mulsd', 'divsd', 'divss', 'test', 'pxor', 'xor', 'xorpd',\
                       'xorps', 'orps', 'and', 'andps', 'andnps', 'or', 'sar', 'cmpl', 'movl', 'xorl', 'subq', 'shl', 'xchg', 'ucomiss',\
                       'ucomisd', 'cvtsd2ss', 'cvtss2sd', 'cvtsi2sd', 'cvttsd2si', 'cvttss2si', 'cvtsi2ss', 'cvtsi2ssl', 'cvtsi2sdl', 'cvtsi2sdq',\
                       'sqrtsd', 'stos', 'punpcklqdq', 'andpd', 'maxsd', 'rol', 'scas', "movsbq", "cmpnless"],
                     ['imul', 'pinsrd']]
    # combine subarrays of self.op_types into one and build id mapping
    self.op_set = [op for op_ls in self.op_types for op in op_ls]
    self.op2id = {}
    self.id2op = {}
    for op in self.op_set:
      self.op2id[op] = len(self.op2id)
      self.id2op[len(self.op2id)-1] = op

    # edge types we support
    # self.edge_types[0] are edges between type 1 nodes
    # self.edge_types[1] are edges between nodes of type 1 and 2
    # self.edge_types[2] are edges between nodes of type 2 and 3
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
    with open(self.asm_file, 'r') as fin:
      lines = fin.readlines()
      # filter out lines without any operation
      # e.g.  ddd:   00 00 00
      lines = [line for line in lines if len(line) > 30]
      # filter out lines that are not instructions
      lines = [line for line in lines if line[0] == ' ']
      self.lines = lines

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
    for i, op in enumerate(line.split()):
      if op in self.op2id:
        #op_id = self.op2id[op]
        if i+1 < len(line.split()):
          operands_str = line.split()[i+1]
        else:
          operands_str = ''
        break

    # create operand nodes from operands_str
    # i is the number of operands attached
    for i in range(len(self.op_types)):
      if op in self.op_types[i]:
        self.get_nodes(operands_str, i, op)
        break
    return

  def get_nodes(self, operands_str, num_op, op):
    # get nodes for operands
    # if no operand, no nothing
    # 1 operand

    # 2 operands
    if num_op == 2:
      operands = operands_str.split('),')
      if len(operands) == 2:
        src, tgt = operands
      else:
        try:
          operands = operands_str.split(',', 1)
          src, tgt = operands
        except:
          # num_op = 1
          return 

      # clean up string
      src = self.__clean_name__(src)
      tgt = self.__clean_name__(tgt)
      self.__get_nodes__(src, is_tgt=False)
      self.__get_nodes__(tgt, is_tgt=True)
    if num_op == 1:
      tgt = operands_str
      # clean up string
      tgt = self.__clean_name__(tgt)
      self.__get_nodes__(tgt, is_tgt=True)

    self.num_ins += 1

  def __ins_node__(self):
    for i, line in enumerate(self.lines):
      flag = 0
      op_id = -1
      for op in line.split():
        if op in self.op2id:
          flag = 1
          op_id = self.op2id[op]
          break
      if not flag:
        # if not flagged, this is a operation we are not supporting, add that
        print(line)
        self.lines[i] += 'nop'
        op_id = self.op2id['nop']

      # add instruction node
      pc = line.split(':')[0].strip()
      node = Node(name='ins_'+self.id2op[op_id],
                  node_id=self.num_nodes,
                  node_type='ins',
                  ins_id=i,
                  pc=pc)

      # add subarray for this pc and add node
      self.nodes.append([])
      self.nodes[i].append(node)

      # build pc to node mapping for building jump edges
      self.pc2node[pc] = node

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
      for j, op in enumerate(line.split()):
        if self.__get_op_type__(op) == 0 and len(line.split()) > j+1 and '%' not in line.split()[j+1] and '$' not in line.split()[j+1]:
          pc = line.split()[j+1].strip()
          if isInt(pc):
            if pc in self.pc2node:
              self.__add_edge__(Edge(name='',
                                     edge_type=self.edge_type_2_id['next-ins'],
                                     src=self.nodes[i][0].id,
                                     tgt=self.pc2node[pc].id))
            break

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
      # add non-mem-reg node
      reg = operand
      reg_node = self.__add_node__(Node(name=reg,
                                        node_id=self.num_nodes,
                                        node_type='reg',
                                        ins_id=self.num_ins,
                                        is_var=True))

      self.__add_edge__(Edge(name='',
                             edge_type=self.edge_type_2_id['non-mem-reg'],
                            #  src=pseudo_node.id,
                             src=self.num_ins,
                             tgt=reg_node.id))
      # pseudo_node.edges[self.edge_type_2_id['non-mem-reg']] = reg_node

      # add usage edges
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
                                             node_type='offset',
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
                                           node_type='base',
                                           ins_id=self.num_ins,
                                           is_var=True))
        self.__add_edge__(Edge(name='',
                               edge_type=self.edge_type_2_id['mem-base'],
                               src=self.num_ins,
                               tgt=base_node.id))
      # array indexing
      else:
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

def Graphs_build_x86(asm_file, task):
  gb = Graph_builder(asm_file, task)
  subset = True

  # print('==============Nodes=============')
  selected_nodes = []
  selected_nodes_dict = {}
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

  # print('==============Edges=============')
  selected_edges = []
  for edge in gb.edges:
    src = gb.id2node[edge.src]
    tgt = gb.id2node[edge.tgt]
    selected_edges.append((src.id, edge.type, tgt.id))

  for nodes in gb.nodes:
      for node in nodes:
        selected_nodes_dict[int(node.id)] = node.name
  selected_nodes_tuple_list = sorted(selected_nodes_dict.items(), key=lambda k: k[0], reverse=False)

  del gb, selected_nodes
  return list(zip(*selected_nodes_tuple_list))[1], selected_edges 
