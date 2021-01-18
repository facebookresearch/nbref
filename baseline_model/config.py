# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import argparse
import os

def get_arg_parser(title):
    parser = argparse.ArgumentParser()
    if title == 'vulnerability-detection':
        data_path = "../data/vul/"
        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument('--train', action='store_true' )
        common_parser.add_argument('--warmup','--n_warmup_steps', type=int, default=8000)

        common_parser.add_argument('--asm', action='store_true')
        common_parser.add_argument('--one_hot_label', action='store_true' )
        common_parser.add_argument('--disable_cuda', action='store_true' )
        common_parser.add_argument('--cache_path', type=str, default ='../data/dataset-vul/cache/')
        common_parser.add_argument('--batch-size', '-bs', type=int, default=8)
        common_parser.add_argument('--epoch-num', '-en', type=int, default=100)

        common_parser.add_argument('--seed', '-s', type=int, default=0)
        common_parser.add_argument('--disable-cuda', action='store_true')
        common_parser.add_argument('--dataset-dir', '-f', type=str, default="../data/dataset-vul")

        common_parser.add_argument('--hid_dim', type=int, default=256)
        common_parser.add_argument('--mem_dim', type=int, default=64)
        common_parser.add_argument('--n_layers', type=int, default=2) 
        common_parser.add_argument('--n_gnn_layers', type=int, default=2) 
        common_parser.add_argument('--max_tolerate_len', type=int, default=1000) #for small GPU memory
        common_parser.add_argument('--n_heads', type=int, default=8) 
        common_parser.add_argument('--pf_dim', type=int, default=512)
        common_parser.add_argument('--dropout', type=float, default=0.1)
        common_parser.add_argument('--graph_aug', action='store_true', default=True)
        common_parser.add_argument('--embedding_flag',type=int, default=1)
        common_parser.add_argument('--lr_ratio', type=float, default=0.1) 

        #control
        common_parser.add_argument('--training', action='store_true', default=True)
        common_parser.add_argument('--eval', action='store_true', default=False)
        common_parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_model')
        common_parser.add_argument('--log-dir', type=str, required=False, default='./log-dir')
        return common_parser

    elif title == 'similarity-check':
        data_path = "../data/dataset-gnn/"
        common_parser = argparse.ArgumentParser(add_help=False)
        model_path_group = common_parser.add_mutually_exclusive_group(
            required=False)
        model_path_group.add_argument('--save', type=str, default=None, required=False)
        model_path_group.add_argument('--load', type=str, default=None,required=False)
        common_parser.add_argument('--train', action='store_true' )
        common_parser.add_argument('--warmup','--n_warmup_steps', type=int, default=12000)
        common_parser.add_argument('--asm', action='store_true' )
        common_parser.add_argument('--disable_cuda', action='store_true' )
        common_parser.add_argument('--split', type=str, default = '../data/split.pkl')
        common_parser.add_argument('--cache_path', type=str, default = os.path.join(data_path,'cache/'))
        common_parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_model')
        common_parser.add_argument('--p', '-p', type=int, default=8)
        common_parser.add_argument('--k', '-k', type=int, default=4)

        common_parser.add_argument('--gamma', '-g', type=float, default=32)
        common_parser.add_argument('--margin', '-m', type=float, default=0.4)

        common_parser.add_argument('--batch-size', '-bs', type=int, default=32)
        common_parser.add_argument('--epoch-num', '-en', type=int, default=100)
        common_parser.add_argument('--train-epoch-size', '-tes',
                                type=int, default=1000)
        common_parser.add_argument('--valid-epoch-size', '-ves',
                                type=int, default=200)

        common_parser.add_argument('--seed', '-s', type=int, default=0)
        common_parser.add_argument('--output-size', '-os', type=int, default=128)
        common_parser.add_argument('--log-dir', type=str, required=False, default='./log-dir')
        common_parser.add_argument('--dataset-dir', '-f', type=str, default=data_path)

        #model control
        common_parser.add_argument('--hid_dim', type=int, default=256)
        common_parser.add_argument('--mem_dim', type=int, default=64)
        common_parser.add_argument('--n_layers', type=int, default=2) 
        common_parser.add_argument('--n_gnn_layers', type=int, default=2) 
        common_parser.add_argument('--n_heads', type=int, default=4) 
        common_parser.add_argument('--pf_dim', type=int, default=512)
        common_parser.add_argument('--max_tolerate_len', type=int, default=1200) #for small GPU mem 
        common_parser.add_argument('--dropout', type=float, default=0.1)
        common_parser.add_argument('--depth_dim', type=int, default=40)
        common_parser.add_argument('--output_dim', type=int, default=128)
        common_parser.add_argument('--graph_aug', action='store_true', default=True)
        common_parser.add_argument('--embedding_flag',type=int, default=1)
        common_parser.add_argument('--lr_ratio', type=float, default=0.15) 
        return common_parser

    if title == 'trf-tree':
        data_path = "../data/re/tst_1/"
        cache_ast_path = "../data/re/cache_tst_ast_1/"
        cache_asm_path = "../data/re/cache_tst_asm_1/"
        parser.add_argument('--gen_num', type=int, default = 10000)
        parser.add_argument('--golden_c_path', type=str, default = os.path.join(data_path, 'golden_c/samples.obj'))
        parser.add_argument('--input_g_path', type=str, default = os.path.join(data_path, 'golden_obj/graphs-3.npy'))
        parser.add_argument('--input_f_path', type=str, default = os.path.join(data_path, 'golden_obj/feats-3.npy'))
        parser.add_argument('--cache_path', type=str, default = cache_ast_path)
        parser.add_argument('--cache_asm_path', type=str, default = cache_asm_path)
        parser.add_argument('--training_flag', action='store_true', default=True)
        parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_model')

        # training argument
        parser.add_argument('--warmup', type=int, default=10000)
        parser.add_argument('--n_epoch', type=int, default=250)
        parser.add_argument('--bsz',type=int, default= 16)
        parser.add_argument('--sample_len', type=int, default=5) # samples partial tree to save your GPU memory
        parser.add_argument('--clip', type=int, default=1) 
        parser.add_argument('--label_smoothing', action='store_true', default=False)
        parser.add_argument('--lr_ratio', type=float, default=1.0)

        # model controls
        parser.add_argument('--hid_dim', type=int, default=256)
        parser.add_argument('--mem_dim', type=int, default=64)
        parser.add_argument('--n_layers', type=int, default=2) 
        parser.add_argument('--n_gnn_layers', type=int, default=2) 
        parser.add_argument('--n_heads', type=int, default=8) 
        parser.add_argument('--pf_dim', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--depth_dim', type=int, default=40)
        parser.add_argument('--output_dim', type=int, default=45)
        parser.add_argument('--graph_aug', action='store_true', default=False)

        # controls
        parser.add_argument('--dump_trace', action='store_true', default=True)
        parser.add_argument('--embedding_flag',type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--training', action='store_true', default=True)
        parser.add_argument('--eval', action='store_true', default=False)
        parser.add_argument('--parallel_gpu', action='store_true', default=True)
        parser.add_argument('--dist_gpu', action='store_true', default=True)
        parser.add_argument('--n_dist_gpu', type=int, default=4, choices=[2,4,8])  
        parser.add_argument('--log-dir', type=str, required=False, default='./log-dir')

    if title == 'dump-trace':
        data_path = "./data/re/tst_1"
        cache_ast_path = "./data/re/cache_tst_ast_1/"
        parser.add_argument('--gen_num', type=int, default = 10000)
        parser.add_argument('--golden_c_path', type=str, default = os.path.join(data_path, 'golden_c/samples.obj'))
        parser.add_argument('--input_g_path', type=str, default = os.path.join(data_path, 'golden_obj/graphs-3.npy'))
        parser.add_argument('--input_f_path', type=str, default = os.path.join(data_path, 'golden_obj/feats-3.npy'))
        parser.add_argument('--cache_path', type=str, default = cache_ast_path)
        parser.add_argument('--training_flag', action='store_true', default=True)
    return parser
