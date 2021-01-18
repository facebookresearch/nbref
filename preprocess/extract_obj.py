# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import argparse
import multiprocessing
import itertools
import os
import tqdm
import subprocess
import re

ignore_x86_asm_block = [".text", "deregister_tm_clones", "register_tm_clones", "register_tm_clones", "__do_global_dtors_aux", \
                    "frame_dummy", "_Z41__static_initialization_and_destruction_0ii", "_GLOBAL__sub_I_main", "_start", \
                    "__libc_csu_init", "__libc_csu_fini" , "_ZS","_GLOBAL", "_ZN"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poj-dir', '-j', type=str, required=False,
                         help='The directory containing POJ-104', default="data/ProgramData")
    parser.add_argument('--filter-list', '-f', type=str, default="./preprocess/filter_list.txt")
    parser.add_argument('--asm_type', '-asm_t', type=str, required=False,
                        help='The output directory', default='mips',choices=['x86','mips'])
    parser.add_argument('--output-asm-dir', '-asm', type=str, required=False,
                        help='The output asm directory',default="data/obj-mips-2/")                        
    parser.add_argument('--num-workers', '-p', type=int, default=None,
                        required=False, help='Number of workers to use')
    args = parser.parse_args()
    return args

def check_sense(arr):
    sensitive_words = ['array']
    for elem in sensitive_words:
        if elem in arr:
            arr = re.sub(elem, elem + '_loc',arr)
    return arr

def check_arr_def(arr):
    return re.findall(r'.*(int|float|char|double|node|const int|struct|}).+?([\w\.-]*)\[([a-zA-Z_]+[0-9]*)\](.*)',arr)

def check_arr_def_dim(arr):
    return re.findall(r'([\w\.-]*)\[([a-zA-Z_]+[0-9]*)\](.*)',arr)

def check_malloc_def(arr):
    return re.findall(r'alloc\(([a-zA-Z_]*)(\)|\*)',arr)

def check_var(arr):
    return re.findall(r'\W*(for|int|const int|long|short|char)\s*(.*);',arr, re.MULTILINE)

def extract_from_solution(arg):
    solution_path, output_asm_path, asm_type = arg
    os.makedirs(os.path.split(output_asm_path)[0], exist_ok=True)

    output_c_path = output_asm_path[:-3] +'c'
    output_bc_path = output_asm_path[:-3] +'bc'
    output_obj_path = output_asm_path[:-3] +'o'
    output_asm_path = output_asm_path[:-3] +'s'

    lcs = []
    errs = []
    pre_df = set(['LEN'])
    pre_var = []
    total_large_asm = 0
    try:
        with open(solution_path, 'rb') as f:
            content = f.read()
        try:
            raw = content.decode('utf-8')
        except:
            raw = content.decode('iso-8859-1')
            
        z = check_var(raw)

        #get variable
        for elem in z:
            l = list(filter(None, re.split('[, = [;]', elem[1])))
            pre_var += l

        #get definition
        z =  check_malloc_def(raw)
        for elem in z:
            if not (elem[0].islower() and len(elem[0]) == 1):
                pre_df.add(elem[0])
        z =  check_arr_def(raw)
        for elem in z:
            if not (elem[2].islower() and len(elem[2]) == 1):
                pre_df.add(elem[2])
            pp = check_arr_def_dim(elem[3])
            if len(pp)!=0:
                if not (pp[0][1].islower() and len(pp[0][1]) == 1):
                    pre_df.add(pp[0][1])

        if 'PI' not in pre_var:
            raw = "\n#define PI 3.14\n" +raw
        pre_df = pre_df.difference(set(pre_var))
        file_c = open(output_c_path, "w")
        raw = check_sense(raw)

        # Fix-1 : error null 
        raw = re.sub('null','NULL',raw)
        raw = re.sub('Null','NULL',raw)
        
        # Fix-2 : missing definition
        raw = ('\n').join(['#define '+ elem +' 100' for elem in pre_df]) + '\n' + raw
        raw = "\n#define INT_MAX 2147483647\n #define INT_MIN -2147483648\n#define MAX 100\n" + raw
        
        # Fix-3 : check compilers
        if asm_type =='x86':
            if('cin' in raw or 'cout' in raw):
                compiler = 'g++'
                raw = '#include <iostream>\n#include <string.h>\n#include <math.h>\n' + \
                    '#include <algorithm>\n#include <iomanip>\n' + \
                    'using namespace std;\n' + raw
                compile_list = [compiler, '-w','-o', output_obj_path, '-O0', output_c_path, '-std=c++11','-lm']
            else:
                compiler = 'gcc'
                raw = '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdbool.h>\n' +  \
                    '#include <math.h>\n#include <stddef.h>\n' + raw
                compile_list = [compiler, '-w','-o', output_obj_path, '-O0', output_c_path, '-lm']

            file_c.write(raw)
            file_c.close()
            subprocess.check_output(compile_list)
            result = subprocess.check_output(["objdump",'-dS', output_obj_path,'--section', '.text']).decode("utf-8")
            result_denoise = []
            result_denoise_len = 0
            # De-noise asm
            for i, elem in enumerate(result.split("\n\n")):
                elem_split = elem.splitlines()
                if i == 0 or elem_split[0] == "":
                    continue 
                elif not all([re.search(x, elem_split[0]) is None for x in ignore_x86_asm_block]):
                    continue
                else:
                    result_denoise += elem_split[1:]
                    result_denoise_len += len(elem_split)
        
            if (result_denoise_len > 1000):
                total_large_asm +=1

            file_asm = open(output_asm_path, 'w')
            file_asm.write("\n".join(result_denoise))
            file_asm.close()

            subprocess.call(['rm', output_obj_path])
            subprocess.call(['rm', output_c_path])

        if asm_type  =='mips':
            if('cin' in raw or 'cout' in raw):
                compiler = 'clang++'
                raw = '#include <iostream>\n#include <string.h>\n#include <math.h>\n' + \
                    '#include <algorithm>\n#include <iomanip>\n' + \
                    'using namespace std;\n' + raw
                compile_list_clang = [compiler, '-emit-llvm', output_c_path, '-c', '-o', output_bc_path, '-std=c++11','-lm']
                compile_list_llc = ['llc', output_bc_path, '-march=mipsel', '-relocation-model=static', '-o', output_asm_path]
            else:
                compiler = 'clang'
                raw = '#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdbool.h>\n' +  \
                    '#include <math.h>\n#include <stddef.h>\n' + raw
                compile_list = [compiler, '-w','-o', output_obj_path, '-O0', output_c_path, '-lm']
                compile_list_clang = [compiler, '-emit-llvm', output_c_path, '-c', '-o', output_bc_path,'-lm']
                compile_list_llc = ['llc', output_bc_path, '-march=mipsel', '-relocation-model=static', '-o', output_asm_path]
            file_c.write(raw)
            file_c.close()

            with open(os.devnull, "w") as f:
                subprocess.check_output(compile_list_clang, stderr=f)
                subprocess.check_output(compile_list_llc, stderr=f)
            subprocess.call(['rm', output_bc_path])
            subprocess.call(['rm', output_c_path])

    except subprocess.CalledProcessError:
        errs.append(solution_path)
    return lcs, errs, total_large_asm

def main():
    args = parse_args()
    filter_set = set()
    if args.filter_list is not None:
        with open(args.filter_list) as f:
            for line in f:
                p, s = line.strip().split('/')
                filter_set.add((p, s))

    problem_list = []
    for problem in os.listdir(args.poj_dir):
        problem_dir = os.path.join(args.poj_dir, problem)
        for solution in os.listdir(problem_dir):
            if (problem, solution) in filter_set:
                continue
            solution_path = os.path.join(problem_dir, solution)
            problem_list.append(
                (solution_path, 
                os.path.join(args.output_asm_dir, problem, solution),
                args.asm_type
                ))

    lcs = []
    errs = []

    total_large_asm = 0
    pool = multiprocessing.Pool(args.num_workers)
    for lcs_problem, errs_problem, total_large_asm_local in tqdm.tqdm(
        pool.imap_unordered(
            extract_from_solution,
            problem_list),
            total=len(problem_list)):
        lcs += lcs_problem
        errs += errs_problem
        total_large_asm += total_large_asm_local

    print(f'Err solutions: {len(errs)}/{len(problem_list)}')
    print(f'Large solutions: {total_large_asm}/{len(problem_list)}')


if __name__ == "__main__":
    main()
