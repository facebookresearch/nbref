# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import os
from git import Repo
import subprocess
import json
import traceback
import re
import argparse
from shutil import copy2

repo_url = {
    "FFmpeg": "https://github.com/FFmpeg/FFmpeg.git",
    "qemu": "https://github.com/qemu/qemu.git"
}

parser = argparse.ArgumentParser()
parser.add_argument('--parallel_num', type=int, default=5)
parser.add_argument('--optimization_level', type=int, default=0, choices=[0, 1, 2, 3, 4], help="-O0, -O1, -O2, -O3]")
parser.add_argument('--dataset', type=str, default="FFmpeg", choices=["FFmpeg", "qemu"])
parser.add_argument('--data_file', type=str, default="function.json")
parser.add_argument('--vulnerable', dest='vulnerable', action='store_true', help="whther to compile unvulnerable cpp (label=0)")
parser.set_defaults(vulnerable=False)
args = parser.parse_args()

output_dir = "output"
output_source_dir = os.path.join(output_dir, "source")
output_assembly_dir = os.path.join(output_dir, "assembly")
output_object_dir = os.path.join(output_dir, "object")
output_meta_dir = os.path.join(output_dir, "meta")
fail_dir = "failed"

finished = set() # the commit hash of previously finished data points


def process(d, repo):
    idx = d["idx"]
    commit, label = d["commit_id"], d["target"]

    # if this data point have been processed
    if commit in finished:
        print(commit, "has finished. Omitting")
        return

    if args.vulnerable and label == "0":
        return

    print("Start processing", idx, commit)

    # get the whole code to pair with assembly when finished
    all_code = d["func"]

    # get the first line of code to extract the function name
    # FUNC_TYPE FUNC_NAME(Parameters)
    code = d["func"].split("\n")[0]
    func_name = code.split('(')[0].split(' ')[-1]

    # Just in case the first line is not the declaration
    if len(code) < 10:
        print(f"Warning code length {idx} {commit} {code}\n")

    # get the changed files from currect commit, only one contains the file name for the target function
    repo.git.checkout(commit, force=True)
    previous_commit = repo.git.rev_list(
        '--parents', '-n', '1', commit).split()[1]
    diffs = repo.git.diff('HEAD~1', name_only=True).split('\n')

    # restore to parent commit
    previous_commit = repo.git.rev_list(
        '--parents', '-n', '1', commit).split()[1]
    repo.git.checkout(previous_commit, force=True)
    correct = None

    # find out the file name with target function by exactly matching the first line of it
    for d in diffs:
        try:
            if code in open(os.path.join(repo_dir, d)).read():
                correct = d
                break
        except:
            # Sometimes it would report decode error, only very few times
            traceback.print_exc()
            print("Error reading", file, repo_dir, d, "\n\n")

    # if not find the file with target functions
    # never happened till now
    if not correct:
        print(f"Error find file {commit} {previous_commit}\n")
        return

    print(f"Correct {commit} {previous_commit} {correct}")

    # store every build history with the commit as folder name (not the previous)
    build_dir = os.path.join(repo_dir, "build", commit)

    # change optimization level
    configure_file = os.path.join(repo_dir, "configure")
    configure_text = re.sub(r"-O[0-5sz]", f"-O{args.optimization_level}", open(configure_file).read())
    open(configure_file, "w").write(configure_text)

    print("Start building")

    try:
        os.mkdir(build_dir)
        subprocess.check_output(
            [f"cd {build_dir} && ../../configure --disable-werror && make -j{args.parallel_num}"], stderr=subprocess.STDOUT, shell=True)
    except:
        # if the os.mkdir fails, it means this commit has been build
        print("Omit building")

    # find the object file
    obj_path = os.path.join(build_dir, correct.replace(".c", ".o"))

    try:
        results = subprocess.check_output(
            [f"nm -f sysv {obj_path} | grep {func_name}"], 
            stderr=subprocess.STDOUT, shell=True).decode("utf-8").split('\n')
    except:
        # Exception means make failed
        # Simply rebuild
        # If still failed, just abort and record error
        print("Rebuild")
        results = subprocess.check_output(
            [f"rm -rf {build_dir} && mkdir {build_dir} && cd {build_dir} && ../../configure && make -j{args.parallel_num}"], 
            stderr=subprocess.STDOUT, shell=True)

        open(os.path.join(fail_dir, f"{commit}.log"), "wb").write(results)

        results = subprocess.check_output(
            [f"nm -f sysv {obj_path} | grep {func_name}"], 
            stderr=subprocess.STDOUT, shell=True).decode("utf-8").split('\n')

    start, end = None, None

    # find out the function name by exact match
    # then extract the address
    for r in results:
        r = r.split('|')

        if len(r) != 7:
            continue

        if r[0].strip() == func_name:
            start, end = int(r[1].strip(), 16), int(r[4].strip(), 16)
            end = hex(start + end)
            start = hex(start)

    # if not find the fucntion address
    # never happened till now
    if not start:
        print(f"Error finding start {commit} {previous_commit} {results}")
        return

    # get the assembly
    results = subprocess.check_output(
        [f"objdump -w -d --start-address={start} --stop-address={end} {obj_path} --section=.text"], 
        stderr=subprocess.STDOUT, shell=True).decode("utf-8").split('\n')
    assembly = '\n'.join(results[6:])

    copy2(obj_path, os.path.join(output_object_dir, f"{idx}_{commit}.o"))

    data = {
        "idx": idx,
        "commit_hash": commit,
        "previsous_hash": previous_commit,
        "source": all_code,
        "assembly": assembly,
        "label": label,
        "filename": correct
    }
    json.dump(data, open(os.path.join(output_meta_dir, f"{idx}_{commit}.json"), "w"))
    open(os.path.join(output_source_dir, f"{idx}_{commit}.cpp"), "w").write(all_code)
    open(os.path.join(output_assembly_dir, f"{idx}_{commit}.s"), "w").write(assembly)
    print("Success", commit, '\n\n')


if __name__ == "__main__":
    repo_dir = args.dataset

    # preparation
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(output_source_dir):
        os.mkdir(output_source_dir)
    if not os.path.isdir(output_assembly_dir):
        os.mkdir(output_assembly_dir)
    if not os.path.isdir(output_object_dir):
        os.mkdir(output_object_dir)
    if not os.path.isdir(output_meta_dir):
        os.mkdir(output_meta_dir)
    if not os.path.isdir(fail_dir):
        os.mkdir(fail_dir)

    if not os.path.isdir(repo_dir):
        print("Repo not exist. Cloning...")
        repo = Repo.clone_from(repo_url[args.dataset], repo_dir)
    else:
        repo = Repo(repo_dir)

    if not os.path.isdir(os.path.join(repo_dir, "build")):
        os.mkdir(os.path.join(repo_dir, "build"))

    # get the finished data points
    for file in os.listdir(output_meta_dir):
        if file.endswith(".json"):
            finished.add(file.replace(".json", "").split('_')[0])

    data = json.load(open(args.data_file))

    print("Start building")
    print(f"Using optimization level at {args.optimization_level}")

    for i, d in enumerate(data):
        try:
            if str(i) in finished or d['project'] != args.dataset:
                continue

            d["idx"] = i
            process(d, repo)

        except Exception as e:
            traceback.print_exc()
            print("Error", i, d["commit_id"], "\n\n")
