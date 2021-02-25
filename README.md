# A neural-based binary analysis tool

## Introduction

This directory contains the demo of a neural-based binary analysis tool. We test the framework using multiple binary analysis tasks: (i) vulnerability detection. (ii) code similarity measures. (iii) decompilations. (iv) malware analysis (coming later).

## Requirements

- Python 3.7.6
- Python packages
    * dgl 0.6.0
    * numpy 1.18.1
    * pandas 1.2.0
    * scipy 1.4.1
    * sklearn 0.0
    * tensorboard 2.2.1
    * torch 1.5.0
    * torchtext 0.2.0
    * tqdm 4.42.1
    * wget 3.2
- C++14 compatible compiler
- Clang++ 3.7.1 

## Tasks and Dataset preparation

### Binary code similarity measures
1. Download dataset
    - Download POJ-104 datasets from [here](https://www.dropbox.com/s/33fop57jjq0wwa9/POJ-104.tar.gz?dl=1) and extract them into `data/`.
2. Compile and preprocess
    - Run `python preprocess/extract_obj.py -asm data/obj` (clang++-3.7.1 required) 
    - Run `python preprocess/split_dataset.py -i data/obj -m p -o data/split.pkl` to split the dataset into train/valid/test sets.
    - Run `python preprocess/sim_preprocess.py` to compile the binary code into graphs data.
    - *(part of the preprocessing code are from [1])

### Binary Vulnerability detections

1. Cramming the binary dataset
    - The dataset is built on top of Devign. We compile the entire library based on the commit id and dump the binary code of the vulnerable functions. The cramming code is given in `preprocess/cram_vul_dataset`.
2. Download Preprocessed data
    - Run `./preprocess.sh` (clang++-3.7.1 required), or
    - You can directly download the preprocessed datasets from [here](https://www.dropbox.com/s/xkmfvq1qh63jqnq/vul.tar.gz?dl=1) and extract them into `data/`.
    - Run `python preprocess/vul_preprocess.py` to compile the binary code into graphs data

### Binary decompilation [N-Bref]
1.  Download dataset
    - Download the demo datasets (raw and preprocessed data) from [here](https://www.dropbox.com/s/yorq24i5lrd8wa4/re.tar.gz?dl=1) and extract them into `data/`. (More datasets to come.)
    - No need to compile the code into graph again as the data has already been preprocessed. 

## Training and Evaluation 
### Binary code similarity measures
- Run `cd baseline_model && python run_similarity_check.py` 

### Binary Vulnerability detections
- Run `cd baseline_model && python run_vulnerability_detection.py` 

### Binary decompilation [N-Bref]
1.  Dump the trace of tree expansion:
    - To accelerate the online processing of the tree output, we will dump the trace of the trea data by running `python -m preprocess.dump_trace`
2.  Training scripts:
    - First, `cd baseline model`.
    - To train the model using torch parallel, run `python run_tree_transformer.py`.
    - To train it on multi-gpu using distribute pytorch, run `python run_tree_transformer_multi_gpu.py`
    - To evaluate, run `python run_tree_transformer.py --eval`
    - To evaluate a multi-gpu trained model, run `python run_tree_transformer_multi_gpu.py --eval`

## References

[1] Ye, Fangke, et al. "MISIM: An End-to-End Neural Code Similarity System." arXiv preprint arXiv:2006.05265 (2020).

[2] Zhou, Yaqin, et al. "Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks." Advances in Neural Information Processing Systems. 2019.

[3] Shi, Zhan, et al. "Learning Execution through Neural Code Fusion.", ICLR (2019).

## License
This repo is CC-BY-NC licensed, as found in the [LICENSE file](./LICENSE).


