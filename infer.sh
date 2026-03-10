#!/bin/bash

# 设置环境变量
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1

# 运行训练命令
torchrun --nproc_per_node=1 --master_port=29514  basicsr/test.py -opt options/TVQRAP_test.yml --launcher pytorch

