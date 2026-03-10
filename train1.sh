#!/bin/bash

# 设置环境变量
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 运行训练命令
torchrun --nproc_per_node=4 --master_port=29501 basicsr/train.py \
    -opt options/TVQRAP_stage1.yml --launcher pytorch

