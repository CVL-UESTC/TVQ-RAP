#!/bin/bash

# 设置环境变量
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=4,5


torchrun --nproc_per_node=2 --master_port=29501 basicsr/train.py \
    -opt options/TVQRAP_stage2.yml --launcher pytorch
