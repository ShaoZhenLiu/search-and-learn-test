#!/bin/bash

# 这里是gcc-5.4.0需要配置的环境变量
# export PATH=/path/to/install/gcc-5.4/bin:$PATH
# export LD_LIBRARY_PATH=/path/to/install/gcc-5.4/lib/:/path/to/install/gcc-5.4/lib64:$LD_LIBRARY_PATH
#export PATH=/home/shaozhen.liu/gcc/gcc/bin:$PATH
#export LD_LIBRARY_PATH=/home/shaozhen.liu/gcc/gcc/lib/:/home/shaozhen.liu/gcc/gcc/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2,3

python scripts/test_time_compute.py recipes/gemma-2-27b-it/diff_of_n.yaml
