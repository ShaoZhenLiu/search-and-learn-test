#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=2,3

export VLLM_WORKER_MULTIPROC_METHOD=spawn
python direct_gen.py \
    --prompt_type deepseek-math \
    --seed 0 \
    --start 0 \
    --end 20000 \
    --temperature 0.6 \
    --n_sampling 1 \
    --max_tokens_per_call 20480