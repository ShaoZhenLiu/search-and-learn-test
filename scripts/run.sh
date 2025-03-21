#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#VLLM_WORKER_MULTIPROC_METHOD=spawn \
#    python scripts/test_time_compute.py \
#        recipes/gemma-2-27b-it/diff_of_n.yaml

VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python scripts/test_time_compute.py \
        recipes/iter_gen_multi_turn.yaml \
        2>&1 | tee inference_multi_turn_16k_32b.log

#VLLM_WORKER_MULTIPROC_METHOD=spawn \
#    python scripts/inference_data_collect.py \
#        recipes/iter_gen_multi_turn.yaml \
#        2>&1 | tee inference_multi_turn_2.log