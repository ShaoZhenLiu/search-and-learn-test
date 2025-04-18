#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#python scripts/test_time_compute.py \
#    recipes/val_multi_turn.yaml \
#    2>&1 | tee log/val_multi_turn_1k_7b.log

#python scripts/test_time_compute.py \
#    recipes/think2.yaml

#python scripts/test_time_compute.py \
#    recipes/iter_gen_multi_turn.yaml \
#    2>&1 | tee log/inference_multi_turn_1k_7b.log

#python scripts/test_time_compute.py \
#    recipes/mid_fin_gen.yaml \
#    2>&1 | tee log/mid_fin_gen_7b.log

python scripts/test_time_compute.py \
    recipes/rejection_sampling.yaml \
    2>&1 | tee log/rejection_sampling.log