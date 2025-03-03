#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7

VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python scripts/test_time_compute.py \
        recipes/gemma-2-27b-it/diff_of_n.yaml
