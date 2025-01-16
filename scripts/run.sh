#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

python scripts/test_time_compute.py recipes/Llama-3.1-8B-Instruct/diff_of_n.yaml
