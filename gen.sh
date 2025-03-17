#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=2,3

# 20k 7b 直接回答 pass@8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python direct_gen.py \
    --model_name_or_path "/data/shaozhen.liu/python_project/hf_models/Qwen2.5-7B-Instruct" \
    --data_dir "/data/shaozhen.liu/python_project/hf_datasets" \
    --data_names "DeepScaleR-distilled-32b"\
    --data_file "train-distilled_20k.jsonl" \
    --prompt_type qwen25-math \
    --seed 0 \
    --start 0 \
    --end -1 \
    --batch_size 2000000 \
    --temperature 0.6 \
    --n_sampling 8 \
    --max_tokens_per_call 20480 \
     2>&1 | tee direct_gen.log

# 15k 7b 直接回答 pass@8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python direct_gen.py \
    --model_name_or_path "/data/shaozhen.liu/python_project/hf_models/Qwen2.5-7B-Instruct" \
    --data_dir "/data/shaozhen.liu/python_project/hf_datasets" \
    --data_names "DeepScaleR-distilled-32b"\
    --data_file "train-distilled_subset15k.jsonl" \
    --prompt_type qwen25-math \
    --seed 0 \
    --start 0 \
    --end -1 \
    --batch_size 2000000 \
    --temperature 0.6 \
    --n_sampling 8 \
    --max_tokens_per_call 20480 \
     2>&1 | tee direct_gen.log

# 15k 7b 直接回答 pass@1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python direct_gen.py \
    --model_name_or_path "/data/shaozhen.liu/python_project/hf_models/Qwen2.5-7B-Instruct" \
    --data_dir "/data/shaozhen.liu/python_project/hf_datasets" \
    --data_names "DeepScaleR-distilled-32b"\
    --data_file "train-distilled_subset15k.jsonl" \
    --prompt_type qwen25-math \
    --seed 0 \
    --start 0 \
    --end -1 \
    --batch_size 2000000 \
    --temperature 0.6 \
    --n_sampling 1 \
    --max_tokens_per_call 20480 \
     2>&1 | tee direct_gen.log