#!/bin/bash
set -ex

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export VLLM_WORKER_MULTIPROC_METHOD=spawn

# MODEL_PATH="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-7B-Instruct"
MODEL_PATH="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-32B-Instruct"
DATA_DIR="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data"
DATA_NAME="DeepScaler-QwQ_32b"
DATA_FILE_20K="distilled_s0_e20000_20250309005632.jsonl"
DATA_FILE_16K="distilled_s0_e20000_20250309005632_final.json"

TEMPERATURE=0.1
TOP_P=0.95
MAX_TOKENS=8192

# # 16k 32b 直接回答 pass@2
# python direct_gen.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_dir $DATA_DIR \
#     --data_names $DATA_NAME \
#     --data_file $DATA_FILE_16K \
#     --output_dir "${DATA_DIR}/${DATA_NAME}/16k_32b_pass@2_results" \
#     --prompt_type qwen25-math \
#     --seed 0 \
#     --start 0 \
#     --end -1 \
#     --batch_size -1 \
#     --temperature $TEMPERATURE \
#     --top_p $TOP_P \
#     --max_tokens_per_call $MAX_TOKENS \
#     --n_sampling 2 \
#      2>&1 | tee direct_gen_16k_32b_pass@2.log

# # 16k 32b 直接回答 pass@4
# python direct_gen.py \
#     --model_name_or_path $MODEL_PATH \
#     --data_dir $DATA_DIR \
#     --data_names $DATA_NAME \
#     --data_file $DATA_FILE_16K \
#     --output_dir "${DATA_DIR}/${DATA_NAME}/16k_32b_pass@4_results" \
#     --prompt_type qwen25-math \
#     --seed 0 \
#     --start 0 \
#     --end -1 \
#     --batch_size -1 \
#      --temperature $TEMPERATURE \
#     --top_p $TOP_P \
#     --max_tokens_per_call $MAX_TOKENS \
#     --n_sampling 4 \
#      2>&1 | tee direct_gen_16k_32b_pass@4.log

# 16k 32b 直接回答 pass@8
python direct_gen.py \
    --model_name_or_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --data_names $DATA_NAME \
    --data_file $DATA_FILE_16K \
    --output_dir "${DATA_DIR}/${DATA_NAME}/16k_32b_pass@8_results" \
    --prompt_type qwen25-math \
    --seed 0 \
    --start 0 \
    --end -1 \
    --batch_size -1 \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens_per_call $MAX_TOKENS \
    --n_sampling 8 \
     2>&1 | tee direct_gen_16k_32b_pass@8.log
