#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATA_NAME=("math_500" "gsm8k" "svamp" "asdiv" "mawps" "carp_en" "tabmwp" "minerva_math" "gaokao2023en" "olympiadbench" "college_math")

for ((i=0; i<${#DATA_NAME[@]}; i++))
do
  echo "=================== Processing dataset: ${DATA_NAME[i]} ==================="
  python scripts/test_time_compute.py \
    recipes/val.yaml \
    --dataset_name="/data/shaozhen.liu/python_project/Qwen2.5-Math/evaluation/data/${DATA_NAME[i]}" \
    --dataset_files="test.jsonl" \
    --output_dir="/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-1k-7b-sft/${DATA_NAME[i]}" \
    --n=1 \
    2>&1 | tee -a "log/val_7b_${DATA_NAME[i]}.log"
done

#DATA_NAME2=("aime24" "amc23")
#for ((i=0; i<${#DATA_NAME2[@]}; i++))
#do
#  echo "=================== Processing dataset: ${DATA_NAME2[i]} ==================="
#  python scripts/test_time_compute.py \
#    recipes/val.yaml \
#    --dataset_name="/data/shaozhen.liu/python_project/Qwen2.5-Math/evaluation/data/${DATA_NAME2[i]}" \
#    --dataset_files="test.jsonl" \
#    --output_dir="/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-1k-7b-sft/${DATA_NAME2[i]}" \
#    --n=8 \
#    2>&1 | tee -a "log/val_7b_${DATA_NAME2[i]}.log"
#done
