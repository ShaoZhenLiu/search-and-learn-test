"""
这个文档是将所有dataset标准化的文档
标准化后的dataset应该有如下标签：
* question：问题文本
* gt_cot：原始回答文本（如果有）
* gt：原始答案（如果有）
* pred_cot_token_len：模型回答的token长度（如果是多轮回答，就是一个列表）
* message：提问与回答的输入，是字典列表，可以直接交给llama-factory微调
"""

import json
from datasets import load_dataset
from huggingface_hub import HfApi


def create_repo(repo_id):
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
    )

def upload_dataset(dataset_path, data_file_name, repo_id):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"{dataset_path}/{data_file_name}",
        path_in_repo="train.jsonl",
        repo_id=repo_id,  # "tttonyyy/DeepScale-qwen2.5_7b-multi"
        repo_type="dataset",
    )

if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/DeepScaler-QwQ_32b/multi_turn_results_16k_32b"
    data_file_name = "bon_completions_s0_eNone_accNone.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)


    repo_id = "tttonyyy/DeepScaleR_16k-32b-multi-pass1"
    # 如果repo不存在，则创建
    create_repo(repo_id)
    upload_dataset(dataset_path, data_file_name, repo_id)

