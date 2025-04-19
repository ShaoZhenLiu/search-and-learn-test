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
import os
import argparse
# from datasets import load_dataset
from huggingface_hub import HfApi, login


def create_repo(repo_id, repo_type="dataset"):
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
    )

def upload_dataset(dataset_path, data_file_name, repo_id):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=f"{dataset_path}/{data_file_name}",
        path_in_repo="train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )

def upload_model(model_path, repo_id):
    api = HfApi()
    # 上传模型文件夹中的所有文件
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="上传数据集或模型到HuggingFace")
    parser.add_argument("--type", type=str, choices=["dataset", "model"], default="model", help="上传类型：dataset 或 model")
    parser.add_argument("--path", type=str, default="/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/full_sft/Qwen2.5-7B-Instruct-1k_filter", help="数据集路径或模型路径")
    parser.add_argument("--repo_id", type=str, default="tttonyyy/Qwen2.5-7B-Instruct-cot", help="HuggingFace上的仓库ID")
    parser.add_argument("--data_file", type=str, default="train.jsonl", help="数据集文件名（仅用于数据集上传）")

    args = parser.parse_args()

    if args.type == "dataset":
        if args.data_file is None:
            raise ValueError("上传数据集时需要提供 --data_file 参数")

        # 加载数据集
        dataset = load_dataset(args.path, data_files=args.data_file, split='train')
        print(f"加载的数据集: {dataset}")

        # 如果repo不存在，则创建
        create_repo(args.repo_id, repo_type="dataset")
        upload_dataset(args.path, args.data_file, args.repo_id)
        print(f"数据集已成功上传到 {args.repo_id}")
    else:  # model
        # 检查模型路径是否存在
        if not os.path.exists(args.path):
            raise ValueError(f"模型路径 {args.path} 不存在")

        # 如果repo不存在，则创建
        # try:
        #     create_repo(args.repo_id, repo_type="model")
        # except e:
        #     print(e)
        upload_model(args.path, args.repo_id)
        print(f"模型已成功上传到 {args.repo_id}")

