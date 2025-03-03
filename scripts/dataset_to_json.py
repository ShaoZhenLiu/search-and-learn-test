import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GemmaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import torch


# 数据预处理函数
def format_data(example):
    # 合并指定字段
    merged_content = f"Different Solutions:\n"

    # 添加k_diff_solutions
    for i, solution in enumerate(example['k_diff_solutions'], 1):
        merged_content += f"Solution {i}: {solution}\n"

    merged_content += (
        f"\nEvaluation: {example['consistency_evaluation']}\n"
        f"Conclusion: {example['conclusion']}\n"
        # f"Final Answer: {example['pred_result']}"  # 去掉这个，因为可能会有多步提问，需要多步的回答
    )

    # 构造对话格式
    messages = [
        {"role": "user",
         "content": f"Solve this math problem and show your reasoning.\n\nMath Problem: {example['problem']}"},
        {"role": "assistant", "content": merged_content},
    ]

    return {
        # "formatted_input": merged_content,
        "new_messages": messages,
        # "solution": example['solution']
    }


def format_data_distilled(example):
    # 确保只有一个<think>\n
    pred_cot = example['pred_cot'].split("<think>\n")[-1]
    # 构造对话格式
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
        {"role": "user",
         "content": f"Solve this math problem and show your reasoning.\n\nMath Problem: {example['question']}"},
        {"role": "assistant", "content": "<think>\n" + pred_cot},
    ]

    return {
        "messages": messages,
        "pred_cot": "<think>\n" + pred_cot,
    }


if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/NuminaMath-CoT-cn_k12/"
    data_file_name = "distilled_s0_e20000.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    dataset = dataset.map(
        format_data_distilled,
        batched=False,
        desc="generate training example",
        load_from_cache_file=False,
    )
    print(dataset[0]["pred_cot"])
    print(dataset)
    dataset.to_json(f"{dataset_path}/{data_file_name}")
    # 转换为列表字典
    list_dict = []
    for example in dataset:
        list_dict.append({'messages': example['messages']})

    # 保存为 JSON 文件
    with open(f"{dataset_path}/distilled_dataset_lf.json", "w", encoding="utf-8") as f:
        json.dump(list_dict, f, ensure_ascii=False, indent=2)

    from huggingface_hub import HfApi

    api = HfApi()
    # api.create_repo(
    #     repo_id="tttonyyy/NuminaMath-CoT-cn_k12-20000",
    #     repo_type="dataset",
    # )
    api.upload_file(
        path_or_fileobj=f"{dataset_path}/{data_file_name}",
        path_in_repo=data_file_name,
        repo_id="tttonyyy/NMC-cn_k12-20k-r1_32b_distilled",
        repo_type="dataset",
    )
