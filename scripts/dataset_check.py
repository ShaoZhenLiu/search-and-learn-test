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

from sal.utils.rewards.math_reward import _sal_reward_fn


def add_message(example):
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


def add_gt(example):
    pass


def add_pred_cot(example):
    pass

token_len_sum = []
def add_pred_cot_token_len(example, tokenizer):
    # token_len_ls = [len(tokenizer.tokenize(messages["content"])) for messages in example["messages"] if messages["role"] == "assistant"]
    # token_len_sum.append(sum(token_len_ls))
    token_len_sum.append(sum(example["pred_cot_token_len"]))
    # print(token_len_ls)
    # return {
    #     "pred_cot_token_len": sum(token_len_ls)
    # }


def add_correct(example):
    correctness = _sal_reward_fn(
        solution_str=example["messages"][-1]["content"],  # 最后一个回答会输出在\boxed{}中的答案
        ground_truth=example["answer"],
        enable_llm=False, check_think=False,
    )
    return {
        "correct": correctness
    }


if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-distilled-32b"
    data_file_name = "bon_completions_s0_e20000_accNone.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    # dataset = dataset.map(
    #     add_correct,
    #     batched=False,
    #     desc="add correct label",
    #     load_from_cache_file=False,
    # )

    from transformers import AutoTokenizer

    model_path = "/data/shaozhen.liu/python_project/hf_models/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = dataset.map(
        add_pred_cot_token_len,
        batched=False,
        desc="add token len",
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
    )
    print(dataset["pred_cot_token_len"][0])
    print(sum(token_len_sum) / len(token_len_sum))

    # dataset.to_json(f"{dataset_path}/{data_file_name}")

    # # 转换为列表字典
    # list_dict = []
    # for example in dataset:
    #     list_dict.append({'messages': example['messages']})
    #
    # # 保存为 JSON 文件
    # with open(f"{dataset_path}/llama-factory-sft-format.json", "w", encoding="utf-8") as f:
    #     json.dump(list_dict, f, ensure_ascii=False, indent=2)
    #
    # from huggingface_hub import HfApi
    #
    # api = HfApi()
    # # api.create_repo(
    # #     repo_id="tttonyyy/NuminaMath-CoT-cn_k12-20000",
    # #     repo_type="dataset",
    # # )
    # api.upload_file(
    #     path_or_fileobj=f"{dataset_path}/{data_file_name}",
    #     path_in_repo="train.jsonl",
    #     repo_id="tttonyyy/DeepScale-qwen2.5_7b-multi",
    #     repo_type="dataset",
    # )
