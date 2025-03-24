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
    if example.get("pred_cot_token_len", None) is not None:
        if isinstance(example["pred_cot_token_len"], list):
            token_len_sum.append(sum(example["pred_cot_token_len"]) / len(example["pred_cot_token_len"]))
        else:
            token_len_sum.append(example["pred_cot_token_len"])
        return {
            "pred_cot_token_len": example["pred_cot_token_len"]
        }
    else:
        token_len_ls = [len(tokenizer.tokenize(messages["content"])) for messages in example["messages"] if messages["role"] == "assistant"]
        token_len_sum.append(sum(token_len_ls))
        print(token_len_ls)
        return {
            "pred_cot_token_len": sum(token_len_ls)
        }

correct_count = 0
def add_correct(example, turn_idx=2):
    if example.get("messages", None) is not None:
        correctness = _sal_reward_fn(
            solution_str=example["messages"][turn_idx]["content"],  # 最后一个回答会输出在\boxed{}中的答案
            ground_truth=example["answer"] if example.get("answer", None) is not None else example["gt"],
            enable_llm=False, check_think=False,
        )
    else:
        correctness = _sal_reward_fn(
            solution_str=example["pred_cot"],
            ground_truth=example["gt"],
            enable_llm=False, check_think=False,
        )

    global correct_count
    correct_count += correctness

    return {
        "correct": correctness
    }


if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/DeepScaler-QwQ_32b/multi_turn_results_16k_32b"
    data_file_name = "bon_completions_s0_eNone_accNone.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    correct_count = 0
    turn_idx = 2
    dataset = dataset.map(
        add_correct,
        batched=False,
        desc="add correct label",
        fn_kwargs={"turn_idx": turn_idx},
        load_from_cache_file=False,
    )
    print(f"acc for turn {turn_idx}: {correct_count / len(dataset)}")

    correct_count = 0
    turn_idx = 4
    dataset = dataset.map(
        add_correct,
        batched=False,
        desc="add correct label",
        fn_kwargs={"turn_idx": turn_idx},
        load_from_cache_file=False,
    )
    print(f"acc for turn {turn_idx}: {correct_count / len(dataset)}")


    correct_count = 0
    turn_idx = -1
    dataset = dataset.map(
        add_correct,
        batched=False,
        desc="add correct label",
        fn_kwargs={"turn_idx": turn_idx},
        load_from_cache_file=False,
    )
    print(f"acc for turn {turn_idx}: {correct_count / len(dataset)}")

    # from transformers import AutoTokenizer

    # model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-7B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # dataset = dataset.map(
    #     add_pred_cot_token_len,
    #     batched=False,
    #     desc="add token len",
    #     fn_kwargs={"tokenizer": tokenizer},
    #     load_from_cache_file=False,
    # )
    # print(f"avg token len: {sum(token_len_sum) / len(token_len_sum)}")

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
