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
# from math_verify import parse, verify

from sal.utils.rewards.math_reward import _sal_reward_fn
from sal.utils.rewards.math_utils import extract_answer


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

def process_cot_token_len(dataset):
    from transformers import AutoTokenizer

    model_path = "/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = dataset.map(
        add_pred_cot_token_len,
        batched=False,
        desc="add token len",
        fn_kwargs={"tokenizer": tokenizer},
        load_from_cache_file=False,
    )
    print(f"avg token len: {sum(token_len_sum) / len(token_len_sum)}")


# correct_count = 0
def add_correct(example, turn_idx=2, correct_tag="correct"):
    if example.get("messages", None) is not None:
        if isinstance(example["messages"][0], list):  # pass@1+avg8
            answer = example["answer"] if example.get("answer", None) is not None else example["gt"]
            correct_ls = [
                _sal_reward_fn(
                    solution_str=ms[turn_idx]["content"],  # 最后一个回答会输出在\boxed{}中的答案
                    ground_truth=answer,
                    enable_llm=False, check_think=False,
                )
                for ms in example["messages"]
            ]
            # correct_ls = []
            # for ms in example["messages"]:
            #     print(extract_answer(ms[turn_idx]["content"]))
            #     # print(parse("$${}$$".format(answer)))
            #     c = verify(
            #         parse("$${}$$".format(answer)),
            #         parse(extract_answer(ms[turn_idx]["content"]))
            #     )
            #     correct_ls.append(c)
            correctness = sum(map(int, correct_ls)) / len(correct_ls)
        elif isinstance(example["messages"][0], dict):  # pass@1+avg1
            correctness = _sal_reward_fn(
                solution_str=example["messages"][turn_idx]["content"],  # 最后一个回答会输出在\boxed{}中的答案
                ground_truth=example["answer"] if example.get("answer", None) is not None else example["gt"],
                enable_llm=False, check_think=False,
            )
    else:
        if example.get("pred_cot", None) is not None:
            if len(example["pred_cot"]) > 1:
                correct_ls = [
                    _sal_reward_fn(
                        solution_str=pc,  # 最后一个回答会输出在\boxed{}中的答案
                        ground_truth=example["gt"],
                        enable_llm=False, check_think=False,
                    )
                    for pc in example["pred_cot"]
                ]
                correctness = sum(map(int, correct_ls)) / len(correct_ls)
            else:
                correctness = _sal_reward_fn(
                    solution_str=example["pred_cot"],
                    ground_truth=example["gt"],
                    enable_llm=False, check_think=False,
                )
        else:
            correctness = _sal_reward_fn(
                solution_str=example["code"][0],
                ground_truth=example["gt"],
                enable_llm=False, check_think=False,
            )

    # global correct_count
    # correct_count += correctness

    return {
        correct_tag: correctness
    }

def calculate_correct_turn(dataset, turn_idx_ls):
    """

    """
    for turn_id in turn_idx_ls:
        tag = f"correct_turn{turn_id}"
        dataset = dataset.map(
            add_correct,
            batched=False,
            desc="add correct label",
            fn_kwargs={"turn_idx": turn_id, "correct_tag": tag},
            load_from_cache_file=False,
        )
        acc = sum(dataset[tag]) / len(dataset) * 100
        print(f"acc for turn {turn_id}: {acc}%")
    # 对数据进行过滤，过滤出turn2和turn4正确，但是turn-1错误的数据
    new_dataset = dataset.filter(lambda x: x["correct_turn2"] == 0 and x["correct_turn-1"] > 0)
    print(len(new_dataset))
    print(new_dataset["idx"])
    return

def add_correct_sft_dataset_val_data(example):
    split_words = "Wait, but let me think again."
    pred_cot_ls = example["code"][0].split(split_words)
    if len(pred_cot_ls) == 1:
        pred_cot_ls.append("")
    example["messages"] = [
        {
            "content": pred_cot_ls[0],
            "role": "assistant",
        },
        {
            "content": pred_cot_ls[1],
            "role": "assistant",
        }
    ]
    return {
        "correct_turn0": add_correct(example, turn_idx=0)["correct"],
        "correct_turn1": add_correct(example, turn_idx=1)["correct"],
    }

def map_sft_val_data(dataset):
    dataset = dataset.map(add_correct_sft_dataset_val_data, load_from_cache_file=False)
    new_dataset = dataset.filter(lambda x: len(x["code"][0].split("Wait, but let me think again.")) == 1)
    print(len(new_dataset))
    print(new_dataset["idx"])
    print(len(dataset.filter(lambda x: x["correct_turn0"] == x["correct_turn1"])))
    new_dataset = dataset.filter(lambda x: x["correct_turn0"] == True and x["correct_turn1"] == False)
    print(len(new_dataset))
    print(new_dataset["idx"])

def compare_ori_and_sft():
    dataset_path = "/data/shaozhen.liu/python_project/Qwen2.5-Math/evaluation/outputs/data/shaozhen.liu/python_project/hf_models/sft_models/Qwen2.5-7B-Instruct-main/math_eval/math_500/"
    data_file_name = "test_qwen25-math-cot_-1_seed0_t0.6_s0_e-1.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    # print(dataset["correct"][0])

    dataset = dataset.map(
        add_correct,
        batched=False,
        desc="add correct label",
        load_from_cache_file=False,
    )
    correct_ls_sft = dataset["correct"]

    dataset_path = "/data/shaozhen.liu/python_project/Qwen2.5-Math/evaluation/outputs/data/shaozhen.liu/python_project/hf_models/Qwen2.5-7B-Instruct/math_eval/math_500/"
    data_file_name = "test_qwen25-math-cot_-1_seed0_t0.6_s0_e-1.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    # print(dataset["correct"][0])

    dataset = dataset.map(
        add_correct,
        batched=False,
        desc="add correct label",
        load_from_cache_file=False,
    )
    correct_ls_ori = dataset["correct"]

    oriTure_sftFalse = []
    sftTure_oriFalse = []
    for i, (c_sft, c_ori) in enumerate(zip(correct_ls_sft, correct_ls_ori)):
        if c_ori == True and c_sft == False:
            oriTure_sftFalse.append(i)
        elif c_sft == True and c_ori == False:
            sftTure_oriFalse.append(i)
    print(oriTure_sftFalse)
    print(sftTure_oriFalse)

if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/data/shaozhen.liu/python_project/Qwen2.5-Math/evaluation/outputs/data/shaozhen.liu/python_project/hf_models/sft_models/Qwen2.5-7B-Instruct-main/math_eval/math_500/"
    data_file_name = "test_qwen25-math-cot_-1_seed0_t0.6_s0_e-1.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)

    map_sft_val_data(dataset)

    # dataset.to_json(f"{dataset_path}/{data_file_name}")

    # # 转换为列表字典
    # list_dict = []
    # for example in dataset:
    #     list_dict.append({'messages': example['messages']})
    #
    # # 保存为 JSON 文件
    # with open(f"{dataset_path}/llama-factory-sft-format.json", "w", encoding="utf-8") as f:
    #     json.dump(list_dict, f, ensure_ascii=False, indent=2)
