import json
from datasets import load_dataset

from sal.utils.rewards.math_reward import _sal_reward_fn


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


def format_data_7b(example):
    system_prompt = example["messages"][0][0]["content"]
    problems = example["problem"] if example.get("problem", None) is not None else example["question"]

    # 构造对话格式
    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{problems}"},
        {"role": "assistant", "content": f"{example["multi_turn_message"]}"},
    ]

    return {
        "new_messages": messages,
    }


def classify_answers(ans1, ans2, gt_ans):
    ans1_correct, ans2_correct = _sal_reward_fn(ans1, gt_ans), _sal_reward_fn(ans2, gt_ans)
    ans1_correct = "correct" if ans1_correct == True else "error"
    ans2_correct = "correct" if ans2_correct == True else "error"
    return f"{ans1_correct}_{ans2_correct}"

def format_error_correct(error_resp, correct_resp):
    return (
        f"{error_resp}\n"
        f"Wait, but let me think again.\n"
        f"{correct_resp}"
    )

def format_correct_correct(resp1, resp2):
    return (
        f"{resp1}\n"
        f"Wait, but let me think again.\n"
        f"{resp2}"
    )

def format_correct_error(resp1, resp2):
    return resp1


def format_message_from_multi_turn(example):
    answer = example["answer"] if example.get("answer", None) is not None else example["gt"]
    # problems = example["problem"] if example.get("problem", None) is not None else example["question"]
    # system_prompt = example["messages"][0][0]["content"]
    ans_turn1 = example["messages"][0][2]["content"]
    ans_turn2 = example["messages"][0][4]["content"]

    case_type = classify_answers(ans_turn1, ans_turn2, answer)

    # 根据类型处理内容
    global c_c, c_e, e_e, e_c
    if case_type == "error_correct":  # 关键
        assistant_content = format_error_correct(ans_turn1, ans_turn2)
        e_c += 1

    elif case_type == "correct_correct":  # 补充
        assistant_content = format_correct_correct(ans_turn1, ans_turn2)
        c_c += 1

    elif case_type == "correct_error":  # 补充
        assistant_content = format_correct_error(ans_turn1, ans_turn2)
        c_e += 1

    elif case_type == "error_error":
        assistant_content = None
        e_e += 1

    else:
        raise NotImplementedError("出现了非法的故障")

    return {
        "multi_turn_message": assistant_content
    }

if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-1k-7b"
    data_file_name = "bon_completions_s0_e1000_accNone_03301852.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    c_c, c_e, e_e, e_c = 0, 0, 0, 0
    dataset = dataset.map(
        format_message_from_multi_turn,
        batched=False,
        desc="generate training example",
        load_from_cache_file=False,
    )
    dataset = dataset.filter(lambda x: x["multi_turn_message"] is not None)
    print(dataset)
    print("correct to correct:", c_c)
    print("correct to error:", c_e)
    print("error to correct:", e_c)
    print("error to error:", e_e)
    dataset = dataset.map(
        format_data_7b,
        batched=False,
        desc="generate training example",
        load_from_cache_file=False,
    )

    # dataset.to_json(f"{dataset_path}/{data_file_name}")
    # 转换为列表字典
    list_dict = []
    for example in dataset:
        list_dict.append({'messages': example['new_messages']})

    # 保存为 JSON 文件
    with open(f"{dataset_path}/llama-factory-sft-format.json", "w", encoding="utf-8") as f:
        json.dump(list_dict, f, ensure_ascii=False, indent=2)

