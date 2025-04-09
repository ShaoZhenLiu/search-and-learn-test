import json
import random
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

    messages_ls = []
    for multi_turn_mess in example["multi_turn_message"]:
        # 构造对话格式
        messages = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{problems}"},
            {"role": "assistant", "content": f"{multi_turn_mess}"},
        ]
        messages_ls.append(messages)

    return {
        "new_messages": messages_ls,
    }


def classify_answers(ans1, ans2, gt_ans):
    ans1_correct, ans2_correct = _sal_reward_fn(ans1, gt_ans), _sal_reward_fn(ans2, gt_ans)
    # ans1_correct = "correct" if ans1_correct == True else "error"
    # ans2_correct = "correct" if ans2_correct == True else "error"
    return f"{ans1_correct}-to-{ans2_correct}"

def format_error_correct(error_resp, eval_resp, correct_resp):
    return (
        f"{error_resp}\n\n"
        f"{eval_resp}\n\n"
        f"{correct_resp}"
    )

def format_correct_correct(resp1, eval_resp, resp2):
    return (
        f"{resp1}\n\n"
        f"{eval_resp}\n\n"
        f"{resp2}"
    )

def format_correct_error(resp1, eval_resp):
    return (
        f"{resp1}\n\n"
        f"{eval_resp}\n\n"
    )


def format_message_from_multi_turn(example):
    answer = example["answer"] if example.get("answer", None) is not None else example["gt"]
    # problems = example["problem"] if example.get("problem", None) is not None else example["question"]
    # system_prompt = example["messages"][0][0]["content"]

    assistant_content_ls, correct_type_ls = [], []
    for msg in example["messages"]:

        ans_turn1 = msg[2]["content"]
        eval_turn = msg[6]["content"]
        ans_turn2 = msg[4]["content"]

        correct_type = classify_answers(ans_turn1, ans_turn2, gt_ans=answer)

        global c_c, c_e, e_e, e_c
        if correct_type == "False-to-True":  # 关键
            assistant_content = format_error_correct(ans_turn1, eval_turn, ans_turn2)
            e_c += 1

        elif correct_type == "True-to-True":  # 补充
            assistant_content = format_correct_correct(ans_turn1, eval_turn, ans_turn2)
            c_c += 1

        elif correct_type == "True-to-False":  # 补充
            assistant_content = format_correct_error(ans_turn1, eval_turn)
            c_e += 1

        elif correct_type == "False-to-False":
            assistant_content = None
            e_e += 1

        else:
            raise NotImplementedError

        assistant_content_ls.append(assistant_content)
        correct_type_ls.append(correct_type)

    return {
        "multi_turn_message": assistant_content_ls,
        "correct_turn_ls": correct_type_ls,
    }

if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-7b-sft_data"
    data_file_name = "merged.jsonl"
    dataset = load_dataset(dataset_path, data_files=data_file_name, split='train')
    print(dataset)
    c_c, c_e, e_e, e_c = 0, 0, 0, 0
    dataset = dataset.map(
        format_message_from_multi_turn,
        batched=False,
        desc="generate training example",
        load_from_cache_file=False,
    )
    # dataset = dataset.filter(lambda x: len(x["multi_turn_message"]) != 0)
    print(dataset)
    print("correct to correct:", c_c)
    print("correct to error:", c_e)
    print("error to correct:", e_c)
    print("error to error:", e_e)
    dataset = dataset.map(
        format_data_7b,
        batched=False,
        desc="format data",
        load_from_cache_file=False,
    )

    # dataset.to_json(f"{dataset_path}/{data_file_name}")
    list_dict = []
    for example in dataset:
        for message, correct_turn in zip(example['new_messages'], example['correct_turn_ls']):
            if correct_turn == "False-to-False":
                continue
            list_dict.append({
                'messages': message,
                'correct_turn': correct_turn,
            })

    # 分离 True-to-True 和 False-to-True 的数据
    true_to_true = [d for d in list_dict if d['correct_turn'] == 'True-to-True']
    false_to_true = [d for d in list_dict if d['correct_turn'] == 'False-to-True']
    other_data = [d for d in list_dict if d['correct_turn'] == 'True-to-False']

    # False-to-True 和 True-to-False + True-to-True 一样多，
    min_count = min(len(true_to_true), len(false_to_true)) // 2
    random.seed(0)
    sampled_true_to_true = random.sample(true_to_true, min_count)
    sampled_true_single = random.sample(other_data, min_count)

    print(len(sampled_true_to_true), len(false_to_true), len(sampled_true_single))
    filtered_list = sampled_true_to_true + false_to_true + sampled_true_single
    filtered_list = [{'messages': d['messages']} for d in filtered_list]
    random.shuffle(filtered_list)

    # 保存为 JSON 文件
    with open(f"{dataset_path}/llama-factory-sft-format.json", "w", encoding="utf-8") as f:
        json.dump(filtered_list, f, ensure_ascii=False, indent=2)

