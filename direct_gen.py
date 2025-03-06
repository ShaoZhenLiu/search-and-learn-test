import random
import os
import argparse
import time
import json

import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

from sal.rewards.math_reward import sal_reward_fn


# from evaluate import evaluate
# from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
# from parser import *
# from trajectory import *
# from data_loader import load_data
# from python_executor import PythonExecutor
# from model_utils import load_hf_lm_and_tokenizer, generate_completions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="deepscaler.json", type=str)
    parser.add_argument("--data_names", default="DeepScaleR-Preview-Dataset", type=str)
    parser.add_argument("--data_dir", default="/data/shaozhen.liu/python_project/hf_datasets/",
                        type=str)
    parser.add_argument("--model_name_or_path",
                        default="/data/shaozhen.liu/python_project/hf_models/DeepSeek-R1-Distill-Qwen-32B",
                        type=str)
    parser.add_argument("--output_dir", default="/data/shaozhen.liu/python_project/hf_datasets",
                        type=str)
    parser.add_argument("--prompt_type", default="deepseek-math", type=str)
    # parser.add_argument("--split", default="test", type=str)
    # parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=16384, type=int)
    parser.add_argument("--shuffle", default=True)
    # parser.add_argument("--use_vllm", default=True)
    parser.add_argument("--save_outputs", default=True)
    parser.add_argument("--overwrite", default=True)
    parser.add_argument("--correct_answer_only", default=True)
    # parser.add_argument("--use_safetensors", action="store_true")
    # parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    # examples = load_data(data_name, args.split, args.data_dir)
    # examples = load_data(data_name=data_name, split="completion", data_dir=args.data_dir)
    dataset = load_dataset(f"{args.data_dir}/{data_name}", data_files=args.data_file, split='train')
    print(dataset)
    examples = dataset.to_list()[args.start: len(dataset) if args.end == -1 else args.end]

    # shuffle
    if args.shuffle:
        random.shuffle(examples)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]
    examples = sorted(examples, key=lambda x: x["idx"])

    # get out_file name
    # out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        raise FileNotFoundError("输出目录不存在")
        # output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/distilled_s{args.start}_e{args.end}.jsonl"

    return examples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # llm, tokenizer = None, None
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_num_seqs=128,
        # enable_chunked_prefill=True,
    )
    tokenizer = None
    if args.apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))


def construct_prompt(example, data_name, args):
    if args.prompt_type == "NuminaMath-CoT-cn_k12":
        input_template = (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
            "<|im_start|>user\n{input}<|im_end|>\n"
            "<|im_start|>assistant\n"  # + "<think>\n"
        )
    elif args.prompt_type == "deepseek-math":  # r1蒸馏模型的数据生成要用这个
        input_template = (
            "User: {input}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
            "Assistant:<think>\n"
        )
    else:
        raise NotImplementedError("这个数据集的prompt尚未实现")
    context = input_template.format(input=example["problem"])
    return context


def save_jsonl(all_samples, out_file):
    dataset = Dataset.from_list(all_samples)
    dataset.to_json(out_file)


def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        full_prompt = construct_prompt(example, data_name, args)

        if idx == args.start:  # 方便我们查看构造好的prompt
            print(full_prompt)

        # parse question and answer
        sample = {
            "idx": idx,
            "question": example["problem"],
            "gt_cot": example["solution"],
            "gt": example["answer"],
            "prompt": full_prompt,
        }

        # add remain fields
        for key in ["source"]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    input_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    # start inference
    # measure time use
    # start_time = time.time()
    print("-" * 20)

    # get all outputs
    prompts = [item[1] for item in input_prompts]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            stop=stop_words,
            stop_token_ids=(
                [151645, 151643]
                if "qwen" in args.model_name_or_path.lower()
                else None
            ),
        ),
    )

    outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id
    outputs_ls = [output.outputs[0].text for output in outputs]
    # print(f"模型的实际输出: \n{outputs_ls[0]}")
    output_token_ids_ls = [len(output.outputs[0].token_ids) for output in outputs]

    assert len(outputs_ls) == len(input_prompts) == len(output_token_ids_ls)

    # remove input_prompt from end_prompt
    # 我看了一下output[0].text，应该是不用这样的，本来就没有input_prompt，所以token_ids里面就完全都是回答了
    # real_outputs = []
    # for i in range(len(input_prompts)):
    #     end_prompt = outputs_ls[i][1]
    #     real_output = '<think>\n' + end_prompt.split(input_prompts[i][1])[-1].strip()  # 去掉input_prompts部分
    #     for stop_word in stop_words:
    #         if stop_word in real_output:
    #             real_output = real_output.split(stop_word)[0].strip()  # 去掉stop_word部分
    #     real_outputs.append(real_output)

    # extract preds
    # results = real_outputs
    # time_use = time.time() - start_time

    # put the correct generated results back to examples
    correct_count = 0
    all_samples = []
    for i, sample in enumerate(samples):
        real_output = outputs_ls[i * args.n_sampling: (i + 1) * args.n_sampling]
        output_token_ids = output_token_ids_ls[i * args.n_sampling: (i + 1) * args.n_sampling]

        # result checking: if any real_output is incorrect, then the sample will be dropped
        correct_ls = [sal_reward_fn(solution_str=res, ground_truth=sample['gt']) for res in real_output]
        is_correct = False if False in correct_ls else True
        correct_count += is_correct

        sample.pop("prompt")
        sample.update({
            "pred_cot": real_output[0] if len(real_output) == 1 else real_output,
            "pred_cot_token_len": output_token_ids[0] if len(output_token_ids) == 1 else output_token_ids,
            "correct": is_correct
        })
        all_samples.append(sample)

    print(f"模型生成回答的平均token长度为: {sum(output_token_ids_ls) / len(outputs_ls)}")
    print(f"模型生成答案的准确率为: {correct_count / len(examples) * 100} %")
    result_json = {
        "average_token_len": sum(output_token_ids_ls) / len(outputs_ls),
        "llm_response_acc": correct_count / len(examples) * 100,
    }

    if args.correct_answer_only:
        print("Dropping samples with incorrect answer...")
        all_samples = [sample for sample in all_samples if sample["correct"]]

    # save outputs
    if args.save_outputs:
        print(f"Saving data to: {out_file}")
        save_jsonl(all_samples, out_file)
        with open(out_file.replace(".jsonl", f"_metrics.json"), "w") as f:
            json.dump(result_json, f, indent=4)

    return result_json


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
