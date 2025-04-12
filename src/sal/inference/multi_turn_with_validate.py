#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import logging
from collections import defaultdict, deque
import asyncio
import aiohttp
from threading import Lock

import numpy as np
from tqdm import tqdm
from jinja2 import Template
# from math_verify import parse, verify
from vllm import LLM, SamplingParams

from sal.config import Config

logger = logging.getLogger()
from sal.utils.score import aggregate_scores
from sal.inference.direct_gen import ResponseCollector
from sal.utils.rewards.math_reward import _sal_reward_fn
from sal.utils.rewards.math_utils import extract_answer


def vllm_generate(convs_ls, config: Config, llm: LLM):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs_ls,
        add_generation_prompt=True,  # 会自动在最后加上，这样模型就不会先生成"<|start_header_id|>assistant<|end_header_id|>"再回答了
        tokenize=False,
    )  # 将对话 prompt 按照大模型的方式编码

    # print(templated_convs[0])

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,  # 解码过程中，概率累积到多少的时候截断
        # top_k=config.top_k,
        best_of=1,  # 每次都生成1个
        # n=config.n if i == 0 else 1,  # 第一次返回n个，之后就只返回1个
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    responses = sorted(responses, key=lambda x: int(x.request_id))  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in responses]
    output_token_ids_ls = [len(output.outputs[0].token_ids) for output in responses]

    return responses, outputs, output_token_ids_ls


def _multi_turn_with_validate(batch_of_prompts, answers, config: Config, llm: LLM) -> dict:
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": Template(config.step_prompt["turn0"]).render(problem=problem)},
        ]
        for p_index, problem in enumerate(batch_of_prompts) for _ in range(config.n)  # n==8，即每个问题生成8轮对话，正确性取平均
    ]  # 构建对话 prompt
    token_len_ls = []


    def generate_convs(old_convs, prompt_index):
        responses, outputs_ls, output_token_ls = vllm_generate(old_convs, config, llm)
        token_len_ls.append(output_token_ls)
        new_convs = [
            [
                *conv,
                {"role": "assistant", "content": output},
                {"role": "user", "content": Template(config.step_prompt[f"turn{prompt_index}"]).render(
                    correctness=_sal_reward_fn(
                        solution_str=output,
                        ground_truth=answers[conv_index // config.n],
                        enable_llm=False, check_think=False,
                    )
                )},
            ]
            for conv_index, (conv, output) in enumerate(zip(old_convs, outputs_ls))
        ]
        return new_convs

    # for i in range(1, 3):
    #     convs = generate_convs(convs, prompt_index=i)
    convs = generate_convs(convs, prompt_index=1)

    # convs_for_system_user_assistant = convs[:-1]
    convs_for_system_user_assistant = [
        [
            *(conv[:-1]),
            {"role": "user", "content": config.step_prompt["turn2"]},
        ]
        for conv_index, (conv) in enumerate(convs)
    ]
    responses, outputs_ls_for_eval, output_token_ls_for_eval = vllm_generate(convs_for_system_user_assistant, config, llm)
    responses, outputs_ls, output_token_ls = vllm_generate(convs, config, llm)  # 得到最终的答案
    token_len_ls.append(output_token_ls)
    token_len_ls.append(output_token_ls_for_eval)
    final_convs = [
        [
            *conv,
            {"role": "assistant", "content": output},
            {"role": "user", "content": config.step_prompt["turn2"]},
            {"role": "assistant", "content": output_eval},
        ]
        for conv_index, (conv, output, output_eval) in enumerate(zip(convs, outputs_ls, outputs_ls_for_eval))
    ]

    # 将token_len的形状从 [4, batch] 变成 [batch, 4]
    new_token_len_ls = [list(token_ls) for token_ls in zip(*token_len_ls)]

    agg_final_convs = []
    agg_token_len_ls = []
    for i in range(len(batch_of_prompts)):
        real_output = final_convs[i * config.n: (i + 1) * config.n]
        output_token_ids = new_token_len_ls[i * config.n: (i + 1) * config.n]
        agg_final_convs.append(real_output)
        agg_token_len_ls.append(output_token_ids)

    step_result = {
        "messages": agg_final_convs,
        "pred_cot_token_len": agg_token_len_ls,
    }

    return step_result




def multi_turn_with_validate(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"] if examples.get("problem", None) is not None else examples["question"]
    answers = examples["answer"] if examples.get("answer", None) is not None else examples["gt"]
    step_result = _multi_turn_with_validate(problems, answers, config, llm)

    if config.calculate_correct:
        correct_ls = [
            [
                _sal_reward_fn(
                    solution_str=message[-1]["content"],
                    ground_truth=answer,
                    enable_llm=False, check_think=False,
                )
                for message in messages
            ]
            for answer, messages in zip(answers, step_result["messages"])
        ]
        agg_correct_ls = [sum(map(int, correct_sample)) / len(correct_sample) for correct_sample in correct_ls] # 取平均
        print(agg_correct_ls[0])
        step_result["correct_ls"] = correct_ls
        step_result["correct"] = agg_correct_ls


        # 生成完正确和错误两轮数据之后，需要生成中间验证步骤
        # 首先需要找出 incorrect-to-correct 和 correct-to-correct
        # 然后采样8次，取其中和原本结果正确的一项
        correct_turn_ls = [
            [(
                f"{_sal_reward_fn(solution_str=message[2]["content"], ground_truth=answer)}"
                "-to-"
                f"{_sal_reward_fn(solution_str=message[4]["content"], ground_truth=answer)}"
            ) for message in messages]
            for answer, messages in zip(answers, step_result["messages"])
        ]

        evaluate_turn_ls = [
            [
                _sal_reward_fn(solution_str=message[2]["content"], ground_truth=answer) == ("[VERIFY] correct" in message[-1]["content"])
                for message in messages
            ]
            for answer, messages in zip(answers, step_result["messages"])
        ]

        # 最后就提取这几个：
        # correct_turn_ls 为 incorrect-to-correct 和 correct-to-correct 的，同时
        # evaluate_turn_ls 为 True 的，最后查看数量

        step_result["correct_turn_ls"] = correct_turn_ls
        step_result["evaluate_turn_ls"] = evaluate_turn_ls


    # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples
