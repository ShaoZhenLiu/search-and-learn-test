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
from .utils import vllm_generate, generate_convs, agg_data


def _middle_final_generate(batch_of_prompts, answers, config: Config, llm: LLM) -> dict:
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": Template(config.step_prompt["turn0"]).render(problem=problem)},
        ]
        for p_index, problem in enumerate(batch_of_prompts) for _ in range(config.n)  # n==8，即每个问题生成8轮对话，正确性取平均
    ]  # 构建对话 prompt
    token_len_ls = []

    # for i in range(1, 3):
    #     convs = generate_convs(convs, prompt_index=i)
    convs, output_token_ls = generate_convs(convs, prompt_index=1, config=config, llm=llm, answers=answers)
    token_len_ls.append(output_token_ls)
    convs, output_token_ls = generate_convs(convs, prompt_index=2, config=config, llm=llm, answers=answers)
    token_len_ls.append(output_token_ls)

    # convs_for_system_user_assistant = convs[:-1]
    convs_for_system_user_assistant = [
        [
            # conv[0],
            {"role": "user", "content": Template(config.step_prompt["turn3"]).render(
                problem=conv[1]["content"],
                answer1=extract_answer(conv[2]["content"]),
                answer2=extract_answer(conv[4]["content"]),
            )},
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
            {"role": "user", "content": Template(config.step_prompt["turn3"]).render(
                problem=conv[1]["content"],
                answer1=extract_answer(conv[2]["content"]),
                answer2=extract_answer(conv[4]["content"]),
            )},
            {"role": "assistant", "content": output_eval},
        ]
        for conv_index, (conv, output, output_eval) in enumerate(zip(convs, outputs_ls, outputs_ls_for_eval))
    ]

    # 将token_len的形状从 [4, batch] 变成 [batch, 4]
    new_token_len_ls = [list(token_ls) for token_ls in zip(*token_len_ls)]

    agg_final_convs, agg_token_len_ls = agg_data(final_convs, new_token_len_ls, config, len(batch_of_prompts))

    step_result = {
        "messages": agg_final_convs,
        "pred_cot_token_len": agg_token_len_ls,
    }

    return step_result




def middle_final_generate(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"] if examples.get("problem", None) is not None else examples["question"]
    answers = examples["answer"] if examples.get("answer", None) is not None else examples["gt"]
    step_result = _middle_final_generate(problems, answers, config, llm)

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

        # evaluate_turn_ls = [
        #     [
        #         _sal_reward_fn(solution_str=message[2]["content"], ground_truth=answer) == ("[VERIFY] correct" in message[-1]["content"])
        #         for message in messages
        #     ]
        #     for answer, messages in zip(answers, step_result["messages"])
        # ]

        # 最后就提取这几个：
        # correct_turn_ls 为 incorrect-to-correct 和 correct-to-correct 的，同时
        # evaluate_turn_ls 为 True 的，最后查看数量

        step_result["correct_turn_ls"] = correct_turn_ls
        # step_result["evaluate_turn_ls"] = evaluate_turn_ls


    # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples
