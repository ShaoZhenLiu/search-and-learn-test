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
from sal.utils.rewards.math_reward import _sal_reward_fn
from sal.utils.rewards.math_utils import extract_answer
from .utils import vllm_generate, agg_data


def _think_twice(batch_of_prompts, config: Config, llm: LLM) -> dict:
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": Template(config.step_prompt["turn0"]).render(problem=problem)},
        ]
        for p_index, problem in enumerate(batch_of_prompts) for _ in range(config.n)  # n==8，即每个问题生成8轮对话，正确性取平均
    ]  # 构建对话 prompt
    token_len_ls = []

    responses, outputs_ls, output_token_ls = vllm_generate(convs, config, llm)
    token_len_ls.append(output_token_ls)
    convs = [
        [
            conv[0],  # system prompt
            # *conv,
            # {"role": "assistant", "content": output},
            {"role": "user", "content": Template(config.step_prompt[f"turn1"]).render(
                problem=conv[1]["content"],
                answer=extract_answer(output),
            )},
        ]
        for conv_index, (conv, output) in enumerate(zip(convs, outputs_ls))
    ]
    responses, outputs_ls, output_token_ls = vllm_generate(convs, config, llm)  # 得到最终的答案
    token_len_ls.append(output_token_ls)
    final_convs = [
        [
            *conv,
            {"role": "assistant", "content": output},
        ]
        for conv_index, (conv, output) in enumerate(zip(convs, outputs_ls))
    ]

    # 将token_len的形状从 [4, batch] 变成 [batch, 4]
    new_token_len_ls = [list(token_ls) for token_ls in zip(*token_len_ls)]

    agg_final_convs, agg_token_len_ls = agg_data(final_convs, new_token_len_ls, config, len(batch_of_prompts))

    step_result = {
        "messages": agg_final_convs,
        "pred_cot_token_len": agg_token_len_ls,
    }

    return step_result


def think_twice(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"] if examples.get("problem", None) is not None else examples["question"]
    answers = examples["answer"] if examples.get("answer", None) is not None else examples["gt"]
    step_result = _think_twice(problems, config, llm)

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

    # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples
