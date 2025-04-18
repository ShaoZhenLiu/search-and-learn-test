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

import numpy as np
from tqdm import tqdm
from jinja2 import Template
from vllm import LLM, SamplingParams

from sal.config import Config

logger = logging.getLogger()
from sal.utils.rewards.math_reward import _sal_reward_fn
from sal.utils.rewards.math_utils import extract_answer, parse_ground_truth, parse_question


def vllm_generate(convs_ls, config: Config, llm: LLM):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs_ls,
        add_generation_prompt=True,  # 会自动在最后加上，这样模型就不会先生成"<|start_header_id|>assistant<|end_header_id|>"再回答了
        tokenize=False,
    )  # 将对话 prompt 按照大模型的方式编码

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


def _single_turn_sample(batch_of_prompts, answers, config: Config, llm: LLM) -> dict:
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": problem},
        ]
        for p_index, problem in enumerate(batch_of_prompts) for _ in range(config.n)  # n==8，即每个问题生成8轮对话，正确性取平均
    ]  # 构建对话 prompt
    token_len_ls = []

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


def single_turn_sample(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """

    dataset_name = config.dataset_name.split("/")[-1].lower()
    examples_ls = [dict(zip(examples.keys(), item_values)) for item_values in zip(*examples.values())]

    problems = [parse_question(example, dataset_name) for example in examples_ls]
    answers = [parse_ground_truth(example, dataset_name)[1]  for example in examples_ls]
    step_result = _single_turn_sample(problems, answers, config, llm)

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
        step_result["correct_ls"] = correct_ls
        step_result["correct"] = agg_correct_ls

    # # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples
