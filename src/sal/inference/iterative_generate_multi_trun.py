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
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from jinja2 import Template
from vllm import LLM, SamplingParams

from sal.config import Config

logger = logging.getLogger()
from sal.utils.score import aggregate_scores


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
        top_k=config.top_k,
        best_of=1,  # 每次都生成1个
        # n=config.n if i == 0 else 1,  # 第一次返回n个，之后就只返回1个
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    responses = sorted(responses, key=lambda x: int(x.request_id))  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in responses]

    return responses, outputs


def _iterative_generate_multi_turn(batch_of_prompts, config: Config, llm: LLM) -> dict:
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": Template(config.step_prompt["turn0"]).render(problem=problem)},
        ]
        for p_index, problem in enumerate(batch_of_prompts)  # 有几个问题，就构造几个对话
    ]  # 构建对话 prompt

    def generate_convs(old_convs, prompt_index):
        responses, outputs_ls = vllm_generate(old_convs, config, llm)
        new_convs = [
            [
                *conv,
                {"role": "assistant", "content": output},
                {"role": "user", "content": config.step_prompt[f"turn{prompt_index}"]},
            ]
            for conv_index, (conv, output) in enumerate(zip(old_convs, outputs_ls))
        ]
        return new_convs

    for i in range(1, 4):
        convs = generate_convs(convs, prompt_index=i)
    responses, outputs_ls = vllm_generate(convs, config, llm)  # 得到最终的答案
    final_convs = [
        [
            *conv,
            {"role": "assistant", "content": output},
        ]
        for conv_index, (conv, output) in enumerate(zip(convs, outputs_ls))
    ]

    step_result = {
        "messages": final_convs,
    }

    return step_result


def iterative_generate_multi_turn(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"]
    step_result = _iterative_generate_multi_turn(problems, config, llm)

    # # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples
