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


def _iterative_generate(batch_of_prompts, config: Config, llm: LLM) -> dict:
    # 实现一次生成，每次中间需要插入下一个阶段的 prompt（和 llava-cot 类似，之后可以看看有没有什么新的方法）
    # 最长迭代 40 次，除了搜索之外，还有一些值得注意的地方
    # 中间如果有生成完的答案，直接放到 completed_beams 里
    # 如果 40 次后生成万能的答案数量不足 n，就直接复制之前的直到 n 个

    step_result = {
        f"step{i}_res": ["" for _ in range(len(batch_of_prompts))] for i in range(4)
    }

    step_result_tokens = {
        f"step{i}_res": [0 for _ in range(len(batch_of_prompts))] for i in range(4)
    }

    for i in range(4):
        convs = [
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": problem if not i else Template(config.step_prompt[f"step{i}"]).render(
                        problem=problem,
                        approach_1=step_result[f"step0_res"][p_index],
                        approach_2=step_result[f"step1_res"][p_index],
                        comparison=step_result[f"step2_res"][p_index],
                    )
                },
            ]
            for p_index, problem in enumerate(batch_of_prompts)  # 意思就是问 n 次，模型回答 n 次
        ]  # 构建对话 prompt
        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=True,  # 会自动在最后加上，这样模型就不会先生成"<|start_header_id|>assistant<|end_header_id|>"再回答了
            tokenize=False,
        )  # 将对话 prompt 按照大模型的方式编码


        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
        )

        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )  # 一次回答 25 个问题

        for p_index in range(len(step_result[f"step{i}_res"])):
            step_result[f"step{i}_res"] = [
                output.text
                for r in responses
                for output in r.outputs
            ]
            step_result_tokens[f"step{i}_res"] = [
                len(output.token_ids)
                for r in responses
                for output in r.outputs
            ]

    return step_result


def iterative_generate(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"]
    step_result = _iterative_generate(problems, config, llm)

    # # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples
