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

# import re
import copy
import json
import logging
from collections import defaultdict

import regex
import numpy as np
from sympy.strategies.core import switch
from tqdm import tqdm
from jinja2 import Template
from vllm import LLM, SamplingParams

from sal.config import Config

logger = logging.getLogger()
from sal.utils.score import aggregate_scores


def _diff_of_n(batch_of_prompts, config: Config, llm: LLM) -> dict:
    # 这里是特例，batch为1的时候的策略
    # 1) 判断模型多个解题思路中, 具有代表性且有显著不同的解法（2~10种，预先设定）
    # 2) 对比多个解题思路的结果是否一致（直接用一个prompt进行对比，加上问题和前面提出的解法）
    # 3) 如果不一致, 判断哪个结果更为可靠

    # step_result = {
    #     f"step{i}_res": ["" for _ in range(len(batch_of_prompts))] for i in range(4)
    # }

    # step_result_tokens = {
    #     f"step{i}_res": [0 for _ in range(len(batch_of_prompts))] for i in range(4)
    # }

    n_solutions = [[] for _ in range(len(batch_of_prompts))]
    from_n_to_k = ["" for _ in range(len(batch_of_prompts))]
    k_diff_solutions = [[] for _ in range(len(batch_of_prompts))]
    consistency_evaluation = ["" for _ in range(len(batch_of_prompts))]

    for i in range(4):
        convs = [
            [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": problem if not i else Template(config.step_prompt[f"step{i}"]).render(
                        problem=problem,
                        # n=config.n,  # 1-最开始n个答案
                        n_solutions=n_solutions[p_index],  # 1-具体的n个答案
                        # k_diff=config.k_diff,  # 2-筛选后k个答案
                        k_diff_solutions=k_diff_solutions[p_index],  # 2-选择后的k个答案
                        consistency_evaluation=consistency_evaluation[p_index],  # 3-对k个答案的评估
                    )
                },
            ]
            for p_index, problem in enumerate(batch_of_prompts)  # 有几个问题，就构造几个对话
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
            top_p=config.top_p,  # 解码过程中，概率累积到多少的时候截断
            top_k=config.top_k,
            # best_of=config.n,  # 每次都生成n个
            n=config.n if i == 0 else 1,  # 第一次返回n个，之后就只返回1个
        )

        responses = llm.generate(
            templated_convs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )  # 一次回答 25 个问题

        if i == 0:  # 第一次回答n个问题
            n_solutions = [
                [
                    output.text
                    for output in r.outputs  # n个
                ]
                for r in responses  # search_batch_size个
            ]

        if i == 1:  # 第二次选出k个不一样的问题
            from_n_to_k = [
                output.text
                for r in responses  # search_batch_size个
                for output in r.outputs  # 1个
            ]
            def func(text):
                try:
                    return list(map(int,
                        regex.findall(config.reg_for_number, regex.search(config.reg_for_box, text).group(1))
                    ))
                except:
                    return None  # 匹配不出来就返回None，之后就都是None了
            k_diff_solutions_indices = [
                func(text)
                for text in from_n_to_k
            ]  # 提取k个目标
            k_diff_solutions = [
                [
                    solution
                    for s_index, solution in enumerate(solution_ls)
                    if (k_diff_solutions_index is not None) and (s_index + 1 in k_diff_solutions_index)  # 因为s_index从0开始，而k_diff_solutions_index从1开始
                ]  # 如果 k_diff_solutions_index 是 None，就一个都选不出，变成[]
                for solution_ls, k_diff_solutions_index in zip(n_solutions, k_diff_solutions_indices)
            ]

        if i == 2:
            consistency_evaluation = [
                output.text
                for r in responses
                for output in r.outputs
            ]

        if i == 3:  # 该收集结果了
            conclusion = [
                output.text
                for r in responses
                for output in r.outputs
            ]
            def func(text):
                try:
                    return regex.search(config.reg_for_box, text).group(1)
                except:
                    return None
            pred_res = [
                func(text)
                for text in conclusion
            ]
            step_result = {
                "n_solutions": n_solutions,
                "from_n_to_k": from_n_to_k,
                "k_diff_solutions": k_diff_solutions,
                "consistency_evaluation": consistency_evaluation,
                "conclusion": conclusion,
                "pred_result": pred_res,
            }

    return step_result


def diff_of_n(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"]
    step_result = _diff_of_n(problems, config, llm)

    # Group together alike beams and store in the dataset
    for step, res_ls in step_result.items():
        examples[step] = res_ls  # 外面再套一层列表，这是因为datasets默认吧n_solutions里面的5个元素当成5个问题的回答了，放到一个列表中就好了

    return examples
