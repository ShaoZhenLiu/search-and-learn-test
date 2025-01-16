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
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import regex

from huggingface_hub import get_full_repo_name

from sal.utils.hub import get_dataset_revisions


@dataclass
class Config:
    approach: Literal["best_of_n", "beam_search", "dvts", "iter_gen", "diff_of_n"] = "diff_of_n"
    model_path: str = "/data/shaozhen.liu/python_project/hf_models/gemma-2-27b-it/"
    gpu_memory_utilization: float = (
        0.5  # vllm is allocated 0.5 of GPU memory, the PRM uses the rest
    )
    prm_path: str = "/data/shaozhen.liu/python_project/hf_models/Llama3.1-8B-PRM-Deepseek-Data"
    # Output Related Options
    output_dir: str = "./data"
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    dataset_name: str = "/data/shaozhen.liu/python_project/hf_datasets/MATH-500"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = None

    # Chat template related options
    # todo 换一下
    system_prompt: str = (
        "If you need to solve the math problem, solve it efficiently and clearly:"
        "\n\n- For simple problems (2 steps or fewer):"
        "\nProvide a concise solution with minimal explanation."
        "\n\n- For complex problems (3 steps or more):"
        "\nUse this step-by-step format:"
        "\n\n## Step 1: [Concise description]"
        "\n[Brief explanation and calculations]"
        "\n\n## Step 2: [Concise description]"
        "\n[Brief explanation and calculations]"
        "\n\n..."
        "\n\nRegardless of the approach, always conclude with:"
        "\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct."
        "\n\nWhere [answer] is just the final number or expression that solves the problem.")
    custom_chat_template: str = '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
    step_prompt = {
        "step1": "Generate a new problem-solving approach:"
                 "\nPlease propose a new problem-solving approach that is significantly different from the previous solution."
                 "\nThe previous solution is: {{ approach_1 }}."
                 "\n\nProvide a detailed explanation of the logic and steps of the new approach, ensuring it does not rely on the previous method."
                 "\nThe problem is: {{ problem }}.",
        "step2": "Compare the results of two problem-solving approaches:"
                 "\nPlease compare two approaches for the following problem: {{ problem }}"
                 "\n- Approach 1: {{ approach_1 }}"
                 "\n- Approach 2: {{ approach_2 }}"
                 "\n\nAnalyze whether the results of these two methods are consistent and provide your reasoning.",
        "step3": "Determine which result is more reliable:"
                 "\nIf the results of Approach 1 and Approach 2 are inconsistent, please analyze which result is more reliable based on the following criteria: "
                 "\n- Is the logic of the approach rigorous?"
                 "\n- Is the diff_of_n_data used complete and reliable?"
                 "\n- Does the method align with practical applications?"
                 "\n\n- Problem description: {{ problem }}"
                 "\n- Approach 1: {{ approach_1 }}"
                 "\n- Approach 2: {{ approach_2 }}"
                 "\n- Comparison analysis: {{ comparison }}"
                 "\n\nRefer to the above information to provide a step-by-step analysis and your conclusion.",
    }

    # Search Related Options
    n: int = 4
    temperature: float = 0.6  # 更低一点 0.6 或 0.4
    top_p: float = 0.95
    top_k: int = 100
    prm_batch_size: int = 4
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 2048
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    # DVTS / Beam Search options
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40  # 如果分步实现，整个搜索的深度
    lookahead: int = 1  # 搜一次会让模型迭代几次，或者说，几句话

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False

    # diff_of_n options:
    reg_for_box = regex.compile(r"\\boxed{((?:[^{}]+|{(?1)})*)}")
    reg_for_number = regex.compile(r"\d+")

    def __post_init__(self):
        if self.approach == "diff_of_n":
            self.step_prompt = {  # todo 让模型先分析，然后再给出答案，强调顺序
                "step1": "You are given a math problem: {{ problem }}."
                         "\n\nAnd the following {{ n_solutions | length }} solution ideas:\n"
                         "{% for solution in n_solutions %}"
                         "- solution {{ loop.index }}: {{ solution }}\n"
                         "{% endfor %}"
                         "\nBefore selecting solutions, please analyze the provided solution ideas. Consider their unique aspects and how representative they are of different approaches to the problem. You should address the following points for each solution:\n"
                         "\n- What makes this solution unique?"
                         "\n- How does it differ from other solutions?"
                         "\n- In what way does it represent a distinct approach to the problem?"
                         "\nAfter your analysis, please select k (1 < k < {{ n_solutions | length }}) solutions that are both representative and significantly different from each other. Explain briefly why you chose each solution."
                         "\n\nRegardless of the approach, always conclude with:"
                         "\nTherefore, the indices of the selected solutions are: $\\boxed{answer}$. I hope you like it."
                         "\nWhere [answer] is an indices sequence indicating the indices of the selected solutions in the previous solution list, which is ranging from 1 to {{ n_solutions | length }}."
                         "\nMultiple indices must be separated by a comma and a space. For example: $\\boxed{1, 3, 5}$. Do not concatenate the digits (e.g., do not output $\\boxed{135}$).",
                "step2": "Given the math problem: {{ problem }}."
                         "\n\nAnd the {{ k_diff_solutions | length }} representative solution ideas:\n"
                         "{% for solution in k_diff_solutions %}"
                         "- solution {{ loop.index }}: {{ solution }}\n"
                         "{% endfor %}"
                         "\n\nPlease compare the final results or outcomes of these k solutions to determine if they are consistent with each other. Provide your reasoning in detail.",
                "step3": "**Given the following math problem:**"
                         "\n{{ problem }}."
                         "\n\n**We have {{ k_diff_solutions | length }} representative solution approaches:**\n"
                         "{% for solution in k_diff_solutions %}"
                         "- **Approach {{ loop.index }}:** {{ solution }}\n"
                         "{% endfor %}"
                         "\n**Here is the consistency evaluation from Step 2:** "
                         "\n{{ consistency_evaluation }}"
                         "\n\nIf all these approaches are consistent, please directly provide the final answer to the math problem."
                         "\nIf the approaches are not consistent, please determine which approach or approaches seem most reliable, briefly explain why you believe these are most reliable, and then give the final solution to the math problem."
                         "\n\nRegardless of the approach, always conclude with:"
                         "\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct."
                         "\n\nWhere [answer] is just the final number or expression that solves the math problem."
            }

        if self.approach == "dvts":
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
            self.n_beams = self.n // self.beam_width

        if self.approach == "beam_search":
            # : implemented a batched version
            if self.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for beam_search")

        # Setting up push to hub dataset
        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                # Set default based on model name. We prepend the username for compatibility with the repo checks below.
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            if self.approach == "beam_search" or self.approach == "dvts":
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--m-{self.beam_width}--iters-{self.num_iterations}--look-{self.lookahead}--seed-{self.seed}--agg_strategy--{self.agg_strategy}"
            elif self.approach == "best_of_n":
                self.revision = f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}--n-{self.n}--seed-{self.seed}--agg_strategy-{self.agg_strategy}"
            else:
                raise ValueError(f"Unknown approach {self.approach}")
            if self.dataset_start is not None and self.dataset_end is not None:
                self.revision = (
                    f"{self.revision}--chunk-{self.dataset_start}_{self.dataset_end}"
                )

            # Early exit if the revision on the Hub already exists
            if not self.overwrite_hub_revision and self.revision in revisions:
                # logger.info(f"Revision {revision} already exists on the Hub. Exiting.")
                exit()


if __name__ == '__main__':
    config = Config
    print(config.system_prompt)

"""
If you need to solve the math problem, solve it efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
"""

"""
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}

{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}

{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""
