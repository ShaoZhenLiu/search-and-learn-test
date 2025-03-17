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
from vllm import LLM, SamplingParams

from sal.config import Config

logger = logging.getLogger()
from sal.utils.score import aggregate_scores
from sal.inference.direct_gen import ResponseCollector
from sal.utils.rewards.math_reward import _sal_reward_fn


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
        use_tqdm=True,
    )
    responses = sorted(responses, key=lambda x: int(x.request_id))  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in responses]
    output_token_ids_ls = [len(output.outputs[0].token_ids) for output in responses]

    return responses, outputs, output_token_ids_ls


def _iterative_generate_multi_turn(batch_of_prompts, config: Config, llm: LLM) -> dict:
    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": Template(config.step_prompt["turn0"]).render(problem=problem)},
        ]
        for p_index, problem in enumerate(batch_of_prompts)  # 有几个问题，就构造几个对话
    ]  # 构建对话 prompt
    token_len_ls = []


    def generate_convs(old_convs, prompt_index):
        responses, outputs_ls, output_token_ls = vllm_generate(old_convs, config, llm)
        token_len_ls.append(output_token_ls)
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

    step_result = {
        "pred_cot_token_len": new_token_len_ls,
        "messages": final_convs,
    }

    return step_result


def iterative_generate_multi_turn(examples, config: Config, llm: LLM):
    """
    examples: 根据 config.search_batch_size 调整里面的个数，默认为25
    """
    problems = examples["problem"] if examples.get("problem", None) is not None else examples["question"]
    answers = examples["answer"] if examples.get("answer", None) is not None else examples["gt"]
    step_result = _iterative_generate_multi_turn(problems, config, llm)

    if config.calculate_correct:
        correctness = [
            _sal_reward_fn(
                solution_str=messages[-1]["content"],  # 最后一个回答会输出在\boxed{}中的答案
                ground_truth=answer,
                enable_llm=False, check_think=False,
            )
            for answer, messages in zip(answers, step_result["messages"])
        ]
        step_result["correct"] = correctness

    # # Group together alike beams and store in the dataset
    # grouped_results = defaultdict(list)
    for step, res_ls in step_result.items():
        examples[step] = res_ls

    return examples


class MultiTurnResponseCollector(ResponseCollector):
    def __init__(self,
                 dataset,
                 server_manager,
                 turn_count=4,
                 prompt_dict=None,
                 max_retries=3,
                 ):
        super().__init__(dataset, server_manager)
        self.turn_count = turn_count  # 总轮次
        self.prompt_dict = prompt_dict or {}  # 每轮的prompt字典
        self.conversation_history = {}  # 保存每条数据的对话历史
        self.max_retries = max_retries  # 最大重试次数
        self.retries = {}  # 记录每条数据的重试次数

    async def send_request(self, session, idx, question, turn=1, conversation_id=None, retry_count=0):
        # 如果是新的对话，初始化对话历史和重试计数
        if conversation_id is None:
            conversation_id = idx
            self.conversation_history[conversation_id] = []
            self.retries[conversation_id] = 0
            # 设置第一轮的prompt
            question = Template(self.prompt_dict["turn0"]).render(problem=question)

        # 检查是否超过最大重试次数
        if self.retries[conversation_id] >= self.max_retries:
            exception_entry = {
                "exception": {
                    "message": f"Max retries ({self.max_retries}) exceeded for conversation {conversation_id}",
                    "conversation_id": conversation_id,
                    "turn": turn
                }
            }
            with self.lock:
                self.file.write(json.dumps(exception_entry) + "\n")
                print(f"Exception for conversation {conversation_id}: Max retries exceeded")
            self.completed_requests += 1
            self.active_requests -= 1
            return

        # 构造请求数据
        messages = self.conversation_history[conversation_id].copy()
        messages.append({"role": "user", "content": question})
        data = {
            "model": self.server_manager.model_name,
            "messages": messages
        }

        # try:
        async with session.post(
            f"{self.server_manager.base_url}/chat/completions",
            headers=self.server_manager.headers,
            json=data,
            # timeout=30
        ) as response:
            if response.status == 200:
                reply = await response.json()
                # 更新对话历史
                messages.append(reply['choices'][0]['message'])
                self.conversation_history[conversation_id] = messages
                print(f"Conversation {conversation_id} - Turn {turn} completed")

                # 如果未达到总轮次，获取下一个prompt并继续对话
                if turn < self.turn_count:
                    next_prompt = self.prompt_dict[f"turn{turn}"]
                    await self.send_request(session, idx, next_prompt, turn + 1, conversation_id)
                else:
                    # 达到总轮次，保存对话历史
                    with self.lock:
                        self.file.write(json.dumps({
                            "conversation_id": conversation_id,
                            "messages": messages
                        }) + "\n")
                        print(f"Conversation {conversation_id} saved")
                    self.completed_requests += 1
            else:
                error_entry = {
                    "error": {
                        "status_code": response.status,
                        "message": await response.text(),
                        "conversation_id": conversation_id,
                        "turn": turn
                    }
                }
                with self.lock:
                    self.file.write(json.dumps(error_entry) + "\n")
                    print(f"Error in conversation {conversation_id} - Turn {turn}: {response.status} - {await response.text()}")
                self.retries[conversation_id] += 1
                print(f"Retrying conversation {conversation_id} - Turn {turn} (Retry {self.retries[conversation_id]})")
                await self.send_request(session, idx, question, turn, conversation_id, retry_count + 1)
        # except Exception as e:
        #     print(e)
        #     exception_entry = {
        #         "exception": {
        #             "message": str(e),
        #             "conversation_id": conversation_id,
        #             "turn": turn
        #         }
        #     }
        #     with self.lock:
        #         self.file.write(json.dumps(exception_entry) + "\n")
        #         print(f"Exception in conversation {conversation_id} - Turn {turn}: {str(e)}")
        #     self.retries[conversation_id] += 1
        #     print(f"Retrying conversation {conversation_id} - Turn {turn} (Retry {self.retries[conversation_id]})")
        #     await self.send_request(session, idx, question, turn, conversation_id, retry_count + 1)
        # finally:
        #     self.active_requests -= 1
        self.active_requests -= 1

    async def worker(self, session):
        while self.queue:
            # if not self.running:
            #     break
            if self.queue:
                with self.lock:
                    idx, question = self.queue.popleft()
                    self.active_requests += 4
                # 启动多轮对话
                await self.send_request(session, idx, question, turn=1)
            else:
                await asyncio.sleep(0.1)

    async def start(self):
        self.running = True
        async with aiohttp.ClientSession() as session:
            tasks = [self.worker(session) for _ in range(self.worker_num)]  # 使用10个并发工作者
            await asyncio.gather(*tasks)

    def stop(self):
        self.running = False
        while self.active_requests > 0:
            time.sleep(0.1)
        self.file.close()
        print("多轮响应收集器已停止")


if __name__ == "__main__":
    import os
    from sal.inference.direct_gen import VLLMServerManager

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # 准备数据集
    dataset = [f"Test {i+1}" for i in range(20)]

    # 初始化服务器管理器
    server_manager = VLLMServerManager(
        model_name="/data/shaozhen.liu/python_project/hf_models/Qwen2.5-1.5B-Instruct",
        gpu_count=2,
        seed=0,
    )
    # server_manager.start()

    # 定义每轮的prompt字典
    prompt_dict = {
        1: "Please elaborate on your previous answer.",
        2: "Can you provide an example?",
        3: "Thank you for the information."
    }

    # 创建多轮响应收集器
    collector = MultiTurnResponseCollector(
        dataset,
        server_manager,
        turn_count=4,
        prompt_dict=prompt_dict
    )

    # 使用 asyncio.run() 启动异步任务
    async def main():
        await collector.start()

    # 如果是 Windows 系统，需要设置事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行异步任务
    asyncio.run(main())

    # 停止收集器和服务器
    collector.stop()
    server_manager.stop()