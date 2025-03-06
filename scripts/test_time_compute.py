#!/usr/bin/env python
# encoding=utf-8
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

import logging

import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.inference import iterative_generate, diff_of_n, diff_of_n_multi_turn, iterative_generate_multi_trun
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from sal.utils.rewards import sal_reward_fn

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "iter_gen": iterative_generate,
    "diff_of_n": diff_of_n,
    "iter_gen_multi_turn": iterative_generate_multi_trun,
    "diff_of_n_multi_turn": diff_of_n_multi_turn,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]  # 根据不同的搜索策略，选择不同的搜索函数

    num_gpus = torch.cuda.device_count()
    print('available gpu number:', num_gpus)
    # num_gpus = 2  # 给别人留
    llm = LLM(
        model=config.model_path,
        # gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
    )
    prm = None if config.approach in ["iter_gen", "diff_of_n"] else load_prm(config)

    dataset = get_dataset(config)

    # 首先生成特定的解和 prm 的打分，保存到 dataset 里面
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm} if config.approach in ["iter_gen", "diff_of_n"] else {"config": config, "llm": llm, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    # # 然后根据 dataset 中的解和打分，生成最好的答案
    # dataset = score(dataset, config)
    acc = None
    if config.calculate_correct:
        dataset, acc = sal_reward_fn(dataset, config)  # 判断输出正误，同时，过滤掉错误的数据
        logger.info(f"模型生成答案的准确性为: {acc}%")

    if config.approach == "diff_of_n":
        # 如果属性 k_diff_solutions 或 pred_res 分别是 [] 和 None 的话，说明该目标生成失败，需要过滤掉
        dataset = dataset.filter(lambda x: (x["k_diff_solutions"] != []) and (x["pred_result"] is not None))

    save_dataset(dataset, config, acc)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
