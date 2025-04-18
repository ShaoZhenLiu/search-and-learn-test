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
from datasets import Value

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.inference import (
    iterative_generate,
    diff_of_n,
    diff_of_n_multi_turn,
    iterative_generate_multi_turn,
    multi_turn_with_validate,
    think_twice,
    middle_final_generate,
)
from sal.inference.rejection_sampling import do_rejection_sampling
from sal.inference.test_time_scaling import test_time_scaling
from sal.inference.direct_gen import VLLMServerManager
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
    "iter_gen_multi_turn": iterative_generate_multi_turn,
    "diff_of_n_multi_turn": diff_of_n_multi_turn,
    "val_multi_turn": multi_turn_with_validate,
    "think2": think_twice,
    "mid_fin_gen": middle_final_generate,
    "tts": test_time_scaling,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    print(config.approach)
    if config.step_prompt is not None:
        print("step_prompt length:", len(config.step_prompt))

    num_gpus = torch.cuda.device_count()
    print('available gpu number:', num_gpus)
    # num_gpus = 2  # 给别人留
    # if config.use_vllm_server:
    #     pass
    # else:
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        tensor_parallel_size=num_gpus,
        enforce_eager=True,
        # max_num_seqs=1024,  # 一次最多生成512个序列
    )

    dataset = get_dataset(config)

    if config.approach in ["val_multi_turn"]:
        if "correct" in dataset.column_names and dataset.features["correct"].dtype == "bool":
            print("change dtype to float32")
            dataset = dataset.cast_column("correct", Value("float32"))

    if config.approach not in ["rej_sample"]:
        dataset = dataset.map(
            APPROACHES[config.approach],
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm},
            desc="Running search",
            load_from_cache_file=False,
        )
    else:
        dataset = do_rejection_sampling(dataset=dataset, llm=llm, config=config)

    save_dataset(dataset, config)

    # # 然后根据 dataset 中的解和打分，生成最好的答案
    # dataset = score(dataset, config)
    acc = None
    if config.calculate_correct:
        # dataset, acc = sal_reward_fn(dataset, config)  # 判断输出正误，同时，过滤掉错误的数据
        acc = sum(dataset["correct"]) / len(dataset) * 100 if len(dataset) != 0 else 0
        logger.info(f"模型生成答案的准确性为: {acc}%")

    if config.approach == "diff_of_n":
        # 如果属性 k_diff_solutions 或 pred_res 分别是 [] 和 None 的话，说明该目标生成失败，需要过滤掉
        dataset = dataset.filter(lambda x: (x["k_diff_solutions"] != []) and (x["pred_result"] is not None))

    save_dataset(dataset, config, acc)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
