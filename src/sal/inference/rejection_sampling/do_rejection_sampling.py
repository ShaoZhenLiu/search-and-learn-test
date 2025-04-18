import sys

import torch
from vllm import LLM
from datasets import Dataset
from sal.config import Config
from sal.utils.data import get_dataset, save_dataset

from .single_turn_generation import single_turn_sample

def sample_and_save(dataset: Dataset, llm: LLM, config: Config, sample_config):
    config.n = sample_config["sample_num"][sample_config["turn"]]
    config.temperature = sample_config["temperature"][sample_config["turn"]]
    dataset = dataset.map(
        single_turn_sample,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "llm": llm},
        desc=f"Rejection Sampling Turn {sample_config["turn"]}",
        load_from_cache_file=False,
    )
    save_dataset(dataset, config)
    return dataset


def do_rejection_sampling(dataset: Dataset, llm: LLM, config: Config):
    """
    难度感知拒绝采样：https://zhuanlan.zhihu.com/p/708371895

    做三轮采样
    第一轮sample 2次，保存所有数据；过滤数据，保留全部错误的题目，进入下一轮
    第二轮sample 10次，保存所有数据；过滤数据，保留全部错误的题目，进入下一轮
    第三轮sample 100次，保存所有数据

    在三轮采样的过程中，采样的参数也会进行相应的改变：
    * temperature在这三轮中依次为：0.6, 1.0, 1.6 (llama2 + DART-Math)
    * max_token = 20480 (虽然原来是2048，但是我想更大一点，提升到蒸馏数据平均最长的水平上)
    * top_p = 0.95
    """
    rejection_sample_config = {
        "turn": 0,
        "temperature": [0.6, 1.0, 1.3],
        "sample_num": [2, 10, 100],
    }
    for i in range(3):
        rejection_sample_config["turn"] = i
        dataset = sample_and_save(dataset, llm, config, rejection_sample_config)
        dataset = dataset.filter(lambda x: True not in x["correct_ls"])  # 过滤，只保留全错的数学问题
        print(dataset)
    return dataset

if __name__ == '__main__':
    from sal.utils.parser import H4ArgumentParser

    parser = H4ArgumentParser(Config)
    config = parser.parse()

    print(config.approach)
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
        # max_num_seqs=1024,  # 一次最多生成512个序列
    )

    dataset = get_dataset(config)
    dataset = do_rejection_sampling(dataset, llm, config)

