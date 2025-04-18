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

import os
import logging
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    get_full_repo_name,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()
date_time = time.strftime("%m%d%H%M", time.localtime())


def get_dataset(config: Config) -> Dataset:
    dataset = load_dataset(config.dataset_name, data_files=config.dataset_files, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset = dataset.select(range(config.dataset_start, config.dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config, acc=None, file_name=None):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has diff_of_n_data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=True,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            config.output_dir = f"data/{config.model_path}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        target_file = file_name if file_name is not None else f"{config.output_dir}/bon_completions_s{config.dataset_start}_e{config.dataset_end}_n{config.n}_len{len(dataset)}_acc{acc}_{date_time}.jsonl"
        dataset.to_json(target_file, lines=True)
        logger.info(f"Saved completions to {target_file}")


def get_json_files(directory):
    """
    获取指定路径下所有JSON和JSONL文件的路径列表

    参数：
    directory (str): 要搜索的目录路径，也可以是文件路径

    返回：
    list: 包含所有匹配文件路径的列表，找不到文件时返回空列表
    """
    json_files = []

    # 检查路径是否存在
    if not os.path.exists(directory):
        return json_files

    # 处理单个文件的情况
    if os.path.isfile(directory):
        if directory.lower().endswith(('.json', '.jsonl')):
            return [directory]
        return json_files

    # 遍历目录树
    for root, _, files in os.walk(directory):
        for file in files:
            # 不区分大小写检查扩展名
            if file.lower().endswith(('.json', '.jsonl')):
                # full_path = os.path.join(root, file)
                # json_files.append(full_path)
                json_files.append(file)

    return json_files


# 使用示例
if __name__ == "__main__":
    target_path = "/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-7b-rej_sample_data/"  # 替换为你的目标路径
    result = get_json_files(target_path)
    print("找到的JSON/JSONL文件：")
    for file in result:
        print(file)
