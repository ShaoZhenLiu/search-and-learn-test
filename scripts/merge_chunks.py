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
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import List

from datasets import concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from sal.utils.hub import get_dataset_revisions
from sal.utils.data import get_json_files

"""Merge revisions of a dataset into a single config.

Usage:

# Merge all revisions of a dataset
python scripts/merge_chunks.py \
    --dataset_name reliable-agents/Qwen2.5-Math-1.5B-Instruct-bon-prm-completions

# Merge only revisions that contain "last" or "T-0.0" in their name
python scripts/merge_chunks.py \
    --dataset_name reliable-agents/Qwen2.5-Math-1.5B-Instruct-bon-prm-completions \
    --filter_strings last T-0.0
"""


@dataclass
class Args:
    dataset_name: str = "/data/shaozhen.liu/python_project/hf_datasets/DeepScaleR-7b-rej_sample_data/turn1/"
    dataset_split: str = "train"
    filter_strings: List[str] = field(default_factory=list)
    download_from_remote: bool = False
    push_to_hub: bool = False


def load_single_revision(args):
    """Load a single dataset revision."""
    dataset_name, revision, dataset_split = args
    return load_dataset(
        dataset_name,
        revision=revision,
        trust_remote_code=True,
        split=dataset_split,
        download_mode="force_redownload",
    )


def prepare_remote_data(args):
    revisions = get_dataset_revisions(args.dataset_name)

    if args.filter_strings:
        revisions = [
            revision
            for revision in revisions
            if all(filter_string in revision for filter_string in args.filter_strings)
        ]

    merged_config = revisions[0].split("--chunk")[0]
    print(f"Merging {len(revisions)} revisions to create config `{merged_config}`")

    # Prepare arguments for multiprocessing
    pool_args = [
        (args.dataset_name, revision, args.dataset_split) for revision in revisions
    ]

    # Use multiprocessing to load datasets in parallel
    with Pool(cpu_count()) as pool:
        datasets = list(
            tqdm(
                pool.imap(load_single_revision, pool_args),
                total=len(revisions),
                desc="Loading datasets",
            )
        )

    return datasets, merged_config



def load_single_local_file(args):
    """加载本地的一个json或jsonl文件"""
    dataset_path, data_file, dataset_split = args
    return load_dataset(
        dataset_path,
        data_files=data_file,
        split=dataset_split,
    )


def prepare_local_data(args):
    files_ls = get_json_files(args.dataset_name)
    args.filter_strings = ["bon_completions"]

    if args.filter_strings:
        files_ls = [
            file_name
            for file_name in files_ls
            if all(filter_string in file_name for filter_string in args.filter_strings)
        ]

    merged_config = files_ls[0].split("_s")[0]
    print(files_ls)
    print(f"Merging {len(files_ls)} revisions to create config `{merged_config}`")

    # Prepare arguments for multiprocessing
    pool_args = [
        (args.dataset_name, file, args.dataset_split) for file in files_ls
    ]
    # print(pool_args)

    # Use multiprocessing to load datasets in parallel
    with Pool(cpu_count()) as pool:
        datasets_ls = list(
            tqdm(
                pool.imap(load_single_local_file, pool_args),
                total=len(files_ls),
                desc="Loading datasets",
            )
        )

    # datasets_ls = [load_single_local_file(arg) for arg in pool_args]

    return datasets_ls, merged_config

def main():
    parser = HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]

    if args.download_from_remote:
        datasets, merged_config = prepare_remote_data(args)
    else:
        print("merge local data")
        datasets, merged_config = prepare_local_data(args)

    # Concatenate datasets
    print("start concatenate dataset")
    merged_dataset = concatenate_datasets(datasets)

    # Sanity check
    print(merged_dataset)

    if args.push_to_hub:
        # Push merged dataset to the hub
        url = merged_dataset.push_to_hub(
            args.dataset_name,
            config_name=merged_config,
            split=args.dataset_split,
            private=True,
        )
        print(f"Pushed merged dataset to {url}")
    else:
        merged_dataset.to_json(f"{args.dataset_name}/merged_dataset.jsonl")


if __name__ == "__main__":
    main()
