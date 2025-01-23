from datasets import load_dataset


# if __name__ == '__main__':
#     dataset = load_dataset("../../hf_datasets/NuminaMath-CoT", split="train")  # 自动加载parquet数据集
#     print(dataset)
#
#     filtered_dataset = dataset.filter(lambda x: x["source"] == "cn_k12")
#
#     # 将过滤后的数据集保存为新的 JSONL 文件
#     print(filtered_dataset)
#     filtered_dataset.to_parquet("./filtered_dataset.parquet")  # 276554

if __name__ == '__main__':
    dataset = load_dataset(path="./cn_k12", split="train")
    print(dataset)