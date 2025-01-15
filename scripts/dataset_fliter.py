from datasets import Dataset, load_dataset
from sal.config import Config
from sal.utils.parser import H4ArgumentParser


if __name__ == '__main__':
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    dataset = load_dataset(config.dataset_name)
    print(dataset)
    # 过滤 out level 为 4 和 5 的样本
    filtered_dataset = dataset.filter(lambda x: x['level'] == 5)

    # 将过滤后的数据集保存为新的 JSONL 文件
    print(filtered_dataset)
    filtered_dataset['test'].to_json("./filtered_dataset.jsonl")
