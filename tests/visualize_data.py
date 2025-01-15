from datasets import load_dataset

if __name__ == '__main__':
    # 加载数据集
    dataset = load_dataset(path="../diff_of_n_data", split="train")

    # 选择要查看的索引
    sample_index = 13

    # 获取指定索引的数据（返回一个 dict）
    sample = dataset[sample_index]

    # 循环打印
    print(f"=== Attributes for sample index {sample_index} ===")
    for key, value in sample.items():
        if key in ["n_solutions", "k_diff_solutions"]:
            if key == "k_diff_solutions":
                continue
            print("="*10, f"{key}: ", "="*10)
            for i, solution in enumerate(value):
                print("#"*10, f"solution {i+1}", "#"*10)
                print(solution)
        else:
            print("="*10, f"{key}:", "="*10)
            print(value)

