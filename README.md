本项目基于 [Search and Learn](https://github.com/huggingface/search-and-learn) 制作
# 项目名字

> 描述

# Installation

```shell
conda create -n sal python=3.10 && conda activate sal
```

创建万环境后，安装pytorch，然后运行下面的命令

```shell
pip install -e '.[dev]'
```

环境就基本上创建完成了

# Configuration

# Replicating Scaling Test Time Compute results:
The [recipes readme](recipes/README.md) includes launch commands and config files in order to replicate our results.

基本的参数都在 `src/sal/config.py` 中，如果有参数想要设置，可以设置在 `recipes` 中，并在运行的时候将该路径作为第二个参数，比如：

```
python scripts/test_time_compute.py <YAML_CONFIG>
# for example:
python scripts/test_time_compute.py recipes/Llama-3.1-8B-Instruct/diff_of_n.yaml
```

上访的代码也是直接运行程序的基本代码，`scripts/test_time_compute.py` 是入口，目前默认运行的是 diff_of_n 方法。

本仓库中有有关 PRM 的方法，不过因为我们在生成数据的时候用不上它们，所以不用管它们。

别的就没啥需要注意的了，目前主要改写的代码是在 `src/sal/inference` 中，基本上只需要查看它们即可。