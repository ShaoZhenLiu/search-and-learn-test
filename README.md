本项目基于 [Search and Learn](https://github.com/huggingface/search-and-learn) 制作
# 项目名字

> 描述

# Installation

建议按照下面的格式进行环境的配置
* gcc和gxx是11.4是bitsandbytes库的建议版本
* cuda是12.1是vllm的建议版本

```shell
mamba create -n sal python=3.12 -y
mamba activate sal

# gcc gxx
mamba install gcc=11.4
mamba install gxx=11.4

# cuda
mamba install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0

# pytorch
mamba install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia/label/cuda-12.1.0

# xgrammar (from vllm) 手动下载
wget https://files.pythonhosted.org/packages/48/27/9afa14acb41d765427c52639a24ec0c2b0e979c0b85b537dc26db629cc85/xgrammar-0.1.11-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
# 根据命令 pip debug --verbose 输入后出现的 Compatible tags
# 选择和当前 xgrammar .whl包最相近的后缀，重命名xgrammar
mv xgrammar-0.1.11-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl xgrammar-0.1.11-cp312-cp312-manylinux_2_17_x86_64.whl
pip install ./xgrammar-0.1.11-cp312-cp312-manylinux_2_17_x86_64.whl

# vllm 手动下载
wget https://files.pythonhosted.org/packages/51/70/6fc00dca2e9f53a76b7792d788cb2efbb9d2587ed0ca9a71d5ccf7fc7543/vllm-0.7.0-cp38-abi3-manylinux1_x86_64.whl
pip install ./vllm-0.7.0-cp38-abi3-manylinux1_x86_64.whl

# 最后安装sal包，同时安装别的边边角角的东西即可，在工作目录下运行以下代码
pip install -e '.[dev]'

# 微调内存不够，需要下面的库，不微调的话不安装下面库的也可以
pip install flash-attn --no-build-isolation
pip install bitsandbyts
```

环境就基本上创建完成了

# Configuration

The [recipes readme](recipes/README.md) includes launch commands and config files in order to replicate our results.

基本的参数都在 `src/sal/config.py` 中，如果有参数想要设置，可以设置在 `recipes` 中，并在运行的时候将该路径作为第二个参数，比如：

```bash
python scripts/test_time_compute.py <YAML_CONFIG>
# for example:
python scripts/test_time_compute.py recipes/Llama-3.1-8B-Instruct/diff_of_n.yaml
# 或者运行bash脚本:
sh ./scripts/run.sh
```

上访的代码也是直接运行程序的基本代码，`scripts/test_time_compute.py` 是入口，目前默认运行的是 diff_of_n 方法。

本仓库中有有关 PRM 的方法，不过因为我们在生成数据的时候用不上它们，所以不用管它们。

别的就没啥需要注意的了，目前主要改写的代码是在 `src/sal/inference` 中，基本上只需要查看它们即可。

# Finetune

微调代码在 `scripts/finetune.py` 中，还没有整合到 config 和 sal 库中，之后稳定了再整合。

```bash
sh ./scripts/finetune.sh
```