from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.pipelines import TextGenerationPipeline

model = LlamaForCausalLM.from_pretrained("/data/shaozhen.liu/python_project/hf_models/Llama3.1-8B-Instruct")
tokenizer = PreTrainedTokenizerFast.from_pretrained("/data/shaozhen.liu/python_project/hf_models/Llama3.1-8B-Instruct")
model.to("cuda:0")
# tokenizer.to("cuda:0")
# print(config, tokenizer, model)
# pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)

system_prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
prompt = "Simplify: $\\frac{\\sqrt{2.5^2-0.7^2}}{2.7-2.5}$."
"12"
conversation = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
]

conversation = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
)
conversation = tokenizer(conversation, return_tensors="pt").to("cuda:0")
# conversation = tokenizer(conversation, return_tensors="pt")
# print(conversation)
# print(conversation.shape)
outputs = model.generate(
    conversation['input_ids'],
    attention_mask=conversation['attention_mask'],
    pad_token_id=tokenizer.eos_token_id,
    max_length=2048,  # 设置较大的 max_length
    num_return_sequences=1,  # 只返回一个序列
    temperature=0.7,  # 可选，控制输出的多样性
    # top_k=50,  # 可选，控制采样
    top_p=0.95,  # 可选，控制采样
    do_sample=True,  # 启用采样，避免贪婪解码
)

ans = tokenizer.decode(outputs[0])
print(ans)

# from vllm import LLM, SamplingParams
# from sal.config import Config
# from sal.utils.parser import H4ArgumentParser
#
#
# parser = H4ArgumentParser(Config)
# config = parser.parse()
# llm = LLM(
#     model=config.model_path,
#     gpu_memory_utilization=config.gpu_memory_utilization,
#     enable_prefix_caching=True,
#     seed=config.seed,
#     tensor_parallel_size=1,
# )
#
# system_prompt = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
# prompt = "Compute: $1-2+3-4+5- \\dots +99-100$."
#
# convs = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": prompt},
# ]
#
# tokenizer = llm.get_tokenizer()
# if config.custom_chat_template is not None:
#     tokenizer.chat_template = config.custom_chat_template
# templated_convs = tokenizer.apply_chat_template(
#     convs,
#     tokenize=False,
# )  # 将对话 prompt 按照大模型的方式编码
# sampling_params = SamplingParams(
#     temperature=config.temperature,  #
#     max_tokens=config.max_tokens,  # 最大 token 是几个
#     top_p=config.top_p,  # 选 top 几个
#     n=1,  # 采样一次
# )
# responses = llm.generate(
#     templated_convs,
#     sampling_params=sampling_params,
#     use_tqdm=True,
# )
# print(responses[0].outputs[0].text)
