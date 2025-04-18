from jinja2 import Template
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.utils.rewards.math_reward import _sal_reward_fn


def vllm_generate(convs_ls, config: Config, llm: LLM):
    tokenizer = llm.get_tokenizer()
    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template
    templated_convs = tokenizer.apply_chat_template(
        convs_ls,
        add_generation_prompt=True,  # 会自动在最后加上，这样模型就不会先生成"<|start_header_id|>assistant<|end_header_id|>"再回答了
        tokenize=False,
    )  # 将对话 prompt 按照大模型的方式编码

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,  # 解码过程中，概率累积到多少的时候截断
        # top_k=config.top_k,
        best_of=1,  # 每次都生成1个
        # repetition_penalty=1.1
        # n=config.n if i == 0 else 1,  # 第一次返回n个，之后就只返回1个
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    responses = sorted(responses, key=lambda x: int(x.request_id))  # sort outputs by request_id
    outputs = [output.outputs[0].text for output in responses]
    output_token_ids_ls = [len(output.outputs[0].token_ids) for output in responses]

    return responses, outputs, output_token_ids_ls


def generate_convs(old_convs, prompt_index, config: Config, llm: LLM, **kwargs):
    answers = kwargs.get("answers", None)
    responses, outputs_ls, output_token_ls = vllm_generate(old_convs, config, llm)
    new_convs = [
        [
            *conv,
            {"role": "assistant", "content": output},
            {"role": "user", "content": Template(config.step_prompt[f"turn{prompt_index}"]).render(
                # correctness=True if "[VERIFY] correct" in output else False
                correctness=_sal_reward_fn(
                    solution_str=output,
                    ground_truth=answers[conv_index // config.n],
                    enable_llm=False, check_think=False,
                )
            )},
        ]
        for conv_index, (conv, output) in enumerate(zip(old_convs, outputs_ls))
    ]
    return new_convs, output_token_ls


def agg_data(final_convs, new_token_len_ls, config, batch_of_prompts_len):
    agg_final_convs = []
    agg_token_len_ls = []
    for i in range(batch_of_prompts_len):
        real_output = final_convs[i * config.n: (i + 1) * config.n]
        output_token_ids = new_token_len_ls[i * config.n: (i + 1) * config.n]
        # agg_final_convs.append(real_output[0] if len(real_output) == 1 else real_output)
        # agg_token_len_ls.append(output_token_ids[0] if len(output_token_ids) == 1 else output_token_ids)
        agg_final_convs.append(real_output)
        agg_token_len_ls.append(output_token_ids)
    return agg_final_convs, agg_token_len_ls