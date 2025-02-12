import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GemmaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
import torch


# 数据预处理函数
def format_data(example):
    # 合并指定字段
    merged_content = (
        f"Math Problem: {example['problem']}\n\n"
        f"Different Solutions:\n"
    )

    # 添加k_diff_solutions
    for i, solution in enumerate(example['k_diff_solutions'], 1):
        merged_content += f"Solution {i}: {solution}\n"

    merged_content += (
        f"\nEvaluation: {example['consistency_evaluation']}\n"
        f"Conclusion: {example['conclusion']}\n"
        f"Final Answer: {example['pred_result']}"
    )

    # 构造对话格式
    messages = [
        {"role": "user", "content": "Solve this math problem and show your reasoning."},
        {"role": "user", "content": merged_content},
        {"role": "assistant", "content": example['solution']}
    ]

    return {
        "formatted_input": merged_content,
        "messages": messages,
        "solution": example['solution']
    }


if __name__ == '__main__':
    # 加载数据集
    dataset_path = "/data/shaozhen.liu/python_project/search-and-learn/diff_of_n_data"
    dataset = load_dataset(dataset_path, data_files='bon_completions.jsonl', split='train')
    dataset = dataset.map(
        format_data,
        batched=False,
        desc="generate training example",
        load_from_cache_file=False,
    )

    # 划分训练验证集
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 加载模型和tokenizer
    model_id = "/data/shaozhen.liu/python_project/hf_models/gemma-2-27b-it/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 量化配置（4bit量化）
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True
    # )

    model = GemmaForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        device_map="auto",  # 自动分配设备
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"  # 启用Flash Attention
    )

    # 设置特殊token
    DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}<start_of_turn>{% if message['role'] == 'user' %}user\n{{ message['content'] }}<end_of_turn>\n{% else %}assistant\n{{ message['content'] }}<end_of_turn>\n{% endif %}{% endfor %}"
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    print(tokenizer.chat_template)


    # 数据转换函数
    def process_data(example):
        messages = example["messages"]

        # 应用chat template
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        )

        return {
            "input_ids": tokenized_chat[0],
            "labels": tokenized_chat[0].clone()
        }


    # 处理数据集
    tokenized_train = train_dataset.map(
        process_data,
        remove_columns=train_dataset.column_names,
        batched=False,
        load_from_cache_file=False,
    )
    tokenized_eval = eval_dataset.map(
        process_data,
        remove_columns=eval_dataset.column_names,
        batched=False,
        load_from_cache_file=False,
    )

    # 数据收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./gemma-math-finetuned",
        per_device_train_batch_size=1,  # 单卡batch_size降为1
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # 增大梯度累积步数
        optim="adamw_bnb_8bit",  # 使用分页优化器
        learning_rate=1e-5,  # 降低学习率
        weight_decay=0.01,
        fp16=False,  # 禁用fp16
        bf16=True,  # 启用bfloat16
        max_grad_norm=0.3,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        gradient_checkpointing=True,
        # deepspeed="./configs/deepspeed_zero3.json",  # 使用DeepSpeed
        report_to="none",
        torch_compile=True
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    model.save_pretrained("./gemma-math-finetuned-final")
    tokenizer.save_pretrained("./gemma-math-finetuned-final")
