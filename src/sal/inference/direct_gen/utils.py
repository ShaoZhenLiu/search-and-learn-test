import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="deepscaler.json", type=str)
    parser.add_argument("--data_names", default="DeepScaleR-Preview-Dataset", type=str)
    parser.add_argument("--data_dir", default="/data/shaozhen.liu/python_project/hf_datasets/",
                        type=str)
    parser.add_argument("--model_name_or_path",
                        default="/data/shaozhen.liu/python_project/hf_models/QwQ-32B",
                        type=str)
    parser.add_argument("--output_dir", default="/data/shaozhen.liu/python_project/hf_datasets",
                        type=str)
    parser.add_argument("--checkpoint_file", default=".checkpoint", type=str)
    parser.add_argument("--prompt_type", default="deepseek-math", type=str)
    # parser.add_argument("--split", default="test", type=str)
    # parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=16384, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--shuffle", default=True)
    # parser.add_argument("--use_vllm", default=True)
    parser.add_argument("--overwrite", default=True)
    parser.add_argument("--correct_answer_only", default=True)
    parser.add_argument("--resume", default=False)
    # parser.add_argument("--use_safetensors", action="store_true")
    # parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args
