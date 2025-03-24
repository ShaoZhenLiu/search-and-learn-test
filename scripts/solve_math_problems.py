import os
import argparse
from datasets import load_dataset
from sal.inference.math_problem_solver import MathProblemSolver, MultiThreadedMathProblemSolver, AsyncMathProblemSolver
import asyncio


def process_questions(solver, questions, output_file, mode, num_threads):
    if mode == 'single':
        print(f"开始处理 {len(questions)} 个数学问题...")
        results = solver.solve_batch(questions, output_file=output_file)
    elif mode == 'multi':
        print(f"开始使用 {num_threads} 个线程处理 {len(questions)} 个数学问题...")
        results = solver.solve_batch_multithreaded(questions, output_file=output_file, num_threads=num_threads)
    elif mode == 'async':
        print(f"开始异步处理 {len(questions)} 个数学问题...")
        results = asyncio.run(solver.solve_batch_async(questions, output_file=output_file))
    else:
        raise ValueError("无效的模式选择")

    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n处理完成！")
    print(f"总问题数: {len(results)}")
    print(f"成功数: {success_count}")
    print(f"失败数: {len(results) - success_count}")
    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='解决数学问题数据集')
    parser.add_argument('--dataset_path', type=str,
                        default='/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/DeepScaler-QwQ_32b',
                        help='数据集路径（可以是Hugging Face数据集名称或本地路径）')
    parser.add_argument('--dataset_file', type=str, default='distilled_s0_e20000_20250309005632_final.json',
                        help='数据集文件名称')
    parser.add_argument('--question_column', type=str, default='question',
                        help='问题列所在的列名')
    parser.add_argument('--output_dir', type=str,
                        default='/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/data/DeepScaler-QwQ_32b/vllm_results',
                        help='输出目录')
    parser.add_argument('--model', type=str,
                        default='/apdcephfs_sh3/share_302139670/hunyuan/berlinni/liushaozhen/models/Qwen2.5-7B-Instruct',
                        help='使用的模型名称')
    parser.add_argument('--api_base', type=str, default='http://localhost:8000/v1',
                        help='VLLM服务地址')
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='top_p')
    parser.add_argument('--num_threads', type=int, default=100,
                        help='线程数')
    parser.add_argument('--mode', type=str, choices=['single', 'multi', 'async'], default='multi',
                        help='选择处理模式：single, multi, async')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据集
    dataset = load_dataset(args.dataset_path, data_files=args.dataset_file)['train']
    # 获取所有问题
    questions = dataset[args.question_column]

    # 初始化求解器
    if args.mode == 'single':
        solver = MathProblemSolver(
            api_base=args.api_base,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    elif args.mode == 'multi':
        solver = MultiThreadedMathProblemSolver(
            api_base=args.api_base,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    elif args.mode == 'async':
        solver = AsyncMathProblemSolver(
            api_base=args.api_base,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

    # 设置输出文件路径
    output_file = os.path.join(args.output_dir, 'results.json')

    # 处理问题
    process_questions(solver, questions, output_file, args.mode, args.num_threads)


if __name__ == "__main__":
    main()