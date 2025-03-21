import json
import requests
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

class BaseMathProblemSolver:
    def __init__(self, 
                 api_base: str = "http://localhost:8000/v1",
                 model: str = "facebook/opt-125m",
                 max_tokens: int = 1024,
                 temperature: float = 0.7,
                 top_p: float = 0.95):
        """
        数学问题求解器
        
        Args:
            api_base: VLLM服务地址
            model: 使用的模型名称
            max_tokens: 最大生成token数
            temperature: 采样温度
        """
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        
    def format_prompt(self, question: str) -> List[Dict[str, str]]:
        """
        格式化问题提示词
        
        Args:
            question: 数学问题文本
            
        Returns:
            包含system和user消息的列表
        """
        return [
            {
                "role": "system",
                "content": "Let's think step by step and output the final answer within \\boxed{}."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    
    def solve_single_problem(self, question: str) -> Dict[str, Any]:
        """
        解决单个数学问题
        
        Args:
            question: 数学问题文本
            
        Returns:
            包含问题、回答和状态的字典
        """
        try:
            # 构建请求数据
            data = {
                "model": self.model,
                "messages": self.format_prompt(question),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            # 发送请求
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                return {
                    "question": question,
                    "answer": answer,
                    "status": "success"
                }
            else:
                return {
                    "question": question,
                    "answer": None,
                    "status": f"error: {response.status_code}",
                    "error_message": response.text
                }
                
        except Exception as e:
            print(f"处理问题 '{question}' 时发生错误: {e}")
            return {
                "question": question,
                "answer": None,
                "status": "error",
                "error_message": str(e)
            }
    
    def solve_batch(self, 
                   questions: List[str], 
                   output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        批量解决数学问题，实时保存结果
        
        Args:
            questions: 数学问题列表
            output_file: 输出文件路径，如果提供则保存结果
            
        Returns:
            结果列表
        """
        results = []
        
        for question in tqdm(questions, desc="Processing math problems"):
            # 处理单个问题
            result = self.solve_single_problem(question)
            results.append(result)
            
            # 实时保存结果
            if output_file:
                # 如果文件不存在，创建新文件并写入
                if not os.path.exists(output_file):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    # 如果文件存在，读取现有结果并追加
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
        return results 

class MathProblemSolver(BaseMathProblemSolver):
    def solve_batch(self, 
                   questions: List[str], 
                   output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for question in tqdm(questions, desc="Processing math problems"):
            result = self.solve_single_problem(question)
            results.append(result)
            if output_file:
                if not os.path.exists(output_file):
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return results

class MultiThreadedMathProblemSolver(BaseMathProblemSolver):
    def solve_batch_multithreaded(self, 
                                  questions: List[str], 
                                  output_file: Optional[str] = None,
                                  num_threads: int = 5) -> List[Dict[str, Any]]:
        """
        使用多线程批量解决数学问题，实时保存结果，并显示进度条
        
        Args:
            questions: 数学问题列表
            output_file: 输出文件路径，如果提供则保存结果
            num_threads: 线程数
            
        Returns:
            结果列表
        """
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_question = {executor.submit(self.solve_single_problem, question): question for question in questions}
            
            # 使用tqdm显示进度条
            for future in tqdm(as_completed(future_to_question), total=len(questions), desc="Processing math problems"):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                    if output_file:
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"问题 {question} 处理时出错: {e}")
        return results 

class AsyncMathProblemSolver(BaseMathProblemSolver):
    async def solve_single_problem_async(self, session: aiohttp.ClientSession, question: str) -> Dict[str, Any]:
        try:
            data = {
                "model": self.model,
                "messages": self.format_prompt(question),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            async with session.post(
                f"{self.api_base}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result["choices"][0]["message"]["content"]
                    return {
                        "question": question,
                        "answer": answer,
                        "status": "success"
                    }
                else:
                    return {
                        "question": question,
                        "answer": None,
                        "status": f"error: {response.status}",
                        "error_message": await response.text()
                    }
        except Exception as e:
            print(f"处理问题 '{question}' 时发生错误: {e}")
            return {
                "question": question,
                "answer": None,
                "status": "error",
                "error_message": str(e)
            }

    async def solve_batch_async(self, 
                                questions: List[str], 
                                output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = [self.solve_single_problem_async(session, question) for question in questions]
            for task in tqdm(asyncio.as_completed(tasks), total=len(questions), desc="Processing math problems"):
                result = await task
                results.append(result)
                if output_file:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return results 