import os
import time
import json
import requests
import asyncio
import aiohttp
from collections import deque
from threading import Thread, Lock
import subprocess

# 1. 启动并管理 vLLM 服务
class VLLMServerManager:
    def __init__(self, model_name="/data/shaozhen.liu/python_project/hf_models/DeepSeek-R1-Distill-Qwen-1.5B",
                 api_key="token-abc123",
                 gpu_count=None, gpu_ids=None, node_count=1):
        self.model_name = model_name
        self.api_key = api_key
        self.process = None
        self.port = 8000
        self.base_url = f"http://localhost:{self.port}/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.running = False
        self.gpu_count = gpu_count
        self.gpu_ids = gpu_ids
        self.node_count = node_count


    def start(self):
        command = ["vllm", "serve", self.model_name, "--dtype", "auto", "--api-key", self.api_key]

        # 添加 GPU 相关参数
        if self.gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        if self.gpu_count is not None:
            command.extend(["--tensor-parallel-size", str(self.gpu_count)])
        if self.node_count > 1:
            command.extend(["--pipeline-parallel-size", str(self.node_count)])

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        self.running = True
        print("vLLM 服务已启动，正在等待初始化完成...")

        # 等待服务器初始化完成
        test_data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "Hello! This is a test message."}]
        }
        while True:
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=test_data,
                    timeout=30
                )
                if response.status_code == 200:
                    print("vLLM 服务已初始化完成")
                    break
            except requests.exceptions.ConnectionError as e:
                print(e)
                time.sleep(1)

    def stop(self):
        if self.running and self.process:
            self.process.terminate()
            self.process.wait()
            self.running = False
            print("vLLM 服务已停止")

# 2. 数据收集程序
class ResponseCollector:
    def __init__(self, dataset, server_manager):
        self.dataset = dataset
        self.server_manager = server_manager
        self.queue = deque(enumerate(self.dataset))
        self.lock = Lock()
        self.running = False
        self.active_requests = 0
        self.completed_requests = 0

        # 创建保存回复的文件夹
        if not os.path.exists("responses"):
            os.makedirs("responses")

        # 打开文件以追加模式写入
        self.file = open("responses/all_responses.json", "a")

    async def send_request(self, session, idx, question):
        data = {
            "model": self.server_manager.model_name,
            "messages": [{"role": "user", "content": question}]
        }

        try:
            async with session.post(
                f"{self.server_manager.base_url}/chat/completions",
                headers=self.server_manager.headers,
                json=data,
                timeout=300  # 超时时间改为5分钟
            ) as response:
                self.completed_requests += 1
                if response.status == 200:
                    reply = await response.json()
                    with self.lock:
                        self.file.write(json.dumps(reply) + "\n")
                        print(f"Saved response {idx+1}")
                else:
                    error_entry = {
                        "error": {
                            "status_code": response.status,
                            "message": await response.text(),
                            "question_index": idx + 1
                        }
                    }
                    with self.lock:
                        self.file.write(json.dumps(error_entry) + "\n")
                        print(f"Error for question {idx+1}: {response.status} - {await response.text()}")
        except Exception as e:
            exception_entry = {
                "exception": {
                    "message": str(e),
                    "question_index": idx + 1
                }
            }
            with self.lock:
                self.file.write(json.dumps(exception_entry) + "\n")
                print(f"Exception for question {idx+1}: {str(e)}")
        finally:
            self.active_requests -= 1

    async def worker(self, session):
        while True:
            if not self.running:
                break
            if self.queue:
                with self.lock:
                    idx, question = self.queue.popleft()
                    self.active_requests += 1
                await self.send_request(session, idx, question)
            else:
                await asyncio.sleep(0.1)

    async def start(self):
        self.running = True
        async with aiohttp.ClientSession() as session:
            tasks = [self.worker(session) for _ in range(10)]  # 使用10个并发工作者
            await asyncio.gather(*tasks)

    def stop(self):
        self.running = False
        while self.active_requests > 0:
            time.sleep(0.1)
        self.file.close()
        print("响应收集器已停止")

# 3. 主函数
if __name__ == "__main__":
    # 准备数据集
    dataset = [f"Question {i+1}" for i in range(10)]

    # 初始化服务器管理器
    server_manager = VLLMServerManager(
        model_name="NousResearch/Meta-Llama-3-8B-Instruct",
        api_key="token-abc123",
        # gpu_count=2,  # 使用2个GPU
        # gpu_ids=[0, 1],  # 指定使用GPU 0和1
        # node_count=1  # 单节点部署
    )
    server_manager.start()

    # 创建响应收集器
    collector = ResponseCollector(dataset, server_manager)

    # 使用 asyncio.run() 启动异步任务
    async def main():
        await collector.start()

    # 如果是 Windows 系统，需要设置事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行异步任务
    asyncio.run(main())

    # 等待所有任务完成
    while True:
        if collector.completed_requests >= len(dataset):
            break
        time.sleep(1)

    # 停止收集器和服务器
    collector.stop()
    server_manager.stop()