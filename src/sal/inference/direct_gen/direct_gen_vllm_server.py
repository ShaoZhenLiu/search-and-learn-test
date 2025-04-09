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
    def __init__(self, model_name,
                 api_key: str =None,
                 gpu_memory_utilization: float =None,
                 enable_prefix_caching: bool =None,
                 gpu_count: int =None,
                 node_count: int =1,
                 seed: int =None,):
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

        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.gpu_count = gpu_count
        self.node_count = node_count
        self.seed = seed


    def start(self):
        command = ["vllm", "serve", self.model_name, "--dtype", "auto"]

        # 添加 GPU 相关参数
        if self.api_key is not None:
            command.extend(["--api-key", self.api_key])
        if self.gpu_memory_utilization is not None:
            command.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        if self.enable_prefix_caching is not None:
            command.extend(["--enable-prefix-caching"])
        if self.gpu_count is not None:
            command.extend(["--tensor-parallel-size", str(self.gpu_count)])
        if self.node_count > 1:
            command.extend(["--pipeline-parallel-size", str(self.node_count)])
        if self.seed is not None:
            command.extend(["--seed", str(self.seed)])
        print(command)

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
                time.sleep(10)

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

        self.worker_num = 10
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
            tasks = [self.worker(session) for _ in range(self.worker_num)]  # 使用10个并发工作者
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

class MultiTurnResponseCollector(ResponseCollector):
    def __init__(self,
                 dataset,
                 server_manager,
                 turn_count=4,
                 prompt_dict=None,
                 max_retries=3,
                 ):
        super().__init__(dataset, server_manager)
        self.turn_count = turn_count  # 总轮次
        self.prompt_dict = prompt_dict or {}  # 每轮的prompt字典
        self.conversation_history = {}  # 保存每条数据的对话历史
        self.max_retries = max_retries  # 最大重试次数
        self.retries = {}  # 记录每条数据的重试次数

    async def send_request(self, session, idx, question, turn=1, conversation_id=None, retry_count=0):
        # 如果是新的对话，初始化对话历史和重试计数
        if conversation_id is None:
            conversation_id = idx
            self.conversation_history[conversation_id] = []
            self.retries[conversation_id] = 0
            # 设置第一轮的prompt
            question = Template(self.prompt_dict["turn0"]).render(problem=question)

        # 检查是否超过最大重试次数
        if self.retries[conversation_id] >= self.max_retries:
            exception_entry = {
                "exception": {
                    "message": f"Max retries ({self.max_retries}) exceeded for conversation {conversation_id}",
                    "conversation_id": conversation_id,
                    "turn": turn
                }
            }
            with self.lock:
                self.file.write(json.dumps(exception_entry) + "\n")
                print(f"Exception for conversation {conversation_id}: Max retries exceeded")
            self.completed_requests += 1
            self.active_requests -= 1
            return

        # 构造请求数据
        messages = self.conversation_history[conversation_id].copy()
        messages.append({"role": "user", "content": question})
        data = {
            "model": self.server_manager.model_name,
            "messages": messages
        }

        # try:
        async with session.post(
            f"{self.server_manager.base_url}/chat/completions",
            headers=self.server_manager.headers,
            json=data,
            # timeout=30
        ) as response:
            if response.status == 200:
                reply = await response.json()
                # 更新对话历史
                messages.append(reply['choices'][0]['message'])
                self.conversation_history[conversation_id] = messages
                print(f"Conversation {conversation_id} - Turn {turn} completed")

                # 如果未达到总轮次，获取下一个prompt并继续对话
                if turn < self.turn_count:
                    next_prompt = self.prompt_dict[f"turn{turn}"]
                    await self.send_request(session, idx, next_prompt, turn + 1, conversation_id)
                else:
                    # 达到总轮次，保存对话历史
                    with self.lock:
                        self.file.write(json.dumps({
                            "conversation_id": conversation_id,
                            "messages": messages
                        }) + "\n")
                        print(f"Conversation {conversation_id} saved")
                    self.completed_requests += 1
            else:
                error_entry = {
                    "error": {
                        "status_code": response.status,
                        "message": await response.text(),
                        "conversation_id": conversation_id,
                        "turn": turn
                    }
                }
                with self.lock:
                    self.file.write(json.dumps(error_entry) + "\n")
                    print(f"Error in conversation {conversation_id} - Turn {turn}: {response.status} - {await response.text()}")
                self.retries[conversation_id] += 1
                print(f"Retrying conversation {conversation_id} - Turn {turn} (Retry {self.retries[conversation_id]})")
                await self.send_request(session, idx, question, turn, conversation_id, retry_count + 1)
        # except Exception as e:
        #     print(e)
        #     exception_entry = {
        #         "exception": {
        #             "message": str(e),
        #             "conversation_id": conversation_id,
        #             "turn": turn
        #         }
        #     }
        #     with self.lock:
        #         self.file.write(json.dumps(exception_entry) + "\n")
        #         print(f"Exception in conversation {conversation_id} - Turn {turn}: {str(e)}")
        #     self.retries[conversation_id] += 1
        #     print(f"Retrying conversation {conversation_id} - Turn {turn} (Retry {self.retries[conversation_id]})")
        #     await self.send_request(session, idx, question, turn, conversation_id, retry_count + 1)
        # finally:
        #     self.active_requests -= 1
        self.active_requests -= 1

    async def worker(self, session):
        while self.queue:
            # if not self.running:
            #     break
            if self.queue:
                with self.lock:
                    idx, question = self.queue.popleft()
                    self.active_requests += 4
                # 启动多轮对话
                await self.send_request(session, idx, question, turn=1)
            else:
                await asyncio.sleep(0.1)

    async def start(self):
        self.running = True
        async with aiohttp.ClientSession() as session:
            tasks = [self.worker(session) for _ in range(self.worker_num)]  # 使用10个并发工作者
            await asyncio.gather(*tasks)

    def stop(self):
        self.running = False
        while self.active_requests > 0:
            time.sleep(0.1)
        self.file.close()
        print("多轮响应收集器已停止")


if __name__ == "__main__":
    import os
    from sal.inference.direct_gen import VLLMServerManager

    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # 准备数据集
    dataset = [f"Test {i+1}" for i in range(20)]

    # 初始化服务器管理器
    server_manager = VLLMServerManager(
        model_name="/data/shaozhen.liu/python_project/hf_models/Qwen2.5-1.5B-Instruct",
        gpu_count=2,
        seed=0,
    )
    # server_manager.start()

    # 定义每轮的prompt字典
    prompt_dict = {
        1: "Please elaborate on your previous answer.",
        2: "Can you provide an example?",
        3: "Thank you for the information."
    }

    # 创建多轮响应收集器
    collector = MultiTurnResponseCollector(
        dataset,
        server_manager,
        turn_count=4,
        prompt_dict=prompt_dict
    )

    # 使用 asyncio.run() 启动异步任务
    async def main():
        await collector.start()

    # 如果是 Windows 系统，需要设置事件循环策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # 运行异步任务
    asyncio.run(main())

    # 停止收集器和服务器
    collector.stop()
    server_manager.stop()