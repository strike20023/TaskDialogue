import asyncio
from multiwoz_agent import MultiWOZLitAgent
from agentlightning import LLM, NamedResources

async def main():
    # 构造模拟 task 和资源
    task = {
        "task_id": "PMUL0959.json"
    }
    # 替换为你本地服务地址和模型信息
    resources = {
        "main_llm": LLM(endpoint="http://20.66.31.2:12013", model="Qwen3-8B")
    }
    rollout = {}

    agent = MultiWOZLitAgent()
    score = await agent.training_rollout_async(task, resources, rollout)
    print("Reward Score:", score)

if __name__ == "__main__":
    asyncio.run(main())