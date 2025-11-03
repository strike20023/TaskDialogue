"""
MultiWOZ Benchmark

多领域任务导向对话 benchmark，包含：
- Restaurant: 餐厅查询和预订
- Hotel: 酒店查询和预订
- Attraction: 景点查询
- Train: 火车查询和订票
- Taxi: 出租车预订

主要组件：
- agents: MultiWOZ 对话 agents
- prompts: MultiWOZ 专用 prompts
- evaluators: MultiWOZ 评估器
- tools: 数据库查询和预订工具
- data: 数据加载器
- pipeline: 完整运行流程
"""

from taskdialogue.benchmarks.multiwoz.pipeline import MultiWOZPipeline

__all__ = ["MultiWOZPipeline"]

