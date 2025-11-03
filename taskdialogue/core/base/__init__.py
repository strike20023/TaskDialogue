"""
基础抽象类 - 定义所有 benchmark 必须实现的接口

这些抽象类确保：
- 统一的接口设计
- 代码复用性
- 易于扩展新 benchmark
"""

from taskdialogue.core.base.agent import BaseAgent
from taskdialogue.core.base.user_simulator import BaseUserSimulator
from taskdialogue.core.base.evaluator import BaseEvaluator
from taskdialogue.core.base.pipeline import BasePipeline

__all__ = [
    "BaseAgent",
    "BaseUserSimulator",
    "BaseEvaluator",
    "BasePipeline",
]

