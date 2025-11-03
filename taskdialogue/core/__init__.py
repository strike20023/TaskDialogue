"""
TaskDialogue Core - 统一的对话系统框架核心模块

这个模块提供了所有 benchmark 共享的核心组件：
- 基础抽象类 (base/)
- 统一数据模型 (schemas/)
- 模型系统 (models/)
- Prompt 管理 (prompts/)
- 工具函数 (utils/)

设计理念：
- 清晰的抽象：定义统一的接口
- 类型安全：使用 Pydantic 进行数据验证
- 易于扩展：新 benchmark 只需实现基类
- 统一格式：推理和评估输出格式一致
"""

__version__ = "2.0.0"

from taskdialogue.core.schemas import (
    Message,
    DialogueTurn,
    InferenceResult,
    EvaluationResult,
    EvaluationMetric,
)

from taskdialogue.core.base import (
    BaseAgent,
    BaseUserSimulator,
    BaseEvaluator,
    BasePipeline,
)

__all__ = [
    # Schemas
    "Message",
    "DialogueTurn",
    "InferenceResult",
    "EvaluationResult",
    "EvaluationMetric",
    # Base classes
    "BaseAgent",
    "BaseUserSimulator",
    "BaseEvaluator",
    "BasePipeline",
]

