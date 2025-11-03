"""
统一数据模型 - 使用 Pydantic 定义所有数据结构

这个模块定义了跨 benchmark 的统一数据格式：
- Message: 消息格式
- DialogueTurn: 对话轮次
- InferenceResult: 推理输出
- EvaluationResult: 评估输出
"""

from taskdialogue.core.schemas.message import Message, ToolCall, ToolResponse
from taskdialogue.core.schemas.dialogue import DialogueTurn
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult, EvaluationMetric

__all__ = [
    "Message",
    "ToolCall",
    "ToolResponse",
    "DialogueTurn",
    "InferenceResult",
    "EvaluationResult",
    "EvaluationMetric",
]

