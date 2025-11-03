"""
vLLM 推理工具库

支持两种部署模式：
1. 直接 Load 模式：使用 vLLM 的 LLM 类直接加载模型到内存
2. OpenAI 服务模式：通过 OpenAI 客户端连接到 vLLM 服务器

用于 AutoTOD 和 tau-bench 项目的模型推理部署
"""

from .vllm_inference import VLLMInference, VLLMServerInference
from .config import VLLMConfig, SamplingConfig

__version__ = "0.1.0"

__all__ = [
    "VLLMInference",
    "VLLMServerInference",
    "VLLMConfig",
    "SamplingConfig",
]

