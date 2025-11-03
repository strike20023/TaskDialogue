"""
Tau2 Benchmark

客服场景对话评测 benchmark，包含：
- Airline: 航空客服
- Retail: 零售客服
- Telecom: 电信客服

这个包提供了 τ²-bench 在 TaskDialogue 中的完整集成。

Structure:
- adapters/: TaskDialogue integration adapters
- prompts/: Centralized prompt management
- config/: Configuration management
- agent/: Tau2 agent framework
- user/: Tau2 user simulator
- environment/: Environment and tool abstractions
- evaluator/: Task evaluation framework
- metrics/: Performance metrics computation
- domains/: Domain-specific implementations (airline, retail, telecom)
- data_model/: Data models (Message, Task, Simulation)
- orchestrator.py: Multi-turn dialogue orchestration
- registry.py: Domain and task registration
- run.py: Main execution entry point
- cli.py: Configuration-driven CLI interface
- pipeline.py: Unified Pipeline interface for TaskDialogue
"""

# 导出主要的 Pipeline 接口（用于 CLI）
from taskdialogue.benchmarks.tau2.pipeline import Tau2Pipeline

# 导出内部模块（用于其他组件使用）
__all__ = [
    "Tau2Pipeline",  # 统一接口
    # 内部模块（保持向后兼容）
    "adapters",
    "prompts",
    "config",
    "agent",
    "user",
    "environment",
    "evaluator",
    "metrics",
    "domains",
    "data_model",
    "orchestrator",
    "registry",
    "run",
    "cli",
]
