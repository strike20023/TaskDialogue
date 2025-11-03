"""
Prompt 管理模块 - 公共 Prompt 模板和工具

提供跨 benchmark 共享的 prompt 组件：
- 公共模板片段
- Prompt 格式化工具
- Prompt 工程最佳实践
"""

from taskdialogue.core.prompts.base import (
    PromptTemplate,
    format_messages,
    format_dialogue_history,
)

__all__ = [
    "PromptTemplate",
    "format_messages",
    "format_dialogue_history",
]

