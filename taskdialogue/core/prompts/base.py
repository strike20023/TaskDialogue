"""
公共 Prompt 工具和模板
"""

from typing import List, Dict, Any, Optional
from taskdialogue.core.schemas.message import Message


class PromptTemplate:
    """Prompt 模板类
    
    提供简单的模板变量替换功能。
    
    Example:
        >>> template = PromptTemplate("Hello, {name}! Your goal is: {goal}")
        >>> prompt = template.format(name="Alice", goal="Find a restaurant")
    """
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """格式化模板"""
        return self.template.format(**kwargs)


def format_messages(messages: List[Message]) -> List[Dict[str, Any]]:
    """将 Message 对象列表转换为标准消息格式
    
    Args:
        messages: Message 对象列表
        
    Returns:
        标准消息格式列表
    """
    return [msg.to_openai_format() for msg in messages]


def format_dialogue_history(
    messages: List[Message],
    format_type: str = "text"
) -> str:
    """格式化对话历史为文本
    
    Args:
        messages: Message 对象列表
        format_type: 格式类型 ("text", "markdown")
        
    Returns:
        格式化后的对话历史文本
    """
    lines = []
    
    for msg in messages:
        if msg.role == "user":
            prefix = "User:"
        elif msg.role == "assistant":
            prefix = "Assistant:"
        elif msg.role == "system":
            prefix = "System:"
        else:
            prefix = "Tool:"
        
        content = msg.content or ""
        
        if format_type == "markdown":
            lines.append(f"**{prefix}** {content}")
        else:
            lines.append(f"{prefix} {content}")
    
    return "\n".join(lines)


# 公共 Prompt 片段
COMMON_GUIDELINES = {
    "be_helpful": "Always be helpful and respectful to the user.",
    "use_tools": "Use the provided tools to access information and perform actions.",
    "no_hallucination": "Do not make up information. If you don't know, say so.",
    "be_proactive": "Be proactive in completing tasks when you have sufficient information.",
}


def get_common_guideline(key: str) -> Optional[str]:
    """获取公共指导原则"""
    return COMMON_GUIDELINES.get(key)

