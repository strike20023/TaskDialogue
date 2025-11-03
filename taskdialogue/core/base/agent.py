"""
Agent 基础抽象类

所有对话 Agent 都应该继承这个基类并实现其抽象方法。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from taskdialogue.core.schemas.message import Message
from taskdialogue.core.models.base import BaseModel


class BaseAgent(ABC):
    """所有 Agent 的基类
    
    这个抽象类定义了 Agent 的标准接口，所有具体的 Agent 实现
    （MultiWOZ Agent, Tau2 Agent 等）都应该继承这个类。
    
    设计理念：
    - 统一接口：所有 Agent 有相同的调用方式
    - 状态管理：Agent 负责管理对话状态
    - 模型无关：可以使用任何模型后端
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        config: Dict[str, Any],
        **kwargs
    ):
        """初始化 Agent
        
        Args:
            model: 模型实例（BaseModel）
            config: Agent 配置
            **kwargs: 额外参数
        """
        self.model = model
        self.config = config
        self.dialogue_history: List[Message] = []
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_response(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成对用户输入的回复
        
        这是 Agent 的核心方法，必须由子类实现。
        
        Args:
            user_input: 用户输入文本
            context: 可选的上下文信息（工具、状态等）
            
        Returns:
            Agent 的回复文本
            
        Example:
            >>> agent = MultiWOZAgent(model, config)
            >>> response = agent.generate_response("I need a restaurant")
            >>> print(response)
            "I can help you find a restaurant..."
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """重置 Agent 状态
        
        清空对话历史和内部状态，准备处理新对话。
        """
        pass
    
    def add_message(self, message: Message) -> None:
        """添加消息到对话历史"""
        self.dialogue_history.append(message)
    
    def get_dialogue_history(self) -> List[Message]:
        """获取对话历史"""
        return self.dialogue_history
    
    def get_last_n_messages(self, n: int) -> List[Message]:
        """获取最近 n 条消息"""
        return self.dialogue_history[-n:] if n > 0 else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            包含 token 使用、对话轮数等统计信息的字典
        """
        total_tokens = sum(
            msg.tokens for msg in self.dialogue_history 
            if msg.tokens is not None
        )
        
        return {
            "num_messages": len(self.dialogue_history),
            "total_tokens": total_tokens if total_tokens > 0 else None,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.model_name})"

