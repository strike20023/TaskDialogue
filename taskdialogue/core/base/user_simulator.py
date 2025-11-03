"""
User Simulator 基础抽象类

所有用户模拟器都应该继承这个基类。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from taskdialogue.core.schemas.message import Message
from taskdialogue.core.models.base import BaseModel


class BaseUserSimulator(ABC):
    """所有 User Simulator 的基类
    
    用户模拟器负责：
    - 根据任务目标生成用户话语
    - 模拟真实用户的对话行为
    - 判断任务是否完成
    
    设计理念：
    - 目标驱动：基于任务目标生成话语
    - 自然对话：模拟真实用户的语言风格
    - 状态感知：了解对话进度和完成状态
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        config: Dict[str, Any],
        **kwargs
    ):
        """初始化 User Simulator
        
        Args:
            model: 模型实例
            config: Simulator 配置
            **kwargs: 额外参数
        """
        self.model = model
        self.config = config
        self.dialogue_history: List[Message] = []
        self.task_config: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_response(
        self, 
        agent_output: Optional[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成用户回复
        
        Args:
            agent_output: Agent 的输出（None 表示对话开始）
            context: 可选的上下文信息
            
        Returns:
            用户的回复文本
            
        Example:
            >>> simulator = MultiWOZUserSimulator(model, config)
            >>> simulator.reset(task_config)
            >>> user_input = simulator.generate_response(None)  # 第一句
            >>> print(user_input)
            "I need a restaurant in the centre"
        """
        pass
    
    @abstractmethod
    def is_dialogue_complete(self) -> bool:
        """判断对话是否完成
        
        Returns:
            True 如果任务已完成或应该终止对话
            
        Example:
            >>> if simulator.is_dialogue_complete():
            ...     print("Dialogue finished!")
        """
        pass
    
    @abstractmethod
    def reset(self, task_config: Dict[str, Any]) -> None:
        """重置并加载新任务
        
        Args:
            task_config: 任务配置，包含目标、约束等信息
            
        Example:
            >>> task = {
            ...     "goals": {"restaurant": {...}, "hotel": {...}},
            ...     "dialogue_id": "MUL0001"
            ... }
            >>> simulator.reset(task)
        """
        pass
    
    def add_message(self, message: Message) -> None:
        """添加消息到对话历史"""
        self.dialogue_history.append(message)
    
    def get_dialogue_history(self) -> List[Message]:
        """获取对话历史"""
        return self.dialogue_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
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

