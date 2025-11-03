"""
对话轮次数据模型
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from taskdialogue.core.schemas.message import Message, ToolCall, ToolResponse


class DialogueTurn(BaseModel):
    """单轮对话
    
    记录一个完整的交互轮次，包括：
    - 用户输入
    - Agent 响应
    - 工具调用（如果有）
    - 工具响应（如果有）
    """
    model_config = ConfigDict(extra="allow")
    
    turn_id: int = Field(..., description="轮次 ID（从 1 开始）")
    
    # 消息
    user_message: Message = Field(..., description="用户消息")
    agent_message: Message = Field(..., description="Agent 消息")
    
    # 工具调用（可选）
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用列表")
    tool_responses: Optional[List[ToolResponse]] = Field(None, description="工具响应列表")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="轮次元数据")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DialogueTurn":
        """从字典创建"""
        return cls.model_validate(data)
    
    def get_all_messages(self) -> List[Message]:
        """获取本轮所有消息（按顺序）"""
        messages = [self.user_message]
        
        if self.agent_message:
            messages.append(self.agent_message)
        
        # 工具消息
        if self.tool_responses:
            for resp in self.tool_responses:
                messages.append(Message(
                    role="tool",
                    content=resp.content,
                    tool_call_id=resp.tool_call_id
                ))
        
        return messages

