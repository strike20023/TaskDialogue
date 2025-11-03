"""
消息数据模型 - 统一的消息格式定义

使用 Pydantic 确保类型安全和数据验证
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict


class ToolCall(BaseModel):
    """工具调用"""
    model_config = ConfigDict(extra="allow")
    
    id: Optional[str] = Field(None, description="工具调用 ID")
    name: str = Field(..., description="工具名称")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
    requestor: Literal["user", "assistant"] = Field("assistant", description="调用者角色")


class ToolResponse(BaseModel):
    """工具响应"""
    model_config = ConfigDict(extra="allow")
    
    tool_call_id: str = Field(..., description="对应的工具调用 ID")
    name: str = Field(..., description="工具名称")
    content: str = Field(..., description="工具返回内容")
    error: Optional[str] = Field(None, description="错误信息（如果有）")


class Message(BaseModel):
    """统一的消息格式
    
    支持所有角色的消息：
    - user: 用户消息
    - assistant: Agent 消息
    - system: 系统消息
    - tool: 工具消息
    """
    model_config = ConfigDict(extra="allow")
    
    role: Literal["user", "assistant", "system", "tool"] = Field(
        ..., description="消息角色"
    )
    content: Optional[str] = Field(None, description="消息内容")
    
    # 工具调用相关
    tool_calls: Optional[List[ToolCall]] = Field(None, description="工具调用列表")
    tool_call_id: Optional[str] = Field(None, description="工具调用 ID（tool 消息）")
    
    # 元数据
    timestamp: Optional[datetime] = Field(None, description="时间戳")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")
    
    # 统计信息
    tokens: Optional[int] = Field(None, description="Token 数量")
    cost: Optional[float] = Field(None, description="调用成本")
    
    def is_tool_call(self) -> bool:
        """判断是否包含工具调用"""
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于保存和传输）"""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建消息（兼容 OpenAI 格式）"""
        # 转换 tool_calls 格式（OpenAI 嵌套格式 → 扁平格式）
        if 'tool_calls' in data and data['tool_calls']:
            converted_tool_calls = []
            for tc in data['tool_calls']:
                if isinstance(tc, dict) and 'function' in tc:
                    # OpenAI 格式: {id, type, function: {name, arguments}}
                    # 转换为扁平格式: {id, name, arguments}
                    import json
                    args = tc['function']['arguments']
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {}
                    
                    converted_tool_calls.append({
                        'id': tc.get('id'),
                        'name': tc['function']['name'],
                        'arguments': args
                    })
                else:
                    # 已经是扁平格式
                    converted_tool_calls.append(tc)
            
            data = data.copy()
            data['tool_calls'] = converted_tool_calls
        
        return cls.model_validate(data)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI API 格式"""
        import json
        
        result = {
            "role": self.role,
            "content": self.content,
        }
        
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id or f"call_{tc.name}",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, str) 
                                    else json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in self.tool_calls
            ]
        
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        
        return result

