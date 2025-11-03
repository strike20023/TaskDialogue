"""
推理结果数据模型 - 统一的推理输出格式
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
import json

from taskdialogue.core.schemas.dialogue import DialogueTurn


class InferenceResult(BaseModel):
    """统一的推理输出格式
    
    所有 benchmark 的推理结果都使用这个统一格式，
    确保输出一致性和可比较性。
    """
    model_config = ConfigDict(extra="allow")
    
    # ==================== 基础信息 ====================
    benchmark: str = Field(..., description="Benchmark 名称 (multiwoz, tau2, etc.)")
    task_id: str = Field(..., description="任务 ID")
    dialogue_id: str = Field(..., description="对话 ID")
    
    # ==================== 对话内容 ====================
    dialogue_history: List[DialogueTurn] = Field(
        default_factory=list, 
        description="完整对话历史"
    )
    
    # 原始消息历史（包含完整的 tool_calls 和 tool 消息）
    raw_messages: Optional[List] = Field(
        None,
        description="原始消息列表（OpenAI 格式，包含 tool_calls）"
    )
    
    # ==================== 模型配置 ====================
    model_config_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="模型配置信息",
        alias="model_config"  # 避免与 Pydantic 的 model_config 冲突
    )
    
    # ==================== 元数据 ====================
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="推理时间戳"
    )
    
    # ==================== 统计信息 ====================
    num_turns: int = Field(0, description="对话轮数")
    total_tokens: Optional[int] = Field(None, description="总 token 数")
    inference_time: Optional[float] = Field(None, description="推理时间（秒）")
    
    # ==================== Benchmark 特定数据 ====================
    benchmark_specific: Optional[Dict[str, Any]] = Field(
        None, 
        description="Benchmark 特定的额外数据"
    )
    
    # ==================== 状态信息 ====================
    success: bool = Field(True, description="推理是否成功")
    error_message: Optional[str] = Field(None, description="错误信息（如果失败）")
    termination_reason: Optional[str] = Field(None, description="终止原因")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于保存）"""
        data = self.model_dump(exclude_none=True, by_alias=True)
        # 处理 datetime
        if "timestamp" in data:
            data["timestamp"] = data["timestamp"].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResult":
        """从字典创建"""
        # 处理 datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls.model_validate(data)
    
    def save_to_file(self, filepath: str | Path) -> None:
        """保存到 JSON 文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str | Path) -> "InferenceResult":
        """从 JSON 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "dialogue_id": self.dialogue_id,
            "num_turns": self.num_turns,
            "total_tokens": self.total_tokens,
            "success": self.success,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

