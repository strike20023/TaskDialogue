"""
评估结果数据模型 - 统一的评估输出格式
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
import json


class EvaluationMetric(BaseModel):
    """单个评估指标
    
    例如：
    - inform: 0.9
    - success: 0.8
    - combined_score: 0.85
    """
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., description="指标名称")
    value: float = Field(..., description="指标值")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump(exclude_none=True)


class EvaluationResult(BaseModel):
    """统一的评估输出格式
    
    所有 benchmark 的评估结果都使用这个统一格式。
    """
    model_config = ConfigDict(extra="allow")
    
    # ==================== 基础信息 ====================
    benchmark: str = Field(..., description="Benchmark 名称")
    task_id: str = Field(..., description="任务 ID")
    dialogue_id: str = Field(..., description="对话 ID")
    
    # ==================== 评估结果 ====================
    overall_score: float = Field(..., description="总分 (0-1)")
    metrics: List[EvaluationMetric] = Field(
        default_factory=list,
        description="详细指标列表"
    )
    
    # ==================== 元数据 ====================
    evaluator_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="评估器配置"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="评估时间戳"
    )
    
    # ==================== Benchmark 特定数据 ====================
    benchmark_specific: Optional[Dict[str, Any]] = Field(
        None,
        description="Benchmark 特定的评估数据"
    )
    
    # ==================== 额外信息 ====================
    error_message: Optional[str] = Field(None, description="错误信息（如果评估失败）")
    evaluation_time: Optional[float] = Field(None, description="评估耗时（秒）")
    
    def _serialize_value(self, obj: Any, seen: Optional[set] = None) -> Any:
        """递归地将对象转换为可序列化的形式
        
        处理 SQLAlchemy 对象、datetime 等不可序列化的类型
        
        Args:
            obj: 需要序列化的对象
            seen: 已访问对象的 ID 集合（用于检测循环引用）
        """
        if seen is None:
            seen = set()
            
        # 基础类型直接返回
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # datetime 转为 ISO 字符串
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # 检测循环引用（对于可变对象）
        obj_id = id(obj)
        if obj_id in seen:
            return f"<circular reference: {type(obj).__name__}>"
        
        # 列表/元组递归处理
        if isinstance(obj, (list, tuple)):
            seen.add(obj_id)
            try:
                return [self._serialize_value(item, seen) for item in obj]
            finally:
                seen.discard(obj_id)
        
        # 字典递归处理（跳过内部属性）
        if isinstance(obj, dict):
            seen.add(obj_id)
            try:
                result = {}
                for k, v in obj.items():
                    # 跳过以 _ 开头的内部属性（如 _sa_instance_state）
                    if isinstance(k, str) and k.startswith('_'):
                        continue
                    result[k] = self._serialize_value(v, seen)
                return result
            finally:
                seen.discard(obj_id)
        
        # SQLAlchemy 对象或有 as_dict 方法的对象
        if hasattr(obj, 'as_dict') and callable(getattr(obj, 'as_dict')):
            try:
                dict_result = obj.as_dict()
                # 递归处理返回的字典（可能还包含不可序列化的对象）
                return self._serialize_value(dict_result, seen)
            except Exception:
                # 如果 as_dict 失败，尝试其他方法
                pass
        
        # 尝试 JSON 序列化测试
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # 无法序列化，转换为字符串表示
            return f"<{type(obj).__name__}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = self.model_dump(exclude_none=True)
        # 递归处理所有值，确保可序列化
        return {k: self._serialize_value(v) for k, v in data.items()}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
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
    def load_from_file(cls, filepath: str | Path) -> "EvaluationResult":
        """从 JSON 文件加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """获取指定指标的值"""
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric.value
        return None
    
    def add_metric(self, name: str, value: float, details: Optional[Dict] = None) -> None:
        """添加评估指标"""
        self.metrics.append(EvaluationMetric(
            name=name,
            value=value,
            details=details
        ))
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "dialogue_id": self.dialogue_id,
            "overall_score": self.overall_score,
            "metrics": {m.name: m.value for m in self.metrics},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

