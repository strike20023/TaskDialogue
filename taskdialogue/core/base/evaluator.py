"""
Evaluator 基础抽象类

所有评估器都应该继承这个基类。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult


class BaseEvaluator(ABC):
    """所有 Evaluator 的基类
    
    评估器负责：
    - 评估对话质量
    - 计算各种指标
    - 生成评估报告
    
    设计理念：
    - 标准化评估：统一的评估接口
    - 多维度评估：支持多种评估指标
    - 可扩展性：易于添加新的评估方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 Evaluator
        
        Args:
            config: 评估器配置
        """
        self.config = config
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def evaluate(
        self, 
        inference_result: InferenceResult,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """评估推理结果
        
        Args:
            inference_result: 推理结果
            ground_truth: 标准答案或真值（如果有）
            
        Returns:
            评估结果
            
        Example:
            >>> evaluator = MultiWOZEvaluator(config)
            >>> eval_result = evaluator.evaluate(inference_result)
            >>> print(eval_result.overall_score)
            0.85
        """
        pass
    
    def evaluate_batch(
        self,
        inference_results: list[InferenceResult],
        ground_truths: Optional[list[Dict[str, Any]]] = None
    ) -> list[EvaluationResult]:
        """批量评估
        
        Args:
            inference_results: 推理结果列表
            ground_truths: 标准答案列表（可选）
            
        Returns:
            评估结果列表
        """
        evaluation_results = []
        
        for i, inf_result in enumerate(inference_results):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            eval_result = self.evaluate(inf_result, gt)
            evaluation_results.append(eval_result)
        
        return evaluation_results
    
    def get_aggregate_statistics(
        self, 
        evaluation_results: list[EvaluationResult]
    ) -> Dict[str, Any]:
        """获取聚合统计信息
        
        Args:
            evaluation_results: 评估结果列表
            
        Returns:
            聚合统计信息（平均分、标准差等）
        """
        if not evaluation_results:
            return {}
        
        # 计算平均分
        avg_score = sum(r.overall_score for r in evaluation_results if r and r.overall_score is not None) / len([r for r in evaluation_results if r])
        
        # 收集所有指标
        all_metrics: Dict[str, list[float]] = {}
        for result in evaluation_results:
            for metric in result.metrics:
                if metric.name not in all_metrics:
                    all_metrics[metric.name] = []
                all_metrics[metric.name].append(metric.value)
        
        # 计算每个指标的平均值
        avg_metrics = {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }
        
        return {
            "num_evaluations": len(evaluation_results),
            "average_overall_score": avg_score,
            "average_metrics": avg_metrics,
            "success_rate": sum(
                1 for r in evaluation_results if r.overall_score >= 0.5
            ) / len(evaluation_results)
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

