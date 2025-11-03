"""
MultiWOZ 完整评估器

使用 LLM 从对话中提取信息，与数据库对比，计算完整的评估指标。
包含 6 个指标：inform, success, combined_score, book, jga, f1
"""

from typing import Dict, Any, Optional, List
import time

from taskdialogue.core.base.evaluator import BaseEvaluator
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult, EvaluationMetric
from taskdialogue.core.models.factory import create_model
from taskdialogue.core.utils.logger import get_logger

# MultiWOZ 特定导入
from taskdialogue.benchmarks.multiwoz.evaluators.llm_extractor import (
    prepare_dialog_string,
    evaluate_by_domain_taxi,
    evaluate_by_domain_train,
    evaluate_by_domain_others,
    evaluate_single_dialogue,
    llm_qa
)
from taskdialogue.benchmarks.multiwoz.evaluators.metrics import (
    calculate_jga,
    calculate_slot_f1,
    calculate_domain_metrics,
    calculate_dialogue_metrics
)

logger = get_logger(__name__)


class MultiWOZEvaluator(BaseEvaluator):
    """MultiWOZ 完整评估器
    
    功能：
    - 使用 LLM 从对话中提取槽位值
    - 与数据库查询结果对比
    - 计算 6 个指标：inform, success, combined_score, book, jga, f1
    - 支持分领域评估
    - 支持 strict/relaxed 模式
    
    Example:
        >>> evaluator = MultiWOZEvaluator(config)
        >>> result = evaluator.evaluate(inference_result, ground_truth)
        >>> print(result.overall_score)  # combined_score
    """
    
    def __init__(self, config: Dict[str, Any]):
        # 提取 _full_config（避免循环引用）
        # 使用 get 而非 pop，避免修改原始 config（多进程共享）
        full_config = config.get("_full_config", None)
        
        super().__init__(config)
        
        # 创建评估模型（需要传递完整 config 以获取 API key）
        eval_model_name = config.get("eval_model", "deepseek-chat")
        eval_model_type = config.get("eval_model_type", "deepseek")
        model_params = config.get("model_params", {})
        
        # 使用之前提取的完整配置（包含 API keys）
        if full_config:
            from taskdialogue.core.utils.config import Config as ConfigClass
            config_obj = ConfigClass(full_config) if isinstance(full_config, dict) else full_config
        else:
            config_obj = None
        
        self.eval_model = create_model(
            provider=eval_model_type,
            model_name=eval_model_name,
            config=config_obj,
            temperature=model_params.get("temperature", 0.01),
            max_tokens=model_params.get("max_tokens", 4096)
        )
        
        # 评估配置
        self.success_mode = config.get("success_mode", "strict")
        self.provide_db_hints = config.get("provide_db_hints", True)
        self.per_domain_evaluation = config.get("per_domain_evaluation", True)
        self.save_detailed_logs = config.get("save_detailed_logs", True)
        self.enabled_metrics = config.get("metrics", [
            "inform", "success", "combined_score", "book", "jga", "f1"
        ])
        
        logger.info(f"MultiWOZ Evaluator initialized:")
        logger.info(f"  Eval model: {eval_model_name}")
        logger.info(f"  Success mode: {self.success_mode}")
        logger.info(f"  Metrics: {self.enabled_metrics}")
    
    def evaluate(
        self, 
        inference_result: InferenceResult,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """评估推理结果
        
        Args:
            inference_result: 推理结果
            ground_truth: 包含 goal 的完整对话数据
            
        Returns:
            评估结果
        """
        start_time = time.time()
        
        # 提取对话和目标
        dialogue = self._convert_to_dialog_format(inference_result)
        goal = ground_truth.get("goal", {}) if ground_truth else {}
        
        # 调试：检查dialogue是否为空
        if not dialogue:
            logger.error(f"Empty dialogue for {inference_result.dialogue_id}, dialogue_history length: {len(inference_result.dialogue_history)}")
            return self._create_error_result(inference_result, "Empty dialogue content")
        
        if not goal:
            logger.warning(f"No goal found for {inference_result.dialogue_id}")
            return self._create_error_result(inference_result, "No goal provided")
        
        # 使用旧代码的完整评估逻辑
        try:
            eval_result_dict = evaluate_single_dialogue(
                dialog=dialogue,
                goal=goal,
                eval_model=self.eval_model,
                success_mode=self.success_mode,
                provide_db_hints=self.provide_db_hints
            )
        except Exception as e:
            logger.error(f"Evaluation error for {inference_result.dialogue_id}: {e}")
            return self._create_error_result(inference_result, str(e))
        
        # 转换为统一格式
        evaluation_result = self._convert_to_unified_format(
            eval_result_dict,
            inference_result,
            time.time() - start_time,
            goal=goal
        )
        
        return evaluation_result
    
    def _convert_to_dialog_format(self, inference_result: InferenceResult) -> List[Dict]:
        """将 InferenceResult 转换为旧格式的对话"""
        dialogue = []
        for turn in inference_result.dialogue_history:
            dialogue.append({
                "user": turn.user_message.content or "",
                "agent": turn.agent_message.content or ""
            })
        return dialogue
    
    def _convert_to_unified_format(
        self,
        eval_dict: Dict[str, Any],
        inference_result: InferenceResult,
        evaluation_time: float,
        goal: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """将旧格式的评估结果转换为统一的 EvaluationResult"""
        
        # 提取 overall 指标
        overall = eval_dict.get('overall', {})
        combined_score = overall.get('combined_score', 0.0)
        
        # 准备 evaluator_config（排除 _full_config 避免循环引用）
        evaluator_config = {k: v for k, v in self.config.items() if k != '_full_config'}
        
        # 创建结果
        result = EvaluationResult(
            benchmark="multiwoz",
            task_id=inference_result.task_id,
            dialogue_id=inference_result.dialogue_id,
            overall_score=combined_score,
            evaluator_config=evaluator_config,
            evaluation_time=evaluation_time
        )
        
        # 添加指标
        if "inform" in self.enabled_metrics:
            result.add_metric("inform", overall.get('inform_rate', 0.0))
        
        if "success" in self.enabled_metrics:
            result.add_metric("success", overall.get('success_rate', 0.0))
        
        if "combined_score" in self.enabled_metrics:
            result.add_metric("combined_score", combined_score)
        
        if "book" in self.enabled_metrics and overall.get('book_rate') is not None:
            result.add_metric("book", overall.get('book_rate'))
        
        if "jga" in self.enabled_metrics:
            result.add_metric("jga", overall.get('jga', 0.0))
        
        if "f1" in self.enabled_metrics:
            result.add_metric("f1", overall.get('slot_f1', 0.0))
        
        # Benchmark 特定数据（分领域结果 + inference 信息）
        if self.per_domain_evaluation:
            # 保留 inference result 的信息
            result.benchmark_specific = inference_result.benchmark_specific.copy() if inference_result.benchmark_specific else {}
            
            # 添加inference原始数据（用于保存完整记录）
            # 提取turns (简化格式)
            turns = []
            for turn in inference_result.dialogue_history:
                turns.append({
                    "user": turn.user_message.content or "",
                    "agent": turn.agent_message.content or ""
                })
            
            # 提取conversation_data (完整messages)
            messages = []
            for turn in inference_result.dialogue_history:
                messages.append(turn.user_message.to_dict())
                if turn.agent_message:
                    messages.append(turn.agent_message.to_dict())
            
            conversation_data = {
                "system_prompt": None,  # 从inference_result中无法获取，后续可以改进
                "messages": messages,
                "tool_schemas": None,
                "model_name": inference_result.model_config_data.get("agent", {}).get("model_name"),
                "provider": inference_result.model_config_data.get("agent", {}).get("provider"),
                "token_usage": {
                    "total_tokens": inference_result.total_tokens
                }
            }
            
            # 统计信息
            stats = {
                "num_turns": inference_result.num_turns,
                "total_tokens": inference_result.total_tokens,
                "inference_time": inference_result.inference_time,
                "function_calls": result.benchmark_specific.get("function_calls", 0)
            }
            
            # 添加完整数据到benchmark_specific
            result.benchmark_specific.update({
                "domain_metrics": eval_dict.get('domains', {}),
                "dialogue_metrics": eval_dict.get('dialogue', {}),
                "goal": goal if goal else {},
                # 添加inference原始数据
                "turns": turns,
                "stats": stats,
                "conversation_data": conversation_data
            })
        
        return result
    
    def _create_error_result(
        self,
        inference_result: InferenceResult,
        error_message: str
    ) -> EvaluationResult:
        """创建错误结果"""
        # 排除 _full_config 避免循环引用
        evaluator_config = {k: v for k, v in self.config.items() if k != '_full_config'}
        
        return EvaluationResult(
            benchmark="multiwoz",
            task_id=inference_result.task_id,
            dialogue_id=inference_result.dialogue_id,
            overall_score=0.0,
            evaluator_config=evaluator_config,
            error_message=error_message
        )
    
    def evaluate_batch(
        self,
        inference_results: List[InferenceResult],
        ground_truths: Optional[List[Dict[str, Any]]] = None
    ) -> List[EvaluationResult]:
        """批量评估（覆盖基类方法以添加进度显示）"""
        evaluation_results = []
        total = len(inference_results)
        
        for i, inf_result in enumerate(inference_results):
            logger.info(f"Evaluating {i+1}/{total}: {inf_result.dialogue_id}")
            
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            eval_result = self.evaluate(inf_result, gt)
            evaluation_results.append(eval_result)
            
            # 显示进度
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{total} completed")
        
        return evaluation_results
