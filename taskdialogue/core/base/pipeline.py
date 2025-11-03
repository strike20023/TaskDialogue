"""
Pipeline 基础抽象类

定义 benchmark 运行流程的标准接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult
from taskdialogue.core.utils.logger import get_logger

logger = get_logger(__name__)


class BasePipeline(ABC):
    """Benchmark 运行流程的基类
    
    Pipeline 负责：
    - 运行完整的推理流程
    - 运行评估流程
    - 保存和加载结果
    - 生成报告
    
    设计理念：
    - 标准化流程：统一的运行方式
    - 模块化设计：推理和评估分离
    - 结果持久化：统一的保存格式
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化 Pipeline
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.benchmark_name = config.get("benchmark", {}).get("type", "unknown")
        
        # 输出目录
        self.inference_dir = Path(config.get("output", {}).get(
            "inference_dir", 
            f"results/{self.benchmark_name}/inference/"
        ))
        self.evaluation_dir = Path(config.get("output", {}).get(
            "evaluation_dir", 
            f"results/{self.benchmark_name}/evaluation/"
        ))
        
        # 创建目录
        self.inference_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized for {self.benchmark_name}")
    
    def _json_serializer(self, obj: Any) -> Any:
        """JSON序列化器（处理datetime等特殊对象）
        
        这是一个通用的JSON序列化器，用于保存结果文件时处理特殊类型。
        
        Args:
            obj: 需要序列化的对象
            
        Returns:
            可序列化的对象
            
        Raises:
            TypeError: 如果对象无法序列化
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳（格式化的字符串）
        
        Returns:
            格式化的时间戳字符串 (YYYYMMDD_HHMMSS)
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @abstractmethod
    def run_inference(
        self, 
        task_ids: Optional[List[str]] = None
    ) -> List[InferenceResult]:
        """运行推理
        
        Args:
            task_ids: 要运行的任务 ID 列表（None 表示运行全部）
            
        Returns:
            推理结果列表
            
        Example:
            >>> pipeline = MultiWOZPipeline(config)
            >>> results = pipeline.run_inference(task_ids=["MUL0001", "MUL0002"])
        """
        pass
    
    @abstractmethod
    def run_evaluation(
        self, 
        inference_results: List[InferenceResult]
    ) -> List[EvaluationResult]:
        """运行评估
        
        Args:
            inference_results: 推理结果列表
            
        Returns:
            评估结果列表
        """
        pass
    
    def run_full_pipeline(
        self, 
        task_ids: Optional[List[str]] = None
    ) -> tuple[List[InferenceResult], List[EvaluationResult]]:
        """运行完整流程：推理 + 评估
        
        Args:
            task_ids: 要运行的任务 ID 列表
            
        Returns:
            (推理结果列表, 评估结果列表)
        """
        logger.info(f"Starting full pipeline for {self.benchmark_name}")
        
        # 1. 推理
        logger.info("Phase 1: Running inference...")
        inference_results = self.run_inference(task_ids)
        logger.info(f"Inference completed: {len(inference_results)} results")
        
        # 2. 保存推理结果
        logger.info("Saving inference results...")
        self.save_inference_results(inference_results)
        
        # 3. 评估
        logger.info("Phase 2: Running evaluation...")
        evaluation_results = self.run_evaluation(inference_results)
        logger.info(f"Evaluation completed: {len(evaluation_results)} results")
        
        # 4. 保存评估结果
        logger.info("Saving evaluation results...")
        # 检查子类是否设置了自定义文件名
        output_basename = getattr(self, '_eval_output_basename', None)
        self.save_evaluation_results(evaluation_results, output_basename)
        
        # 5. 生成报告（已禁用）
        # logger.info("Generating report...")
        # self.generate_report(inference_results, evaluation_results)
        
        logger.info("Pipeline completed successfully!")
        
        return inference_results, evaluation_results
    
    def save_inference_results(self, results: List[InferenceResult]) -> None:
        """保存推理结果
        
        注意：MultiWOZ Pipeline 已经在 run_inference 中保存为 JSONL 格式，
        metadata 也已经融入到 JSONL 文件的第一行，不再单独保存 summary.json
        """
        # ✅ 已禁用 summary.json 保存，metadata 已融入 predictions.jsonl 的第一行
        pass
    
    def save_evaluation_results(self, results: List[EvaluationResult], output_basename: Optional[str] = None) -> None:
        """保存评估结果到一个大的 JSON 文件（参考 backup）
        
        Args:
            results: 评估结果列表
            output_basename: 输出文件基础名（不含扩展名），如果为 None 则自动生成
        """
        if not results:
            return
            
        timestamp = self._get_timestamp()
        
        # 使用基类的JSON序列化器
        
        # 计算统计信息
        successful_results = [r for r in results if not r.error_message]
        total_score = sum(r.overall_score for r in successful_results) if successful_results else 0
        avg_score = total_score / len(successful_results) if successful_results else 0
        
        # 收集所有指标
        all_metrics = {}
        if successful_results:
            for metric in successful_results[0].metrics:
                metric_name = metric.name
                metric_values = [m.value for r in successful_results for m in r.metrics if m.name == metric_name and m.value is not None]
                if metric_values:
                    all_metrics[metric_name] = sum(metric_values) / len(metric_values)
        
        # 构建per_dialogue数据（参考格式）
        per_dialogue = []
        for r in results:
            # 构建符合参考格式的数据结构
            dialogue_entry = {
                "dialogue_id": r.dialogue_id
            }
            
            # 从benchmark_specific提取数据
            if r.benchmark_specific:
                bs = r.benchmark_specific
                
                # 添加inference原始数据到顶层
                if 'turns' in bs:
                    dialogue_entry['turns'] = bs['turns']
                if 'stats' in bs:
                    dialogue_entry['stats'] = bs['stats']
                if 'conversation_data' in bs:
                    dialogue_entry['conversation_data'] = bs['conversation_data']
                if 'goal' in bs:
                    dialogue_entry['goal'] = bs['goal']
                
                # 构建evaluation字段
                dialogue_entry['evaluation'] = {
                    "dialogue_id": r.dialogue_id,
                    "num_turns": bs.get('stats', {}).get('num_turns', 0),
                    "num_function_calls": bs.get('stats', {}).get('function_calls', 0),
                    "token_usage": {
                        "total_tokens": bs.get('stats', {}).get('total_tokens', 0)
                    },
                    "goals": bs.get('goal', {}),
                    "domains": bs.get('domain_metrics', {}),
                    "overall": {
                        "inform_rate": next((m.value for m in r.metrics if m.name == 'inform'), 0),
                        "success_rate": next((m.value for m in r.metrics if m.name == 'success'), 0),
                        "book_rate": next((m.value for m in r.metrics if m.name == 'book'), None),
                        "combined_score": r.overall_score,
                        "jga": next((m.value for m in r.metrics if m.name == 'jga'), 0),
                        "slot_f1": next((m.value for m in r.metrics if m.name == 'f1'), 0)
                    }
                }
            
            per_dialogue.append(dialogue_entry)
        
        # 构建评估结果数据结构（参考格式）
        evaluation_data = {
            "evaluation_time": timestamp,
            "num_dialogues": len(results),
            "statistics": {
                "total": len(results),
                "successful": len(successful_results),
                "failed": len(results) - len(successful_results),
                "average_score": avg_score
            },
            "metrics": all_metrics,
            "metadata": {
            "benchmark": self.benchmark_name,
                "timestamp": timestamp
            },
            "per_dialogue": per_dialogue
        }
        
        # 使用指定的文件名或生成默认文件名
        if output_basename:
            output_file = self.evaluation_dir / f"{output_basename}.json"
        else:
            output_file = self.evaluation_dir / f"evaluation_{timestamp}.json"
        
        # 保存到一个大的 JSON 文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        logger.info(f"Saved {len(results)} evaluation results to {output_file}")
    
    def load_inference_results(
        self, 
        result_files: Optional[List[str]] = None
    ) -> List[InferenceResult]:
        """加载推理结果
        
        Args:
            result_files: 结果文件路径列表（None 表示加载最新的jsonl文件）
            
        Returns:
            推理结果列表
        """
        if result_files is None:
            # ✅ 查找最新的 json 或 jsonl 文件
            json_files = list(self.inference_dir.glob("predictions_*.json"))
            jsonl_files = list(self.inference_dir.glob("*.jsonl"))
            all_files = json_files + jsonl_files
            
            if not all_files:
                logger.error(f"No inference files found in {self.inference_dir}")
                return []
            # 按修改时间排序，取最新的
            result_files = [max(all_files, key=lambda f: f.stat().st_mtime)]
            logger.info(f"Loading latest inference results: {result_files[0]}")
        
        results = []
        for filepath in result_files:
            filepath = Path(filepath)
            
            # ✅ 根据文件扩展名选择加载方式
            if filepath.suffix == '.json':
                # 加载大 JSON 格式（新格式）
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    logger.info(f"Loaded {data.get('num_dialogues', 0)} dialogues from {filepath}")
                    
                    # 从 dialogues 字段加载
                    for dialogue_data in data.get('dialogues', []):
                        # 从 messages 构造 dialogue_history
                        dialogue_history = []
                        messages = dialogue_data.get('messages', [])
                        
                        # 跳过 system 消息，按 user/assistant 对组织
                        i = 0
                        if i < len(messages) and messages[i].get('role') == 'system':
                            i += 1  # 跳过 system
                        
                        turn_id = 1
                        while i < len(messages):
                            from taskdialogue.core.schemas.dialogue import DialogueTurn
                            from taskdialogue.core.schemas.message import Message
                            
                            user_msg = None
                            agent_msg = None
                            
                            # 读取 user 消息
                            if i < len(messages) and messages[i].get('role') == 'user':
                                user_msg = Message.from_dict(messages[i])
                                i += 1
                            
                            # 读取 assistant 消息（可能包含 tool_calls）
                            if i < len(messages) and messages[i].get('role') == 'assistant':
                                agent_msg = Message.from_dict(messages[i])
                                i += 1
                                
                                # 跳过后续的 tool 消息
                                while i < len(messages) and messages[i].get('role') == 'tool':
                                    i += 1
                                    # 如果有 tool 响应后的 assistant 消息，更新 agent_msg
                                    if i < len(messages) and messages[i].get('role') == 'assistant':
                                        agent_msg = Message.from_dict(messages[i])
                                        i += 1
                            
                            if user_msg:
                                dialogue_history.append(DialogueTurn(
                                    turn_id=turn_id,
                                    user_message=user_msg,
                                    agent_message=agent_msg
                                ))
                                turn_id += 1
                        
                        # 构造 InferenceResult
                        stats = dialogue_data.get('stats', {})
                        
                        # 将 messages 转换为 Message 对象（保留为 raw_messages）
                        raw_messages = []
                        for msg_data in dialogue_data.get('messages', []):
                            raw_messages.append(Message.from_dict(msg_data))
                        
                        result = InferenceResult(
                            benchmark="multiwoz",
                            task_id=dialogue_data['dialogue_id'],
                            dialogue_id=dialogue_data['dialogue_id'],
                            dialogue_history=dialogue_history,
                            raw_messages=raw_messages,  # 保存完整的 messages
                            num_turns=stats.get('num_turns', len(dialogue_history)),
                            total_tokens=stats.get('total_tokens', 0),
                            inference_time=stats.get('inference_time', 0),
                            success=True,
                            timestamp=datetime.now(),
                            benchmark_specific={"goals": dialogue_data.get('goal', {})}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif filepath.suffix == '.jsonl':
                # 加载JSONL格式（每行一个对话）
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line)
                            
                            # 转换为InferenceResult
                            # 从turns构造dialogue_history
                            dialogue_history = []
                            for idx, turn in enumerate(data.get('turns', []), 1):
                                from taskdialogue.core.schemas.dialogue import DialogueTurn
                                from taskdialogue.core.schemas.message import Message
                                dialogue_history.append(DialogueTurn(
                                    turn_id=idx,
                                    user_message=Message(role='user', content=turn.get('user', '')),
                                    agent_message=Message(role='assistant', content=turn.get('agent', ''))
                                ))
                            
                            # 构造InferenceResult
                            stats = data.get('stats', {})
                            result = InferenceResult(
                                benchmark='multiwoz',
                                task_id=data.get('dialogue_id'),
                                dialogue_id=data.get('dialogue_id'),
                                dialogue_history=dialogue_history,
                                model_config_data=data.get('conversation_data', {}).get('model_name', {}),
                                num_turns=stats.get('num_turns', len(dialogue_history)),
                                total_tokens=stats.get('total_tokens'),
                                inference_time=stats.get('inference_time'),
                                benchmark_specific={
                                    'goals': data.get('goal', {}),
                                    'domains': list(data.get('goal', {}).keys()),
                                    'function_calls': stats.get('function_calls', 0)
                                }
                            )
                            results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                # 加载JSON格式（单个文件）
                try:
                    result = InferenceResult.load_from_file(filepath)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
        
        logger.info(f"Loaded {len(results)} inference results")
        return results
    
    def generate_report(
        self,
        inference_results: List[InferenceResult],
        evaluation_results: List[EvaluationResult]
    ) -> None:
        """生成评估报告
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 聚合指标
        aggregated_metrics = self._aggregate_metrics(evaluation_results)
        
        # 计算成功和失败的对话数
        successful_evaluations = [r for r in evaluation_results if not r.error_message]
        failed_evaluations = [r for r in evaluation_results if r.error_message]
        
        # 序列化 config，避免循环引用
        serializable_config = self._make_config_serializable(self.config)
        
        report = {
            "benchmark": self.benchmark_name,
            "timestamp": timestamp,
            "config": serializable_config,
            "inference_summary": {
                "num_dialogues": len(inference_results),
                "successful": len([r for r in inference_results if r.success]),
                "failed": len([r for r in inference_results if not r.success]),
                "average_turns": sum(r.num_turns for r in inference_results) / len(inference_results) if inference_results else 0,
                "total_tokens": sum(r.total_tokens or 0 for r in inference_results),
                "average_tokens_per_dialogue": sum(r.total_tokens or 0 for r in inference_results) / len(inference_results) if inference_results else 0,
            },
            "evaluation_summary": {
                "num_evaluations": len(evaluation_results),
                "successful": len(successful_evaluations),
                "failed": len(failed_evaluations),
                "average_score": sum(r.overall_score for r in successful_evaluations) / len(successful_evaluations) if successful_evaluations else 0,
                "metrics": aggregated_metrics,
                "errors": [
                    {
                        "dialogue_id": r.dialogue_id,
                        "error": r.error_message
                    } 
                    for r in failed_evaluations
                ] if failed_evaluations else []
            }
        }
        
        report_file = self.evaluation_dir.parent / f"report_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("📊 Evaluation Report Summary")
        logger.info(f"{'='*80}")
        logger.info(f"Inference: {len(inference_results)} dialogues ({len([r for r in inference_results if r.success])} successful)")
        logger.info(f"Evaluation: {len(successful_evaluations)} successful, {len(failed_evaluations)} failed")
        if aggregated_metrics:
            logger.info(f"\nMetrics:")
            for name, value in aggregated_metrics.items():
                logger.info(f"  • {name:20s}: {value:.4f}")
        else:
            logger.warning("⚠️  No metrics found in evaluation results!")
        logger.info(f"\nFull report saved to: {report_file}")
        logger.info(f"{'='*80}\n")
    
    def _make_config_serializable(self, obj: Any, seen: Optional[set] = None) -> Any:
        """递归地将配置对象转换为可序列化的格式
        
        Args:
            obj: 需要序列化的对象
            seen: 已访问对象的 ID 集合（用于检测循环引用）
            
        Returns:
            可序列化的对象
        """
        if seen is None:
            seen = set()
        
        # 基础类型直接返回
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # 检查循环引用（对于可变对象）
        obj_id = id(obj)
        if obj_id in seen:
            return f"<circular reference: {type(obj).__name__}>"
        
        # 列表/元组递归处理
        if isinstance(obj, (list, tuple)):
            seen.add(obj_id)
            try:
                result = [self._make_config_serializable(item, seen) for item in obj]
                return result
            finally:
                seen.discard(obj_id)
        
        # 字典递归处理
        if isinstance(obj, dict):
            seen.add(obj_id)
            try:
                result = {}
                for key, value in obj.items():
                    try:
                        # 尝试序列化键和值
                        serializable_key = str(key) if not isinstance(key, str) else key
                        result[serializable_key] = self._make_config_serializable(value, seen)
                    except Exception as e:
                        # 如果无法序列化，使用字符串表示
                        logger.debug(f"Cannot serialize config key '{key}': {e}")
                        result[serializable_key] = f"<error: {type(value).__name__}>"
                return result
            finally:
                seen.discard(obj_id)
        
        # 其他类型转换为字符串表示
        try:
            # 尝试 JSON 序列化测试（不递归）
            json.dumps(obj)
            return obj
        except (TypeError, ValueError, RecursionError):
            return f"<{type(obj).__name__}>"
    
    def _aggregate_metrics(self, evaluation_results: List[EvaluationResult]) -> Dict[str, float]:
        """聚合评估指标"""
        if not evaluation_results:
            return {}
        
        # 过滤掉有错误的结果
        valid_results = [r for r in evaluation_results if not r.error_message]
        
        if not valid_results:
            logger.warning("No valid evaluation results to aggregate metrics")
            return {}
        
        all_metrics: Dict[str, List[float]] = {}
        for result in valid_results:
            if not result.metrics:
                logger.debug(f"No metrics found in result for {result.dialogue_id}")
                continue
            
            for metric in result.metrics:
                if metric.name not in all_metrics:
                    all_metrics[metric.name] = []
                all_metrics[metric.name].append(metric.value)
        
        if not all_metrics:
            logger.warning(f"No metrics collected from {len(valid_results)} valid results")
        
        return {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
            if values  # 确保列表非空
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(benchmark={self.benchmark_name})"

