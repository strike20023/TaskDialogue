"""
评估结果格式化器

负责：
- 打印评估报告
- 保存评估结果
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from taskdialogue.core.schemas.evaluation import EvaluationResult
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.utils.logger import get_logger
from taskdialogue.benchmarks.multiwoz.constants import DOMAINS, DOMAIN_NAMES

logger = get_logger(__name__)


class ResultFormatter:
    """评估结果格式化器"""
    
    DOMAINS = DOMAINS
    DOMAIN_NAMES = DOMAIN_NAMES
    
    @classmethod
    def print_evaluation_report(
        cls,
        overall_metrics: Dict[str, Any],
        dialogue_metrics: Dict[str, Any],
        domain_averages: Dict[str, Any],
        metrics_by_domain: Dict[str, List],
        total_function_calls: int = 0
    ):
        """打印评估报告"""
        logger.info("\n📊 综合评估报告")
        logger.info("=" * 80)
        logger.info("\n📊 评估指标总览:\n")
        
        # 表头
        header = f"{'Level/Domain':<15} {'Count':>7} {'Inform':>8} {'Success':>8} {'JGA':>8} {'Slot-F1':>9} {'Combined':>9}"
        logger.info(header)
        logger.info("=" * 80)
        
        # Overall 行 (Domain Level汇总)
        logger.info(f"{'Overall':<15} {overall_metrics['num_domains']:>7} "
                   f"{overall_metrics['inform']:>7.2%} {overall_metrics['success']:>7.2%} "
                   f"{overall_metrics['jga']:>7.2%} {overall_metrics['f1']:>9.4f} "
                   f"{overall_metrics['combined_score']:>8.2%}")
        
        # Dialogue 行 (Dialogue Level)
        logger.info(f"{'Dialogue':<15} {dialogue_metrics['num_dialogues']:>7} "
                   f"{dialogue_metrics['inform']:>7.2%} {dialogue_metrics['success']:>7.2%} "
                   f"{dialogue_metrics['jga']:>7.2%} {dialogue_metrics['f1']:>9.4f} "
                   f"{dialogue_metrics['combined_score']:>8.2%}")
        
        # 分隔线
        logger.info("-" * 80)
        
        # 各领域行
        for domain in cls.DOMAINS:
            if domain in domain_averages:
                d_metrics = domain_averages[domain]
                jga_str = f"{d_metrics['jga']:>7.2%}" if d_metrics['jga'] > 0 else "   N/A"
                f1_str = f"{d_metrics['f1']:>9.4f}" if d_metrics['f1'] > 0 else "      N/A"
                
                logger.info(f"{cls.DOMAIN_NAMES[domain]:<15} {d_metrics['count']:>7} "
                           f"{d_metrics['inform']:>7.2%} {d_metrics['success']:>7.2%} "
                           f"{jga_str:>8} {f1_str:>9} {d_metrics['combined_score']:>8.2%}")
        
        logger.info("=" * 80)
        
        # 打印总体结果
        logger.info("\n📊 Overall Results (Domain Level):")
        logger.info(f"  • Domain Instances:   {overall_metrics['num_domains']}")
        logger.info(f"  • Inform Rate:        {overall_metrics['inform']:.2%}")
        logger.info(f"  • Success Rate:       {overall_metrics['success']:.2%}")
        logger.info(f"  • Combined Score:     {overall_metrics['combined_score']:.2%}")
        logger.info(f"  • JGA:                {overall_metrics['jga']:.2%}")
        logger.info(f"  • Slot F1:            {overall_metrics['f1']:.4f}")
        
        logger.info("\n📊 Dialogue Level Results:")
        logger.info(f"  • Dialogue Count:     {dialogue_metrics['num_dialogues']}")
        logger.info(f"  • Inform Rate:        {dialogue_metrics['inform']:.2%}")
        logger.info(f"  • Success Rate:       {dialogue_metrics['success']:.2%}")
        logger.info(f"  • Combined Score:     {dialogue_metrics['combined_score']:.2%}")
        logger.info(f"  • JGA:                {dialogue_metrics['jga']:.2%}")
        logger.info(f"  • Slot F1:            {dialogue_metrics['f1']:.4f}")
        
        if total_function_calls > 0:
            logger.info(f"\n📊 Function Calls:")
            logger.info(f"  • Total:              {total_function_calls}")
            logger.info(f"  • Avg per Dialogue:   {total_function_calls / dialogue_metrics['num_dialogues']:.1f}")
    
    @classmethod
    def save_evaluation_results(
        cls,
        inference_results: List[InferenceResult],
        evaluation_results: List[EvaluationResult],
        overall_metrics: Dict[str, Any],
        dialogue_metrics: Dict[str, Any],
        domain_averages: Dict[str, Any],
        output_file: Path,
        config: Dict[str, Any],
        model_name: str,
        provider: str,
        split: str,
        timestamp: str,
        tools: Any = None,
        system_prompt: str = None
    ):
        """保存评估结果为 JSON"""
        # 构建 inference_result -> evaluation_result 的映射
        eval_map = {eval_result.dialogue_id: eval_result for eval_result in evaluation_results}
        
        # 构建对话列表
        dialogues_data = []
        for inf_result in inference_results:
            eval_result = eval_map.get(inf_result.dialogue_id)
            
            # Inference 部分（messages）
            messages = []
            if inf_result.raw_messages:
                for msg in inf_result.raw_messages:
                    messages.append(msg.to_openai_format())
            
            # Stats
            stats = {
                "num_turns": inf_result.num_turns,
                "total_tokens": inf_result.total_tokens,
                "inference_time": inf_result.inference_time,
                "function_calls": inf_result.benchmark_specific.get("function_calls", 0) if inf_result.benchmark_specific else 0
            }
            
            # Evaluation 部分
            evaluation_data = None
            if eval_result and not eval_result.error_message:
                # 提取评估指标
                eval_metrics = {}
                for metric in eval_result.metrics:
                    eval_metrics[metric.name] = metric.value
                
                # 领域级别评估
                domain_eval = {}
                if eval_result.benchmark_specific and 'domain_metrics' in eval_result.benchmark_specific:
                    for domain, domain_data in eval_result.benchmark_specific['domain_metrics'].items():
                        inform_val = domain_data.get('inform', {}).get('complete')
                        success_val = domain_data.get('success', {}).get('complete')
                        
                        if inform_val is not None:
                            domain_eval[domain] = {
                                'inform': int(inform_val),
                                'success': int(success_val) if success_val is not None else None,
                                'details': domain_data
                            }
                
                evaluation_data = {
                    **eval_metrics,
                    'domains': domain_eval
                }
            
            # 组合数据
            dialogue_data = {
                "dialogue_id": inf_result.dialogue_id,
                "messages": messages,
                "stats": stats,
                "goal": inf_result.benchmark_specific.get("goals", {}) if inf_result.benchmark_specific else {},
                "evaluation": evaluation_data
            }
            
            dialogues_data.append(dialogue_data)
        
        # 如果没有传入 tools 和 system_prompt，尝试从 inference_results 获取
        if tools is None or system_prompt is None:
            if inference_results and inference_results[0].raw_messages:
                first_msg = inference_results[0].raw_messages[0]
                if first_msg.role == 'system' and system_prompt is None:
                    system_prompt = first_msg.content
        
        # 最终 JSON 结构（使用更强大的序列化器）
        def json_serializer(obj):
            """处理不可序列化的对象"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            # 处理数据库对象（SQLAlchemy）
            if hasattr(obj, '__dict__') and hasattr(obj, '__table__'):
                # SQLAlchemy 对象，转换为 dict
                return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
            # 处理其他对象
            if hasattr(obj, '__dict__'):
                return str(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        final_json = {
            "benchmark": "multiwoz",
            "timestamp": timestamp,
            "model_name": model_name,
            "provider": provider,
            "split": split,
            "num_dialogues": len(inference_results),
            "config": config,
            "tools": tools,
            "system_prompt": system_prompt,
            "overall_metrics": overall_metrics,
            "dialogue_metrics": dialogue_metrics,
            "domain_metrics": domain_averages,
            "dialogues": dialogues_data
        }
        
        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False, default=json_serializer)
        
        logger.info(f"\nEvaluation results saved to: {output_file}")
        logger.info(f"   - {len(inference_results)} dialogues")
        logger.info(f"   - Overall Combined Score: {overall_metrics['combined_score']:.2%}")
        logger.info(f"   - Dialogue Combined Score: {dialogue_metrics['combined_score']:.2%}")

