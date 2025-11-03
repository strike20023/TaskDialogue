"""
评估指标聚合器

负责聚合和计算评估指标：
- Overall metrics (domain-level average)
- Dialogue metrics (dialogue-level average)
- Domain-specific metrics
"""

from typing import List, Dict, Any, Tuple
from taskdialogue.core.schemas.evaluation import EvaluationResult
from taskdialogue.benchmarks.multiwoz.constants import DOMAINS


class MetricsAggregator:
    """评估指标聚合器"""
    
    DOMAINS = DOMAINS
    
    @staticmethod
    def _avg_metric(metrics: List[Dict], key: str) -> float:
        """计算指标平均值（忽略 None）"""
        values = [m[key] for m in metrics if m.get(key) is not None]
        return sum(values) / len(values) if values else 0
    
    @classmethod
    def aggregate_metrics(
        cls, 
        evaluation_results: List[EvaluationResult]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, List], List[Dict]]:
        """聚合评估指标
        
        Args:
            evaluation_results: 评估结果列表
            
        Returns:
            (overall_metrics, dialogue_metrics, domain_averages, metrics_by_domain, dialogue_level_metrics)
        """
        dialogue_level_metrics = []
        metrics_by_domain = {}
        
        for result in evaluation_results:
            # 跳过失败的评估结果
            if result is None or result.error_message:
                continue
            
            # 收集 dialogue-level metrics
            dialogue_informs, dialogue_successes = [], []
            
            if result.benchmark_specific and 'domain_metrics' in result.benchmark_specific:
                domain_metrics = result.benchmark_specific['domain_metrics']
                
                for domain, domain_data in domain_metrics.items():
                    inform_val = domain_data.get('inform', {}).get('complete')
                    success_val = domain_data.get('success', {}).get('complete')
                    
                    if inform_val is not None:
                        dialogue_informs.append(int(inform_val))
                        if success_val is not None:
                            dialogue_successes.append(int(success_val))
                    
                    # 收集 domain-level metrics
                    if domain not in metrics_by_domain:
                        metrics_by_domain[domain] = []
                    
                    if inform_val is not None and success_val is not None:
                        # 计算 JGA 和 F1
                        jga_val, f1_val = cls._compute_slot_metrics(
                            domain_data, 
                            result.benchmark_specific.get('goal', {}),
                            domain
                        )
                        
                        metrics_by_domain[domain].append({
                            'inform': int(inform_val),
                            'success': int(success_val),
                            'combined_score': 0.5 * int(inform_val) + 0.5 * int(success_val),
                            'jga': jga_val,
                            'f1': f1_val
                        })
            
            # Dialogue-level
            if dialogue_informs:
                dialogue_inform = 1 if all(dialogue_informs) else 0
                dialogue_success = 1 if all(dialogue_successes) else 0
                
                # 获取 JGA 和 F1
                dialogue_jga, dialogue_f1 = None, None
                for metric in result.metrics:
                    if metric.name == 'jga':
                        dialogue_jga = metric.value
                    elif metric.name == 'f1':
                        dialogue_f1 = metric.value
                
                dialogue_level_metrics.append({
                    'inform': dialogue_inform,
                    'success': dialogue_success,
                    'combined_score': 0.5 * dialogue_inform + 0.5 * dialogue_success,
                    'jga': dialogue_jga,
                    'f1': dialogue_f1
                })
        
        # 计算总体指标
        all_domain_metrics = []
        for domain_metrics_list in metrics_by_domain.values():
            all_domain_metrics.extend(domain_metrics_list)
        
        overall_metrics = {
            'inform': cls._avg_metric(all_domain_metrics, 'inform') if all_domain_metrics else 0,
            'success': cls._avg_metric(all_domain_metrics, 'success') if all_domain_metrics else 0,
            'combined_score': cls._avg_metric(all_domain_metrics, 'combined_score') if all_domain_metrics else 0,
            'jga': cls._avg_metric(all_domain_metrics, 'jga') if all_domain_metrics else 0,
            'f1': cls._avg_metric(all_domain_metrics, 'f1') if all_domain_metrics else 0,
            'num_domains': len(all_domain_metrics)
        }
        
        dialogue_metrics = {
            'inform': cls._avg_metric(dialogue_level_metrics, 'inform') if dialogue_level_metrics else 0,
            'success': cls._avg_metric(dialogue_level_metrics, 'success') if dialogue_level_metrics else 0,
            'combined_score': cls._avg_metric(dialogue_level_metrics, 'combined_score') if dialogue_level_metrics else 0,
            'jga': cls._avg_metric(dialogue_level_metrics, 'jga') if dialogue_level_metrics else 0,
            'f1': cls._avg_metric(dialogue_level_metrics, 'f1') if dialogue_level_metrics else 0,
            'num_dialogues': len(dialogue_level_metrics)
        }
        
        # Domain-level averages
        domain_averages = {}
        for domain in cls.DOMAINS:
            if domain in metrics_by_domain:
                d_metrics = metrics_by_domain[domain]
                domain_averages[domain] = {
                    'inform': cls._avg_metric(d_metrics, 'inform'),
                    'success': cls._avg_metric(d_metrics, 'success'),
                    'combined_score': cls._avg_metric(d_metrics, 'combined_score'),
                    'jga': cls._avg_metric(d_metrics, 'jga'),
                    'f1': cls._avg_metric(d_metrics, 'f1'),
                    'count': len(d_metrics)
                }
        
        return overall_metrics, dialogue_metrics, domain_averages, metrics_by_domain, dialogue_level_metrics
    
    @staticmethod
    def _compute_slot_metrics(domain_data: Dict, ground_truth_goal: Dict, domain: str) -> Tuple:
        """计算 JGA 和 F1"""
        jga_val, f1_val = None, None
        
        if 'inform' in domain_data and 'slot_values' in domain_data['inform']:
            inform_slot_values = domain_data['inform']['slot_values']
            domain_goal = ground_truth_goal.get(domain, {})
            gt_slots = domain_goal.get('info', {})
            
            if inform_slot_values and gt_slots:
                from taskdialogue.benchmarks.multiwoz.evaluators.metrics import calculate_jga, calculate_slot_f1
                jga_val = calculate_jga(inform_slot_values, gt_slots)
                f1_result = calculate_slot_f1(inform_slot_values, gt_slots)
                f1_val = f1_result['f1']
        
        return jga_val, f1_val

