"""
增强的评估指标计算模块
包括 Domain Level、Dialogue Level、JGA 和 Slot-F1

评估指标说明 (参考 MultiWOZ 论文和相关研究):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Inform Rate: 系统是否提供了满足用户约束的正确实体
   - 定义: 系统在对话中提供至少一个符合用户约束的实体
   - 公式: Inform_d = 1 if 提供了正确实体, else 0

2. Success Rate: 系统是否完成了完整任务（综合指标）
   - 定义: Inform=1 + 回答所有请求属性 + (如需预订)预订成功
   - 公式: S_d = 1 if (Inform=1 AND 所有reqt已回答 AND Book=1(如需)), else 0
   - **关键**: Success 包含 Book（Book 是 Success 的子指标）

3. Book Rate: Success 的子指标（只统计需要预订的对话）
   - 定义: 在需要预订的对话中，预订是否成功
   - 公式: Book_d = 1 if 预订实体符合用户目标, else 0
   - **关系**: Book ⊆ Success ⊆ Inform

4. Combined Score: 综合得分
   - 公式: Combined = 0.5 * Inform + 0.5 * Success
   - **Success 已包含 Book**，不需要单独加权

核心原则:
- Success ⊆ Inform (Success 依赖于 Inform)
- Book ⊆ Success (Book 是 Success 的组成部分)
- Book ⊆ Inform (传递性)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict


def validate_evaluation_consistency(result: Dict[str, Any]) -> Dict[str, str]:
    """
    验证评估结果的一致性
    
    确保满足以下约束 (参考 TD-Eval):
    1. Success ⊆ Inform: Success = 1 时，Inform 必须 = 1
    2. Book ⊆ Inform: Book = 1 时，Inform 必须 = 1
    
    Args:
        result: 单个领域的评估结果
    
    Returns:
        验证警告字典 (如果有问题)
    """
    warnings = {}
    
    if 'domains' not in result:
        return warnings
    
    for domain, domain_data in result['domains'].items():
        if 'error' in domain_data:
            continue
        
        # run_evaluation.py 会把嵌套结构展平
        inform_complete = domain_data.get('inform_complete', 0)
        success_complete = domain_data.get('success_complete', None)
        book_complete = domain_data.get('book_complete', None)
        
        # 检查 Success ⊆ Inform
        if success_complete == 1 and inform_complete != 1:
            warnings[f'{domain}_success'] = (
                f"⚠️  不一致: {domain} Success=1 但 Inform=0 "
                f"(违反了 TD-Eval 的严格标准)"
            )
        
        # 检查 Book ⊆ Inform
        if book_complete == 1 and inform_complete != 1:
            warnings[f'{domain}_book'] = (
                f"⚠️  不一致: {domain} Book=1 但 Inform=0 "
                f"(违反了 TD-Eval 的严格标准)"
            )
    
    return warnings


def calc_combine_score(inform: Optional[float], success: Optional[float], book: Optional[float]) -> float:
    """
    计算 Combined Score
    
    Formula:
        score = 0.5 * inform + 0.5 * success_book
        success_book = (success + book) / (success_f + book_f)
    
    When all three metrics are available:
        score = 0.5 * inform + 0.25 * success + 0.25 * book
    
    Args:
        inform: Inform Rate (0-1)
        success: Success Rate (0-1)
        book: Book Rate (0-1)
    
    Returns:
        Combined score (0-1)
    """
    if inform is None:
        inform = 0.0
    if success is None:
        success = 0.0
    
    # 标准公式: Combined = 0.5 * Inform + 0.5 * Success
    # Success 已包含 Book（如果需要预订，Book 必须成功才算 Success=1）
    # 所以不需要单独加权 Book
    score = 0.5 * inform + 0.5 * success
    
    return score


def calculate_slot_f1(predicted_slots: Dict[str, str], 
                      ground_truth_slots: Dict[str, str]) -> Dict[str, float]:
    """
    计算槽位级别的 Precision, Recall 和 F1
    
    Args:
        predicted_slots: 预测的槽位值 {slot_name: value}
        ground_truth_slots: 真实的槽位值 {slot_name: value}
    
    Returns:
        {precision, recall, f1}
    """
    # 过滤掉 'none' 值
    pred_valid = {k: v for k, v in predicted_slots.items() 
                  if v and str(v).lower() not in ['none', '']}
    gt_valid = {k: v for k, v in ground_truth_slots.items() 
                if v and str(v).lower() not in ['none', '']}
    
    if not pred_valid and not gt_valid:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    
    if not pred_valid:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not gt_valid:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # 使用smart_match进行智能匹配（处理时间标准化、大小写等）
    from taskdialogue.benchmarks.multiwoz.evaluators.matching.slot_matcher import smart_match
    
    correct = 0
    for slot_name, pred_value in pred_valid.items():
        if slot_name in gt_valid:
            gt_value = gt_valid[slot_name]
            # 使用智能匹配代替简单字符串比较
            if smart_match(slot_name, gt_value, pred_value):
                correct += 1
    
    precision = correct / len(pred_valid) if pred_valid else 0.0
    recall = correct / len(gt_valid) if gt_valid else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'predicted': len(pred_valid),
        'ground_truth': len(gt_valid)
    }


def calculate_jga(predicted_slots: Dict[str, str],
                  ground_truth_slots: Dict[str, str]) -> float:
    """
    计算 Joint Goal Accuracy (JGA)
    只有当所有槽位都正确时才算1，否则为0
    
    使用 smart_match 进行智能匹配，与 Slot-F1 保持一致
    
    Args:
        predicted_slots: 预测的槽位值
        ground_truth_slots: 真实的槽位值
    
    Returns:
        1.0 if all slots match, 0.0 otherwise
    """
    # 使用 smart_match 进行智能匹配
    from taskdialogue.benchmarks.multiwoz.evaluators.matching.slot_matcher import smart_match
    
    if not ground_truth_slots:
        return 1.0 if not predicted_slots else 0.0
    
    # 检查所有 ground truth 槽位是否都被正确预测
    for slot, gt_value in ground_truth_slots.items():
        pred_value = predicted_slots.get(slot, 'none')
        # 使用 smart_match 代替简单字符串比较
        if not smart_match(slot, gt_value, pred_value):
            return 0.0
    
    # 检查是否有多余的预测槽位
    for slot, pred_value in predicted_slots.items():
        if pred_value and str(pred_value).lower() not in ['none', ''] and slot not in ground_truth_slots:
            return 0.0
    
    return 1.0


def calculate_domain_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    计算每个领域的详细指标
    
    Returns:
        {
            'restaurant': {
                'count': int,
                'inform_rate': float,
                'success_rate': float,
                'book_rate': float,
                'jga': float,
                'slot_f1': float
            },
            ...
        }
    """
    from taskdialogue.benchmarks.multiwoz.tools.utils import DOMAINS
    
    domain_metrics = {}
    
    for domain in DOMAINS:
        domain_results = []
        jga_scores = []
        slot_f1_scores = []
        
        for result in evaluation_results:
            # 数据结构: result['evaluation']['domains'][domain]
            evaluation = result.get('evaluation', {})
            if 'error' not in result and domain in evaluation.get('domains', {}):
                domain_data = evaluation['domains'][domain]
                if 'error' not in domain_data:
                    domain_results.append(domain_data)
                    
                    # 计算 JGA
                    if 'inform_slot_values' in domain_data:
                        # 从goal获取ground truth（goal在result顶层，不在evaluation中）
                        goal = result.get('goal', {}).get(domain, {})
                        gt_slots = goal.get('info', {})
                        pred_slots = domain_data['inform_slot_values']
                        
                        # 统一使用 calculate_jga 计算
                        # 现在 LLM 会提取 info slots，可以真正评估系统对约束的理解
                        jga = calculate_jga(pred_slots, gt_slots)
                        jga_scores.append(jga)
                        
                        # 计算 Slot F1
                        slot_f1_result = calculate_slot_f1(pred_slots, gt_slots)
                        slot_f1_scores.append(slot_f1_result['f1'])
        
        if domain_results:
            # 参考 TD-Eval 第 142-148 行: 使用相同的分母计算 Inform 和 Success
            # 核心逻辑: for domain in set(match) | set(success)
            # 即：所有有 Inform 或 Success 的对话都参与统计，没有的记为 0
            
            inform_sum = 0
            success_sum = 0
            book_sum = 0
            book_count = 0
            
            for r in domain_results:
                inform_val = r.get('inform_complete')
                success_val = r.get('success_complete')
                book_val = r.get('book_complete')
                
                # Inform 和 Success 使用相同的分母（所有对话）
                inform_sum += inform_val if inform_val is not None else 0
                success_sum += success_val if success_val is not None else 0
                
                # Book 单独统计（只统计有 book 的对话）
                if book_val is not None:
                    book_sum += book_val
                    book_count += 1
            
            # 使用相同的分母 (total)
            total = len(domain_results)
            
            if total > 0:
                inform_rate = inform_sum / total
                success_rate = success_sum / total
                book_rate = book_sum / book_count if book_count > 0 else None
            else:
                inform_rate = 0
                success_rate = 0
                book_rate = None
            
            # Calculate combined score using the formula
            combined_score = calc_combine_score(
                inform=inform_rate if total > 0 else None,
                success=success_rate if total > 0 else None,
                book=book_rate
            )
            
            domain_metrics[domain] = {
                'count': len(domain_results),
                'inform_rate': inform_rate,
                'success_rate': success_rate,
                'book_rate': book_rate,
                'jga': sum(jga_scores) / len(jga_scores) if jga_scores else 0,
                'slot_f1': sum(slot_f1_scores) / len(slot_f1_scores) if slot_f1_scores else 0,
                'combined_score': combined_score,
            }
        else:
            domain_metrics[domain] = {
                'count': 0,
                'inform_rate': 0,
                'success_rate': 0,
                'book_rate': None,
                'jga': 0,
                'slot_f1': 0,
                'combined_score': 0,
            }
    
    return domain_metrics


def calculate_dialogue_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算对话级别的指标
    对话级别：一个对话中所有领域都成功才算成功
    
    注意：Dialogue Level 指标通常会低于 Domain Level 平均指标，
    因为它要求对话中的所有领域都成功，这是一个更严格的标准。
    这反映了系统完成整个多领域对话任务的能力。
    
    Returns:
        {
            'dialogue_inform_rate': float,
            'dialogue_success_rate': float,
            'dialogue_jga': float,
            'dialogue_combined_score': float,
            'avg_function_calls': float,
            'total_function_calls': int
        }
    """
    dialogue_inform = []
    dialogue_success = []
    dialogue_book = []
    dialogue_jga = []
    dialogue_slot_f1 = []
    function_calls = []
    
    for result in evaluation_results:
        # 数据结构: result['evaluation']['domains']
        evaluation = result.get('evaluation', {})
        if 'error' in result or 'domains' not in evaluation:
            continue
        
        # 收集工具调用次数
        if 'num_function_calls' in evaluation:
            function_calls.append(evaluation['num_function_calls'])
        
        # 收集该对话所有领域的指标
        domain_informs = []
        domain_successes = []
        domain_books = []
        domain_jgas = []
        domain_slot_f1s = []
        
        for domain, domain_data in evaluation['domains'].items():
            if 'error' not in domain_data:
                if domain_data.get('inform_complete') is not None:
                    domain_informs.append(domain_data['inform_complete'])
                if domain_data.get('success_complete') is not None:
                    domain_successes.append(domain_data['success_complete'])
                if domain_data.get('book_complete') is not None:
                    domain_books.append(domain_data['book_complete'])
                
                # JGA 和 Slot-F1
                if 'inform_slot_values' in domain_data:
                    goal = result.get('goal', {}).get(domain, {})
                    gt_slots = goal.get('info', {})
                    pred_slots = domain_data['inform_slot_values']
                    
                    # JGA
                    jga = calculate_jga(pred_slots, gt_slots)
                    domain_jgas.append(jga)
                    
                    # Slot-F1
                    slot_f1_result = calculate_slot_f1(pred_slots, gt_slots)
                    domain_slot_f1s.append(slot_f1_result['f1'])
        
        # 对话级别：所有领域都要成功（参考 TD-Eval 第 253 行）
        # 注意：只统计有值的指标（None 不参与）
        if domain_informs:
            dialogue_inform.append(int(all(domain_informs)))
        
        if domain_successes:
            dialogue_success.append(int(all(domain_successes)))
        
        # Book 可能为 None，只有有 book 的才算
        valid_books = [b for b in domain_books if b is not None]
        if valid_books:
            dialogue_book.append(int(all(valid_books)))
        
        # JGA 和 Slot-F1: 连续指标，取平均值
        # 对话级别 = 该对话所有领域的平均值
        if domain_jgas:
            dialogue_jga.append(sum(domain_jgas) / len(domain_jgas))
        
        if domain_slot_f1s:
            dialogue_slot_f1.append(sum(domain_slot_f1s) / len(domain_slot_f1s))
    
    inform_rate = sum(dialogue_inform) / len(dialogue_inform) if dialogue_inform else 0
    success_rate = sum(dialogue_success) / len(dialogue_success) if dialogue_success else 0
    book_rate = sum(dialogue_book) / len(dialogue_book) if dialogue_book else None
    jga_rate = sum(dialogue_jga) / len(dialogue_jga) if dialogue_jga else 0
    slot_f1_rate = sum(dialogue_slot_f1) / len(dialogue_slot_f1) if dialogue_slot_f1 else 0
    
    # Calculate dialogue combined score using the formula
    dialogue_combined = calc_combine_score(
        inform=inform_rate if dialogue_inform else None,
        success=success_rate if dialogue_success else None,
        book=book_rate
    )
    
    return {
        'dialogue_count': len(evaluation_results),
        'dialogue_inform_rate': inform_rate,
        'dialogue_success_rate': success_rate,
        'dialogue_book_rate': book_rate,
        'dialogue_jga': jga_rate,
        'dialogue_slot_f1': slot_f1_rate,
        'avg_function_calls': sum(function_calls) / len(function_calls) if function_calls else 0,
        'total_function_calls': sum(function_calls) if function_calls else 0,
        'dialogue_combined_score': dialogue_combined
    }


def generate_comprehensive_report(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成全面的评估报告
    
    Returns:
        {
            'overall': {...},
            'domain_level': {...},
            'dialogue_level': {...},
            'detailed_stats': {...}
        }
    """
    # Domain Level Metrics
    domain_metrics = calculate_domain_metrics(evaluation_results)
    
    # Dialogue Level Metrics  
    dialogue_metrics = calculate_dialogue_metrics(evaluation_results)
    
    # Overall Metrics (跨领域加权平均)
    # 使用加权平均（按各领域对话数加权），而不是简单平均
    total_dialogues = sum(m['count'] for m in domain_metrics.values() if m['count'] > 0)
    
    weighted_inform = 0
    weighted_success = 0
    weighted_book = 0
    weighted_jga = 0
    weighted_slot_f1 = 0
    weighted_combined = 0
    book_dialogue_count = 0
    
    for domain, metrics in domain_metrics.items():
        if metrics['count'] > 0:
            count = metrics['count']
            weight = count / total_dialogues if total_dialogues > 0 else 0
            
            weighted_inform += metrics['inform_rate'] * weight
            weighted_success += metrics['success_rate'] * weight
            weighted_jga += metrics['jga'] * weight
            weighted_slot_f1 += metrics['slot_f1'] * weight
            weighted_combined += metrics['combined_score'] * weight
            
            # Book 单独统计（只统计有 book 的领域）
            if metrics['book_rate'] is not None:
                # 需要知道这个领域有多少对话有 book
                # 简化：用 book_rate * count 估算
                weighted_book += metrics['book_rate'] * weight
                book_dialogue_count += count
    
    overall_inform = weighted_inform
    overall_success = weighted_success
    overall_book = weighted_book if book_dialogue_count > 0 else None
    
    overall_metrics = {
        'inform_rate': overall_inform,
        'success_rate': overall_success,
        'book_rate': overall_book,
        'jga': weighted_jga,
        'slot_f1': weighted_slot_f1,
        'combined_score': weighted_combined,
    }
    
    return {
        'overall': overall_metrics,
        'domain_level': domain_metrics,
        'dialogue_level': dialogue_metrics,
        'summary': {
            'total_dialogues': len(evaluation_results),
            'total_domains_evaluated': sum(m['count'] for m in domain_metrics.values()),
            'valid_dialogues': len([r for r in evaluation_results if 'error' not in r])
        }
    }


def print_comprehensive_report(report: Dict[str, Any]):
    """打印详细的评估报告"""
    print("\n" + "=" * 80)
    print("📊 综合评估报告")
    print("=" * 80)
    
    # 统一表格展示
    print("\n📊 评估指标总览:")
    
    # 表头
    header = f"{'Level/Domain':<14} {'Count':>6} {'Inform':>8} {'Success':>8} {'JGA':>8} {'Slot-F1':>8} {'Combined':>8}"
    print(header)
    print("═" * len(header))
    
    # Overall行 (Domain 加权平均)
    overall = report['overall']
    total_count = sum(m['count'] for m in report['domain_level'].values())
    overall_row = (f"{'Overall':<14} "
                   f"{total_count:>6} "
                   f"{overall['inform_rate']:>7.2%} "
                   f"{overall['success_rate']:>7.2%} "
                   f"{overall['jga']:>7.2%} "
                   f"{overall['slot_f1']:>8.4f} "
                   f"{overall['combined_score']:>7.2%}")
    print(overall_row)
    
    # Dialogue Level (所有领域都成功才算成功)
    dialogue = report['dialogue_level']
    dialogue_row = (f"{'Dialogue':<14} "
                    f"{dialogue['dialogue_count']:>6} "
                    f"{dialogue['dialogue_inform_rate']:>7.2%} "
                    f"{dialogue['dialogue_success_rate']:>7.2%} "
                    f"{dialogue['dialogue_jga']:>7.2%} "
                    f"{dialogue['dialogue_slot_f1']:>8.4f} "
                    f"{dialogue['dialogue_combined_score']:>7.2%}")
    print(dialogue_row)
    print("─" * len(header))
    
    # 各领域数据
    domains_order = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
    for domain in domains_order:
        if domain in report['domain_level']:
            stats = report['domain_level'][domain]
            if stats['count'] > 0:
                row = (f"{domain.capitalize():<14} "
                       f"{stats['count']:>6} "
                       f"{stats['inform_rate']:>7.2%} "
                       f"{stats['success_rate']:>7.2%} "
                       f"{stats['jga']:>7.2%} "
                       f"{stats['slot_f1']:>8.4f} "
                       f"{stats['combined_score']:>7.2%}")
                print(row)
    
    # Book 作为补充信息
    print(f"\n📋 Book Rate (Success 的子指标，仅统计有预订需求的对话):")
    for domain in domains_order:
        if domain in report['domain_level']:
            stats = report['domain_level'][domain]
            if stats.get('book_rate') is not None:
                print(f"  {domain.capitalize():<12s}: {stats['book_rate']:>6.2%}")
    
    # 说明
    print(f"\n💡 指标说明:")
    print(f"  • Overall: 各领域加权平均（按对话数加权）")
    print(f"  • Dialogue: 所有领域都成功的对话占比（更严格）")
    print(f"  • Success 已包含 Book（如需预订，Book 必须成功）")
    print(f"  • Combined = 0.5 * Inform + 0.5 * Success")
    
    # Tool Calls统计
    print("\n🛠️  Tool Call Statistics:")
    print(f"  • Total Calls:   {dialogue['total_function_calls']}")
    print(f"  • Avg per Dialog: {dialogue['avg_function_calls']:.2f}")
    
    # Token Statistics (如果有的话)
    if 'inference_tokens' in dialogue and 'eval_tokens' in dialogue:
        print("\n📊 Token Usage Statistics:")
        print("\n  Inference Stage:")
        inf_tokens = dialogue['inference_tokens']
        print(f"    • Prompt Tokens:     {inf_tokens['prompt_tokens']:,}")
        print(f"    • Completion Tokens: {inf_tokens['completion_tokens']:,}")
        print(f"    • Total Tokens:      {inf_tokens['total_tokens']:,}")
        
        print("\n  Evaluation Stage:")
        eval_tokens = dialogue['eval_tokens']
        print(f"    • Prompt Tokens:     {eval_tokens['prompt_tokens']:,}")
        print(f"    • Completion Tokens: {eval_tokens['completion_tokens']:,}")
        print(f"    • Total Tokens:      {eval_tokens['total_tokens']:,}")
        
        print("\n  Overall:")
        total_tokens = dialogue['total_tokens']
        print(f"    • Prompt Tokens:     {total_tokens['prompt_tokens']:,}")
        print(f"    • Completion Tokens: {total_tokens['completion_tokens']:,}")
        print(f"    • Total Tokens:      {total_tokens['total_tokens']:,}")
        if dialogue.get('total_dialogues', 0) > 0:
            print(f"    • Avg per Dialog:    {total_tokens['total_tokens'] / dialogue['total_dialogues']:,.0f}")
    
    print("\n" + "=" * 80)

