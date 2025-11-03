#!/usr/bin/env python3
"""
转换 TaskDialogue 推理结果为 LLaMA-Factory SFT 训练格式

输入: predictions_*.json (TaskDialogue 推理结果)
输出: training_data.jsonl (LLaMA-Factory SFT 格式)

使用方法:
    python scripts/convert_to_llama_factory.py \
    --input results/multiwoz/evaluation/evaluation_deepseek_chat_valid_256samples_20251031_170146.json \
    --output data/llama_factory/multiwoz_perfect_256samples_fixed.jsonl \
    --split train \
    --filter-perfect \
    --tools-as-string
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def convert_dialogue_to_sft_format(
    dialogue: Dict[str, Any],
    tools: List[Dict],
    data_source: str,
    split: str = "train",
    enable_thinking: bool = False,
    tools_as_string: bool = True
) -> Dict[str, Any]:
    """
    转换单个对话为 LLaMA-Factory OpenAI 格式（支持工具调用）
    
    保持原始的 OpenAI 格式，包含 tool_calls 字段
    LLaMA-Factory 的 OpenAIDatasetConverter 会自动处理 tool_calls
    
    Args:
        dialogue: TaskDialogue 对话数据（OpenAI 格式）
        tools: 工具定义列表
        data_source: 数据来源（如 "multiwoz_valid"）
        split: 数据集划分
        enable_thinking: 是否启用思考功能
        tools_as_string: 是否将 tools 转换为 JSON 字符串（默认 True）
        
    Returns:
        LLaMA-Factory OpenAI 格式的数据
    """
    # 直接提取 messages（已经是 OpenAI 格式）
    messages = dialogue.get('messages', [])
    
    # 提取 system 消息并清理 None 值字段
    system_prompt = None
    filtered_messages = []
    
    for msg in messages:
        if msg.get('role') == 'system':
            system_prompt = msg.get('content', '')
        else:
            # 清理消息：移除值为 None 的字段，但保留必要的空数组
            cleaned_msg = {}
            for key, value in msg.items():
                if value is not None:
                    cleaned_msg[key] = value
            
            # 确保 assistant 消息有 tool_calls 字段（避免 LLaMA-Factory 的 None 检查 bug）
            # 如果没有 tool_calls，添加空数组
            if cleaned_msg.get('role') == 'assistant' and 'tool_calls' not in cleaned_msg:
                cleaned_msg['tool_calls'] = []
            
            filtered_messages.append(cleaned_msg)

    
    # 处理 tools 字段（LLaMA-Factory 会自动转换，但我们提前转换更安全）
    # 始终转换为字符串格式
    if isinstance(tools, (list, dict)):
        tools_value = json.dumps(tools, ensure_ascii=False)
    else:
        tools_value = tools
    
    # 构建 OpenAI 格式的数据
    result = {
        "messages": filtered_messages  # 保持原始 OpenAI 格式（包含 tool_calls）
    }
    
    # 添加 system 提示词（如果有）
    if system_prompt:
        result["system"] = system_prompt
    
    # 添加 tools（如果有）
    if tools_value:
        result["tools"] = tools_value
    
    return result


def convert_file(
    input_file: str,
    output_file: str,
    split: str = "train",
    enable_thinking: bool = False,
    max_samples: int = None,
    include_evaluation: bool = False,
    filter_perfect: bool = False,
    min_inform: float = None,
    min_success: float = None,
    min_combined: float = None,
    tools_as_string: bool = False
):
    """
    转换整个文件
    
    Args:
        input_file: 输入文件路径（predictions_*.json 或 evaluation_*.json）
        output_file: 输出文件路径（JSONL 格式）
        split: 数据集划分（train/valid/test）
        enable_thinking: 是否启用思考功能
        max_samples: 最大样本数（None 表示全部）
        include_evaluation: 是否包含评估分数（用于 DPO/RL）
        filter_perfect: 是否只保留完美样本（dialogue-level success=1.0 且 inform=1.0）
        min_inform: 最小 inform 分数（过滤阈值）
        min_success: 最小 success 分数（过滤阈值）
        min_combined: 最小 combined_score（过滤阈值）
        tools_as_string: 是否将 tools 转换为 JSON 字符串（默认 False，LLaMA-Factory 会自动转换）
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取输入文件
    print(f"读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取信息
    benchmark = data.get('benchmark', 'unknown')
    model_name = data.get('model_name', 'unknown')
    original_split = data.get('split', 'unknown')
    tools = data.get('tools', [])
    dialogues = data.get('dialogues', [])
    
    print(f"  Benchmark: {benchmark}")
    print(f"  Model: {model_name}")
    print(f"  Split: {original_split}")
    print(f"  Tools: {len(tools)} 个")
    print(f"  Dialogues: {len(dialogues)} 个")
    
    # 数据来源标识
    data_source = f"{benchmark}_{original_split}"
    
    # 转换对话
    converted_count = 0
    skipped_count = 0
    filtered_by_quality = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, dialogue in enumerate(dialogues):
            # 检查是否达到最大样本数
            if max_samples and converted_count >= max_samples:
                break
            
            # 跳过没有 messages 的对话
            if not dialogue.get('messages'):
                skipped_count += 1
                continue
            
            # 质量过滤（基于评估结果）
            if 'evaluation' in dialogue and dialogue['evaluation']:
                evaluation = dialogue['evaluation']
                
                # 过滤完美样本
                if filter_perfect:
                    dialogue_inform = evaluation.get('inform', 0)
                    dialogue_success = evaluation.get('success', 0)
                    
                    # 要求对话级别的 inform 和 success 都是 1.0
                    if dialogue_inform != 1.0 or dialogue_success != 1.0:
                        filtered_by_quality += 1
                        continue
                
                # 自定义阈值过滤
                if min_inform is not None and evaluation.get('inform', 0) < min_inform:
                    filtered_by_quality += 1
                    continue
                
                if min_success is not None and evaluation.get('success', 0) < min_success:
                    filtered_by_quality += 1
                    continue
                
                if min_combined is not None and evaluation.get('combined_score', 0) < min_combined:
                    filtered_by_quality += 1
                    continue
            
            # 转换格式
            sft_data = convert_dialogue_to_sft_format(
                dialogue, tools, data_source, split, enable_thinking, tools_as_string
            )
            
            # 如果是评估文件且需要包含评估分数
            if include_evaluation and 'evaluation' in dialogue:
                evaluation = dialogue['evaluation']
                if evaluation:
                    # 添加评估分数到 metadata（用于 DPO/RL）
                    sft_data['metadata']['evaluation'] = {
                        'inform': evaluation.get('inform'),
                        'success': evaluation.get('success'),
                        'combined_score': evaluation.get('combined_score')
                    }
            
            # 写入文件（JSONL 格式）
            f.write(json.dumps(sft_data, ensure_ascii=False) + '\n')
            converted_count += 1
            
            # 进度显示
            if (idx + 1) % 50 == 0:
                print(f"  已处理: {idx + 1}/{len(dialogues)} 个对话...")
    
    print(f"\n✅ 转换完成！")
    print(f"  输出文件: {output_path}")
    print(f"  转换成功: {converted_count} 个")
    print(f"  跳过（无 messages）: {skipped_count} 个")
    if filtered_by_quality > 0:
        print(f"  过滤（质量不符）: {filtered_by_quality} 个")
    print(f"  文件格式: JSONL (每行一个 JSON 对象)")
    
    # 统计信息
    if filter_perfect or min_inform or min_success or min_combined:
        total_candidates = len(dialogues) - skipped_count
        if total_candidates > 0:
            print(f"\n📊 质量过滤统计:")
            print(f"  候选样本: {total_candidates}")
            print(f"  通过过滤: {converted_count} ({converted_count/total_candidates*100:.1f}%)")
            print(f"  被过滤: {filtered_by_quality} ({filtered_by_quality/total_candidates*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='转换 TaskDialogue 推理结果为 LLaMA-Factory SFT 训练格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 1. 转换推理结果（基础用法）
    python scripts/convert_to_llama_factory.py \\
        --input results/multiwoz/inference/predictions_deepseek_chat_valid_256samples.json \\
        --output data/llama_factory/multiwoz_train.jsonl \\
        --split train
    
    # 2. 只保留完美样本（inform=1.0 且 success=1.0）⭐
    python scripts/convert_to_llama_factory.py \\
        --input results/multiwoz/evaluation/evaluation_deepseek_chat_valid_256samples.json \\
        --output data/llama_factory/multiwoz_perfect.jsonl \\
        --split train \\
        --filter-perfect
    
    # 3. 自定义质量阈值过滤
    python scripts/convert_to_llama_factory.py \\
        --input results/multiwoz/evaluation/evaluation_*.json \\
        --output data/llama_factory/multiwoz_high_quality.jsonl \\
        --split train \\
        --min-inform 0.8 \\
        --min-success 0.8
    
    # 4. 包含评估分数（用于 DPO/RL 训练）
    python scripts/convert_to_llama_factory.py \\
        --input results/multiwoz/evaluation/evaluation_*.json \\
        --output data/llama_factory/multiwoz_with_scores.jsonl \\
        --split train \\
        --include-evaluation
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入文件路径（predictions_*.json 或 evaluation_*.json）'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出文件路径（JSONL 格式）'
    )
    parser.add_argument(
        '--split',
        default='train',
        choices=['train', 'valid', 'test'],
        help='数据集划分（默认: train）'
    )
    parser.add_argument(
        '--enable-thinking',
        action='store_true',
        help='启用思考功能（默认: false）'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大样本数（默认: 全部）'
    )
    parser.add_argument(
        '--include-evaluation',
        action='store_true',
        help='包含评估分数到 metadata（用于 DPO/RL，仅对 evaluation_*.json 有效）'
    )
    parser.add_argument(
        '--filter-perfect',
        action='store_true',
        help='只保留完美样本（dialogue-level inform=1.0 且 success=1.0，需要 evaluation 文件）'
    )
    parser.add_argument(
        '--min-inform',
        type=float,
        default=None,
        help='最小 inform 分数（例如: 0.8）'
    )
    parser.add_argument(
        '--min-success',
        type=float,
        default=None,
        help='最小 success 分数（例如: 0.8）'
    )
    parser.add_argument(
        '--min-combined',
        type=float,
        default=None,
        help='最小 combined_score（例如: 0.8）'
    )
    parser.add_argument(
        '--tools-as-string',
        action='store_true',
        help='将 tools 字段转换为 JSON 字符串（默认: false，LLaMA-Factory 会自动转换）'
    )
    
    args = parser.parse_args()
    
    # 执行转换
    convert_file(
        input_file=args.input,
        output_file=args.output,
        split=args.split,
        enable_thinking=args.enable_thinking,
        max_samples=args.max_samples,
        include_evaluation=args.include_evaluation,
        filter_perfect=args.filter_perfect,
        min_inform=args.min_inform,
        min_success=args.min_success,
        min_combined=args.min_combined,
        tools_as_string=args.tools_as_string
    )


if __name__ == '__main__':
    main()

