"""
MultiWOZ Pipeline

完整的 MultiWOZ benchmark 运行流程：
1. 加载数据
2. 初始化 Agent 和 User Simulator
3. 运行对话推理
4. 评估结果
5. 保存输出
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import time
import json

from taskdialogue.core.base.pipeline import BasePipeline
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult
from taskdialogue.core.schemas.dialogue import DialogueTurn
from taskdialogue.core.schemas.message import Message
from taskdialogue.core.models.factory import create_model_from_config
from taskdialogue.core.utils.logger import get_logger
from taskdialogue.core.utils.parallel import parallel_process_with_saving

from taskdialogue.benchmarks.multiwoz.agents import (
    MultiWOZFunctionAgent,
    CustomFormatAgent,
    MultiWOZUserSimulator
)
from taskdialogue.benchmarks.multiwoz.evaluators import MultiWOZEvaluator
from taskdialogue.benchmarks.multiwoz.data import load_multiwoz_data
from taskdialogue.benchmarks.multiwoz.evaluators.aggregator import MetricsAggregator
from taskdialogue.benchmarks.multiwoz.evaluators.formatter import ResultFormatter
from taskdialogue.benchmarks.multiwoz.constants import DOMAINS, DOMAIN_NAMES

logger = get_logger(__name__)


def _run_single_dialogue_worker(dialogue_data: Dict[str, Any], idx: int, config: Dict[str, Any], max_turns: int) -> Tuple[int, InferenceResult]:
    """
    单个对话推理（多进程/多线程 worker 函数）
    
    每个 worker 创建自己的 agent 和 user simulator 实例
    
    Args:
        dialogue_data: 对话数据
        idx: 索引（用于排序）
        config: 配置字典
        max_turns: 最大对话轮数
        
    Returns:
        (idx, InferenceResult)
    """
    dialogue_id = dialogue_data.get("dialogue_idx", f"dialogue_{idx}")
    start_time = time.time()
    
    try:
        agent_model = create_model_from_config(config, "agent")
        user_model = create_model_from_config(config, "user")
        
        # 创建 Agent
        agent_config = config.get("agent", {})
        agent_type = agent_config.get("type", "function_calling")
        db_path = config.get("database", {}).get("path", "data/db/multiwoz.db")
        book_db_path = config.get("database", {}).get("booking_path", "data/db/multiwoz_book.db")
        
        # Convert to absolute paths if relative
        from pathlib import Path
        import os
        if not os.path.isabs(db_path):
            # Get project root: pipeline.py -> multiwoz -> benchmarks -> taskdialogue -> TaskDialogue
            project_root = Path(__file__).parent.parent.parent.parent
            db_path = str(project_root / db_path)
        if not os.path.isabs(book_db_path):
            project_root = Path(__file__).parent.parent.parent.parent
            book_db_path = str(project_root / book_db_path)
        
        if agent_type == "custom_format":
            agent = CustomFormatAgent(agent_model, agent_config, database_path=db_path, booking_database_path=book_db_path)
        else:
            agent = MultiWOZFunctionAgent(agent_model, agent_config, database_path=db_path, booking_database_path=book_db_path)
        
        # 创建 User Simulator
        user_config = config.get("model", {}).get("user", {})
        user_simulator = MultiWOZUserSimulator(user_model, user_config)
        
        # 重置
        agent.reset()
        user_simulator.reset(dialogue_data)
        
        # 对话历史
        dialogue_turns = []
        
        # 第一句用户输入
        user_input = user_simulator.generate_response(None)
        
        # 打印对话开始
        from taskdialogue.core.constants import Colors
        from taskdialogue.core.utils.logger import logger
        logger.info('')
        logger.info("=" * 80)
        logger.info(f"Dialogue: {dialogue_id}")
        logger.info("=" * 80)
        
        # 对话循环
        for turn_idx in range(1, max_turns + 1):
            # 打印用户输入
            logger.info('')
            logger.info(f"{Colors.BLUE}[Turn {turn_idx}] User:{Colors.RESET} {user_input}")
            
            # Agent 回复
            agent_output = agent.generate_response(user_input)
            
            # 打印 Agent 回复
            logger.info(f"{Colors.YELLOW}Agent:{Colors.RESET} {agent_output}")
            
            # 记录本轮对话
            turn = DialogueTurn(
                turn_id=turn_idx,
                user_message=Message(role="user", content=user_input),
                agent_message=Message(role="assistant", content=agent_output)
            )
            dialogue_turns.append(turn)
            
            # 用户回复
            user_input = user_simulator.generate_response(agent_output)
            
            # 检查用户是否结束
            if user_simulator.is_dialogue_complete():
                logger.info(f"{Colors.GREEN}✓ Dialogue complete (user ended){Colors.RESET}")
                break
        
        # 计算统计信息
        agent_stats = agent.get_statistics()
        total_tokens = agent_stats["total_tokens"]
        function_calls = getattr(agent, 'function_call_count', 0)
        
        # 获取 agent 的原始消息历史（包含完整的 tool_calls 和 tool 消息）
        raw_messages = agent.get_dialogue_history()
        
        # 构建结果
        result = InferenceResult(
            benchmark="multiwoz",
            task_id=dialogue_id,
            dialogue_id=dialogue_id,
            dialogue_history=dialogue_turns,
            raw_messages=raw_messages,  # 保存原始消息
            model_config_data={
                "agent": {
                    "provider": agent_model.provider,
                    "model_name": agent_model.model_name,
                },
                "user": {
                    "provider": user_model.provider,
                    "model_name": user_model.model_name,
                }
            },
            num_turns=len(dialogue_turns),
            total_tokens=total_tokens,
            inference_time=time.time() - start_time,
            benchmark_specific={
                "goals": dialogue_data.get("goal", {}),
                "domains": list(dialogue_data.get("goal", {}).keys()),
                "function_calls": function_calls
            }
        )
        
        return idx, result
        
    except Exception as e:
        # 打印详细错误信息以便调试
        import traceback
        from taskdialogue.core.utils.logger import logger
        logger.error(f"Error in dialogue {dialogue_id}: {e}")
        logger.error(traceback.format_exc())
        
        # 创建错误结果
        error_result = InferenceResult(
            benchmark="multiwoz",
            task_id=dialogue_id,
            dialogue_id=dialogue_id,
            success=False,
            error_message=str(e)
        )
        return idx, error_result


def _evaluate_single_dialogue_worker_wrapper(item: Tuple[InferenceResult, Dict], idx: int, eval_config: Dict[str, Any]) -> Tuple[int, EvaluationResult]:
    """Worker wrapper for pickle compatibility"""
    inf_result, ground_truth = item
    return _evaluate_single_dialogue_worker(inf_result, idx, eval_config, ground_truth)


def _evaluate_single_dialogue_worker(inf_result: InferenceResult, idx: int, eval_config: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[int, EvaluationResult]:
    """
    单个对话评估（多进程 worker 函数）
    
    Args:
        inf_result: 推理结果
        idx: 索引
        eval_config: 评估配置
        ground_truth: Ground truth
        
    Returns:
        (idx, EvaluationResult)
    """
    try:
        evaluator = MultiWOZEvaluator(eval_config)
        eval_result = evaluator.evaluate(inf_result, ground_truth)
        return idx, eval_result
    except Exception as e:
        # 创建错误结果
        error_result = EvaluationResult(
            benchmark="multiwoz",
            task_id=inf_result.dialogue_id,
            dialogue_id=inf_result.dialogue_id,
            overall_score=0.0,
            metrics=[],
            success=False,
            error_message=str(e)
        )
        return idx, error_result


class MultiWOZPipeline(BasePipeline):
    """MultiWOZ 完整运行流程
    
    Example:
        >>> config = load_config("configs/multiwoz/default.yaml")
        >>> pipeline = MultiWOZPipeline(config)
        >>> inf_results, eval_results = pipeline.run_full_pipeline()
    """
    
    # 常量定义
    # 领域常量（从 constants 导入）
    DOMAINS = DOMAINS
    DOMAIN_NAMES = DOMAIN_NAMES
    
    @staticmethod
    def _filter_serializable(obj):
        """过滤出可JSON序列化的数据"""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: MultiWOZPipeline._filter_serializable(v) for k, v in obj.items() 
                   if not k.startswith('_') and not callable(v)}
        elif isinstance(obj, (list, tuple)):
            return [MultiWOZPipeline._filter_serializable(item) for item in obj]
        else:
            return None
    
    @staticmethod
    def _avg_metric(metrics_list, key):
        """从 metrics 列表中提取指定键的平均值
        
        支持两种格式：
        1. EvaluationMetric 对象列表的列表（对话级别）
        2. 字典列表（域级别）
        """
        values = []
        for item in metrics_list:
            if isinstance(item, dict):
                # 域级别的 metrics（字典格式）
                if key in item and item[key] is not None:
                    values.append(item[key])
            else:
                # 对话级别的 metrics（EvaluationMetric 对象列表）
                for metric in item:
                    if metric.name == key and metric.value is not None:
                        values.append(metric.value)
                        break
        return sum(values) / len(values) if values else 0.0
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        logger.info("Initializing MultiWOZ Pipeline...")
        
        # 加载数据
        data_config = config.get("data", {})
        domains_config = config.get("domains", {})
        
        self.dataset = load_multiwoz_data(
            data_path=data_config.get("path", "data/multiwoz/data.json"),
            split=data_config.get("split", "test"),
            num_samples=data_config.get("num_tasks"),
            remove_police_hospital=data_config.get("remove_police_hospital", True),
            enabled_domains=domains_config.get("enabled")
        )
        logger.info(f"Loaded {len(self.dataset)} dialogues")
        
        # 初始化模型
        self.agent_model = create_model_from_config(config, "agent")
        self.user_model = create_model_from_config(config, "user")
        logger.info(f"Agent model: {self.agent_model.model_name}")
        logger.info(f"User model: {self.user_model.model_name}")
        
        # 初始化 Agent（根据类型选择）
        agent_config = config.get("agent", {})
        agent_type = agent_config.get("type", "function_calling")
        
        # 获取数据库路径（用于工具初始化）
        db_path = config.get("database", {}).get("path", "data/db/multiwoz.db")
        
        if agent_type == "custom_format":
            logger.info("Using Custom Format Agent")
            self.agent = CustomFormatAgent(
                self.agent_model, 
                agent_config,
                database_path=db_path  # 传递数据库路径
            )
        else:
            logger.info("Using Function Calling Agent")
            self.agent = MultiWOZFunctionAgent(
                self.agent_model, 
                agent_config,
                database_path=db_path  # 传递数据库路径
            )
        
        user_config = config.get("user", {})
        self.user_simulator = MultiWOZUserSimulator(self.user_model, user_config)
        
        # 初始化评估器（传入完整配置以获取 API key）
        eval_config = config.get("evaluation", {})
        eval_config["_full_config"] = config  # 传入完整配置
        self.evaluator = MultiWOZEvaluator(eval_config)
        
        # 运行配置
        self.max_turns = config.get("agent", {}).get("max_turns", 30)
        
        logger.info("MultiWOZ Pipeline initialized successfully")
    
    def run_inference(
        self, 
        task_ids: Optional[List[str]] = None
    ) -> List[InferenceResult]:
        """运行推理（支持多进程/多线程）
        
        对每个对话：
        1. 重置 agent 和 user simulator
        2. 进行多轮对话直到完成
        3. 记录对话历史
        4. 批量保存结果
        """
        logger.info("="*80)
        logger.info("Starting Inference Phase")
        logger.info("="*80)
        
        # 确定要运行的对话
        dialogues_to_run = self._get_dialogues_to_run(task_ids)
        
        # 获取并发配置
        inf_config = self.config.get("inference", {})
        num_workers = inf_config.get("num_workers", 1)
        save_batch_size = inf_config.get("save_batch_size", 32)
        
        # 准备输出路径（参考 backup 格式）
        output_dir = Path(self.config.get("output", {}).get("inference_dir", "results/multiwoz/inference"))
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成包含模型和配置信息的文件名
        dataset_name = self.config.get("data", {}).get("dataset_name", "multiwoz24")
        model_name = self.agent_model.model_name.replace('/', '_').replace('-', '_')
        split = self.config.get("data", {}).get("split", "test")
        num_samples = len(dialogues_to_run)
        
        filename = f"predictions_{model_name}_{split}_{num_samples}samples_{timestamp}.json"
        output_file = output_dir / filename
        
        logger.info(f"Output file: {output_file}")
        logger.info(f"Num workers: {num_workers}")
        logger.info(f"Save batch size: {save_batch_size}")
        
        # 改成收集所有结果，最后一次性保存为大 JSON
        all_dialogue_data = []  # 收集所有对话数据
        
        def save_func(results: List[InferenceResult]):
            """收集推理结果（批量收集，最后统一保存）"""
            def convert_to_reference_format(result: InferenceResult) -> dict:
                """转换为优化后的格式（去除冗余，保留有效信息）"""
                
                # 使用 raw_messages（agent 的原始消息历史，包含完整的 tool_calls）
                messages = []
                
                # 添加系统消息（如果有）
                if hasattr(self.agent, 'system_prompt') and self.agent.system_prompt:
                    messages.append({
                        "role": "system",
                        "content": self.agent.system_prompt
                    })
                
                # 使用 raw_messages 而不是从 dialogue_history 重建
                if result.raw_messages:
                    for msg in result.raw_messages:
                        messages.append(msg.to_openai_format())
                
                # 统计信息
                stats = {
                    "num_turns": result.num_turns,
                    "total_tokens": result.total_tokens,
                    "inference_time": result.inference_time,
                    "function_calls": result.benchmark_specific.get("function_calls", 0) if result.benchmark_specific else 0
                }
                
                # 最终数据结构（保留有效信息，去除冗余）
                data = {
                    "dialogue_id": result.dialogue_id,
                    "messages": messages,  # 完整的 OpenAI 格式（包含 tool_calls）
                    "stats": stats,  # 统计信息
                }
                
                # 保留 goal（评估需要）
                if result.benchmark_specific and result.benchmark_specific.get("goals"):
                    data["goal"] = result.benchmark_specific["goals"]
                
                # 保留其他 benchmark 特定信息
                if result.benchmark_specific:
                    for key in ["original_data", "extra_info"]:
                        if key in result.benchmark_specific and result.benchmark_specific[key]:
                            data[key] = result.benchmark_specific[key]
                
                return data
            
            # 批量收集数据（不立即写入文件）
            for result in results:
                data = convert_to_reference_format(result)
                all_dialogue_data.append(data)
        
        # 定义处理函数（带配置）
        # 将 Config 对象转换为 dict 以便在多线程中传递
        config_dict = self.config if isinstance(self.config, dict) else self.config.to_dict()
        def process_func(dialogue_data: Dict[str, Any], idx: int) -> Tuple[int, InferenceResult]:
            return _run_single_dialogue_worker(dialogue_data, idx, config_dict, self.max_turns)
        
        # 并行处理
        show_progress = self.config.get("logging", {}).get("show_progress", True)
        inference_results = parallel_process_with_saving(
            items=dialogues_to_run,
            process_func=process_func,
            save_func=save_func,
            num_workers=num_workers,
            save_batch_size=save_batch_size,
            use_processes=False,  # 使用线程池（API 调用 I/O 密集）
            desc="Running inference",
            show_progress=show_progress,
            final_save=False  # 禁用 parallel 中的 final_save，改为手动保存大 JSON
        )
        
        # 所有对话处理完成后，一次性保存为大 JSON 文件
        if inference_results:
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            # 过滤并序列化配置
            config_serializable = self._filter_serializable(self.config)
            
            # 构造最终的大 JSON
            final_json = {
                "benchmark": "multiwoz",
                "timestamp": timestamp,
                "model_name": self.agent_model.model_name,
                "provider": self.agent_model.provider,
                "split": split,
                "num_dialogues": len(all_dialogue_data),
                "config": config_serializable,  # 保存完整配置
                "tools": getattr(self.agent, 'tools', None),  # 全局工具定义
                "system_prompt": getattr(self.agent, 'system_prompt', None),  # 全局 system prompt
                "dialogues": all_dialogue_data  # 所有对话数据
            }
            
            # 一次性写入
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            logger.info(f"Saved {len(all_dialogue_data)} dialogues to: {output_file}")
        
        # 打印统计信息
        logger.info(f"\n{'='*80}")
        logger.info(f"Inference Phase Complete: {len(inference_results)} dialogues")
        logger.info(f"Results saved to: {output_file}")
        
        if inference_results:
            # 计算统计信息
            total_turns = sum(r.num_turns for r in inference_results if r.success)
            total_tokens = sum(r.total_tokens or 0 for r in inference_results if r.success)
            successful_dialogues = sum(1 for r in inference_results if r.success)
            
            # 统计工具调用（从 benchmark_specific 中提取）
            total_function_calls = 0
            for r in inference_results:
                if r.success and r.benchmark_specific:
                    total_function_calls += r.benchmark_specific.get('function_calls', 0)
            
            logger.info("")
            logger.info("📊 Inference Statistics:")
            logger.info(f"  • Total Dialogues:    {len(inference_results)}")
            logger.info(f"  • Successful:         {successful_dialogues}")
            logger.info(f"  • Failed:             {len(inference_results) - successful_dialogues}")
            logger.info(f"  • Total Turns:        {total_turns}")
            logger.info(f"  • Total Function Calls: {total_function_calls}")
            logger.info(f"  • Total Tokens:       {total_tokens:,}")
            if successful_dialogues > 0:
                logger.info(f"  • Avg Tokens/Dialogue: {total_tokens / successful_dialogues:.0f}")
                logger.info(f"  • Avg Turns/Dialogue:  {total_turns / successful_dialogues:.1f}")
                logger.info(f"  • Avg Calls/Dialogue:  {total_function_calls / successful_dialogues:.1f}")
        
        logger.info(f"{'='*80}\n")
        
        # 缓存结果（供 save_evaluation_results 使用）
        self._last_inference_results = inference_results
        
        return inference_results
    
    def run_evaluation(
        self, 
        inference_results: List[InferenceResult]
    ) -> List[EvaluationResult]:
        """运行评估（支持多进程）"""
        logger.info("\n" + "="*80)
        logger.info("Starting Evaluation Phase")
        logger.info("="*80)
        
        # 获取并发配置【
        eval_config = self.config.get("evaluation", {}).copy()
        # 传入完整配置的深拷贝（确保可序列化）
        import copy
        try:
            eval_config["_full_config"] = copy.deepcopy(self.config)
        except Exception:
            # 如果深拷贝失败，只传递必要的 API 配置
            eval_config["_full_config"] = {
                "models": self.config.get("models", {}),
                "api_keys": self.config.get("api_keys", {})
            }
        num_workers = eval_config.get("num_workers", 1)
        save_batch_size = eval_config.get("save_batch_size", 32)
        
        # 准备输出路径（参考 backup 格式）
        output_dir = Path(self.config.get("output", {}).get("evaluation_dir", "results/multiwoz/evaluation"))
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成包含模型和配置信息的文件名
        dataset_name = self.config.get("data", {}).get("dataset_name", "multiwoz24")
        eval_model_name = eval_config.get("eval_model", "deepseek_chat").replace('/', '_').replace('-', '_')
        split = self.config.get("data", {}).get("split", "test")
        num_samples = len(inference_results)
        
        # 生成文件名基础名（会在 save_evaluation_results 中使用）
        self._eval_output_basename = f"evaluation_{eval_model_name}_{split}_{num_samples}samples_{timestamp}"
        
        logger.info(f"Num workers: {num_workers}")
        
        # 准备评估数据（加载 ground truth）
        eval_items = []
        for inf_result in inference_results:
            dialogue_data = self.dataset.get_by_id(inf_result.dialogue_id)
            ground_truth = {
                "goal": dialogue_data.get("goal", {}) if dialogue_data else {}
            }
            eval_items.append((inf_result, ground_truth))
        
        # 定义处理函数（使用 lambda 避免 pickle 问题）
        from functools import partial
        process_func = partial(_evaluate_single_dialogue_worker_wrapper, eval_config=eval_config)
        
        # 并行处理（不中间保存，最后统一保存为一个大JSON）
        show_progress = self.config.get("logging", {}).get("show_progress", True)
        from taskdialogue.core.utils.parallel import parallel_process
        evaluation_results = parallel_process(
            items=eval_items,
            process_func=process_func,
            num_workers=num_workers,
            use_processes=True,  # 使用进程池（CPU 密集）
            desc="Running evaluation",
            show_progress=show_progress
        )
        
        # 使用新的 aggregator 和 formatter 模块
        if evaluation_results:
            # 1. 聚合指标
            overall_metrics, dialogue_metrics, domain_averages, metrics_by_domain, dialogue_level_metrics = \
                MetricsAggregator.aggregate_metrics(evaluation_results)
            
            # 2. 打印报告
            total_function_calls = sum(
                r.benchmark_specific.get('function_calls', 0) 
                for r in evaluation_results if not r.error_message and r.benchmark_specific
            )
            ResultFormatter.print_evaluation_report(
                overall_metrics, dialogue_metrics, domain_averages, 
                metrics_by_domain, total_function_calls
            )
            
            # 3. 保存结果（存储到实例变量，供 save_evaluation_results 使用）
            self._cached_metrics = {
                'overall': overall_metrics,
                'dialogue': dialogue_metrics,
                'domain': domain_averages,
                'timestamp': timestamp,
                'split': split,
                'num_samples': num_samples
            }
        
        return evaluation_results
    
    def save_evaluation_results(self, results: List[EvaluationResult], output_basename: Optional[str] = None):
        """保存评估结果（覆盖基类方法，使用新格式）"""
        # 获取 inference_results（从缓存或重新加载）
        if not hasattr(self, '_last_inference_results'):
            logger.warning("No cached inference results, loading from latest file...")
            inference_results = self.load_inference_results()
        else:
            inference_results = self._last_inference_results
        
        # 使用缓存的 metrics（如果有）
        if hasattr(self, '_cached_metrics'):
            metrics = self._cached_metrics
        else:
            # 重新计算（如果需要）
            overall_metrics, dialogue_metrics, domain_averages, _, _ = \
                MetricsAggregator.aggregate_metrics(results)
            metrics = {
                'overall': overall_metrics,
                'dialogue': dialogue_metrics,
                'domain': domain_averages,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'split': self.config.get("data", {}).get("split", "test"),
                'num_samples': len(results)
            }
        
        # 文件名
        output_dir = Path(self.config.get("output", {}).get("evaluation_dir", "results/multiwoz/evaluation"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_basename:
            filename = f"{output_basename}.json"
        else:
            model_name = self.agent_model.model_name.replace('/', '_').replace('-', '_')
            filename = f"evaluation_{model_name}_{metrics['split']}_{metrics['num_samples']}samples_{metrics['timestamp']}.json"
        
        output_file = output_dir / filename
        config_serializable = self._filter_serializable(self.config)
        
        # 调用 ResultFormatter 保存
        ResultFormatter.save_evaluation_results(
            inference_results, results,
            metrics['overall'], metrics['dialogue'], metrics['domain'],
            output_file, config_serializable,
            self.agent_model.model_name, self.agent_model.provider,
            metrics['split'], metrics['timestamp'],
            tools=getattr(self.agent, 'tools', None),
            system_prompt=getattr(self.agent, 'system_prompt', None)
        )
    
    def _get_dialogues_to_run(self, task_ids: Optional[List[str]]) -> List[Dict]:
        """获取要运行的对话列表"""
        if task_ids is None:
            return list(self.dataset.data)
        
        dialogues = []
        for task_id in task_ids:
            dialogue = self.dataset.get_by_id(task_id)
            if dialogue:
                dialogues.append(dialogue)
        
        return dialogues
    
    def _get_model_config(self) -> Dict[str, Any]:
        """获取模型配置信息"""
        return {
            "agent": {
                "provider": self.agent_model.provider,
                "model_name": self.agent_model.model_name,
                "temperature": self.agent.temperature,
            },
            "user": {
                "provider": self.user_model.provider,
                "model_name": self.user_model.model_name,
                "temperature": self.user_simulator.temperature,
            }
        }

