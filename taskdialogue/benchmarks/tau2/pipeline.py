"""
Tau2 Pipeline

Tau2 benchmark 的完整运行流程，适配器连接到核心系统。

注意：这是一个适配层，底层使用 taskdialogue.tau2_bench 的现有实现，
但提供统一的 BasePipeline 接口。
参考 MultiWOZ 的实现，支持多进程推理和评估。
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import json

from taskdialogue.core.base.pipeline import BasePipeline
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult
from taskdialogue.core.utils.logger import get_logger
from taskdialogue.core.utils.parallel import parallel_process, parallel_process_with_saving

# 导入现有的 tau2_bench 组件
from taskdialogue.benchmarks.tau2.config.manager import RunConfig
from taskdialogue.benchmarks.tau2.evaluator.evaluator import evaluate_simulation, EvaluationType
from taskdialogue.benchmarks.tau2.run import run_single_task
from taskdialogue.benchmarks.tau2.registry import registry
from taskdialogue.benchmarks.tau2.data_model.tasks import Task
from taskdialogue.benchmarks.tau2.data_model.simulation import SimulationRun

logger = get_logger(__name__)


def _serialize_reward_info(reward_info: Any) -> Dict[str, Any]:
    """序列化 reward_info 为字典格式（公共函数，避免重复代码）
    
    Args:
        reward_info: RewardInfo 对象或字典
        
    Returns:
        序列化后的字典
    """
    if reward_info is None:
        return None
    
    if hasattr(reward_info, 'model_dump'):
        try:
            return reward_info.model_dump()
        except:
            # 如果 model_dump 失败，尝试手动提取
            return {
                "reward": getattr(reward_info, 'reward', None),
                "reward_breakdown": getattr(reward_info, 'reward_breakdown', None),
                "info": getattr(reward_info, 'info', None),
            }
    elif isinstance(reward_info, dict):
        return reward_info
    else:
        # 其他情况，尝试提取属性
        return {
            "reward": getattr(reward_info, 'reward', None),
            "reward_breakdown": getattr(reward_info, 'reward_breakdown', None),
            "info": getattr(reward_info, 'info', None),
        }


def _build_benchmark_specific_from_sim(sim: Any, include_tokens: bool = False) -> Dict[str, Any]:
    """从 SimulationRun 构建 benchmark_specific 数据（公共函数，避免重复代码）
    
    Args:
        sim: SimulationRun 对象
        include_tokens: 是否包含 token 统计
        
    Returns:
        benchmark_specific 字典
    """
    benchmark_specific = {
        "simulation_id": getattr(sim, 'id', None),
        "trial": getattr(sim, 'trial', None),
        "seed": getattr(sim, 'seed', None),
        "duration": getattr(sim, 'duration', None),
    }
    
    # 添加 token 统计（如果需要）
    if include_tokens:
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        
        if hasattr(sim, 'messages') and sim.messages:
            from taskdialogue.benchmarks.tau2.utils.llm_utils import get_token_usage
            try:
                token_usage = get_token_usage(sim.messages)
                if token_usage:
                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                    completion_tokens = token_usage.get('completion_tokens', 0)
                    total_tokens = prompt_tokens + completion_tokens
            except:
                pass
        
        benchmark_specific["total_tokens"] = total_tokens
        benchmark_specific["prompt_tokens"] = prompt_tokens
        benchmark_specific["completion_tokens"] = completion_tokens
        
        # 统计工具调用次数
        function_calls_count = 0
        if hasattr(sim, 'messages') and sim.messages:
            for msg in sim.messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    function_calls_count += len(msg.tool_calls)
                elif isinstance(msg, dict) and 'tool_calls' in msg:
                    function_calls_count += len(msg.get('tool_calls', []))
        benchmark_specific["function_calls"] = function_calls_count
    
    # 保存 reward_info（如果存在）
    reward_info = getattr(sim, 'reward_info', None)
    if reward_info:
        benchmark_specific["reward_info"] = _serialize_reward_info(reward_info)
    
    return benchmark_specific


def _evaluate_single_result_worker(
    item: Tuple[InferenceResult, Dict, str, bool, str],
    idx: int
) -> Tuple[int, EvaluationResult]:
    """单个结果评估 worker 函数（模块级，用于 pickle）
    
    Args:
        item: (inf_result, tasks_map, evaluation_type, solo_mode, domain)
        idx: 索引
        
    Returns:
        (idx, EvaluationResult)
    """
    from taskdialogue.core.schemas.evaluation import EvaluationMetric
    from taskdialogue.benchmarks.tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
    from taskdialogue.benchmarks.tau2.utils.utils import get_now
    from taskdialogue.benchmarks.tau2.evaluator.evaluator import EvaluationType as EvalType, evaluate_simulation
    from taskdialogue.benchmarks.tau2.data_model.message import UserMessage, AssistantMessage, ToolMessage, ToolCall
    from datetime import datetime
    
    inf_result, tasks_map, evaluation_type, solo_mode, domain = item
    
    try:
        # 获取 reward_info 或重新评估
        reward_info = None
        if inf_result.benchmark_specific:
            reward_info_data = inf_result.benchmark_specific.get("reward_info")
            if reward_info_data:
                if isinstance(reward_info_data, dict):
                    reward_info = RewardInfo.model_validate(reward_info_data)
                else:
                    reward_info = reward_info_data
        
        # 如果没有 reward_info，重新评估
        if reward_info is None:
            # 匹配任务
            task = None
            for tid, t in tasks_map.items():
                if str(tid) == str(inf_result.task_id) or tid == inf_result.task_id:
                    task = t
                    break
            
            if not task:
                return idx, EvaluationResult(
                    benchmark="tau2",
                    task_id=inf_result.task_id,
                    dialogue_id=inf_result.dialogue_id,
                    overall_score=0.0,
                    error_message=f"Task {inf_result.task_id} not found"
                )
            
            # 重建 SimulationRun 并评估
            tau2_messages = []
            if inf_result.raw_messages:
                # 用于推断 ToolMessage 的 requestor
                last_user_tool_call_ids = set()
                
                for msg_data in inf_result.raw_messages:
                    if isinstance(msg_data, dict):
                        role = msg_data.get('role', 'unknown')
                        if role == 'user':
                            user_msg = UserMessage(
                                role="user", content=msg_data.get('content', ''),
                                turn_idx=msg_data.get('turn_idx')
                            )
                            # 检查用户消息是否有工具调用
                            if 'tool_calls' in msg_data and msg_data['tool_calls']:
                                last_user_tool_call_ids = {tc.get('id', '') for tc in msg_data['tool_calls'] if tc.get('id')}
                            else:
                                last_user_tool_call_ids = set()
                            tau2_messages.append(user_msg)
                        elif role == 'assistant':
                            tau2_msg = AssistantMessage(
                                role="assistant", content=msg_data.get('content', ''),
                                turn_idx=msg_data.get('turn_idx')
                            )
                            if 'tool_calls' in msg_data and msg_data['tool_calls']:
                                tool_calls = []
                                for tc_data in msg_data['tool_calls']:
                                    tool_calls.append(ToolCall(
                                        id=tc_data.get('id', ''),
                                        name=tc_data.get('name', ''),
                                        arguments=tc_data.get('arguments', {}),
                                        requestor=tc_data.get('requestor', 'assistant')
                                    ))
                                tau2_msg.tool_calls = tool_calls
                            tau2_messages.append(tau2_msg)
                            # 清除用户工具调用 ID 缓存（assistant 消息后不再是用户调用）
                            last_user_tool_call_ids = set()
                        elif role == 'tool':
                            tool_id = msg_data.get('id', '')
                            # 推断 requestor：如果在用户的工具调用 ID 集合中，则为 "user"，否则为 "assistant"
                            requestor = 'user' if tool_id in last_user_tool_call_ids else msg_data.get('requestor', 'assistant')
                            tau2_messages.append(ToolMessage(
                                role="tool", id=tool_id,
                                content=msg_data.get('content', ''),
                                error=msg_data.get('error', False),
                                requestor=requestor
                            ))
            
            sim_id = inf_result.benchmark_specific.get('simulation_id') if inf_result.benchmark_specific else inf_result.dialogue_id
            term_reason_str = inf_result.termination_reason or "user_stop"
            try:
                term_reason = TerminationReason(term_reason_str)
            except ValueError:
                term_reason = TerminationReason.USER_STOP
            
            timestamp_str = inf_result.timestamp.isoformat() if hasattr(inf_result.timestamp, 'isoformat') else (str(inf_result.timestamp) if inf_result.timestamp else get_now())
            
            sim_run = SimulationRun(
                id=sim_id, task_id=str(task.id),
                timestamp=timestamp_str, start_time=timestamp_str,
                end_time=get_now(), duration=inf_result.inference_time or 0.0,
                termination_reason=term_reason, messages=tau2_messages,
                seed=inf_result.benchmark_specific.get('seed') if inf_result.benchmark_specific else None,
            )
            
            eval_type_map = {"env": EvalType.ENV, "action": EvalType.ACTION, "communicate": EvalType.COMMUNICATE, "all": EvalType.ALL}
            eval_type = eval_type_map.get(evaluation_type, EvalType.ALL)
            
            try:
                reward_info = evaluate_simulation(
                    simulation=sim_run, task=task,
                    evaluation_type=eval_type, solo_mode=solo_mode, domain=domain,
                )
            except ValueError as ve:
                # 捕获 set_state 时的工具调用验证失败
                # 这通常发生在环境状态在执行过程中改变，导致重新执行工具调用时结果不一致
                error_msg = str(ve)
                if "Tool call:" in error_msg or "Returned:" in error_msg:
                    logger.warning(
                        f"工具调用验证失败 (dialogue_id={inf_result.dialogue_id}): "
                        f"环境重建时工具调用返回结果与期望不一致。这可能是因为环境状态在执行过程中发生了改变。"
                    )
                    logger.debug(f"详细错误: {error_msg}")
                    # 返回一个默认的评估结果，reward 为 0（因为无法验证）
                    reward_info = RewardInfo(
                        reward=0.0,
                        info={
                            "note": "Environment state reconstruction failed during evaluation",
                            "error": error_msg[:500]  # 限制错误信息长度
                        }
                    )
                else:
                    # 其他类型的 ValueError，重新抛出
                    raise
        
        # 转换为 EvaluationResult
        overall_score = getattr(reward_info, 'reward', 0.0)
        metrics = [EvaluationMetric(name="reward", value=overall_score)]
        
        if hasattr(reward_info, 'reward_breakdown') and reward_info.reward_breakdown:
            for key, value in reward_info.reward_breakdown.items():
                if isinstance(value, (int, float)):
                    metrics.append(EvaluationMetric(name=key, value=float(value), details={"breakdown": True}))
        
        benchmark_specific = {}
        if hasattr(reward_info, 'db_check') and reward_info.db_check:
            benchmark_specific['db_check'] = {
                "db_match": getattr(reward_info.db_check, 'db_match', False),
                "db_reward": getattr(reward_info.db_check, 'db_reward', 0.0),
            }
        if hasattr(reward_info, 'env_assertions'):
            benchmark_specific['env_assertions'] = reward_info.env_assertions
        if hasattr(reward_info, 'action_checks'):
            benchmark_specific['action_checks'] = reward_info.action_checks
        if hasattr(reward_info, 'communicate_checks'):
            benchmark_specific['communicate_checks'] = reward_info.communicate_checks
        if hasattr(reward_info, 'reward_basis'):
            benchmark_specific['reward_basis'] = reward_info.reward_basis
        if hasattr(reward_info, 'info'):
            benchmark_specific['evaluation_info'] = reward_info.info
        
        eval_result = EvaluationResult(
            benchmark="tau2",
            task_id=inf_result.task_id,
            dialogue_id=inf_result.dialogue_id,
            overall_score=overall_score,
            metrics=metrics,
            evaluator_config={"evaluation_type": evaluation_type, "solo_mode": solo_mode},
            benchmark_specific=benchmark_specific if benchmark_specific else None,
            timestamp=datetime.now(),
        )
        
        return idx, eval_result
        
    except Exception as e:
        logger.error(f"评估失败 (dialogue_id={inf_result.dialogue_id}): {e}")
        import traceback
        traceback.print_exc()
        return idx, EvaluationResult(
            benchmark="tau2",
            task_id=inf_result.task_id,
            dialogue_id=inf_result.dialogue_id,
            overall_score=0.0,
            error_message=str(e)
        )


def _run_single_task_worker(
    task_data: Tuple[Task, int, str, Optional[int]], 
    idx: int, 
    config: Dict[str, Any]
) -> Tuple[int, SimulationRun]:
    """
    单个任务推理（多进程/多线程 worker 函数）
    
    Args:
        task_data: (task, trial_idx, domain, seed) 元组
        idx: 索引（用于排序）
        config: 配置字典
        
    Returns:
        (idx, SimulationRun)
    """
    task, trial_idx, domain, seed = task_data
    task_id = task.id if hasattr(task, 'id') else f"task_{idx}"
    
    try:
        # 获取环境构造函数（传递 config 以支持 profile 切换）
        get_env = registry.get_domain(domain)
        # 尝试传递 config 参数（如果 get_environment 支持）
        try:
            # 将 config 转换为 dict（如果它是 Config 对象）
            config_dict = config
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            elif hasattr(config, '__dict__') and not isinstance(config, dict):
                config_dict = dict(config) if config else {}
            environment = get_env(config=config_dict) if config_dict else get_env()
        except TypeError:
            # 向后兼容：如果 get_environment 不支持 config 参数，使用默认调用
            environment = get_env()
        
        # 获取配置参数
        tau2_config = config.get("tau2", {})
        limits = tau2_config.get("limits", {})
        max_steps = limits.get("max_steps", 50)
        max_errors = limits.get("max_errors", 5)
        
        # 运行单个任务
        simulation_run = run_single_task(
            domain=domain,
            task=task,
            trial_idx=trial_idx,
            environment=environment,
            taskdialogue_config=config,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=seed,
        )
        
        return idx, simulation_run
        
    except Exception as e:
        logger.error(f"Error in task {task_id} (trial {trial_idx}): {e}")
        import traceback
        traceback.print_exc()
        
        # 创建错误结果
        from taskdialogue.benchmarks.tau2.data_model.simulation import SimulationRun, TerminationReason
        from taskdialogue.benchmarks.tau2.utils.utils import get_now
        error_sim = SimulationRun(
            id=f"error_{task_id}_{trial_idx}",
            task_id=task_id,
            start_time=get_now(),
            end_time=get_now(),
            duration=0.0,
            termination_reason=TerminationReason.TOO_MANY_ERRORS,
            messages=[],
        )
        return idx, error_sim


class Tau2Pipeline(BasePipeline):
    """Tau2 Pipeline - 适配器模式
    
    这个类是一个适配器，将现有的 tau2_bench 实现包装成
    统一的 BasePipeline 接口。
    
    Example:
        >>> config = load_config("configs/tau2/airline.yaml")
        >>> pipeline = Tau2Pipeline(config)
        >>> inf_results, eval_results = pipeline.run_full_pipeline()
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        logger.info("Initializing Tau2 Pipeline...")
        
        # 从配置创建 RunConfig
        tau2_config = config.get("tau2", {})
        self.run_config = RunConfig.from_config({"tau2": tau2_config})
        
        # 保存完整配置用于模型初始化
        self.full_config = config
        
        all_domains = tau2_config.get("all_domains", False)
        if all_domains:
            logger.info(f"Domain: 全部 (airline, retail, telecom)")
        else:
            logger.info(f"Domain: {self.run_config.domain}")
        logger.info(f"Num tasks: {self.run_config.num_tasks or 'all'}")
        logger.info("Tau2 Pipeline initialized successfully")
    
    def run_inference(
        self, 
        task_ids: Optional[List[str]] = None
    ) -> List[InferenceResult]:
        """运行推理（支持多进程/多线程）
        
        参考 MultiWOZ 的实现，支持并行处理和文件保存。
        """
        logger.info("="*80)
        logger.info("Starting Tau2 Inference Phase")
        logger.info("="*80)
        
        # 获取并发配置
        inf_config = self.config.get("inference", {})
        num_workers = inf_config.get("num_workers", 1)
        save_batch_size = inf_config.get("save_batch_size", 32)
        use_processes = inf_config.get("use_processes", False)
        show_progress = self.config.get("logging", {}).get("show_progress", True)
        
        # 准备输出路径
        base_output_dir = Path(self.config.get("output", {}).get("inference_dir", "results/tau2/inference"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 检查是否运行所有domain
        all_domains = self.config.get("tau2", {}).get("all_domains", False)
        if all_domains:
            # 运行所有domain
            domains = ["airline", "retail", "telecom"]
            all_results = []
            
            for domain in domains:
                logger.info("="*80)
                logger.info(f"🚀 运行 Domain: {domain}")
                logger.info("="*80)
                
                # 为每个domain创建子目录
                output_dir = base_output_dir / domain
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # 运行单个domain
                domain_results = self._run_single_domain_inference(
                    domain, output_dir, timestamp, num_workers, save_batch_size, use_processes, show_progress
                )
                all_results.extend(domain_results)
            
            return all_results
        else:
            # 运行单个domain
            domain = self.run_config.domain
            output_dir = base_output_dir / domain  # 二级文件夹：按domain保存
            output_dir.mkdir(parents=True, exist_ok=True)
            return self._run_single_domain_inference(
                domain, output_dir, timestamp, num_workers, save_batch_size, use_processes, show_progress
            )
    
    def _run_single_domain_inference(
        self,
        domain: str,
        output_dir: Path,
        timestamp: str,
        num_workers: int,
        save_batch_size: int,
        use_processes: bool,
        show_progress: bool
    ) -> List[InferenceResult]:
        """运行单个domain的推理"""
        # 加载任务数据
        try:
            get_tasks_fn = registry.get_tasks(self.run_config.task_set_name or domain)
            tasks_data = get_tasks_fn()
            
            # 转为 Task 对象
            tasks = []
            for t in tasks_data:
                if isinstance(t, dict):
                    tasks.append(Task.model_validate(t))
                else:
                    tasks.append(t)
            
            # 限制任务数量
            if self.run_config.num_tasks is not None:
                tasks = tasks[:self.run_config.num_tasks]
            
            total_runs = len(tasks) * self.run_config.num_trials
            logger.info(f"📋 测试任务: {len(tasks)} 个 (来自 {domain} test set)")
            logger.info(f"🔄 每个任务运行 {self.run_config.num_trials} 次 trial (用于评估随机性)")
            logger.info(f"📊 总计: {total_runs} 次对话将运行")
        except FileNotFoundError as e:
            logger.error(f"❌ 数据文件缺失: {e}")
            logger.error("")
            logger.error("请确保 Tau2 数据文件存在。可以通过以下方式设置数据路径：")
            logger.error("  1. 设置环境变量: export TAU2_DATA_DIR=/path/to/data")
            logger.error("  2. 或将数据放在: project_root/data/tau2/")
            logger.error("")
            logger.error("数据目录结构应为:")
            logger.error("  data/tau2/domains/airline/db.json")
            logger.error("  data/tau2/domains/airline/tasks.json")
            logger.error("  data/tau2/domains/airline/policy.md")
            raise
        except Exception as e:
            logger.error(f"❌ 加载任务失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 准备任务列表（包含所有 trial）
        task_items = []
        for task in tasks:
            for trial_idx in range(self.run_config.num_trials):
                # 获取随机种子
                seed = self.run_config.seed
                if seed is not None:
                    seed = seed + trial_idx  # 每个trial使用不同的种子
                task_items.append((task, trial_idx, domain, seed))
        
        num_samples = len(task_items)
        
        # 生成文件名（对齐 MultiWOZ 格式）
        model_name = self.config.get("model", {}).get("agent", {}).get("name", "unknown").replace('/', '_').replace('-', '_')
        filename = f"predictions_{model_name}_{domain}_{num_samples}samples_{timestamp}.json"
        output_file = output_dir / filename
        
        logger.info(f"输出文件: {output_file}")
        logger.info(f"并发数: {num_workers}")
        logger.info(f"批量保存大小: {save_batch_size}")
        
        # 收集所有对话数据（用于批量保存，参考 MultiWOZ）
        all_dialogue_data = []  # 收集所有对话数据
        
        def save_func(results: List[SimulationRun]):
            """批量收集推理结果（参考 MultiWOZ 的实现）
            
            注意：这里只是收集数据，最后统一保存为大 JSON
            为了避免 pickle 问题，使用模块级转换函数
            """
            # 批量转换为对话数据（使用辅助函数避免 self 引用）
            for sim in results:
                if sim:
                    try:
                        # 直接从 SimulationRun 提取对话数据（避免调用实例方法）
                        # 获取原始消息
                        messages = []
                        if hasattr(sim, 'messages') and sim.messages:
                            for msg in sim.messages:
                                # 转换为字典格式
                                if hasattr(msg, 'model_dump'):
                                    try:
                                        msg_dict = msg.model_dump()
                                    except:
                                        msg_dict = {"role": getattr(msg, 'role', 'unknown'), "content": str(msg)}
                                elif hasattr(msg, 'to_dict'):
                                    msg_dict = msg.to_dict()
                                elif isinstance(msg, dict):
                                    msg_dict = msg
                                else:
                                    msg_dict = {"role": getattr(msg, 'role', 'unknown'), "content": str(msg)}
                                
                                # 转换为 OpenAI 格式
                                if msg_dict.get('role') == 'assistant' and 'tool_calls' in msg_dict:
                                    # 处理 tool_calls
                                    tool_calls = msg_dict.get('tool_calls', [])
                                    if tool_calls:
                                        openai_tool_calls = []
                                        for tc in tool_calls:
                                            if isinstance(tc, dict):
                                                openai_tool_calls.append({
                                                    "id": tc.get('id', ''),
                                                    "type": "function",
                                                    "function": {
                                                        "name": tc.get('name', ''),
                                                        "arguments": json.dumps(tc.get('arguments', {})) if isinstance(tc.get('arguments'), dict) else str(tc.get('arguments', ''))
                                                    }
                                                })
                                        msg_dict["tool_calls"] = openai_tool_calls
                                
                                messages.append(msg_dict)
                        
                        # 构建对话数据
                        sim_id = getattr(sim, 'id', f"tau2_{getattr(sim, 'task_id', 'unknown')}")
                        task_id = str(getattr(sim, 'task_id', 'unknown'))
                        
                        # 统计工具调用次数
                        function_calls_count = 0
                        for msg in messages:
                            if isinstance(msg, dict) and 'tool_calls' in msg:
                                function_calls_count += len(msg.get('tool_calls', []))
                        
                        # 统计 token 使用（从消息中提取）
                        total_tokens = 0
                        prompt_tokens = 0
                        completion_tokens = 0
                        if hasattr(sim, 'messages') and sim.messages:
                            from taskdialogue.benchmarks.tau2.utils.llm_utils import get_token_usage
                            try:
                                token_usage = get_token_usage(sim.messages)
                                if token_usage:
                                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                                    completion_tokens = token_usage.get('completion_tokens', 0)
                                    total_tokens = prompt_tokens + completion_tokens
                            except:
                                pass
                        
                        # 统计对话轮次（user + assistant 配对）
                        num_turns = 0
                        user_count = 0
                        for msg in messages:
                            if isinstance(msg, dict):
                                role = msg.get('role', '')
                                if role == 'user':
                                    user_count += 1
                                elif role == 'assistant':
                                    if user_count > 0:
                                        num_turns += 1
                                        user_count = 0
                        # 如果最后还有 user 但没有 assistant，也算一轮
                        if user_count > 0:
                            num_turns += 1
                        
                        dialogue_data = {
                            "dialogue_id": sim_id,
                            "task_id": task_id,
                            "messages": messages,
                            "stats": {
                                "num_turns": num_turns,
                                "total_tokens": total_tokens,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "function_calls": function_calls_count,
                                "inference_time": getattr(sim, 'duration', 0.0) or 0.0,
                            }
                        }
                        
                        # 添加 benchmark_specific 数据（使用公共函数，但使用已统计的值避免重复计算）
                        benchmark_specific = _build_benchmark_specific_from_sim(sim, include_tokens=False)
                        # 覆盖已统计的值（更准确，因为已经在循环中统计了）
                        benchmark_specific["function_calls"] = function_calls_count
                        benchmark_specific["total_tokens"] = total_tokens
                        benchmark_specific["prompt_tokens"] = prompt_tokens
                        benchmark_specific["completion_tokens"] = completion_tokens
                        
                        dialogue_data.update(benchmark_specific)
                        all_dialogue_data.append(dialogue_data)
                    except Exception as e:
                        logger.error(f"转换 SimulationRun 失败: {e}")
                        import traceback
                        traceback.print_exc()
        
        # 定义处理函数（将 Config 转换为 dict 以便在多线程中传递）
        config_dict = self.full_config if isinstance(self.full_config, dict) else (self.full_config.to_dict() if hasattr(self.full_config, 'to_dict') else self.full_config)
        def process_func(task_data: Tuple[Task, int, str, Optional[int]], idx: int) -> Tuple[int, SimulationRun]:
            return _run_single_task_worker(task_data, idx, config_dict)
        
        # 并行处理（使用顺序保存，参考 MultiWOZ）
        simulations = parallel_process_with_saving(
            items=task_items,
            process_func=process_func,
            save_func=save_func,
            num_workers=num_workers,
            save_batch_size=save_batch_size,
            use_processes=use_processes,
            desc=f"Running {domain} inference",
            show_progress=show_progress,
            final_save=False  # 禁用 parallel 中的 final_save，改为手动保存大 JSON
        )
        
        logger.info(f"✓ 完成 {len(simulations)} 个仿真的推理")
        
        # 转换为 InferenceResult（用于返回）
        inference_results = []
        for sim in simulations:
            if sim:
                try:
                    if isinstance(sim, dict):
                        sim = SimulationRun.model_validate(sim)
                    inf_result = self._convert_simulation_to_inference(sim)
                    inference_results.append(inf_result)
                except Exception as e:
                    logger.error(f"转换 SimulationRun 到 InferenceResult 失败: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.info(f"✓ 成功转换 {len(inference_results)} 个推理结果")
        
        # 所有对话处理完成后，一次性保存为大 JSON 文件（参考 MultiWOZ）
        if all_dialogue_data:
            model_name_config = self.config.get("model", {}).get("agent", {}).get("name", "unknown")
            provider = self.config.get("model", {}).get("agent", {}).get("provider", "unknown")
            
            # 使用基类的序列化方法
            config_serializable = self._make_config_serializable(self.config)
            
            final_json = {
                "benchmark": "tau2",
                "timestamp": timestamp,
                "model_name": model_name_config,
                "provider": provider,
                "domain": domain,
                "num_dialogues": len(all_dialogue_data),
                "config": config_serializable,
                "dialogues": all_dialogue_data
            }
            
            # 保存文件
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            logger.info(f"✓ 结果已保存到: {output_file} ({len(all_dialogue_data)} 个对话)")
        
        # 打印统计信息（参考 MultiWOZ 的格式）
        if inference_results:
            self._print_inference_statistics(inference_results, domain, output_file)
        
        return inference_results
    
    def _print_inference_statistics(self, inference_results: List[InferenceResult], domain: str, output_file: Path):
        """打印推理统计信息（参考 MultiWOZ 的格式）
        
        Args:
            inference_results: 推理结果列表
            domain: 域名
            output_file: 输出文件路径
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✅ Inference Phase Complete: {len(inference_results)} dialogues ({domain})")
        logger.info(f"📁 Results saved to: {output_file}")
        logger.info("")
        
        # 计算统计信息
        successful_results = [r for r in inference_results if r.success]
        failed_results = [r for r in inference_results if not r.success]
        
        # 统计对话轮次（从所有结果统计，包括失败的）
        total_turns = sum(r.num_turns or 0 for r in inference_results)
        
        # 统计 token 使用
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # 统计工具调用次数
        total_function_calls = 0
        
        # 从所有结果统计（包括失败的），而不只是成功的
        for result in inference_results:
            # Token 统计：优先从 benchmark_specific 中获取，否则从 result.total_tokens 或 raw_messages
            result_tokens = 0
            result_prompt_tokens = 0
            result_completion_tokens = 0
            
            if result.benchmark_specific:
                # 优先使用 benchmark_specific 中的数据
                if 'total_tokens' in result.benchmark_specific and result.benchmark_specific['total_tokens']:
                    result_tokens = result.benchmark_specific['total_tokens']
                if 'prompt_tokens' in result.benchmark_specific and result.benchmark_specific['prompt_tokens']:
                    result_prompt_tokens = result.benchmark_specific['prompt_tokens']
                if 'completion_tokens' in result.benchmark_specific and result.benchmark_specific['completion_tokens']:
                    result_completion_tokens = result.benchmark_specific['completion_tokens']
            
            # 如果 benchmark_specific 中没有，使用 result.total_tokens
            if result_tokens == 0 and result.total_tokens:
                result_tokens = result.total_tokens
            
            # 如果还是没有，尝试从 raw_messages 中提取
            if result_tokens == 0 and result.raw_messages:
                from taskdialogue.benchmarks.tau2.utils.llm_utils import get_token_usage
                try:
                    # 尝试获取 token usage（如果消息有 usage 字段）
                    if all(hasattr(m, 'usage') for m in result.raw_messages if hasattr(m, 'usage')):
                        token_usage = get_token_usage(result.raw_messages)
                        if token_usage:
                            result_prompt_tokens = token_usage.get('prompt_tokens', 0)
                            result_completion_tokens = token_usage.get('completion_tokens', 0)
                            result_tokens = result_prompt_tokens + result_completion_tokens
                except:
                    pass
            
            # 累加 token 统计
            total_tokens += result_tokens
            total_prompt_tokens += result_prompt_tokens
            total_completion_tokens += result_completion_tokens
            
            # 工具调用统计：优先从 benchmark_specific 中获取，否则从 raw_messages 统计
            func_calls = 0
            if result.benchmark_specific and 'function_calls' in result.benchmark_specific:
                func_calls = result.benchmark_specific.get('function_calls', 0)
            elif result.raw_messages:
                for msg in result.raw_messages:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        func_calls += len(msg.tool_calls)
                    elif isinstance(msg, dict):
                        # 处理 tool_calls 可能为 None 的情况
                        tool_calls = msg.get('tool_calls') or []
                        if isinstance(tool_calls, list):
                            func_calls += len(tool_calls)
            total_function_calls += func_calls
        
        # 打印统计信息
        logger.info("📊 Inference Statistics:")
        logger.info(f"  • Total Dialogues:      {len(inference_results)}")
        logger.info(f"  • Successful:          {len(successful_results)}")
        logger.info(f"  • Failed:              {len(failed_results)}")
        logger.info(f"  • Total Turns:         {total_turns}")
        logger.info(f"  • Total Function Calls: {total_function_calls}")
        
        # Token 统计
        if total_tokens > 0:
            logger.info(f"  • Total Tokens:        {total_tokens:,}")
        if total_prompt_tokens > 0 or total_completion_tokens > 0:
            logger.info(f"  • Prompt Tokens:       {total_prompt_tokens:,}")
            logger.info(f"  • Completion Tokens:  {total_completion_tokens:,}")
        
        # 平均值
        if len(successful_results) > 0:
            logger.info("")
            logger.info("📈 Averages (per successful dialogue):")
            logger.info(f"  • Avg Turns/Dialogue:    {total_turns / len(successful_results):.1f}")
            logger.info(f"  • Avg Calls/Dialogue:    {total_function_calls / len(successful_results):.1f}")
            if total_tokens > 0:
                logger.info(f"  • Avg Tokens/Dialogue:   {total_tokens / len(successful_results):.0f}")
        
        logger.info("=" * 80)
        logger.info("")
    
    def _convert_simulation_to_inference(self, sim: Any) -> InferenceResult:
        """将 SimulationRun 转换为 InferenceResult
        
        参考 MultiWOZ 的实现，正确处理消息转换和错误处理。
        """
        from datetime import datetime
        from taskdialogue.core.schemas.dialogue import DialogueTurn
        from taskdialogue.core.schemas.message import Message as CoreMessage, ToolCall, ToolResponse
        
        # 解析时间戳
        try:
            if isinstance(sim.timestamp, str):
                timestamp = datetime.fromisoformat(sim.timestamp.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
        except:
            timestamp = datetime.now()
        
        # 转换消息为对话轮次（正确处理 Tau2 的消息格式）
        dialogue_history = []
        raw_messages = []
        
        if hasattr(sim, 'messages') and sim.messages:
            # Tau2 消息格式：UserMessage, AssistantMessage, ToolMessage, SystemMessage
            # 按顺序处理，将连续的 user + assistant (+ tool) 组织成一个 turn
            turn_id = 1
            i = 0
            current_user_msg = None
            current_agent_msg = None
            current_tool_calls = []
            current_tool_responses = []
            
            while i < len(sim.messages):
                msg = sim.messages[i]
                
                # 保存原始消息
                if hasattr(msg, 'model_dump'):
                    try:
                        raw_messages.append(msg.model_dump())
                    except:
                        raw_messages.append({"role": getattr(msg, 'role', 'unknown'), "content": str(msg)})
                else:
                    raw_messages.append({"role": getattr(msg, 'role', 'unknown'), "content": str(msg)})
                
                # 处理不同类型的消息
                role = getattr(msg, 'role', None)
                content = getattr(msg, 'content', None) or ""
                
                if role == 'user':
                    # 如果已经有 user + agent 的组合，先保存为一个 turn
                    if current_user_msg and current_agent_msg:
                        turn = DialogueTurn(
                            turn_id=turn_id,
                            user_message=current_user_msg,
                            agent_message=current_agent_msg,
                            tool_calls=current_tool_calls if current_tool_calls else None,
                            tool_responses=current_tool_responses if current_tool_responses else None
                        )
                        dialogue_history.append(turn)
                        turn_id += 1
                        # 重置
                        current_agent_msg = None
                        current_tool_calls = []
                        current_tool_responses = []
                    
                    # 新的 user 消息
                    current_user_msg = CoreMessage(role="user", content=content)
                    
                elif role == 'assistant':
                    # Agent 回复
                    current_agent_msg = CoreMessage(role="assistant", content=content)
                    
                    # 检查是否有 tool_calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_call = ToolCall(
                                id=getattr(tc, 'id', ''),
                                name=getattr(tc, 'name', ''),
                                arguments=getattr(tc, 'arguments', {})
                            )
                            current_tool_calls.append(tool_call)
                            # 更新 agent message 的 tool_calls
                            if current_agent_msg:
                                current_agent_msg.tool_calls = current_tool_calls
                    
                elif role == 'tool':
                    # Tool 响应
                    tool_call_id = getattr(msg, 'id', '')
                    error_flag = getattr(msg, 'error', False)
                    
                    # 从对应的 tool_call 中获取工具名称
                    tool_name = ''
                    if current_tool_calls:
                        # 查找匹配的 tool_call
                        for tc in current_tool_calls:
                            if tc.id == tool_call_id:
                                tool_name = tc.name
                                break
                    
                    # 如果没有找到，尝试从消息的其他属性获取
                    if not tool_name:
                        # 尝试从 msg 的其他属性获取工具名称（如果有）
                        tool_name = getattr(msg, 'name', '') or 'unknown_tool'
                    
                    # 将 error 从布尔值转换为字符串或 None
                    error_str = None
                    if error_flag:
                        # 如果有错误，使用 content 作为错误信息，或提供默认错误信息
                        error_str = content if content else "Tool execution failed"
                    
                    tool_resp = ToolResponse(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        content=content or '',
                        error=error_str
                    )
                    current_tool_responses.append(tool_resp)
                
                i += 1
            
            # 保存最后一个 turn（如果有）
            if current_user_msg and current_agent_msg:
                turn = DialogueTurn(
                    turn_id=turn_id,
                    user_message=current_user_msg,
                    agent_message=current_agent_msg,
                    tool_calls=current_tool_calls if current_tool_calls else None,
                    tool_responses=current_tool_responses if current_tool_responses else None
                )
                dialogue_history.append(turn)
        
        # 构建 benchmark_specific 数据（使用公共函数）
        benchmark_specific = _build_benchmark_specific_from_sim(sim, include_tokens=False)
        
        # 确定是否成功（基于 termination_reason 和 reward_info）
        success = True
        termination_reason = None
        if hasattr(sim, 'termination_reason'):
            termination_reason = str(sim.termination_reason)
            # Tau2 的终止原因枚举值：user_stop, agent_stop, max_steps, too_many_errors
            # user_stop 和 agent_stop 通常表示正常完成
            if termination_reason not in ['user_stop', 'agent_stop']:
                success = False
        
        # 如果有 reward_info，也用来判断成功
        if hasattr(sim, 'reward_info') and sim.reward_info:
            reward = getattr(sim.reward_info, 'reward', None)
            if reward is not None and reward < 0.5:
                success = False
        
        # 创建 InferenceResult
        # 确保 task_id 是字符串
        task_id = str(getattr(sim, 'task_id', 'unknown')) if hasattr(sim, 'task_id') and sim.task_id else 'unknown'
        return InferenceResult(
            benchmark="tau2",
            task_id=task_id,
            dialogue_id=getattr(sim, 'id', f"tau2_{timestamp.isoformat()}"),
            dialogue_history=dialogue_history,
            raw_messages=raw_messages,
            timestamp=timestamp,
            num_turns=len(dialogue_history),
            inference_time=getattr(sim, 'duration', None),
            benchmark_specific=benchmark_specific,
            success=success,
            termination_reason=termination_reason,
        )
    
    def run_evaluation(
        self, 
        inference_results: List[InferenceResult]
    ) -> List[EvaluationResult]:
        """运行评估（支持多进程）
        
        Tau2 的评估在推理阶段已经完成（reward_info 已包含在 SimulationRun 中），
        这里只需要将 reward_info 转换为统一的 EvaluationResult 格式。
        
        如果 reward_info 缺失，可以重新评估（需要重新加载任务数据）。
        """
        logger.info("\n" + "="*80)
        logger.info("Starting Tau2 Evaluation Phase")
        logger.info("="*80)
        
        from taskdialogue.core.schemas.evaluation import EvaluationMetric
        from taskdialogue.benchmarks.tau2.data_model.simulation import RewardInfo, SimulationRun
        
        eval_config = self.config.get("tau2", {}).get("evaluation", {}) or self.config.get("evaluation", {})
        evaluation_type = eval_config.get("evaluation_type", "all")
        solo_mode = eval_config.get("solo_mode", False)
        num_workers = eval_config.get("num_workers", 1)
        use_processes = eval_config.get("use_processes", True)
        show_progress = self.config.get("logging", {}).get("show_progress", True)
        
        logger.info(f"Evaluation type: {evaluation_type}")
        logger.info(f"Num workers: {num_workers}")
        logger.info(f"Processing {len(inference_results)} inference results")
        
        # 准备任务数据（如果需要重新评估）
        tasks_map = {}
        need_reload_tasks = False
        for inf_result in inference_results:
            reward_info = inf_result.benchmark_specific.get("reward_info") if inf_result.benchmark_specific else None
            if reward_info is None:
                need_reload_tasks = True
                break
        
        if need_reload_tasks:
            logger.info("Some results missing reward_info, loading task data for re-evaluation...")
            from taskdialogue.benchmarks.tau2.registry import registry
            get_tasks_fn = registry.get_tasks(self.run_config.task_set_name or self.run_config.domain)
            tasks = get_tasks_fn()
            from taskdialogue.benchmarks.tau2.data_model.tasks import Task
            for task in tasks:
                if isinstance(task, dict):
                    task = Task.model_validate(task)
                tasks_map[task.id] = task
        
        # 准备评估项（用于并行处理）
        eval_items = []
        for inf_result in inference_results:
            eval_items.append((inf_result, tasks_map, evaluation_type, solo_mode, self.run_config.domain))
        
        # 并行处理评估（使用模块级函数，避免 pickle 问题）
        process_func = _evaluate_single_result_worker
        
        evaluation_results = parallel_process(
            items=eval_items,
            process_func=process_func,
            num_workers=num_workers,
            use_processes=use_processes,
            desc="Running evaluation",
            show_progress=show_progress
        )
        
        logger.info(f"✓ 完成 {len(evaluation_results)} 个评估结果")
        
        # 过滤 None 值（处理失败的情况）
        valid_results = [r for r in evaluation_results if r is not None]
        if len(valid_results) < len(evaluation_results):
            failed_count = len(evaluation_results) - len(valid_results)
            logger.warning(f"⚠️  {failed_count} 个评估失败（返回了 None）")
        
        # 打印评估统计信息（参考 MultiWOZ 的格式）
        if valid_results:
            self._print_evaluation_statistics(valid_results)
        
        return valid_results
    
    def _print_evaluation_statistics(self, evaluation_results: List[EvaluationResult]):
        """打印评估统计信息（参考 MultiWOZ 的格式）
        
        Args:
            evaluation_results: 评估结果列表
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✅ Evaluation Phase Complete: {len(evaluation_results)} dialogues")
        logger.info("")
        
        # 计算统计信息
        successful_evals = [r for r in evaluation_results if not r.error_message]
        failed_evals = [r for r in evaluation_results if r.error_message]
        
        # 统计分数
        if successful_evals:
            scores = [r.overall_score for r in successful_evals]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0
        
        # 统计 metrics
        metric_summary = {}
        for result in successful_evals:
            if result.metrics:
                for metric in result.metrics:
                    metric_name = metric.name  # 使用 name 而不是 metric_name
                    if metric_name not in metric_summary:
                        metric_summary[metric_name] = []
                    metric_summary[metric_name].append(metric.value)
        
        # 打印统计信息
        logger.info("📊 Evaluation Statistics:")
        logger.info(f"  • Total Dialogues:      {len(evaluation_results)}")
        logger.info(f"  • Successful:          {len(successful_evals)}")
        logger.info(f"  • Failed:              {len(failed_evals)}")
        
        if successful_evals:
            logger.info("")
            logger.info("📈 Score Statistics:")
            logger.info(f"  • Average Score:       {avg_score:.3f}")
            logger.info(f"  • Max Score:           {max_score:.3f}")
            logger.info(f"  • Min Score:           {min_score:.3f}")
            
            # 打印 metrics 摘要
            if metric_summary:
                logger.info("")
                logger.info("📋 Metrics Summary:")
                for metric_name, values in sorted(metric_summary.items()):
                    if values:
                        avg_val = sum(values) / len(values)
                        logger.info(f"  • {metric_name}:          {avg_val:.3f} (avg)")
        
        logger.info("=" * 80)
        logger.info("")
    
    def save_evaluation_results(self, results: List[EvaluationResult], output_basename: Optional[str] = None) -> None:
        """保存评估结果（覆盖基类方法，添加 domain 二级目录）
        
        Args:
            results: 评估结果列表
            output_basename: 输出文件基础名（不含扩展名），如果为 None 则自动生成
        """
        if not results:
            return
        
        # 获取 domain（从配置或结果中推断）
        domain = self.run_config.domain if hasattr(self, 'run_config') else None
        if not domain and results:
            # 从结果中推断 domain（如果有）
            first_result = results[0]
            if hasattr(first_result, 'benchmark_specific') and first_result.benchmark_specific:
                domain = first_result.benchmark_specific.get('domain')
            if not domain:
                domain = "unknown"
        
        # 创建 domain 二级目录
        base_evaluation_dir = Path(self.config.get("output", {}).get(
            "evaluation_dir", 
            "results/tau2/evaluation/"
        ))
        evaluation_dir = base_evaluation_dir / domain
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = self._get_timestamp()
        
        # 计算统计信息
        successful_results = [r for r in results if not r.error_message]
        avg_score = sum(r.overall_score for r in successful_results) / len(successful_results) if successful_results else 0.0
        
        # 收集所有 metrics
        all_metrics = []
        per_dialogue = []
        
        for result in results:
            # 收集 metrics
            if result.metrics:
                for metric in result.metrics:
                    all_metrics.append({
                        "dialogue_id": result.dialogue_id,
                        "task_id": result.task_id,
                        "metric_name": metric.name,  # 使用 name 而不是 metric_name
                        "value": metric.value,
                        "details": metric.details
                    })
            
            # 构建每个对话的评估条目
            dialogue_entry = {
                "dialogue_id": result.dialogue_id,
                "task_id": result.task_id,
                "overall_score": result.overall_score,
                "error_message": result.error_message,
                "metrics": [{
                    "metric_name": m.name,  # 使用 name 而不是 metric_name
                    "value": m.value,
                    "details": m.details
                } for m in (result.metrics or [])]
            }
            
            if result.benchmark_specific:
                dialogue_entry["benchmark_specific"] = result.benchmark_specific
            
            per_dialogue.append(dialogue_entry)
        
        # 构建评估结果数据结构
        evaluation_data = {
            "evaluation_time": timestamp,
            "domain": domain,
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
                "domain": domain,
                "timestamp": timestamp
            },
            "per_dialogue": per_dialogue
        }
        
        # 使用指定的文件名或生成默认文件名
        if output_basename:
            output_file = evaluation_dir / f"{output_basename}.json"
        else:
            # 从推理结果文件名或配置中获取模型信息（如果有）
            model_name_config = self.config.get("model", {}).get("agent", {}).get("name", "unknown")
            provider = self.config.get("model", {}).get("agent", {}).get("provider", "unknown")
            model_name = model_name_config.replace('/', '_').replace('-', '_')
            output_file = evaluation_dir / f"evaluation_{model_name}_{domain}_{len(results)}samples_{timestamp}.json"
        
        # 保存到一个大的 JSON 文件
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
        
        logger.info(f"✓ 评估结果已保存到: {output_file} ({len(results)} 个对话)")
    

