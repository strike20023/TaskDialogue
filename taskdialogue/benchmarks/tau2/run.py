"""Run entrypoints for τ²-like execution within TaskDialogue with orchestrator."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from taskdialogue.core.utils.logger import get_logger
from taskdialogue.benchmarks.tau2.registry import registry
from taskdialogue.benchmarks.tau2.data_model.tasks import Task
from taskdialogue.benchmarks.tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

logger = get_logger(__name__)


@dataclass
class RunConfig:
    domain: str
    task_set_name: Optional[str] = None
    num_tasks: Optional[int] = None
    agent: str = "taskdialogue_agent"
    user: str = "user_simulator"
    num_trials: int = 1
    max_steps: int = 100
    max_errors: int = 10
    save_to: Optional[str] = None
    max_concurrency: int = 1
    seed: Optional[int] = 300


def run_single_task(
    domain: str,
    task: Task,
    trial_idx: int,
    environment,
    taskdialogue_config: Any,
    max_steps: int,
    max_errors: int,
    seed: Optional[int],
) -> Any:
    """运行单个任务（使用完整的Orchestrator + TaskDialogue模型）。"""
    from taskdialogue.benchmarks.tau2.adapters.agent.taskdialogue_llm_agent import TaskDialogueLLMAgent
    from taskdialogue.benchmarks.tau2.adapters.agent.taskdialogue_user_simulator import TaskDialogueUserSimulator
    from taskdialogue.benchmarks.tau2.orchestrator import Orchestrator
    from taskdialogue.core.models.factory import create_model_from_config
    
    # 创建TaskDialogue模型实例
    agent_model = create_model_from_config(taskdialogue_config, model_type='agent')
    user_model = create_model_from_config(taskdialogue_config, model_type='user')
    
    # 获取工具列表
    tau2_tools = environment.get_tools()
    if isinstance(tau2_tools, dict):
        tau2_tools_list = list(tau2_tools.values())
    else:
        tau2_tools_list = tau2_tools
    
    # 创建使用TaskDialogue模型的Agent和User
    agent = TaskDialogueLLMAgent(
        model=agent_model,
        tools=tau2_tools_list,
        domain_policy=environment.policy,
        config=taskdialogue_config,
    )
    
    user = TaskDialogueUserSimulator(
        model=user_model,
        instructions=task.user_scenario.instructions if hasattr(task, 'user_scenario') and task.user_scenario else None,
        tools=None,
        config=taskdialogue_config,
    )
    
    # 获取 max_result_length 配置（参考 MultiWOZ）
    tau2_config = taskdialogue_config.get("tau2", {})
    limits = tau2_config.get("limits", {})
    max_result_length = limits.get("max_result_length", 2000)
    
    # 使用 Orchestrator 运行多轮对话
    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=False,
        max_result_length=max_result_length,
    )
    
    simulation_run = orchestrator.run()
    
    # 注意：评估阶段已在 Pipeline.run_evaluation 中单独处理
    # 这里只返回仿真结果，不进行评估
    
    return simulation_run


def run_domain(config: RunConfig, taskdialogue_config: Optional[Any] = None) -> Dict[str, Any]:
    """运行 τ² domain。"""
    domain = config.domain
    get_env = registry.get_domain(domain)
    get_tasks_fn = registry.get_tasks(config.task_set_name or domain)
    # 尝试传递 config 参数（如果 get_environment 支持 profile 切换）
    try:
        # 将 taskdialogue_config 转换为 dict（如果它是 Config 对象）
        config_dict = taskdialogue_config
        if hasattr(taskdialogue_config, 'to_dict'):
            config_dict = taskdialogue_config.to_dict()
        elif hasattr(taskdialogue_config, '__dict__') and not isinstance(taskdialogue_config, dict):
            config_dict = dict(taskdialogue_config) if taskdialogue_config else {}
        environment = get_env(config=config_dict) if config_dict else get_env()
    except TypeError:
        # 向后兼容：如果 get_environment 不支持 config 参数，使用默认调用
        environment = get_env()
    tasks_data = get_tasks_fn()

    # 转为 Task 对象
    tasks = []
    for t in tasks_data:
        if isinstance(t, dict):
            tasks.append(Task.model_validate(t))
        else:
            tasks.append(t)

    if config.num_tasks is not None:
        tasks = tasks[: config.num_tasks]
    
    logger.info(f"📋 准备运行 {len(tasks)} 个任务，每个任务 {config.num_trials} 次 trial")

    simulations: List[Any] = []
    
    # 准备任务列表
    task_args = []
    for trial_idx in range(config.num_trials):
        for task in tasks:
            task_args.append((domain, task, trial_idx, environment, taskdialogue_config, config.max_steps, config.max_errors, config.seed))
    
    # 执行
    if config.max_concurrency > 1:
        logger.info(f"⚡ 并发运行 {len(task_args)} 个任务")
        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            futures = [executor.submit(run_single_task, *args) for args in task_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="运行任务"):
                try:
                    simulations.append(future.result())
                except Exception as e:
                    logger.error(f"❌ 失败: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        logger.info(f"🔄 顺序运行 {len(task_args)} 个任务")
        for args in tqdm(task_args, desc="运行任务"):
            try:
                simulations.append(run_single_task(*args))
            except Exception as e:
                logger.error(f"❌ 失败: {e}")
                import traceback
                traceback.print_exc()

    # 转为 dict
    simulations_dict = [s.model_dump() for s in simulations]
    
    # 简化 metrics
    metrics_dict = {}
    if simulations:
        success_count = sum(1 for s in simulations if s.reward_info and getattr(s.reward_info, 'reward', 0) > 0.5)
        metrics_dict = {
            "total_tasks": len(simulations),
            "success_count": success_count,
            "success_rate": success_count / len(simulations),
            "avg_duration": sum(s.duration for s in simulations) / len(simulations),
        }
        logger.info(f"📊 Metrics: {metrics_dict}")

    return {"simulations": simulations_dict, "metrics": metrics_dict}
