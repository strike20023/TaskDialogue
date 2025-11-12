"""
LitAgent wrapper for the MultiWOZ pipeline.

参考 trainer/example.py 的结构，提供一个基于 MultiWOZPipeline 的 LitAgent，
用于对单个对话运行推理与评估，并将评估分数作为训练/验证的 reward。

使用说明：
- 默认读取 `configs/multiwoz/default.yaml` 配置来创建模型与管线。
- 传入的 task 应包含 `dialogue_id`（或 `task_id`）来指定要运行的对话；
  若未提供则从数据集中选择第一个对话作为演示。

注意：此 Agent 直接复用现有的 MultiWOZPipeline 推理与评估流程，
不改变原有结果保存机制。适合作为示例/集成参考。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

from agentlightning import LitAgent, NamedResources, Trainer, reward, configure_logger

from taskdialogue.core.utils.config import load_config, Config
from taskdialogue.benchmarks.multiwoz.pipeline import MultiWOZPipeline
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult


configure_logger()


async def multiwoz_reward(inf_result: InferenceResult, eval_result: EvaluationResult) -> float:
    """简单地将评估的 overall_score 作为 reward。

    如果评估失败（存在 error_message），则返回 0.0。
    """
    if eval_result and not eval_result.error_message:
        return float(eval_result.overall_score or 0.0)
    return 0.0


class MultiWOZLitAgent(LitAgent[Any]):
    """基于 MultiWOZPipeline 的 LitAgent。

    - training_rollout_async: 针对单个对话运行 pipeline 推理与评估，并返回 reward。
    - validation_rollout_async: 复用训练流程，但设置为验证场景（如需可拓展为温度等参数的区分）。
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self._config_path = config_path

    def _prepare_pipeline(self, resources: NamedResources) -> MultiWOZPipeline:
        """加载配置并构建 MultiWOZPipeline。

        参考框架默认配置路径：`configs/multiwoz/default.yaml`。
        若需要与外部资源（如自定义 LLM）对齐，可在此处对 config 进行覆盖。
        """
        cfg: Config = load_config(self._config_path or "configs/multiwoz/default.yaml")
        
        # 如需从 NamedResources 注入自定义模型，可在这里覆盖 config（演示：保留默认配置）。
        # 例如：
        main_llm = resources.get("main_llm")
        if main_llm is not None:
            cfg.set("vllm.model_path", getattr(main_llm, "model", cfg.get("model.agent.name")))
            cfg.set("vllm.server.base_url", main_llm.endpoint)
            cfg.set("model.agent.name", getattr(main_llm, "model", cfg.get("model.agent.name")))
            cfg.set("model.agent.provider", "vllm")
            cfg.set("evaluation.num_workers", 1)
            cfg.set("inference.num_workers", 1)
        else:
            raise ValueError("No main_llm found in resources.")
        print(cfg)
        pipeline = MultiWOZPipeline(cfg.to_dict())
        return pipeline

    async def training_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any
    ) -> Any:
        """对单个 MultiWOZ 对话运行推理与评估，并返回 reward。

        入参 task 期望包含：
        - `dialogue_id`（或 `task_id`）：字符串，目标对话 ID。
        若未提供，则选取数据集中的第一个样本进行演示。
        """
        pipeline = self._prepare_pipeline(resources)

        # 若未提供，则从数据集中取第一个样本
        assert "task_id" in task, "task must contain 'task_id'."
        inf_results, eval_results = pipeline.run_full_pipeline([task["task_id"]])
        reward_score = await multiwoz_reward(inf_results[0], eval_results[0])
        return reward_score

    async def validation_rollout_async(
        self,
        task: Any,
        resources: NamedResources,
        rollout: Any,
    ) -> Any:
        # 复用同样的流程；如需区分温度等参数，可在 _prepare_pipeline / config 中控制。
        return await self.training_rollout_async(task, resources, rollout)


if __name__ == "__main__":
    # macOS/py3.8+ 默认使用 'spawn'，会导致多进程 pickling 本地函数失败。
    # AgentLightning 的默认执行策略在子进程中使用局部函数作为 target，需改为 'fork'。
    try:
        import multiprocessing as mp
        mp.set_start_method("fork")
    except Exception:
        # 若已设置或不可用则忽略，继续尝试运行
        pass

    from os import getenv
    n_workers = int(getenv("AL_WORKERS", "1"))
    Trainer(n_workers=n_workers).fit(MultiWOZLitAgent())