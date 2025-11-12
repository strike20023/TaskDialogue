# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to define a Calc-X agent trainable with Agent-lightning
with latest Agent-lightning API (v0.2+)."""

import asyncio
import os
import re


from taskdialogue.core.utils.config import load_config, Config
from taskdialogue.benchmarks.multiwoz.pipeline import _run_single_dialogue_worker as run_agent
from taskdialogue.benchmarks.multiwoz.pipeline import _evaluate_single_dialogue_worker as evaluate
from taskdialogue.core.schemas.inference import InferenceResult
from taskdialogue.core.schemas.evaluation import EvaluationResult

import agentlightning as agl

from typing import TypedDict, Any, Dict
from pydantic import BaseModel
class MultiwozGoal(BaseModel):
    message: Any
    class Config:
        extra = 'allow'

class MultiwozData(BaseModel):
    dialogue_idx: str
    goal: MultiwozGoal
    log: Any

class TrainingLoader(BaseModel):
    dialogue_data: MultiwozData

@agl.rollout
async def tool_agent(task: MultiwozData, llm: agl.LLM) -> None:
    """agent rollout function.

    It would accept a task and a LLM endpoint resource.
    It's expected to return None, and emit reward via `agl.emit_reward`.
    
    You can choose either way, but not both.
    """
    cfg: Config = load_config("configs/multiwoz/default.yaml")
    cfg.set("model.agent.endpoint", llm.endpoint)
    cfg.set("model.agent.name", llm.model)
    cfg.set("model.agent.temperature", llm.sampling_parameters.get("temperature", 0.7))
    cfg.set("model.agent.provider", "agentlightning")
    eval_config = cfg.get("evaluation", {}).copy()
    max_turns = cfg.get("agent", {}).get("max_turns", 30)
    config = cfg.to_dict()
    # try:
    _, inf_result = run_agent(task.get('dialogue_data'), 0, config, max_turns)
    inf_result: InferenceResult
    _, eval_result = evaluate(inf_result, 0, eval_config, {
            "goal": task.get('dialogue_data').get('goal') if task.get('dialogue_data').get('goal') else {}
        })
    eval_result: EvaluationResult
    reward = float(eval_result.overall_score or 0.0)
    # except Exception as e:
    #     print("Failure:", str(e))
    #     reward = 0.0
    agl.emit_reward(reward)
    print("answer: {} ground_truth: {} reward: {}".format(inf_result.raw_messages[-1], task.get('dialogue_data').get('goal'), reward))


async def debug():
    """Here we show a more manual way for debugging, without Trainer.

    We get the data samples on our own, and run the agent with LitAgentRunner.
    You will need an `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variable set
    to run this function.
    """
    # Manually create a tracer as Runner will need it.
    # Use a dummy OtelTracer if you don't need to trace anything other than reward.
    tracer = agl.OtelTracer()
    # The runner processes MathProblem, which matches the agent's task type.
    runner = agl.LitAgentRunner[TrainingLoader](tracer)

    # A store is required here to store the data collected.
    store = agl.InMemoryLightningStore()

    # This is what needs to be tuned (i.e., LLM)
    resource = agl.LLM(
        endpoint='http://20.66.31.2:12013/v1',
        model="Qwen3-8B", 
        sampling_parameters={"temperature": 0.3}
    )

    from taskdialogue.benchmarks.multiwoz.data.loader import load_multiwoz_data
    from taskdialogue.core.utils.config import load_config, Config


    cfg: Config = load_config("configs/multiwoz/default.yaml")

    data = load_multiwoz_data(
        data_path=cfg.get("data.path", "data/multiwoz/data.json"),
        split=None,
        num_samples=None,
        remove_police_hospital=cfg.get("data.remove_police_hospital", True),
        enabled_domains=cfg.get("domains.enabled")
    ).data
    made_up_task: TrainingLoader = TrainingLoader(
        dialogue_data=data[0]
    )
    another_made_up_task: TrainingLoader = TrainingLoader(
        dialogue_data=data[-1]
    )

    # The agent here must be the same agent that will be used in the real run.
    with runner.run_context(agent=tool_agent, store=store):
        await runner.step(
            made_up_task,
            resources={
                # The key "main_llm" here can be arbitrary
                "main_llm": resource
            },
        )

        # Run another task
        await runner.step(
            another_made_up_task,
            resources={"main_llm": resource},
        )

if __name__ == "__main__":
    asyncio.run(debug())
