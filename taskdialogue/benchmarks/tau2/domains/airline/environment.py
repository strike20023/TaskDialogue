"""Airline domain environment and task loaders (copy-first, thin adapt)."""

import json
from typing import Any, Dict, List, Optional

from taskdialogue.benchmarks.tau2.domains.airline.data_model import FlightDB
from taskdialogue.benchmarks.tau2.domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_TASK_SET_PATH,
)
from taskdialogue.benchmarks.tau2.environment.environment import Environment
from taskdialogue.benchmarks.tau2.prompts.policies import get_domain_policy


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Environment:
    """获取 Airline domain 的 Environment。
    
    Args:
        db: FlightDB 实例（可选，默认从文件加载）
        solo_mode: 是否启用 solo 模式
        config: 完整配置（可选，用于 profile 选择）
    
    Returns:
        Environment 实例
    """
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    
    # Get policy from embedded constants (no file I/O)
    policy = get_domain_policy("airline")
    
    # 根据 profile 选择工具（确保 original 模式不受影响）
    if config:
        profile = config.get("tau2", {}).get("tools", {}).get("profile", "original")
        if profile == "sql":
            # SQL 模式：使用 SQL 工具
            from taskdialogue.benchmarks.tau2.adapters.tools.registry import get_tools
            tools = get_tools(profile="sql", domain="airline", config=config)
        else:
            # Original 模式：使用原有逻辑（向后兼容）
            if db is None:
                db = FlightDB.load(AIRLINE_DB_PATH)
            from taskdialogue.benchmarks.tau2.adapters.tools.original.airline_tools import AirlineTools
            tools = AirlineTools(db)
    else:
        # 没有 config，使用原有逻辑（向后兼容）
        if db is None:
            db = FlightDB.load(AIRLINE_DB_PATH)
        from taskdialogue.benchmarks.tau2.adapters.tools.original.airline_tools import AirlineTools
        tools = AirlineTools(db)
    
    return Environment(domain_name="airline", policy=policy, tools=tools)


def get_tasks() -> List[Dict[str, Any]]:
    with open(AIRLINE_TASK_SET_PATH, "r", encoding="utf-8") as fp:
        tasks = json.load(fp)
    return tasks


