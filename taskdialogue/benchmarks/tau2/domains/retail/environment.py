# Copyright Sierra
import json
from typing import Any, Dict, Optional

from taskdialogue.benchmarks.tau2.data_model.tasks import Task
from taskdialogue.benchmarks.tau2.domains.retail.data_model import RetailDB
from taskdialogue.benchmarks.tau2.domains.retail.tools import RetailTools
from taskdialogue.benchmarks.tau2.domains.retail.utils import (
    RETAIL_DB_PATH,
    RETAIL_TASK_SET_PATH,
)
from taskdialogue.benchmarks.tau2.environment.environment import Environment
from taskdialogue.benchmarks.tau2.prompts.policies import get_domain_policy


def get_environment(
    db: Optional[RetailDB] = None,
    solo_mode: bool = False,
    config: Optional[dict] = None,
) -> Environment:
    """获取 Retail domain 的 Environment。
    
    Args:
        db: RetailDB 实例（可选，默认从文件加载）
        solo_mode: 是否启用 solo 模式
        config: 完整配置（可选，用于 profile 选择）
    
    Returns:
        Environment 实例
    """
    if solo_mode:
        raise ValueError("Retail domain does not support solo mode")
    
    # Get policy from embedded constants (no file I/O)
    policy = get_domain_policy("retail")
    
    # 根据 profile 选择工具（确保 original 模式不受影响）
    if config:
        profile = config.get("tau2", {}).get("tools", {}).get("profile", "original")
        if profile == "sql":
            # SQL 模式：使用 SQL 工具
            from taskdialogue.benchmarks.tau2.adapters.tools.registry import get_tools
            tools = get_tools(profile="sql", domain="retail", config=config)
        else:
            # Original 模式：使用原有逻辑（向后兼容）
            if db is None:
                db = RetailDB.load(RETAIL_DB_PATH)
            tools = RetailTools(db)
    else:
        # 没有 config，使用原有逻辑（向后兼容）
        if db is None:
            db = RetailDB.load(RETAIL_DB_PATH)
        tools = RetailTools(db)
    
    return Environment(
        domain_name="retail",
        policy=policy,
        tools=tools,
    )


def get_tasks() -> list[Task]:
    with open(RETAIL_TASK_SET_PATH, "r") as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]
