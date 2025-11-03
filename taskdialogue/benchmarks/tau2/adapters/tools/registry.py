"""Tool registry for switching between original and SQL tool profiles."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from taskdialogue.benchmarks.tau2.environment.toolkit import ToolKitBase
from taskdialogue.core.utils.logger import get_logger

logger = get_logger(__name__)


def build_original_tools(domain: str = "airline") -> ToolKitBase:
    """构建 original τ² 工具（从域的 ToolKitBase 获取）。
    
    Args:
        domain: 域名称
    
    Returns:
        ToolKitBase 实例（original 模式）
    """
    # 根据域加载对应的工具
    if domain == "airline":
        from taskdialogue.benchmarks.tau2.adapters.tools.original.airline_tools import AirlineTools
        from taskdialogue.benchmarks.tau2.domains.airline.data_model import FlightDB
        from taskdialogue.benchmarks.tau2.domains.airline.utils import AIRLINE_DB_PATH
        
        db = FlightDB.load(AIRLINE_DB_PATH)
        toolkit = AirlineTools(db)
        return toolkit  # 返回 ToolKitBase 实例
    elif domain == "retail":
        from taskdialogue.benchmarks.tau2.domains.retail.tools import RetailTools
        from taskdialogue.benchmarks.tau2.domains.retail.data_model import RetailDB
        from taskdialogue.benchmarks.tau2.domains.retail.utils import RETAIL_DB_PATH
        
        db = RetailDB.load(RETAIL_DB_PATH)
        toolkit = RetailTools(db)
        return toolkit
    elif domain == "telecom":
        from taskdialogue.benchmarks.tau2.domains.telecom.tools import TelecomTools
        from taskdialogue.benchmarks.tau2.domains.telecom.data_model import TelecomDB
        from taskdialogue.benchmarks.tau2.domains.telecom.utils import TELECOM_DB_PATH
        
        db = TelecomDB.load(TELECOM_DB_PATH)
        toolkit = TelecomTools(db)
        return toolkit
    else:
        raise ValueError(f"Unknown domain: {domain}")


def build_sql_tools(sql_cfg: Dict[str, Any], domain: str = "airline") -> ToolKitBase:
    """构建 SQL 工具栈（返回 ToolKitBase 兼容对象）。
    
    Args:
        sql_cfg: SQL 配置（engine, uri, init_sql）
        domain: 域名（用于自动生成数据库路径）
    
    Returns:
        ToolKitBase 实例（SQL 模式）
    """
    from taskdialogue.benchmarks.tau2.adapters.tools.sql.sql_tools import create_sql_tools
    from taskdialogue.benchmarks.tau2.adapters.tools.sql.sql_toolkit import SQLToolKit
    from taskdialogue.benchmarks.tau2.utils.utils import DATA_DIR
    
    # 如果 uri 为空，自动生成（data/tau2/sql/{domain}.db）
    if not sql_cfg.get("uri"):
        sql_cfg = sql_cfg.copy()
        sql_cfg["uri"] = str(DATA_DIR / "tau2" / "sql" / f"{domain}.db")
        
        # 如果数据库文件不存在，尝试自动转换
        sql_path = Path(sql_cfg["uri"])
        if not sql_path.exists():
            json_path = DATA_DIR / "tau2" / "domains" / domain / "db.json"
            if json_path.exists():
                logger.info(f"🔄 自动转换数据库: {json_path} → {sql_path}")
                from taskdialogue.benchmarks.tau2.adapters.tools.sql.db_converter import convert_domain_db_to_sqlite
                try:
                    convert_domain_db_to_sqlite(domain, json_path, sql_path)
                except Exception as e:
                    logger.warning(f"⚠️  数据库自动转换失败: {e}")
                    logger.warning(f"   请手动运行转换工具或确保数据库文件存在: {sql_path}")
            else:
                logger.warning(f"⚠️  JSON 数据文件不存在: {json_path}")
    
    # 创建 SQL 工具列表
    tools_list = create_sql_tools(sql_cfg)
    # 转为 dict {name: {function, schema}}
    tools_dict = {t["name"]: {"function": t["function"], "schema": t["schema"]} for t in tools_list}
    
    # 创建 SQLToolKit 适配器
    return SQLToolKit(tools_dict)


def build_hybrid_tools(sql_cfg: Dict[str, Any], domain: str = "airline") -> ToolKitBase:
    """构建混合工具包：SQL 查询工具 + 原始业务逻辑工具。
    
    策略：
    - 查询类工具（READ）：使用 SQL 工具（更灵活）
    - 业务逻辑工具（WRITE/GENERIC）：保留原始工具（确保正确性）
    
    Args:
        sql_cfg: SQL 配置（engine, uri, init_sql）
        domain: 域名
    
    Returns:
        ToolKitBase 实例（混合模式）
    """
    from taskdialogue.benchmarks.tau2.adapters.tools.hybrid_toolkit import HybridToolKit
    
    # 构建 SQL 工具（查询用）
    sql_toolkit = build_sql_tools(sql_cfg, domain=domain)
    
    # 构建原始工具（业务逻辑用）
    original_toolkit = build_original_tools(domain)
    
    # 创建混合工具包
    return HybridToolKit(sql_toolkit, original_toolkit)


def get_tools(profile: str, domain: str, config: Dict[str, Any]) -> ToolKitBase:
    """获取工具 Toolkit（根据 profile 选择）。
    
    Args:
        profile: 'original', 'sql', 或 'hybrid'
            - 'original': 仅使用原始业务逻辑工具
            - 'sql': 仅使用 SQL 工具（纯 SQL 模式）
            - 'hybrid': SQL 查询工具 + 原始业务逻辑工具（推荐）
        domain: 域名称
        config: 完整配置
    
    Returns:
        ToolKitBase 实例
    """
    if profile == "original":
        # Original 模式：返回原有的 ToolKitBase 实例（不影响现有代码）
        return build_original_tools(domain)
    elif profile == "hybrid":
        # Hybrid 模式：SQL 查询工具 + 原始业务逻辑工具
        sql_cfg = config.get("tau2", {}).get("tools", {}).get("sql", {}) if isinstance(config, dict) else {}
        return build_hybrid_tools(sql_cfg, domain=domain)
    else:
        # SQL 模式：纯 SQL 工具（让 Agent 用 SQL 实现所有操作）
        sql_cfg = config.get("tau2", {}).get("tools", {}).get("sql", {}) if isinstance(config, dict) else {}
        return build_sql_tools(sql_cfg, domain=domain)


