"""Hybrid Toolkit - 混合 SQL 查询工具和原始业务逻辑工具。

策略：
- 查询类工具（READ）：使用 SQL 工具（db_query, db_list_tables, db_get_schema）
- 业务逻辑工具（WRITE/GENERIC）：保留原始工具（book_reservation, cancel_reservation 等）
"""

from typing import Any, Dict

from taskdialogue.benchmarks.tau2.environment.tool import BaseTool
from taskdialogue.benchmarks.tau2.environment.toolkit import ToolKitBase, ToolType


class HybridToolKit(ToolKitBase):
    """混合工具包：SQL 查询工具 + 原始业务逻辑工具。
    
    优势：
    - SQL 查询更灵活，Agent 可以组合复杂查询
    - 业务逻辑工具保留，确保正确性和一致性
    """
    
    def __init__(self, sql_toolkit: ToolKitBase, original_toolkit: ToolKitBase):
        """初始化混合工具包。
        
        Args:
            sql_toolkit: SQL 工具包（只包含查询工具）
            original_toolkit: 原始工具包（包含所有业务逻辑工具）
        """
        super().__init__(db=original_toolkit.db)  # 使用原始 toolkit 的 db
        self.sql_toolkit = sql_toolkit
        self.original_toolkit = original_toolkit
    
    @property
    def tools(self) -> Dict[str, Any]:
        """获取所有工具（SQL + 原始业务逻辑）。
        
        策略：
        - 如果工具是 READ 类型，优先使用 SQL 工具（如果存在）
        - WRITE 和 GENERIC 类型使用原始工具
        """
        all_tools = {}
        
        # 添加 SQL 工具（仅查询工具，参考 MultiWOZ：Agent 不应该直接执行写入操作）
        sql_tools = self.sql_toolkit.tools
        for name, func in sql_tools.items():
            # 只包含查询类工具（db_query, db_list_tables, db_get_schema）
            # 不包含 db_execute，写入操作应该使用原始业务逻辑工具
            if name in ['db_query', 'db_list_tables', 'db_get_schema']:
                all_tools[name] = func
        
        # 添加原始工具（业务逻辑工具）
        original_tools = self.original_toolkit.tools
        for name, func in original_tools.items():
            # 跳过查询类工具，因为已经用 SQL 工具替代
            tool_type = self.original_toolkit.tool_type(name)
            if tool_type in [ToolType.WRITE, ToolType.GENERIC]:
                # WRITE 和 GENERIC 工具保留
                all_tools[name] = func
            elif tool_type == ToolType.READ:
                # READ 工具：如果 SQL 工具不存在同名工具，则保留原始工具
                # 这样向后兼容，如果某些 READ 工具 SQL 无法替代
                if name not in all_tools:
                    all_tools[name] = func
        
        return all_tools
    
    def get_tools(self) -> Dict[str, BaseTool]:
        """获取所有 Tool 对象（SQL + 原始业务逻辑）。
        
        策略（参考 MultiWOZ）：
        - SQL 工具：仅查询工具（db_query, db_list_tables, db_get_schema）
        - 不包含 db_execute：Agent 不应该直接执行 SQL 写入操作
        - 原始工具：只保留 WRITE 和 GENERIC 类型（业务逻辑）
        - READ 类型的原始工具被 SQL 工具替代
        """
        all_tools = {}
        
        # 添加 SQL 工具（仅查询工具，参考 MultiWOZ：Agent 不应该直接执行写入操作）
        sql_tools = self.sql_toolkit.get_tools()
        for name, tool in sql_tools.items():
            # 只包含查询类工具（db_query, db_list_tables, db_get_schema）
            # 不包含 db_execute，写入操作应该使用原始业务逻辑工具
            if name in ['db_query', 'db_list_tables', 'db_get_schema']:
                all_tools[name] = tool
        
        # 添加原始业务逻辑工具（只保留 WRITE 和 GENERIC）
        original_tools = self.original_toolkit.get_tools()
        for name, tool in original_tools.items():
            tool_type = self.original_toolkit.tool_type(name)
            if tool_type in [ToolType.WRITE, ToolType.GENERIC]:
                # WRITE 和 GENERIC 工具保留
                all_tools[name] = tool
            # READ 类型的工具被 SQL 工具替代，不添加
        
        return all_tools
    
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在。"""
        return tool_name in self.tools
    
    def tool_type(self, tool_name: str) -> ToolType:
        """获取工具类型。"""
        # 先检查 SQL 工具
        if tool_name in self.sql_toolkit.tools:
            return self.sql_toolkit.tool_type(tool_name)
        # 再检查原始工具
        if tool_name in self.original_toolkit.tools:
            return self.original_toolkit.tool_type(tool_name)
        raise ValueError(f"Tool '{tool_name}' not found")

