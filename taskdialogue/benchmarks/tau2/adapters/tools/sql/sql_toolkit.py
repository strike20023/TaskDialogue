"""SQL Toolkit Adapter - 将 SQL 工具转换为 ToolKitBase 兼容格式。"""

from typing import Any, Callable, Dict, Optional

from taskdialogue.benchmarks.tau2.environment.tool import BaseTool, Tool
from taskdialogue.benchmarks.tau2.environment.toolkit import ToolKitBase, ToolType, TOOL_TYPE_ATTR


class SQLToolWrapper(BaseTool):
    """SQL 工具包装器 - 使用预定义的 schema 而不是从函数签名生成。
    
    这样可以确保 Agent 看到的参数名和 schema 与我们预定义的完全一致。
    """
    
    def __init__(self, func: Callable, schema: Dict[str, Any]):
        """初始化 SQL 工具包装器。
        
        Args:
            func: 可调用的 SQL 工具函数
            schema: OpenAI 格式的完整 schema（包含 name, description, parameters）
        """
        self._func = func
        self._schema = schema
        # 提取 schema 信息
        func_schema = schema.get("function", schema) if "function" in schema else schema
        super().__init__(name=func_schema.get("name", func.__name__))
        
        # 设置 tool_type（根据工具名称判断）
        if func_schema.get("name") == "db_execute":
            setattr(self, TOOL_TYPE_ATTR, ToolType.WRITE)
        else:
            setattr(self, TOOL_TYPE_ATTR, ToolType.READ)
    
    @property
    def openai_schema(self) -> Dict[str, Any]:
        """返回预定义的 OpenAI schema（而不是自动生成的）。"""
        # 确保返回正确的格式
        if "type" in self._schema and "function" in self._schema:
            return self._schema
        else:
            # 如果没有外层包装，添加 type: function 包装
            return {
                "type": "function",
                "function": self._schema
            }
    
    def _call(self, *args: Any, **kwargs: Any) -> Any:
        """调用原始函数。"""
        return self._func(*args, **kwargs)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """使包装器可调用。"""
        return self._call(*args, **kwargs)


class SQLToolKit(ToolKitBase):
    """SQL 工具适配器 - 将 SQL 工具字典转换为 ToolKitBase 兼容格式。
    
    注意：这个类只在 profile="sql" 时使用，不影响 original 模式。
    """
    
    def __init__(self, sql_tools_dict: Dict[str, Dict[str, Any]]):
        """初始化 SQL Toolkit。
        
        Args:
            sql_tools_dict: SQL 工具字典 {name: {function, schema}}
                - function: 可调用函数
                - schema: OpenAI 格式的 schema
        """
        self.sql_tools_dict = sql_tools_dict
        # 不需要 db，SQL 工具已经通过 client 连接了数据库
        super().__init__(db=None)
        
        # 将 SQL 工具的函数提取出来，绑定到实例
        for name, tool_data in sql_tools_dict.items():
            func = tool_data.get("function")
            if func:
                # 使用 setattr 动态添加方法
                setattr(self, name, func)
    
    @property
    def tools(self) -> Dict[str, Any]:
        """获取工具函数字典。"""
        return {
            name: getattr(self, name)
            for name in self.sql_tools_dict.keys()
            if hasattr(self, name)
        }
    
    def get_tools(self) -> Dict[str, BaseTool]:
        """将 SQL 工具转换为 Tool 对象（使用预定义的 schema）。
        
        Returns:
            {name: BaseTool} 字典
        """
        result = {}
        for name, tool_data in self.sql_tools_dict.items():
            func = tool_data.get("function")
            schema = tool_data.get("schema", {})
            if func and schema:
                # 使用自定义包装器，确保使用预定义的 schema
                result[name] = SQLToolWrapper(func, schema)
            elif func:
                # 向后兼容：如果没有 schema，使用默认的 Tool
                from taskdialogue.benchmarks.tau2.environment.tool import as_tool
                result[name] = as_tool(func)
        return result
    
    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在。"""
        return tool_name in self.sql_tools_dict
    
    def tool_type(self, tool_name: str):
        """获取工具类型（SQL 工具默认是 READ）。"""
        from taskdialogue.benchmarks.tau2.environment.toolkit import ToolType
        # SQL 工具可能包含 READ（query）和 WRITE（execute）
        if tool_name == "db_execute":
            return ToolType.WRITE
        return ToolType.READ

