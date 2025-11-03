"""SQL tools for τ² SQL tool stack。"""

import re
from typing import Any, Dict, List, Optional, Tuple

from taskdialogue.benchmarks.tau2.adapters.tools.sql.db_client import SQLClient


# SQL 工具配置（可扩展）
class SQLToolsConfig:
    """SQL 工具配置"""
    DEFAULT_MAX_ROWS = 50  # 默认最大显示行数
    DEFAULT_MAX_CHARS = 5000  # 默认最大字符数
    FILTERED_MAX_ROWS = 100  # 过滤查询最大行数
    FILTERED_MAX_CHARS = 10000  # 过滤查询最大字符数


def _validate_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """验证 SQL 语句安全性（只允许 SELECT/INSERT/UPDATE/DELETE）。
    
    Args:
        sql: SQL 语句
        
    Returns:
        (is_safe, error_message)
    """
    sql_upper = sql.strip().upper()
    
    # 允许的操作
    allowed_operations = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
    
    # 禁止的操作（防止数据库被破坏）
    forbidden_keywords = [
        'DROP', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC', 'EXECUTE',
        'PRAGMA', 'ATTACH', 'DETACH', 'VACUUM', 'ANALYZE'
    ]
    
    # 检查是否包含禁止的关键字
    for keyword in forbidden_keywords:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"SQL statement contains forbidden keyword: {keyword}"
    
    # 检查第一个关键字是否是允许的操作
    first_word = sql_upper.split()[0] if sql_upper.split() else ""
    if not any(op in first_word for op in allowed_operations):
        return False, f"SQL statement must start with one of: {', '.join(allowed_operations)}"
    
    return True, None


def _format_query_results(
    results: List[Any], 
    columns: List[str],
    max_rows: int = SQLToolsConfig.DEFAULT_MAX_ROWS,
    max_chars: int = SQLToolsConfig.DEFAULT_MAX_CHARS
) -> str:
    """格式化查询结果为可读的表格格式（参考 MultiWOZ）。
    
    Args:
        results: 查询结果列表
        columns: 列名列表
        max_rows: 最大显示行数
        max_chars: 最大字符数
        
    Returns:
        格式化后的字符串
    """
    if not results:
        return "No results found."
    
    total_rows = len(results)
    
    # 构建表格头部
    lines = []
    lines.append(f"Found {total_rows} record(s) in total. Showing up to {max_rows} records:")
    lines.append("")
    
    # 表头
    header = "| " + " | ".join(str(col) for col in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(separator)
    
    # 添加数据行
    char_count = len("\n".join(lines))
    displayed_rows = 0
    
    for i, row in enumerate(results):
        if displayed_rows >= max_rows:
            break
            
        # 格式化行数据（处理 None 和特殊字符）
        row_values = []
        for val in row:
            if val is None:
                row_values.append("NULL")
            else:
                val_str = str(val)
                # 截断过长的值
                if len(val_str) > 100:
                    val_str = val_str[:97] + "..."
                row_values.append(val_str)
        
        row_line = "| " + " | ".join(row_values) + " |"
        char_count += len(row_line) + 1
        
        if char_count <= max_chars:
            lines.append(row_line)
            displayed_rows += 1
        else:
            break
    
    # 添加截断提示
    if total_rows > displayed_rows:
        remaining = total_rows - displayed_rows
        lines.append("")
        lines.append(f"... and {remaining} more record(s) not shown (total: {total_rows} records)")
        lines.append("")
        lines.append("NOTE: The above results are sufficient for answering the query.")
        lines.append("DO NOT query again just to see more records.")
    
    return "\n".join(lines)


def create_sql_tools(sql_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """创建 SQL 工具栈。
    
    Args:
        sql_config: SQL 配置（engine, uri, init_sql, enable_schema_tools）
    
    Returns:
        工具列表（name, function, schema 格式）
    """
    client = SQLClient(
        engine=sql_config.get("engine", "sqlite"),
        uri=sql_config.get("uri", ""),
        init_sql=sql_config.get("init_sql"),
    )
    enable_schema_tools = sql_config.get("enable_schema_tools", True)
    
    def db_query(sql: str, params: Dict[str, Any] = None) -> str:
        """执行 SQL 查询（SELECT 语句）。
        
        Args:
            sql: SQL 查询语句（必须是 SELECT）
            params: 参数化查询参数（可选）
        
        Returns:
            格式化后的查询结果（表格格式）
        """
        # 安全性验证
        is_safe, error_msg = _validate_sql_safety(sql)
        if not is_safe:
            return f"Error: {error_msg}"
        
        try:
            # 执行查询并获取列名（一次执行）
            if params:
                if isinstance(params, dict):
                    if '?' in sql:
                        param_values = tuple(params.values())
                        cursor = client.conn.execute(sql, param_values)
                    else:
                        cursor = client.conn.execute(sql, params)
                else:
                    cursor = client.conn.execute(sql, params)
            else:
                cursor = client.conn.execute(sql)
            
            # 获取列名和结果
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = cursor.fetchall()
            cursor.close()
            
            # 判断查询类型以确定显示限制
            sql_upper = sql.strip().upper()
            if 'WHERE' in sql_upper and sql_upper.count('AND') >= 2:
                max_rows = SQLToolsConfig.FILTERED_MAX_ROWS
                max_chars = SQLToolsConfig.FILTERED_MAX_CHARS
            else:
                max_rows = SQLToolsConfig.DEFAULT_MAX_ROWS
                max_chars = SQLToolsConfig.DEFAULT_MAX_CHARS
            
            # 格式化结果
            return _format_query_results(results, columns, max_rows, max_chars)
            
        except Exception as e:
            return f"Database query error: {str(e)}\n\nSQL: {sql[:200]}..."
    
    def db_execute(sql: str, params: Dict[str, Any] = None) -> str:
        """执行 SQL 写入操作（INSERT/UPDATE/DELETE）。
        
        Args:
            sql: SQL 执行语句
            params: 参数化执行参数（可选）
        
        Returns:
            执行结果消息（包含影响行数）
        """
        # 安全性验证
        is_safe, error_msg = _validate_sql_safety(sql)
        if not is_safe:
            return f"Error: {error_msg}"
        
        try:
            # 执行操作并获取影响行数
            cursor = client.conn.execute(sql, params if params else ())
            affected_rows = cursor.rowcount
            client.conn.commit()
            cursor.close()
            
            return f"Execution successful. {affected_rows} row(s) affected."
            
        except Exception as e:
            # 回滚（虽然 SQLite 可能不需要，但为了一致性）
            try:
                client.conn.rollback()
            except:
                pass
            return f"Database execution error: {str(e)}\n\nSQL: {sql[:200]}..."
    
    def db_list_tables() -> str:
        """列出数据库中的所有表。
        
        Returns:
            表名列表
        """
        try:
            # SQLite 查询所有表
            cursor = client.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            if not tables:
                return "No tables found in the database."
            
            return f"Tables in database:\n" + "\n".join(f"  - {table}" for table in tables)
            
        except Exception as e:
            return f"Error listing tables: {str(e)}"
    
    def db_get_schema(table_name: str) -> str:
        """获取指定表的结构信息。
        
        Args:
            table_name: 表名
        
        Returns:
            表结构信息
        """
        try:
            # SQLite 查询表结构
            cursor = client.conn.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            cursor.close()
            
            if not columns_info:
                return f"Table '{table_name}' not found or has no columns."
            
            # 格式化输出
            lines = [f"Schema for table '{table_name}':", ""]
            lines.append("| Column Name | Type | Not Null | Default | Primary Key |")
            lines.append("|-------------|------|----------|---------|-------------|")
            
            for col_info in columns_info:
                # PRAGMA table_info 返回: (cid, name, type, notnull, dflt_value, pk)
                col_name = col_info[1]
                col_type = col_info[2]
                not_null = "Yes" if col_info[3] else "No"
                default = str(col_info[4]) if col_info[4] is not None else "NULL"
                primary_key = "Yes" if col_info[5] else "No"
                
                lines.append(f"| {col_name} | {col_type} | {not_null} | {default} | {primary_key} |")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error getting schema for table '{table_name}': {str(e)}"
    
    # 返回工具定义（与 TaskDialogue tools/functions.py 格式一致）
    tools = [
        {
            "name": "db_query",
            "function": db_query,
            "schema": {
                "name": "db_query",
                "description": "Execute SQL SELECT query on the database. Returns formatted table results with row limits for readability.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query statement (must be SELECT or WITH ... SELECT)"
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional parameters for parameterized query (for security)"
                        }
                    },
                    "required": ["sql"]
                }
            }
        },
        {
            "name": "db_execute",
            "function": db_execute,
            "schema": {
                "name": "db_execute",
                "description": "Execute SQL write operation (INSERT/UPDATE/DELETE) on the database. Returns execution status and affected row count.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL execution statement (INSERT/UPDATE/DELETE only)"
                        },
                        "params": {
                            "type": "object",
                            "description": "Optional parameters for parameterized execution (for security)"
                        }
                    },
                    "required": ["sql"]
                }
            }
        }
    ]
    
    # 可选：添加 schema 工具（如果启用）
    if enable_schema_tools:
        tools.extend([
            {
                "name": "db_list_tables",
                "function": db_list_tables,
                "schema": {
                    "name": "db_list_tables",
                    "description": "List all tables in the database. Useful for discovering available tables before writing queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "name": "db_get_schema",
                "function": db_get_schema,
                "schema": {
                    "name": "db_get_schema",
                    "description": "Get the schema (column names, types, constraints) for a specific table. Essential for writing correct SQL queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to get schema for"
                            }
                        },
                        "required": ["table_name"]
                    }
                }
            }
        ])
    
    return tools

