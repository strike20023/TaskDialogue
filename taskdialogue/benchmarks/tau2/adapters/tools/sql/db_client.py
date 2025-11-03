"""SQL database client for τ² SQL tool stack（复用 TaskDialogue database.py）。"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class SQLClient:
    """SQL 客户端（复用 TaskDialogue 的 database 查询逻辑）。"""
    
    def __init__(self, engine: str = "sqlite", uri: str = "", init_sql: Optional[str] = None):
        self.engine = engine
        self.uri = uri
        self.init_sql = init_sql
        self.conn = None
        
        if engine == "sqlite":
            self._connect_sqlite()
            if init_sql:
                self._run_init_sql(init_sql)
        else:
            raise NotImplementedError(f"Engine {engine} not implemented yet")
    
    def _connect_sqlite(self):
        """连接 SQLite 数据库。"""
        self.conn = sqlite3.connect(self.uri)
    
    def _run_init_sql(self, init_sql_path: str):
        """执行初始化 SQL。"""
        if Path(init_sql_path).exists():
            with open(init_sql_path, "r") as f:
                sql = f.read()
            self.conn.executescript(sql)
            self.conn.commit()
    
    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """执行查询（复用 TaskDialogue query_database 逻辑）。
        
        Args:
            sql: SQL 查询语句
            params: 参数化查询参数（安全，可以是 dict 或 tuple）
        
        Returns:
            查询结果列表（元组列表）
        """
        try:
            # 处理参数：如果是 dict，需要转换为 tuple 或使用命名参数
            if params:
                if isinstance(params, dict):
                    # SQLite 支持命名参数（:param_name 或 ?）
                    # 对于 ? 占位符，需要转换为 tuple
                    if '?' in sql:
                        # 转换为 tuple（按顺序）
                        param_values = tuple(params.values())
                        cursor = self.conn.execute(sql, param_values)
                    else:
                        # 使用命名参数（:name 或 $name）
                        cursor = self.conn.execute(sql, params)
                else:
                    cursor = self.conn.execute(sql, params)
            else:
                cursor = self.conn.execute(sql)
            results = cursor.fetchall()
            return results
        except Exception as e:
            raise RuntimeError(f"Database query failed: {e}")
    
    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> str:
        """执行写入操作。"""
        try:
            if params:
                self.conn.execute(sql, params)
            else:
                self.conn.execute(sql)
            self.conn.commit()
            return "Execution successful"
        except Exception as e:
            raise RuntimeError(f"Database execution failed: {e}")
    
    def close(self):
        """关闭连接。"""
        if self.conn:
            self.conn.close()

