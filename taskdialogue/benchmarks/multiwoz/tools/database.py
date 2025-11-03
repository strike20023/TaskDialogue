"""Database utilities for tool functions."""

import sqlite3
from pathlib import Path
from typing import Optional


def get_default_db_path() -> str:
    """Get default database path relative to project root."""
    # Get project root (TaskDialogue directory)
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "db" / "multiwoz.db"
    return str(db_path)


def query_database(sql: str, db_path: Optional[str] = None) -> list:
    """
    Execute SQL query on database.
    
    Args:
        sql: SQL query string
        db_path: Path to database file (uses default if None)
        
    Returns:
        List of query results
    """
    if db_path is None:
        db_path = get_default_db_path()
    
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        return results
    except Exception as e:
        raise RuntimeError(f"Database query failed: {e}")
    finally:
        conn.close()

