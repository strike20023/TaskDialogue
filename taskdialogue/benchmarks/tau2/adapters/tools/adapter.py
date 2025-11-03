"""Tool adapter to unify τ² tool calls with TaskDialogue function_call style.

This module provides a thin wrapper so callers can invoke tools via
`call(function_name, parameters)` regardless of the underlying stack.
"""

from typing import Any, Dict, List, Callable


class ToolAdapter:
    def __init__(self, tools: List[Dict[str, Any]]) -> None:
        """Initialize with a list of tool specs or callable bindings.

        Each tool entry can be a dict like:
        {"name": str, "function": Callable, "schema": dict}
        """
        self._func_map: Dict[str, Callable[..., Any]] = {}
        for t in tools:
            name = t.get("name")
            fn = t.get("function")
            if name and callable(fn):
                self._func_map[name] = fn

    def call(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        if function_name not in self._func_map:
            raise KeyError(f"Tool '{function_name}' not found")
        return self._func_map[function_name](**(parameters or {}))


