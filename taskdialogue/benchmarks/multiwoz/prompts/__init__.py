"""
MultiWOZ Prompts

MultiWOZ 专用的 prompt 模板。
"""

from taskdialogue.benchmarks.multiwoz.prompts.prompts import (
    get_agent_system_prompt,
    get_user_simulator_template,
    format_user_simulator_prompt,
    get_evaluation_system_prompt,
    get_evaluation_human_template,
    get_evaluation_answer_format_template,
    get_tool_calling_format_instruction,
)

__all__ = [
    "get_agent_system_prompt",
    "get_user_simulator_template",
    "format_user_simulator_prompt",
    "get_evaluation_system_prompt",
    "get_evaluation_human_template",
    "get_evaluation_answer_format_template",
    "get_tool_calling_format_instruction",
]

