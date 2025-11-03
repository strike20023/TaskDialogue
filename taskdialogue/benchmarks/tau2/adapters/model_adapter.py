"""Adapter to use TaskDialogue models with Tau2 Agent/User components.
"""

from typing import Any, Dict, List, Optional
import json

from taskdialogue.core.models.base import BaseModel
from taskdialogue.benchmarks.tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from taskdialogue.benchmarks.tau2.environment.tool import Tool


def tau2_message_to_dict(message: Message) -> Optional[Dict[str, Any]]:
    """Convert Tau2 message to TaskDialogue message format.
    
    Returns None if message should be skipped (e.g., orphaned tool messages).
    """
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, UserMessage):
        if message.is_tool_call():
            # User tool calls - skip for now (not supported in standard flow)
            return None
        return {"role": "user", "content": message.content}
    elif isinstance(message, AssistantMessage):
        msg_dict = {"role": "assistant", "content": message.content}
        if message.is_tool_call() and message.tool_calls:
            # Convert to OpenAI tool_calls format (not function_call)
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id or f"call_{tc.name}",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                    }
                })
            msg_dict["tool_calls"] = tool_calls
            # Remove content if there are tool calls (OpenAI requirement)
            if msg_dict["content"] is None or msg_dict["content"] == "":
                msg_dict["content"] = None
        return msg_dict
    elif isinstance(message, ToolMessage):
        # Tool message - must have matching tool_call_id
        msg_dict = {
            "role": "tool",
            "content": message.content or "",
        }
        # Get tool_call_id from message
        if hasattr(message, 'id') and message.id:
            msg_dict["tool_call_id"] = message.id
        elif hasattr(message, 'tool_call_id') and message.tool_call_id:
            msg_dict["tool_call_id"] = message.tool_call_id
        else:
            # No id - this will cause error, skip it
            return None
        return msg_dict
    else:
        return {"role": "user", "content": str(message)}


def generate_with_model(
    model: BaseModel,
    messages: List[Message],
    tools: Optional[List[Tool]] = None,
    **kwargs
) -> AssistantMessage:
    """Generate response using TaskDialogue model (adapter for Tau2's generate function).
    
    Args:
        model: TaskDialogue BaseModel instance
        messages: List of Tau2 Message objects
        tools: Optional list of Tau2 Tool objects
        **kwargs: Additional arguments (temperature, max_tokens, etc.)
        
    Returns:
        Tau2 AssistantMessage
    """
    # Convert Tau2 messages to TaskDialogue format (filter out None values)
    messages = [tau2_message_to_dict(msg) for msg in messages]
    messages = [msg for msg in messages if msg is not None]
    
    # Convert Tau2 tools to TaskDialogue/OpenAI format
    # Tau2 tools already have correct OpenAI schema format
    tools_list = None
    if tools and len(tools) > 0:
        tools_list = [tool.openai_schema for tool in tools if hasattr(tool, 'openai_schema')]
    
    # Get temperature and max_tokens from kwargs or use defaults
    temperature = kwargs.pop('temperature', None) or 0.1
    max_tokens = kwargs.pop('max_tokens', None) or 4096
    
    # Call TaskDialogue model (align with MultiWOZ interface)
    response = model.chat_completion(
        messages=messages,
        tools=tools_list,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    # Convert response to Tau2 AssistantMessage
    assistant_msg = AssistantMessage(
        role="assistant",
        content=response.content,
    )
    
    # Handle tool calls (OpenAI uses tool_calls array, not single function_call)
    # First check if we need to get tool_calls from the raw response
    tool_calls = []
    
    # Method 1: Check ModelResponse for tool_calls (preferred, supports multiple calls)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_calls = response.tool_calls
    # Method 2: Check function_call (legacy format, single call only)
    elif response.function_call:
        func_name = response.function_call.get('name')
        func_args_str = response.function_call.get('arguments')
        
        # Parse arguments
        if isinstance(func_args_str, str):
            try:
                func_args = json.loads(func_args_str)
            except:
                func_args = {}
        else:
            func_args = func_args_str or {}
        
        if func_name:
            tool_calls = [{
                'id': f"call_{func_name}",
                'type': 'function',
                'function': {
                    'name': func_name,
                    'arguments': func_args_str if isinstance(func_args_str, str) else json.dumps(func_args)
                }
            }]
    
    # Convert tool_calls to Tau2 ToolCall objects
    if tool_calls:
        tau2_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
                func_info = tc.get('function', {}) if 'function' in tc else tc
                func_name = func_info.get('name') or tc.get('name')
                func_args_str = func_info.get('arguments') or tc.get('arguments', '{}')
                
                # Parse arguments
                if isinstance(func_args_str, str):
                    try:
                        func_args = json.loads(func_args_str)
                    except:
                        func_args = {}
                else:
                    func_args = func_args_str or {}
                
                tau2_tool_calls.append(
                    ToolCall(
                        id=tc.get('id') or f"call_{func_name}",
                        name=func_name,
                        arguments=func_args,
                        requestor="assistant",
                    )
                )
        
        if tau2_tool_calls:
            assistant_msg.tool_calls = tau2_tool_calls
    
    # Add usage info
    if response.usage:
        assistant_msg.usage = response.usage
        # Calculate cost (简化，实际应该根据模型和 token 数计算)
        assistant_msg.cost = response.usage.get('total_tokens', 0) * 0.000001
    
    return assistant_msg


# 别名：为了向后兼容，保留旧的函数名
generate_with_autodia_model = generate_with_model  # 已废弃，请使用 generate_with_model

