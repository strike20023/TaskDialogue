"""
Custom Format Agent for MultiWOZ

用于不支持原生 function calling 的模型（如某些开源模型）。
使用特殊 token (<function_call>, <function_response>) 进行工具调用。
"""

import json
import re
from typing import Dict, List, Optional, Any

from taskdialogue.core.base.agent import BaseAgent
from taskdialogue.core.models.base import BaseModel
from taskdialogue.core.schemas.message import Message
from taskdialogue.core.constants import Colors
from taskdialogue.core.utils.logger import logger
from taskdialogue.benchmarks.multiwoz.prompts import (
    get_agent_system_prompt,
    get_tool_calling_format_instruction
)
from taskdialogue.benchmarks.multiwoz.tools.functions import get_all_function_factories


def generate_tool_definitions_json(db_path: str = None) -> str:
    """生成紧凑 JSON 格式的工具定义"""
    factories = get_all_function_factories(db_path)
    
    tools = []
    for factory in factories:
        result = factory()
        schema = result['schema']
        
        tool_def = {
            "name": schema['name'],
            "description": schema['description'],
            "parameters": schema['parameters']
        }
        tools.append(tool_def)
    
    return json.dumps(tools, ensure_ascii=False, separators=(',', ':'))


class CustomFormatAgent(BaseAgent):
    """Custom Format Agent
    
    用于使用特殊 token 而非原生 function calling 的模型。
    
    特点：
    1. 不使用 OpenAI function calling API
    2. 解析模型生成的 <function_call> 标记
    3. 执行工具并返回 <function_response>
    
    Example:
        >>> model = create_model_from_config(config, "agent")
        >>> agent = CustomFormatAgent(model, config)
        >>> response = agent.generate_response("I need a restaurant")
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(model, config, **kwargs)
        
        # 获取 custom_format 配置
        custom_format_config = config.get('custom_format', {})
        self.tool_call_token = custom_format_config.get('tool_call_token', '<function_call>')
        self.tool_call_end_token = custom_format_config.get('tool_call_end_token', '</function_call>')
        self.tool_response_token = custom_format_config.get('tool_response_token', '<function_response>')
        self.tool_response_end_token = custom_format_config.get('tool_response_end_token', '</function_response>')
        
        # 初始化工具（从根配置读取数据库路径）
        # 需要从完整配置中获取，而不是从 agent config
        db_path = kwargs.get('database_path')  # 从 kwargs 传入
        book_db_path = kwargs.get('booking_database_path')
        
        self.func_map: Dict[str, Any] = {}
        factories = get_all_function_factories(db_path, book_db_path)
        for factory in factories:
            result = factory()
            self.func_map[result['name']] = result['function']
        
        # 生成 system 消息
        self.system_prompt = get_agent_system_prompt()
        self.format_instruction = get_tool_calling_format_instruction()
        self.tool_definitions = generate_tool_definitions_json(db_path)
        
        # 合并 system 内容
        merged_system = (
            f"{self.system_prompt}\n\n"
            f"{self.format_instruction}\n\n"
            f"## Available Tools\n\n{self.tool_definitions}"
        )
        
        # 初始化对话
        self.system_message = Message(role="system", content=merged_system)
        
        # 配置
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)
        self.max_tool_iterations = config.get("max_tool_iterations", 5)
        
        # 上下文管理
        self.max_result_length = config.get("max_result_length", 2000)
        self.max_history = config.get("max_history", 50)
        
        # 统计
        self.function_call_count = 0
    
    def _truncate_result(self, result: str) -> str:
        """截断过长的工具返回结果（observation 可能很大）"""
        if len(result) <= self.max_result_length:
            return result
        
        truncated = result[:self.max_result_length]
        return truncated + f"\n... (truncated from {len(result)} chars)"
    
    def _truncate_messages(self):
        """截断消息历史（对于 custom format，只需简单截断）
        
        注意：custom_format 不使用 tool_calls/tool 消息格式，
        而是将工具调用和响应作为普通 user 消息，因此不需要配对检查
        """
        if len(self.dialogue_history) <= self.max_history:
            return
        
        # Custom format 使用简单截断即可
        self.dialogue_history = self.dialogue_history[-self.max_history:]
    
    def generate_response(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成回复（处理 custom format tool calls）"""
        # 添加用户消息
        user_msg = Message(role="user", content=user_input)
        self.add_message(user_msg)
        
        # 检查是否需要截断消息历史
        self._truncate_messages()
        
        # 构建消息
        messages = self._build_messages()
        
        # 处理可能的多次工具调用
        for iteration in range(self.max_tool_iterations):
            # 调用模型（不使用 tools 参数）
            response = self.model.chat_completion(
                messages=messages,
                tools=None,  # 不使用原生 function calling
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.content or ""
            
            # 解析工具调用
            tool_call_match = self._extract_tool_call(content)
            
            if tool_call_match:
                # 有工具调用
                func_name = tool_call_match['name']
                func_args = tool_call_match['arguments']
                
                # 添加 assistant 消息（包含工具调用）
                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    tokens=response.usage.get("total_tokens") if response.usage else None
                )
                self.add_message(assistant_msg)
                
                # 执行工具
                logger.info('')
                logger.info('Function: ' + Colors.MAGENTA + f'{func_name}' + Colors.RESET)
                logger.info('Arguments: ' + Colors.GREEN + f'{func_args}' + Colors.RESET)
                
                if func_name in self.func_map:
                    func = self.func_map[func_name]
                    try:
                        tool_result = func(**func_args)
                        self.function_call_count += 1
                    except Exception as e:
                        tool_result = {"error": f"Function execution error: {type(e).__name__}: {str(e)}"}
                        logger.warning('')
                        logger.warning('⚠️  Function execution error:')
                        logger.warning(f'   {type(e).__name__}: {str(e)}')
                else:
                    tool_result = {"error": f"Unknown tool: {func_name}"}
                
                # 转换为字符串并截断过长结果
                tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                
                # 打印结果（截断显示）
                display_result = tool_result_str[:200] + '...' if len(tool_result_str) > 200 else tool_result_str
                logger.info('Result: ' + Colors.CYAN + display_result + Colors.RESET)
                
                tool_result_str = self._truncate_result(tool_result_str)
                if len(json.dumps(tool_result, ensure_ascii=False)) > self.max_result_length:
                    logger.info(f"⚠️  Result truncated to {self.max_result_length} characters")
                
                # 格式化工具响应
                tool_response_content = (
                    f"{self.tool_response_token}"
                    f"{tool_result_str}"
                    f"{self.tool_response_end_token}"
                )
                
                # 添加工具消息
                tool_msg = Message(role="user", content=tool_response_content)
                self.add_message(tool_msg)
                
                # 更新消息列表
                messages = self._build_messages()
                
                continue
            else:
                # 没有工具调用，返回回复
                # 清理掉可能的工具标记
                clean_content = self._clean_response(content)
                
                assistant_msg = Message(
                    role="assistant",
                    content=clean_content,
                    tokens=response.usage.get("total_tokens") if response.usage else None
                )
                self.add_message(assistant_msg)
                
                return clean_content
        
        return "I apologize, but I'm having trouble processing your request."
    
    def _extract_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """从内容中提取工具调用"""
        pattern = rf"{re.escape(self.tool_call_token)}(.*?){re.escape(self.tool_call_end_token)}"
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return None
        
        tool_call_json = match.group(1).strip()
        
        try:
            tool_call = json.loads(tool_call_json)
            return {
                'name': tool_call.get('name'),
                'arguments': tool_call.get('parameters', {})
            }
        except:
            return None
    
    def _clean_response(self, content: str) -> str:
        """清理响应内容，去除工具标记"""
        # 去除工具调用标记
        content = re.sub(
            rf"{re.escape(self.tool_call_token)}.*?{re.escape(self.tool_call_end_token)}",
            "",
            content,
            flags=re.DOTALL
        )
        
        # 去除工具响应标记
        content = re.sub(
            rf"{re.escape(self.tool_response_token)}.*?{re.escape(self.tool_response_end_token)}",
            "",
            content,
            flags=re.DOTALL
        )
        
        return content.strip()
    
    def reset(self) -> None:
        """重置 agent 状态"""
        self.dialogue_history = []
        self.metadata = {}
        self.function_call_count = 0
    
    def _build_messages(self) -> List[Dict[str, Any]]:
        """构建消息列表"""
        messages = [
            {"role": "system", "content": self.system_message.content}
        ]
        
        for msg in self.dialogue_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return messages

