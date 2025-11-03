"""
MultiWOZ Function Calling Agent

基于 function calling 的 MultiWOZ 对话 agent，继承自 BaseAgent。
"""

from typing import Dict, Any, Optional, List
import json
import ast

from taskdialogue.core.base.agent import BaseAgent
from taskdialogue.core.models.base import BaseModel
from taskdialogue.core.schemas.message import Message, ToolCall
from taskdialogue.core.constants import Colors
from taskdialogue.core.utils.logger import logger
from taskdialogue.benchmarks.multiwoz.prompts import get_agent_system_prompt
from taskdialogue.benchmarks.multiwoz.tools.functions import get_all_function_factories


class MultiWOZFunctionAgent(BaseAgent):
    """MultiWOZ Function Calling Agent
    
    功能：
    - 使用 function calling 与数据库交互
    - 支持多领域任务（restaurant, hotel, train, taxi, attraction）
    - 自动处理工具调用和结果
    
    Example:
        >>> model = create_model_from_config(config, "agent")
        >>> agent = MultiWOZFunctionAgent(model, config)
        >>> response = agent.generate_response("I need a restaurant")
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__(model, config, **kwargs)
        
        # System prompt
        self.system_prompt = get_agent_system_prompt()
        
        # Tools（从 kwargs 获取数据库路径）
        db_path = kwargs.get('database_path')
        book_db_path = kwargs.get('booking_database_path')
        from taskdialogue.benchmarks.multiwoz.tools.functions import get_all_function_factories
        
        # 构建工具 schemas（OpenAI 格式）
        factories = get_all_function_factories(db_path, book_db_path)
        self.tools = []
        self.func_map = {}
        for factory in factories:
            result = factory()
            # OpenAI function calling 需要的格式
            tool_schema = {
                "type": "function",
                "function": result['schema']
            }
            self.tools.append(tool_schema)
            self.func_map[result['name']] = result['function']
        
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
        """截断消息历史，确保 tool_calls 和 tool response 配对"""
        if len(self.dialogue_history) <= self.max_history:
            return
        
        recent_messages = self.dialogue_history[-self.max_history:]
        
        # 策略 1: 从后往前检查，移除不完整的 tool_calls（没有对应的 tool response）
        end_idx = len(recent_messages)
        for i in range(len(recent_messages) - 1, -1, -1):
            msg = recent_messages[i]
            
            # 如果是带 tool_calls 的 assistant 消息
            if msg.role == 'assistant' and msg.tool_calls:
                # 检查下一条消息是否是 tool response
                if i + 1 >= len(recent_messages) or recent_messages[i + 1].role != 'tool':
                    # 没有对应的 tool response，截断到这里之前
                    end_idx = i
                    continue
            break
        
        recent_messages = recent_messages[:end_idx]
        
        # 策略 2: 从前往后检查，移除孤立的 tool message（没有前面的 tool_calls）
        start_idx = 0
        for i, msg in enumerate(recent_messages):
            # 跳过开头的孤立 tool 消息
            if msg.role == 'tool':
                continue
            
            # 如果是带 tool_calls 的 assistant 消息
            if msg.role == 'assistant' and msg.tool_calls:
                # 确保有对应的 tool response
                if i + 1 < len(recent_messages) and recent_messages[i + 1].role == 'tool':
                    start_idx = i
                    break
            else:
                # 找到第一个非 tool 的普通消息
                start_idx = i
                break
        
        self.dialogue_history = recent_messages[start_idx:]
    
    def generate_response(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成对用户输入的回复
        
        自动处理 function calling 流程：
        1. 添加用户消息
        2. 调用模型
        3. 如果有工具调用，执行工具并继续
        4. 返回最终回复
        """
        # 添加用户消息
        user_msg = Message(role="user", content=user_input)
        self.add_message(user_msg)
        
        # 检查是否需要截断消息历史
        self._truncate_messages()
        
        # 构建消息历史
        messages = self._build_messages()
        
        # 处理可能的多次工具调用
        for iteration in range(self.max_tool_iterations):
            # 调用模型
            response = self.model.chat_completion(
                messages=messages,
                tools=self.tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # 创建 assistant 消息
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                tokens=response.usage.get("total_tokens") if response.usage else None
            )
            
            # 处理 function call
            if response.function_call:
                # 处理 arguments（可能是字符串或字典）
                args = response.function_call["arguments"]
                if isinstance(args, dict):
                    # 已经是 dict，直接使用
                    pass
                elif isinstance(args, str):
                    # 是字符串，尝试解析（可能是 JSON 或 Python dict 字面量）
                    try:
                        # 先尝试 JSON 解析
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            # 如果 JSON 失败，尝试 Python dict 字面量解析
                            args = ast.literal_eval(args)
                        except (ValueError, SyntaxError) as e:
                            logger.warning(f"Failed to parse arguments: {args[:100]}, error: {e}")
                            args = {}
                else:
                    logger.warning(f"Unexpected arguments type: {type(args)}")
                    args = {}
                
                # 添加工具调用到消息
                tool_call = ToolCall(
                    id=f"call_{iteration}",
                    name=response.function_call["name"],
                    arguments=args
                )
                assistant_msg.tool_calls = [tool_call]
                self.add_message(assistant_msg)
                
                # 执行工具
                logger.info('')
                logger.info('Function: ' + Colors.MAGENTA + f'{tool_call.name}' + Colors.RESET)
                logger.info('Arguments: ' + Colors.GREEN + f'{tool_call.arguments}' + Colors.RESET)
                
                if tool_call.name in self.func_map:
                    func = self.func_map[tool_call.name]
                    try:
                        tool_result = func(**tool_call.arguments)
                        self.function_call_count += 1
                    except Exception as e:
                        tool_result = {"error": f"Function execution error: {type(e).__name__}: {str(e)}"}
                        logger.warning('')
                        logger.warning('⚠️  Function execution error:')
                        logger.warning(f'   {type(e).__name__}: {str(e)}')
                else:
                    tool_result = {"error": f"Unknown tool: {tool_call.name}"}
                
                # 转换为字符串并截断过长结果
                tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                
                # 打印结果（截断显示）
                display_result = tool_result_str[:200] + '...' if len(tool_result_str) > 200 else tool_result_str
                logger.info('Result: ' + Colors.CYAN + display_result + Colors.RESET)
                
                tool_result_str = self._truncate_result(tool_result_str)
                if len(json.dumps(tool_result, ensure_ascii=False)) > self.max_result_length:
                    logger.info(f"⚠️  Result truncated to {self.max_result_length} characters")
                
                # 添加工具结果
                tool_msg = Message(
                    role="tool",
                    content=tool_result_str,
                    tool_call_id=tool_call.id
                )
                self.add_message(tool_msg)
                
                # 更新消息列表
                messages = self._build_messages()
                
                # 继续下一轮（模型可能需要再次调用或生成回复）
                continue
            else:
                # 没有工具调用，添加消息并返回
                self.add_message(assistant_msg)
                return assistant_msg.content or ""
        
        # 达到最大迭代次数
        return "I apologize, but I'm having trouble processing your request."
    
    def reset(self) -> None:
        """重置 agent 状态"""
        self.dialogue_history = []
        self.metadata = {}
        self.function_call_count = 0
    
    def _build_messages(self) -> List[Dict[str, Any]]:
        """构建消息列表（OpenAI 格式）"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        for msg in self.dialogue_history:
            messages.append(msg.to_openai_format())
        
        return messages

