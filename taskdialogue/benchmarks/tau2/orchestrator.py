import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from taskdialogue.core.utils.logger import get_logger

logger = get_logger(__name__)

from taskdialogue.benchmarks.tau2.agent.base import BaseAgent, is_valid_agent_history_message
from taskdialogue.benchmarks.tau2.agent.llm_agent import LLMSoloAgent
from taskdialogue.benchmarks.tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from taskdialogue.benchmarks.tau2.data_model.simulation import SimulationRun, TerminationReason
from taskdialogue.benchmarks.tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from taskdialogue.benchmarks.tau2.environment.environment import Environment, EnvironmentInfo
from taskdialogue.benchmarks.tau2.user.base import BaseUser, is_valid_user_history_message
from taskdialogue.benchmarks.tau2.user.user_simulator import DummyUser, UserSimulator, UserState
from taskdialogue.benchmarks.tau2.utils.llm_utils import get_cost
from taskdialogue.benchmarks.tau2.utils.utils import format_time, get_now


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)


class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        max_result_length: int = 2000,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.solo_mode = solo_mode
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.max_result_length = max_result_length  # 工具响应最大长度（参考 MultiWOZ）
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None

        # Add timestamps to the message history
        message_history = self._add_timestamps(message_history)

        if self.solo_mode:
            assert self.environment.solo_mode, "Environment should be in solo mode"
            assert isinstance(self.agent, LLMSoloAgent), (
                "Agent must be a LLMSoloAgent in solo mode"
            )
            assert isinstance(self.user, DummyUser), (
                "User must be a DummyUser in solo mode"
            )

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds for the agent, user
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Initialize the agent and user states
        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            # Last message is an assistant message
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():  # Last message is for the user
                    self.to_role = Role.USER
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            # Last message is a user message
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():  # Last message is for the agent
                    self.to_role = Role.AGENT
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            # Last message is a tool message
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history

        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            if not self.solo_mode:
                first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                first_message.timestamp = get_now()
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.USER
            else:
                first_message, agent_state = self.agent.generate_next_message(
                    None, self.agent_state
                )
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.ENV
                self.done = self.agent.is_stop(first_message)
                if self.done:
                    self.termination_reason = TerminationReason.AGENT_STOP

        self.environment.sync_tools()
    
    def _truncate_result(self, result: str) -> str:
        """截断过长的工具返回结果（observation 可能很大）
        
        参考 MultiWOZ 的实现，防止超长响应导致 token 溢出。
        
        Args:
            result: 工具返回的结果字符串
            
        Returns:
            截断后的结果字符串（如果超过 max_result_length）
        """
        if len(result) <= self.max_result_length:
            return result
        
        truncated = result[:self.max_result_length]
        return truncated + f"\n... (truncated from {len(result)} chars)"

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.initialize()
        
        # 打印对话开始（参考 MultiWOZ 的格式）
        try:
            from taskdialogue.core.constants import Colors
            logger.info("=" * 80)
            logger.info(f"Task: {self.task.id if hasattr(self.task, 'id') else 'unknown'}")
            logger.info("=" * 80)
        except Exception:
            pass
        
        while not self.done:
            self.step()
            if self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
                try:
                    from taskdialogue.core.constants import Colors
                    logger.info(f"{Colors.RED}⚠ Dialogue terminated (max steps reached){Colors.RESET}")
                except Exception:
                    pass
            if self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
                try:
                    from taskdialogue.core.constants import Colors
                    logger.info(f"{Colors.RED}⚠ Dialogue terminated (too many errors){Colors.RESET}")
                except Exception:
                    pass
        duration = time.perf_counter() - start
        messages = self.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res
        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
        )
        return simulation_run

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")
        
        # 导入颜色常量（用于美化输出）
        try:
            from taskdialogue.core.constants import Colors
        except ImportError:
            class Colors:
                RESET = ""
                BLUE = ""
                YELLOW = ""
                GREEN = ""
                CYAN = ""
                PURPLE = ""
        
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        # AGENT/ENV -> USER
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg, self.user_state = self.user.generate_next_message(
                self.message, self.user_state
            )
            user_msg.validate()
            
            # 打印 User 回复（参考 MultiWOZ 的格式）
            try:
                from taskdialogue.core.constants import Colors
                import json
                turn_num = getattr(user_msg, 'turn_idx', self.step_count) or self.step_count
                user_content = user_msg.content or ""
                
                # 过滤 XML 格式的工具调用标签和元提示信息（原始回复只在 DEBUG 模式下打印）
                if user_content:
                    logger.debug(f"[原始 User 回复] {user_content}")
                
                import re
                original_user_content = user_content  # 保存原始内容用于判断
                
                # 移除 <function_results>...</function_results> 块
                user_content = re.sub(r'<function_results>.*?</function_results>', '', user_content, flags=re.DOTALL)
                # 移除其他可能的 XML 标签
                user_content = re.sub(r'<invoke[^>]*>.*?</invoke>', '', user_content, flags=re.DOTALL)
                # 移除括号内的等待提示（元提示信息）- 更全面的模式
                user_content = re.sub(r'\(Waiting[^)]*\)', '', user_content, flags=re.IGNORECASE)
                user_content = re.sub(r'\(Still waiting[^)]*\)', '', user_content, flags=re.IGNORECASE)
                user_content = re.sub(r'\(Continuing to wait[^)]*\)', '', user_content, flags=re.IGNORECASE)
                user_content = re.sub(r'\(.*?waiting.*?\)', '', user_content, flags=re.IGNORECASE)
                user_content = user_content.strip()
                
                # 如果过滤后内容为空或只有空白，标记为空
                if not user_content or user_content.isspace():
                    user_content = ""
                
                # 尝试解析 JSON 格式的内容（User 不应该输出 JSON，但如果出现则尝试提取）
                if user_content and user_content.strip().startswith('{'):
                    try:
                        parsed_content = json.loads(user_content)
                        if isinstance(parsed_content, dict):
                            # 格式1: {"message": "..."}
                            if 'message' in parsed_content:
                                user_content = parsed_content.get('message', '')
                            # 格式2: {"status": "success", "data": {...}} - 尝试转换为自然语言
                            elif 'status' in parsed_content and 'data' in parsed_content:
                                data = parsed_content.get('data', {})
                                # 尝试将结构化数据转换为自然语言描述
                                # 例如: {"status": "success", "data": {"flight": "AA2471"}} -> "My flight is AA2471"
                                if isinstance(data, dict) and data:
                                    # 提取关键信息并转换为自然语言
                                    items = []
                                    for key, value in data.items():
                                        # 处理常见的字段名
                                        if key in ['flight_number', 'flight']:
                                            items.append(f"flight {value}")
                                        elif key in ['departure', 'origin']:
                                            items.append(f"from {value}")
                                        elif key in ['destination', 'arrival']:
                                            items.append(f"to {value}")
                                        elif key in ['departure_date', 'date']:
                                            items.append(f"on {value}")
                                        elif key in ['passenger_name', 'passenger']:
                                            items.append(f"for {value}")
                                        else:
                                            items.append(f"{key.replace('_', ' ')}: {value}")
                                    
                                    if items:
                                        user_content = ", ".join(items)
                                    else:
                                        # 如果无法转换，至少显示警告
                                        logger.warning(f"User output JSON format: {user_content[:100]}")
                                        user_content = "[用户不应输出 JSON 格式，应为自然语言]"
                                else:
                                    user_content = "[用户不应输出 JSON 格式，应为自然语言]"
                            # 格式3: 其他 JSON 格式，尝试提取可能的文本字段
                            else:
                                # 查找任何可能的文本字段
                                text_fields = ['text', 'response', 'reply', 'answer', 'content']
                                found = False
                                for field in text_fields:
                                    if field in parsed_content:
                                        user_content = str(parsed_content[field])
                                        found = True
                                        break
                                
                                if not found:
                                    logger.warning(f"User output unexpected JSON format: {user_content[:100]}")
                                    user_content = "[用户不应输出 JSON 格式，应为自然语言]"
                    except:
                        pass  # 不是 JSON，使用原始内容
                
                # 打印 User 消息
                if user_content:
                    logger.info(f"{Colors.BLUE}[Step {self.step_count}] User:{Colors.RESET} {user_content}")
                
                # 打印工具调用信息（参考 MultiWOZ 的格式，单独显示）
                if user_msg.is_tool_call() and user_msg.tool_calls:
                    for tc in user_msg.tool_calls:
                        logger.info('Function: ' + Colors.MAGENTA + f'{tc.name}' + Colors.RESET)
                        if tc.arguments:
                            try:
                                args_str = json.dumps(tc.arguments, ensure_ascii=False, indent=2)
                                logger.info('Arguments: ' + Colors.GREEN + f'{args_str}' + Colors.RESET)
                            except:
                                logger.info('Arguments: ' + Colors.GREEN + f'{tc.arguments}' + Colors.RESET)
                        else:
                            logger.info('Arguments: ' + Colors.GREEN + '(无参数)' + Colors.RESET)
            except Exception as e:
                logger.debug(f"打印 User 消息失败: {e}")
                pass  # 如果打印失败，继续执行
            
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
                try:
                    from taskdialogue.core.constants import Colors
                    logger.info(f"{Colors.GREEN}✓ Dialogue complete (user ended){Colors.RESET}")
                except Exception:
                    pass
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
        # USER/ENV -> AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            agent_msg, self.agent_state = self.agent.generate_next_message(
                self.message, self.agent_state
            )
            agent_msg.validate()
            
            # 打印 Agent 回复（参考 MultiWOZ 的格式）
            try:
                from taskdialogue.core.constants import Colors
                import json
                turn_num = getattr(agent_msg, 'turn_idx', self.step_count) or self.step_count
                agent_content = agent_msg.content or ""
                
                # 过滤 XML 格式的工具调用标签（原始回复只在 DEBUG 模式下打印）
                if agent_content:
                    logger.debug(f"[原始 Agent 回复] {agent_content}")
                
                # 例如: <function_calls>...</function_calls> 应该在打印时移除，因为工具调用已经单独显示了
                import re
                original_agent_content = agent_content  # 保存原始内容用于判断
                
                # 移除 <function_calls>...</function_calls> 块
                agent_content = re.sub(r'<function_calls>.*?</function_calls>', '', agent_content, flags=re.DOTALL)
                # 移除其他可能的 XML 标签
                agent_content = re.sub(r'<invoke[^>]*>.*?</invoke>', '', agent_content, flags=re.DOTALL)
                agent_content = agent_content.strip()
                
                # 如果过滤后内容为空或只有空白，但原来有内容，说明整条消息都是 XML
                if not agent_content or agent_content.isspace():
                    if original_agent_content and original_agent_content.strip():
                        # 整条消息都是 XML 格式，标记为空，工具调用会单独显示
                        agent_content = ""
                
                # 尝试解析 JSON 格式的内容（某些 Agent 可能返回 JSON 字符串）
                parsed_content = None
                if agent_content and agent_content.strip().startswith('{'):
                    try:
                        parsed_content = json.loads(agent_content)
                        # 提取实际消息内容：支持多种可能的 JSON 格式
                        if isinstance(parsed_content, dict):
                            # 格式1: {"action": "send_message", "content": "..."}
                            if 'action' in parsed_content and 'content' in parsed_content:
                                agent_content = parsed_content.get('content', '')
                            # 格式2: {"message": "..."}
                            elif 'message' in parsed_content:
                                agent_content = parsed_content.get('message', '')
                            # 格式3: 如果是其他格式，尝试提取文本内容或格式化显示
                            else:
                                # 如果 JSON 中没有可用的文本字段，使用格式化 JSON（但不应该发生）
                                logger.warning(f"Agent output JSON without 'message' or 'content' field: {agent_content[:100]}")
                                agent_content = json.dumps(parsed_content, ensure_ascii=False, indent=2)
                    except:
                        pass  # 不是 JSON，使用原始内容
                
                # 打印 Agent 消息
                if agent_content:
                    logger.info(f"{Colors.YELLOW}[Step {self.step_count}] Agent:{Colors.RESET} {agent_content}")
                
                # 打印工具调用信息（参考 MultiWOZ 的格式，单独显示）
                if agent_msg.is_tool_call() and agent_msg.tool_calls:
                    for tc in agent_msg.tool_calls:
                        logger.info('Function: ' + Colors.MAGENTA + f'{tc.name}' + Colors.RESET)
                        if tc.arguments:
                            try:
                                args_str = json.dumps(tc.arguments, ensure_ascii=False, indent=2)
                                logger.info('Arguments: ' + Colors.GREEN + f'{args_str}' + Colors.RESET)
                            except:
                                logger.info('Arguments: ' + Colors.GREEN + f'{tc.arguments}' + Colors.RESET)
                        else:
                            logger.info('Arguments: ' + Colors.GREEN + '(无参数)' + Colors.RESET)
                elif agent_msg.is_tool_call():
                    # 即使 tool_calls 为空，也尝试打印警告
                    logger.warning(f"⚠️  Agent message has is_tool_call()=True but tool_calls is empty or None")
            except Exception as e:
                logger.warning(f"⚠️  打印 Agent 消息失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 如果打印失败，至少尝试打印基本信息
                try:
                    if agent_msg.is_tool_call() and agent_msg.tool_calls:
                        logger.info(f"Function: {agent_msg.tool_calls[0].name if agent_msg.tool_calls else 'unknown'}")
                except:
                    pass
            
            if self.agent.is_stop(agent_msg):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            if agent_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.USER
        # AGENT/USER -> ENV
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            tool_msgs = []
            for tool_call in self.message.tool_calls:
                tool_msg = self.environment.get_response(tool_call)
                
                # 截断过长的工具响应（防止 token 溢出，参考 MultiWOZ）
                if tool_msg.content and len(tool_msg.content) > self.max_result_length:
                    original_length = len(tool_msg.content)
                    tool_msg.content = self._truncate_result(tool_msg.content)
                    logger.debug(f"Tool response truncated: {original_length} -> {len(tool_msg.content)} chars")
                
                tool_msgs.append(tool_msg)
                
                # 打印工具执行结果（参考 MultiWOZ 的格式）
                # 注意：工具调用的 Function 和 Arguments 已在 Agent 消息打印时显示
                # 这里只打印执行结果（Result）
                try:
                    from taskdialogue.core.constants import Colors
                    import json
                    
                    # 获取工具响应（observation）
                    tool_response = tool_msg.content or ""
                    
                    # 处理错误情况
                    if getattr(tool_msg, 'error', False):
                        logger.warning('')
                        logger.warning('⚠️  Tool execution error:')
                        logger.warning(f'   {tool_response if tool_response else "Unknown error"}')
                    
                    # 格式化工具响应（observation）
                    if tool_response:
                        # 尝试解析为 JSON 并格式化
                        try:
                            parsed = json.loads(tool_response)
                            tool_result_str = json.dumps(parsed, ensure_ascii=False)
                        except:
                            # 不是 JSON，直接使用原文本
                            tool_result_str = tool_response
                        
                        # 打印结果（截断显示，参考 MultiWOZ：200字符）
                        display_result = tool_result_str[:200] + '...' if len(tool_result_str) > 200 else tool_result_str
                        logger.info('Result: ' + Colors.CYAN + display_result + Colors.RESET)
                        
                        # 如果结果被截断，提示用户
                        if len(tool_result_str) > 200:
                            logger.info(f"⚠️  Result truncated to 200 characters (total: {len(tool_result_str)} characters)")
                    else:
                        logger.info('Result: ' + Colors.CYAN + '(无响应)' + Colors.RESET)
                except Exception as e:
                    logger.warning(f"⚠️  打印工具响应失败: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    # 即使打印失败，也尝试打印基本信息（只打印 Result，不重复 Function/Arguments）
                    try:
                        logger.info('Result: ' + Colors.CYAN + f"{str(tool_msg.content)[:200] if tool_msg.content else '(无响应)'}" + Colors.RESET)
                    except:
                        pass
                    
            assert len(self.message.tool_calls) == len(tool_msgs), (
                "Number of tool calls and tool messages should be the same"
            )
            self.trajectory.extend(tool_msgs)
            if (
                len(tool_msgs) > 1
            ):  # Packaging multiple tool messages into a MultiToolMessage
                self.message = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                self.message = tool_msgs[0]
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )
        self.step_count += 1
        self.environment.sync_tools()

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages == 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ):
        """
        Initialize the environment.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _get_environment_info(self) -> EnvironmentInfo:
        """
        Get the environment info.
        """
        return self.environment.get_info()

    def _count_errors(self, message_history: list[Message]) -> int:
        """
        Count the number of errors in the message history.
        """
        return sum(
            1 for msg in message_history if isinstance(msg, ToolMessage) and msg.error
        )

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history
