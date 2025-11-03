"""TaskDialogue-powered LLMAgent for Tau2-bench.

This adapter wraps TaskDialogue's model system to work with Tau2's Agent interface.
"""

from typing import List, Optional
from copy import deepcopy

from taskdialogue.core.models.base import BaseModel
from taskdialogue.benchmarks.tau2.agent.base import BaseAgent
from taskdialogue.benchmarks.tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from taskdialogue.benchmarks.tau2.environment.tool import Tool
from taskdialogue.benchmarks.tau2.adapters.model_adapter import generate_with_model
from taskdialogue.benchmarks.tau2.prompts import get_agent_system_prompt, get_agent_system_prompt_sql
from pydantic import BaseModel as PydanticBaseModel


ValidAgentInputMessage = UserMessage | ToolMessage | MultiToolMessage


class LLMAgentState(PydanticBaseModel):
    """Agent state."""
    system_messages: list[SystemMessage]
    messages: list[Message]


class TaskDialogueLLMAgent(BaseAgent):
    """LLM Agent using TaskDialogue's model system."""
    
    def __init__(
        self,
        model: BaseModel,
        tools: List[Tool],
        domain_policy: str,
        config: Optional[dict] = None,
    ):
        self.model = model
        self.tools = tools
        self.domain_policy = domain_policy
        self.config = config or {}
        
        # Get temperature and max_tokens from config
        self.temperature = self.config.get('model.agent.temperature', 0.3)
        self.max_tokens = self.config.get('model.agent.max_tokens', 512)
    
    @property
    def system_prompt(self) -> str:
        # 根据 profile 选择 prompt（确保 original 模式不受影响）
        profile = self.config.get("tau2", {}).get("tools", {}).get("profile", "original")
        if profile == "sql" or profile == "hybrid":
            # SQL/Hybrid 模式：使用 SQL 专用 prompt
            return get_agent_system_prompt_sql(self.domain_policy, schema_info=None)
        else:
            # Original 模式：使用原有 prompt
            return get_agent_system_prompt(self.domain_policy)
    
    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        if message_history is None:
            message_history = []
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
    
    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """Generate next message using TaskDialogue model."""
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        
        messages = state.system_messages + state.messages
        
        # Use TaskDialogue model adapter
        assistant_message = generate_with_model(
            model=self.model,
            messages=messages,
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        state.messages.append(assistant_message)
        return assistant_message, state
    
    def set_seed(self, seed: int):
        """Set seed (not supported by TaskDialogue models currently)."""
        pass


