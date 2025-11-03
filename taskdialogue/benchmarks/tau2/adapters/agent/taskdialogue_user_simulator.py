"""TaskDialogue-powered UserSimulator for Tau2-bench.

This adapter wraps TaskDialogue's model system to work with Tau2's User interface.
"""

from typing import Optional, Tuple

from taskdialogue.core.models.base import BaseModel
from taskdialogue.benchmarks.tau2.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
)
from taskdialogue.benchmarks.tau2.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from taskdialogue.benchmarks.tau2.data_model.tasks import UserInstructions
from taskdialogue.benchmarks.tau2.environment.tool import Tool
from taskdialogue.benchmarks.tau2.adapters.model_adapter import generate_with_model
from taskdialogue.benchmarks.tau2.prompts import get_user_simulator_system_prompt


class TaskDialogueUserSimulator(BaseUser):
    """User simulator using TaskDialogue's model system."""
    
    def __init__(
        self,
        model: BaseModel,
        instructions: Optional[UserInstructions],
        tools: Optional[list[Tool]] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(instructions=instructions, llm=None, llm_args=None)
        self.model = model
        self.tools = tools
        self.config = config or {}
        
        # Get temperature and max_tokens from config
        self.temperature = self.config.get('model.user.temperature', 0.1)
        self.max_tokens = self.config.get('model.user.max_tokens', 512)
    
    @property
    def system_prompt(self) -> str:
        instructions_str = str(self.instructions) if self.instructions else ""
        use_tools = self.tools is not None
        return get_user_simulator_system_prompt(
            instructions=instructions_str,
            use_tools=use_tools
        )
    
    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        if message_history is None:
            message_history = []
        
        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state
    
    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """Check if message is a stop message."""
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )
    
    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Generate next message using TaskDialogue model."""
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        
        messages = state.system_messages + state.flip_roles()
        
        # Use TaskDialogue model adapter
        assistant_message = generate_with_model(
            model=self.model,
            messages=messages,
            tools=self.tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Convert assistant message to user message (flip roles)
        user_message = UserMessage(
            role="user",
            content=assistant_message.content,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
        )
        
        # Handle tool calls
        if assistant_message.tool_calls:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )
        
        state.messages.append(user_message)
        return user_message, state
    
    def set_seed(self, seed: int):
        """Set seed (not supported by TaskDialogue models currently)."""
        pass


