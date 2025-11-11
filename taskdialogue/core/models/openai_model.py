"""
OpenAI API model implementation.

Supports OpenAI, DeepSeek, and other OpenAI-compatible APIs.
"""

import openai
from typing import Dict, List, Optional, Any

from taskdialogue.core.models.base import BaseModel, ModelResponse
from taskdialogue.core.constants import AgentDefaults
from taskdialogue.core.utils.retry import retry_with_exponential_backoff, RetryConfig


class OpenAIModel(BaseModel):
    """
    Model implementation for OpenAI API and compatible APIs.
    
    Supports:
    - OpenAI (api.openai.com)
    - DeepSeek (api.deepseek.com)
    - Any OpenAI-compatible API
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ):
        """
        Initialize OpenAI-compatible model.
        
        Args:
            model_name: Model name
            api_key: API key
            base_url: Base URL for API (None for OpenAI, custom for others)
            timeout: Request timeout in seconds
            retry_config: 重试配置（包含 max_retries, initial_delay 等）
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout or AgentDefaults.TIMEOUT_OPENAI
        
        # 重试配置
        self.retry_config = retry_config or RetryConfig()
        
        # Create client
        client_kwargs = {'api_key': api_key}
        if base_url:
            client_kwargs['base_url'] = base_url
        
        self.client = openai.OpenAI(**client_kwargs)
        
        # Usage tracking
        self._usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """
        Generate chat completion using OpenAI API.
        
        Args:
            messages: Conversation messages
            tools: Tool/function definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse object
        """
        # Prepare API call parameters
        api_params = {
            'model': self.model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'timeout': self.retry_config.timeout,
        }
        
        if tools:
            api_params['tools'] = tools
        
        # Add any additional parameters
        api_params.update(kwargs)
        
        # Call API with retry
        completion = self._call_api_with_retry(api_params)
        
        # Extract response
        message = completion.choices[0].message
        response = ModelResponse(
            content=message.content,
            role=message.role
        )
        
        # Extract tool calls if present (OpenAI returns tool_calls array)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            # Convert OpenAI tool_calls to dict format
            tool_calls_list = []
            for tc in message.tool_calls:
                tool_calls_list.append({
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments
                    }
                })
            response.tool_calls = tool_calls_list
            
            # For backward compatibility, also set function_call (first tool call)
            tool_call = message.tool_calls[0]
            response.function_call = {
                'name': tool_call.function.name,
                'arguments': tool_call.function.arguments
            }
        
        # Track usage
        if hasattr(completion, 'usage') and completion.usage:
            usage = {
                'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_tokens': completion.usage.total_tokens
            }
            response.usage = usage
            
            # Update cumulative usage
            for key in self._usage:
                self._usage[key] += usage.get(key, 0)
        
        return response
    
    def _call_api_with_retry(self, api_params: Dict[str, Any]):
        """调用 API 并自动重试（使用配置的重试策略）"""
        @retry_with_exponential_backoff(
            max_retries=self.retry_config.max_retries,
            initial_delay=self.retry_config.initial_delay,
            exponential_base=self.retry_config.exponential_base,
            max_delay=self.retry_config.max_delay,
            log_attempts=self.retry_config.log_attempts
        )
        def _call():
            return self.client.chat.completions.create(**api_params)
        
        return _call()
    
    def chat(
        self,
        system_message: str,
        user_message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_usage: bool = False
    ) -> Any:
        """
        Simple chat interface for evaluation.
        
        Args:
            system_message: System prompt
            user_message: User message
            temperature: Sampling temperature (default 0.1)
            max_tokens: Maximum tokens (default 4096)
            return_usage: Whether to return usage stats
            
        Returns:
            Response content string or dict with content and usage if return_usage=True
        """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]
        
        # Use defaults if not specified
        if temperature is None:
            temperature = self.config.get('temperature', 0.1)
        if max_tokens is None:
            max_tokens = self.config.get('max_tokens', 4096)
        
        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if return_usage:
            return {
                'content': response.content,
                'usage': response.usage or {}
            }
        return response.content
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get cumulative usage statistics."""
        return self._usage.copy()
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self._usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }
    
    @property
    def provider(self) -> str:
        """Get provider name based on base_url."""
        if self.base_url and 'deepseek' in self.base_url:
            return 'deepseek'
        return 'openai'

