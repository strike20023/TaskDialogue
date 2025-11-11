"""
vLLM model implementation.

Supports vLLM server mode inference.
"""

import json
import threading
from typing import Dict, List, Optional, Any

from taskdialogue.core.models.base import BaseModel, ModelResponse


class VLLMModel(BaseModel):
    """
    Model implementation for vLLM server.
    
    Uses the vllm_utils package for inference.
    """
    
    # Class-level lock for thread safety
    _lock = threading.Lock()
    
    def __init__(
        self,
        model_name: str,
        server_url: str,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize vLLM model.
        
        Args:
            model_name: Model name
            server_url: vLLM server URL
            timeout: Request timeout
            **kwargs: Additional vLLM parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.server_url = server_url
        self.timeout = timeout or 60
        
        # Import vLLM utils
        try:
            from vllm_utils import VLLMServerInference
            from vllm_utils.config import load_sampling_config_from_dict, load_server_config_from_dict
        except ImportError as e:
            raise ImportError(
                "vllm_utils not found. Please install it to use vLLM backend.\n"
                f"Error: {e}"
            )
        
        # Initialize client
        server_config = load_server_config_from_dict({
            'api_base': server_url + ("/v1" if not server_url.endswith("/v1") else ""),
            'timeout': self.timeout
        })
        
        sampling_config = load_sampling_config_from_dict(kwargs.get('sampling', {}))
        
        self.client = VLLMServerInference(
            server_config=server_config,
            model_name=model_name,
            sampling_config=sampling_config,
            verbose=kwargs.get('verbose', False)
        )
        
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
        Generate chat completion using vLLM.
        
        Args:
            messages: Conversation messages
            tools: Tool/function definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            ModelResponse object
        """
        # Use lock for thread safety (vLLM v1 issue)
        with self._lock:
            result = self.client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                return_usage=True
            )
        
        # Parse result
        if isinstance(result, dict):
            content = result.get('content', '')
            tool_calls = result.get('tool_calls')
            usage = result.get('usage', {})
        else:
            content = str(result)
            tool_calls = None
            usage = {}
        
        # Create response
        response = ModelResponse(
            content=content,
            role='assistant'
        )
        
        # Extract function call if present
        if tool_calls:
            tool_call = tool_calls[0]
            # Parse arguments if it's a string (vLLM returns JSON string)
            arguments = tool_call['function']['arguments']
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    # If JSON parsing fails, keep as string (will be handled by agent)
                    pass
            
            response.function_call = {
                'name': tool_call['function']['name'],
                'arguments': arguments
            }
        
        # Track usage
        if usage:
            response.usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            
            for key in self._usage:
                self._usage[key] += response.usage.get(key, 0)
        
        return response
    
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
        return 'vllm'

