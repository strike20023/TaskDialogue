"""
Base model interface for all inference backends.

This module defines the abstract interface that all model implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class ModelResponse:
    """Standard response format from models."""
    
    content: Optional[str]
    role: str = "assistant"
    function_call: Optional[Dict[str, Any]] = None  # Legacy: single function call (first tool_call)
    tool_calls: Optional[List[Dict[str, Any]]] = None  # Modern: array of tool calls
    usage: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            'content': self.content,
            'role': self.role
        }
        if self.function_call:
            result['function_call'] = self.function_call
        return result


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    
    All model backends (OpenAI, vLLM, etc.) must implement this interface.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tool/function definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
            
        Returns:
            ModelResponse object
        """
        pass
    
    @abstractmethod
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get cumulative usage statistics.
        
        Returns:
            Dictionary with 'prompt_tokens', 'completion_tokens', 'total_tokens'
        """
        pass
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        pass
    
    @property
    def provider(self) -> str:
        """Get the provider name (openai, vllm, etc.)."""
        return self.__class__.__name__.lower().replace('model', '')

