"""
Model factory for creating model instances based on configuration.

This module provides a unified interface for creating different model backends.
"""

from typing import Optional

from taskdialogue.core.models.base import BaseModel, ModelResponse
from taskdialogue.core.models.openai_model import OpenAIModel
from taskdialogue.core.models.vllm_model import VLLMModel
from taskdialogue.core.utils.config import Config, get_api_key
from taskdialogue.core.constants import AgentDefaults
from taskdialogue.core.utils.retry import RetryConfig


def create_model(
    provider: str,
    model_name: str,
    config: Optional[Config] = None,
    model_type: str = 'agent',
    **kwargs
) -> BaseModel:
    """
    Create a model instance based on provider.
    
    Args:
        provider: Model provider ('openai', 'deepseek', 'vllm', 'zhipuai')
        model_name: Name of the model
        config: Configuration object (optional)
        model_type: 'agent' or 'user' (for different temperature settings)
        **kwargs: Additional model-specific parameters
        
    Returns:
        BaseModel instance
        
    Raises:
        ValueError: If provider is not supported
        
    Example:
        >>> from taskdialogue.utils import load_config
        >>> config = load_config()
        >>> model = create_model('openai', 'gpt-4', config=config, model_type='agent')
        >>> response = model.chat_completion(messages=[...])
    """
    provider = provider.lower()
    
    # Get common parameters from config if not provided
    if config:
        # Support new config structure: model.agent.* or model.user.*
        config_prefix = f'model.{model_type}.'
        
        if 'temperature' not in kwargs:
            # Try new structure first, fall back to old structure
            kwargs['temperature'] = config.get(
                f'{config_prefix}temperature',
                config.get('model.temperature', AgentDefaults.DEFAULT_TEMPERATURE)
            )
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = config.get(
                f'{config_prefix}max_tokens',
                config.get('model.max_tokens', AgentDefaults.DEFAULT_MAX_TOKENS)
            )
        if 'timeout' not in kwargs:
            kwargs['timeout'] = config.get(
                f'{config_prefix}timeout',
                config.get('model.timeout', AgentDefaults.get_timeout(provider))
            )
    
    # Create model based on provider
    if provider == 'openai':
        return _create_openai_model(model_name, config, **kwargs)
    
    elif provider == 'deepseek':
        return _create_deepseek_model(model_name, config, **kwargs)
    
    elif provider == 'agentlightning':
        return _create_agentlightning_model(model_name, config, **kwargs)
    
    elif provider == 'zhipuai':
        return _create_zhipuai_model(model_name, config, **kwargs)
    
    elif provider == 'vllm':
        return _create_vllm_model(model_name, config, **kwargs)
    

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: openai, deepseek, zhipuai, vllm, agentlightning"
        )


def _create_openai_model(model_name: str, config: Optional[Config], **kwargs) -> OpenAIModel:
    """Create OpenAI model instance."""
    api_key = kwargs.pop('api_key', None) or get_api_key('openai', config)
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add to config.")
    
    # 创建重试配置
    retry_config = RetryConfig.from_config(config, 'openai') if config else RetryConfig()
    
    return OpenAIModel(
        model_name=model_name,
        api_key=api_key,
        retry_config=retry_config,
        **kwargs
    )


def _create_deepseek_model(model_name: str, config: Optional[Config], **kwargs) -> OpenAIModel:
    """Create DeepSeek model instance (uses OpenAI-compatible API)."""
    api_key = kwargs.pop('api_key', None) or get_api_key('deepseek', config)
    
    if not api_key:
        raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable or add to config.")
    
    # 创建重试配置
    retry_config = RetryConfig.from_config(config, 'deepseek') if config else RetryConfig()
    
    # DeepSeek uses OpenAI-compatible API with custom base URL
    return OpenAIModel(
        model_name=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        timeout=kwargs.pop('timeout', AgentDefaults.TIMEOUT_DEEPSEEK),
        retry_config=retry_config,
        **kwargs
    )

class AgentLightningLLMModel(BaseModel):
    def __init__(self, resource: any, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.resource = resource
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def chat_completion(self, messages: list[dict], tools: list[dict] | None = None, temperature: float = 0.1, max_tokens: int = 4096, **kwargs) -> ModelResponse:
        api_params = {
            "model": getattr(self.resource, "model", None) or self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            api_params["tools"] = tools
        api_params.update(kwargs)

        result = None
        if hasattr(self.resource, "client") and hasattr(self.resource.client, "chat") and hasattr(self.resource.client.chat, "completions") and hasattr(self.resource.client.chat.completions, "create"):
            result = self.resource.client.chat.completions.create(**api_params)
            message = result.choices[0].message
            response = ModelResponse(content=message.content, role=message.role)
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls_list = []
                for tc in message.tool_calls:
                    tool_calls_list.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    })
                response.tool_calls = tool_calls_list
                tool_call = message.tool_calls[0]
                response.function_call = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            if hasattr(result, "usage") and result.usage:
                usage = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }
                response.usage = usage
                for k in self._usage:
                    self._usage[k] += usage.get(k, 0)
            return response
        elif hasattr(self.resource, "chat"):
            response_dict = self.resource.chat(messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens, **kwargs)
            content = None
            tool_calls = None
            usage = None
            if isinstance(response_dict, dict):
                content = response_dict.get("content")
                tool_calls = response_dict.get("tool_calls")
                usage = response_dict.get("usage")
            response = ModelResponse(content=content, role="assistant")
            if tool_calls:
                response.tool_calls = tool_calls
                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                    tc = tool_calls[0]
                    if isinstance(tc, dict) and "function" in tc:
                        response.function_call = {
                            "name": tc["function"].get("name"),
                            "arguments": tc["function"].get("arguments"),
                        }
            if usage:
                response.usage = usage
                for k in self._usage:
                    self._usage[k] += usage.get(k, 0)
            return response
        raise RuntimeError("Unsupported LLM resource interface")

    def get_usage_stats(self) -> dict[str, int]:
        return self._usage.copy()

def _create_agentlightning_model(model_name: str, config: Optional[Config], **kwargs) -> BaseModel:
    resource = None
    if config:
        resource = config.get('model.agent.llm_resource', None)
    if resource is not None:
        return AgentLightningLLMModel(resource, model_name, **kwargs)
    retry_config = RetryConfig.from_config(config, 'openai') if config else RetryConfig()
    return OpenAIModel(
        model_name=model_name,
        api_key='sk-123',
        base_url=config.get('model.agent.endpoint', 'http://localhost:8000/v1') if config else None,
        retry_config=retry_config,
        **kwargs
    )

def _create_zhipuai_model(model_name: str, config: Optional[Config], **kwargs) -> BaseModel:
    """Create ZhipuAI model instance."""
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError("zhipuai package not installed. Install with: pip install zhipuai")
    
    api_key = kwargs.pop('api_key', None) or get_api_key('zhipuai', config)
    
    if not api_key:
        raise ValueError("ZhipuAI API key not found. Set ZHIPUAI_API_KEY environment variable or add to config.")
    
    # For now, wrap ZhipuAI in OpenAI model (they have similar interfaces)
    # TODO: Create dedicated ZhipuAI model class if needed
    raise NotImplementedError(
        "ZhipuAI model is not yet implemented in the new architecture. "
        "Please use OpenAI or DeepSeek for now."
    )


def _create_vllm_model(model_name: str, config: Optional[Config], **kwargs) -> VLLMModel:
    """Create vLLM model instance."""
    # Get vLLM configuration
    server_url = kwargs.pop('server_url', None)
    timeout = kwargs.pop('timeout', None)
    
    if not server_url and config:
        server_url = config.get('vllm.server.base_url', 'http://localhost:8000')
    
    if not server_url:
        raise ValueError("vLLM server URL not provided. Set in config or pass as parameter.")
    
    # Get timeout from config if not provided
    if timeout is None and config:
        timeout = config.get('vllm.server.timeout', 360)
    
    # Get sampling configuration
    if config and 'sampling' not in kwargs:
        kwargs['sampling'] = config.get('vllm.sampling', {})
    
    return VLLMModel(
        model_name=model_name,
        server_url=server_url,
        timeout=timeout,
        **kwargs
    )


def create_model_from_config(config: Config, model_type: str = 'agent') -> BaseModel:
    """
    Create model directly from configuration.
    
    Args:
        config: Configuration object
        model_type: 'agent' or 'user' (determines which config section to use)
        
    Returns:
        BaseModel instance
        
    Example:
        >>> config = load_config('configs/my_config.yaml')
        >>> agent_model = create_model_from_config(config, model_type='agent')
        >>> user_model = create_model_from_config(config, model_type='user')
    """
    # 支持 Config 对象或 dict
    from taskdialogue.core.utils.config import Config
    if not isinstance(config, Config):
        from taskdialogue.core.utils.config import Config as ConfigClass
        config = ConfigClass(config)
    
    # Try new config structure first: model.agent.* or model.user.*
    config_prefix = f'model.{model_type}.'
    provider = config.get(
        f'{config_prefix}provider',
        config.get('model.provider', 'openai')
    )
    model_name = config.get(
        f'{config_prefix}name',
        config.get('model.name', 'gpt-3.5-turbo')
    )
    print(f'create {model_type} model {model_name} with provider {provider}')
    return create_model(provider, model_name, config=config, model_type=model_type)

