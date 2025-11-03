"""
重试机制工具

提供统一的重试装饰器，用于处理 API 调用的超时和错误重试。
"""

import time
import functools
from typing import Callable, Optional, Tuple, Type
from taskdialogue.core.utils.logger import get_logger

logger = get_logger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    max_delay: float = 60.0,
    retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    log_attempts: bool = True
):
    """
    带指数退避的重试装饰器
    
    Args:
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        exponential_base: 指数基数
        max_delay: 最大延迟（秒）
        retry_on_exceptions: 需要重试的异常类型（None 表示所有异常）
        log_attempts: 是否记录重试日志
        
    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def call_api():
            return client.chat.completions.create(...)
    """
    if retry_on_exceptions is None:
        # 默认重试常见的网络和 API 错误
        retry_on_exceptions = (
            TimeoutError,
            ConnectionError,
            Exception  # 捕获所有异常，但会在日志中记录
        )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retry_on_exceptions as e:
                    last_exception = e
                    
                    # 最后一次尝试失败，抛出异常
                    if attempt == max_retries:
                        if log_attempts:
                            logger.error(f"❌ All {max_retries + 1} attempts failed for {func.__name__}")
                        raise
                    
                    # 计算延迟时间
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    
                    # 记录日志
                    if log_attempts:
                        logger.warning(
                            f"⚠️  Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {type(e).__name__}: {str(e)[:100]}"
                        )
                        logger.info(f"   Retrying in {delay:.1f}s...")
                    
                    # 等待后重试
                    time.sleep(delay)
            
            # 理论上不会到这里，但为了安全
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class RetryConfig:
    """重试配置类"""
    
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        max_delay: float = 60.0,
        timeout: int = 60,
        log_attempts: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.timeout = timeout
        self.log_attempts = log_attempts
    
    @classmethod
    def from_config(cls, config: dict, provider: str = "default") -> "RetryConfig":
        """从配置字典创建 RetryConfig
        
        Args:
            config: 配置字典（可能包含 api.{provider} 配置）
            provider: 提供商名称（openai, deepseek, zhipuai 等）
            
        Returns:
            RetryConfig 实例
        """
        # 尝试从 api.{provider} 读取
        api_config = config.get("api", {}).get(provider, {})
        
        # 尝试从 retry 全局配置读取
        retry_config = config.get("retry", {})
        
        return cls(
            max_retries=api_config.get("max_retries") or retry_config.get("max_retries", 5),
            initial_delay=retry_config.get("initial_delay", 1.0),
            exponential_base=retry_config.get("exponential_base", 2.0),
            max_delay=retry_config.get("max_delay", 60.0),
            timeout=api_config.get("timeout") or retry_config.get("timeout", 60),
            log_attempts=retry_config.get("log_attempts", True)
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "max_retries": self.max_retries,
            "initial_delay": self.initial_delay,
            "exponential_base": self.exponential_base,
            "max_delay": self.max_delay,
            "timeout": self.timeout,
            "log_attempts": self.log_attempts
        }

