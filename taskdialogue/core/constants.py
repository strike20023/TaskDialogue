"""
常量定义
"""


class AgentDefaults:
    """Agent 默认配置"""
    
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TIMEOUT = 30
    
    # Provider-specific timeouts
    TIMEOUT_OPENAI = 30
    TIMEOUT_DEEPSEEK = 60
    TIMEOUT_ZHIPUAI = 30
    TIMEOUT_VLLM = 360
    
    @staticmethod
    def get_timeout(provider: str) -> int:
        """获取提供商的默认超时时间"""
        timeouts = {
            'openai': AgentDefaults.TIMEOUT_OPENAI,
            'deepseek': AgentDefaults.TIMEOUT_DEEPSEEK,
            'zhipuai': AgentDefaults.TIMEOUT_ZHIPUAI,
            'vllm': AgentDefaults.TIMEOUT_VLLM,
        }
        return timeouts.get(provider.lower(), AgentDefaults.DEFAULT_TIMEOUT)


class Colors:
    """终端颜色"""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    MAGENTA = "\033[95m"  # 同 PURPLE
    CYAN = "\033[96m"

