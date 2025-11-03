"""
vLLM 配置管理模块
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class VLLMConfig:
    """vLLM 模型配置"""
    
    # 模型路径
    model_path: str
    
    # 并行配置
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # 内存配置
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None  # 最大并发序列数
    
    # 数据类型
    dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    
    # 信任远程代码
    trust_remote_code: bool = True
    
    # LoRA 配置
    enable_lora: bool = False
    max_lora_rank: int = 64
    
    # 其他配置
    disable_log_stats: bool = True
    
    # 额外的 vLLM 引擎参数
    extra_engine_args: Dict[str, Any] = field(default_factory=dict)
    
    def to_engine_args(self) -> Dict[str, Any]:
        """转换为 vLLM 引擎参数"""
        args = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "disable_log_stats": self.disable_log_stats,
        }
        
        # 只有明确指定了 max_model_len 才添加
        # 如果为 None，让 vLLM 自动推断
        if self.max_model_len is not None and self.max_model_len > 0:
            args["max_model_len"] = self.max_model_len
        
        # 添加 max_num_seqs（控制并发数，影响内存使用）
        if self.max_num_seqs is not None and self.max_num_seqs > 0:
            args["max_num_seqs"] = self.max_num_seqs
        
        if self.enable_lora:
            args["enable_lora"] = True
            args["max_lora_rank"] = self.max_lora_rank
        
        # 添加额外参数（过滤掉 None 值）
        extra_args = {k: v for k, v in self.extra_engine_args.items() if v is not None}
        args.update(extra_args)
        
        return args


@dataclass
class SamplingConfig:
    """采样配置"""
    
    temperature: float = 0.01
    top_p: float = 0.95
    top_k: int = -1  # -1 表示不使用
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = None
    skip_special_tokens: bool = True
    seed: Optional[int] = None
    
    def to_sampling_params(self, tokenizer=None):
        """转换为 vLLM SamplingParams"""
        try:
            from vllm import SamplingParams
        except ImportError:
            raise ImportError("请先安装 vLLM: pip install vllm")
        
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p if self.top_p > 0 else 1.0,
            "top_k": self.top_k if self.top_k > 0 else -1,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty if self.repetition_penalty > 0 else 1.0,
            "skip_special_tokens": self.skip_special_tokens,
        }
        
        if self.stop_token_ids is not None:
            params["stop_token_ids"] = self.stop_token_ids
        
        if self.seed is not None:
            params["seed"] = self.seed
        
        return SamplingParams(**params)


@dataclass
class ServerConfig:
    """vLLM Server 配置"""
    
    # 服务器地址
    host: str = "0.0.0.0"
    port: int = 8000
    
    # OpenAI API Base URL
    api_base: Optional[str] = None
    
    # API Key（可选）
    api_key: Optional[str] = None
    
    # 超时设置
    timeout: int = 120
    
    def get_base_url(self) -> str:
        """获取完整的 base URL"""
        if self.api_base:
            return self.api_base
        return f"http://{self.host}:{self.port}/v1"
    
    def get_health_url(self) -> str:
        """获取健康检查 URL"""
        if self.api_base:
            # 去掉 /v1 后缀
            base = self.api_base.rstrip('/v1').rstrip('/')
            return f"{base}/health"
        return f"http://{self.host}:{self.port}/health"


def load_vllm_config_from_dict(config_dict: Dict[str, Any]) -> VLLMConfig:
    """从字典加载 vLLM 配置"""
    return VLLMConfig(
        model_path=config_dict.get("model_path", ""),
        tensor_parallel_size=config_dict.get("tensor_parallel_size", 1),
        pipeline_parallel_size=config_dict.get("pipeline_parallel_size", 1),
        gpu_memory_utilization=config_dict.get("gpu_memory_utilization", 0.9),
        max_model_len=config_dict.get("max_model_len"),
        max_num_seqs=config_dict.get("max_num_seqs"),  # 添加 max_num_seqs
        dtype=config_dict.get("dtype", "auto"),
        trust_remote_code=config_dict.get("trust_remote_code", True),
        enable_lora=config_dict.get("enable_lora", False),
        max_lora_rank=config_dict.get("max_lora_rank", 64),
        disable_log_stats=config_dict.get("disable_log_stats", True),
        extra_engine_args=config_dict.get("extra_engine_args", {}),
    )


def load_sampling_config_from_dict(config_dict: Dict[str, Any]) -> SamplingConfig:
    """从字典加载采样配置"""
    return SamplingConfig(
        temperature=config_dict.get("temperature", 0.01),
        top_p=config_dict.get("top_p", 0.95),
        top_k=config_dict.get("top_k", -1),
        max_tokens=config_dict.get("max_tokens", 512),
        repetition_penalty=config_dict.get("repetition_penalty", 1.0),
        stop_token_ids=config_dict.get("stop_token_ids"),
        skip_special_tokens=config_dict.get("skip_special_tokens", True),
        seed=config_dict.get("seed"),
    )


def load_server_config_from_dict(config_dict: Dict[str, Any]) -> ServerConfig:
    """从字典加载服务器配置"""
    return ServerConfig(
        host=config_dict.get("host", "0.0.0.0"),
        port=config_dict.get("port", 8000),
        api_base=config_dict.get("api_base") or config_dict.get("server_url"),
        api_key=config_dict.get("api_key"),
        timeout=config_dict.get("timeout", 120),
    )

