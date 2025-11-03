"""
vLLM 推理核心模块

支持两种模式：
1. VLLMInference: 直接 Load 模式，使用 vLLM LLM 类
2. VLLMServerInference: OpenAI 服务模式，连接 vLLM 服务器
"""

import json
import warnings
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .config import VLLMConfig, SamplingConfig, ServerConfig


class VLLMInference:
    """
    vLLM 直接推理模式
    
    直接加载模型到 GPU 内存，适合单机高性能推理
    """
    
    def __init__(
        self,
        model_path: str,
        vllm_config: Optional[VLLMConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
        verbose: bool = True,
    ):
        """
        初始化 vLLM 推理引擎
        
        Args:
            model_path: 模型路径
            vllm_config: vLLM 配置
            sampling_config: 采样配置
            verbose: 是否显示详细信息
        """
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"请先安装 vLLM 和 transformers: pip install vllm transformers\n原始错误: {e}"
            )
        
        self.model_path = model_path
        self.verbose = verbose
        
        # 默认配置
        if vllm_config is None:
            vllm_config = VLLMConfig(model_path=model_path)
        else:
            vllm_config.model_path = model_path
        
        self.vllm_config = vllm_config
        
        if sampling_config is None:
            sampling_config = SamplingConfig()
        self.sampling_config = sampling_config
        
        # 加载 tokenizer
        if self.verbose:
            print(f"📦 加载 tokenizer: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=vllm_config.trust_remote_code,
        )
        
        # 加载 vLLM 模型
        if self.verbose:
            print(f"🚀 加载 vLLM 模型: {model_path}")
            print(f"   - Tensor Parallel Size: {vllm_config.tensor_parallel_size}")
            print(f"   - GPU Memory Utilization: {vllm_config.gpu_memory_utilization}")
            print(f"   - DType: {vllm_config.dtype}")
        
        engine_args = vllm_config.to_engine_args()
        
        # 添加额外的稳定性参数
        if 'max_model_len' not in engine_args or engine_args['max_model_len'] is None:
            # 如果没有指定，让 vLLM 自动推断
            engine_args.pop('max_model_len', None)
        
        if self.verbose:
            print(f"   - Engine Args: {engine_args}")
        
        try:
            self.llm = LLM(**engine_args)
        except Exception as e:
            print(f"\n❌ vLLM 初始化失败！")
            print(f"错误信息: {e}")
            print(f"\n建议:")
            print(f"  1. 检查 GPU 内存是否足够")
            print(f"  2. 尝试降低 gpu_memory_utilization (当前: {vllm_config.gpu_memory_utilization})")
            print(f"  3. 尝试移除或降低 max_model_len")
            print(f"  4. 检查模型路径是否正确: {model_path}")
            raise
        
        if self.verbose:
            print("✅ vLLM 模型加载完成")
        
        # API 信息
        self.api_type = "vllm_direct"
        self.model_name = Path(model_path).name
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        聊天补全接口（OpenAI 兼容）
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: Top-p 采样
            tools: 工具定义（Function Calling）
            return_usage: 是否返回使用信息
            **kwargs: 其他参数
        
        Returns:
            str 或 dict: 生成的内容或包含使用信息的字典
        """
        # 应用模板转换 messages 为 prompt
        prompt = self._apply_chat_template(messages, tools)
        
        # 准备采样参数
        sampling_config = self._prepare_sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # 生成
        outputs = self.llm.generate([prompt], sampling_config)
        
        # 提取结果
        if len(outputs) == 0:
            raise ValueError("vLLM 未生成任何输出")
        
        output = outputs[0]
        generated_text = output.outputs[0].text
        
        # 处理 Function Calling
        if tools is not None:
            # 尝试解析 tool calls
            parsed_result = self._parse_tool_calls(generated_text)
            if parsed_result.get("tool_calls"):
                if return_usage:
                    return {
                        "content": parsed_result.get("content", ""),
                        "tool_calls": parsed_result["tool_calls"],
                        "usage": self._get_usage_from_output(output),
                    }
                return parsed_result
        
        if return_usage:
            return {
                "content": generated_text,
                "usage": self._get_usage_from_output(output),
            }
        
        return generated_text
    
    def batch_chat_completion(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        批量聊天补全
        
        Args:
            messages_list: 消息列表的列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: Top-p 采样
            **kwargs: 其他参数
        
        Returns:
            List[str]: 生成的内容列表
        """
        # 批量应用模板
        prompts = [self._apply_chat_template(messages) for messages in messages_list]
        
        # 准备采样参数
        sampling_config = self._prepare_sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )
        
        # 批量生成
        outputs = self.llm.generate(prompts, sampling_config)
        
        # 提取结果
        results = [output.outputs[0].text for output in outputs]
        return results
    
    def _apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """应用聊天模板"""
        # 如果有 tools，将其注入到 system message 或使用特定格式
        if tools is not None:
            # 对于支持 Function Calling 的模型（如 Qwen），使用特定格式
            messages = self._inject_tools_to_messages(messages, tools)
        
        # 使用 tokenizer 的 chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            warnings.warn(f"应用 chat template 失败: {e}，使用简单拼接")
            prompt = self._simple_chat_format(messages)
        
        return prompt
    
    def _inject_tools_to_messages(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """将工具定义注入到消息中"""
        # 构建工具描述
        tools_description = "You have access to the following functions:\n\n"
        for tool in tools:
            if "function" in tool:
                func = tool["function"]
                tools_description += f"Function: {func['name']}\n"
                tools_description += f"Description: {func.get('description', 'N/A')}\n"
                tools_description += f"Parameters: {json.dumps(func.get('parameters', {}), indent=2)}\n\n"
        
        tools_description += "\nTo use a function, respond with a JSON object in this format:\n"
        tools_description += '{"name": "function_name", "arguments": {...}}\n\n'
        
        # 注入到第一个 system message，或创建新的 system message
        new_messages = []
        system_added = False
        
        for msg in messages:
            if msg["role"] == "system" and not system_added:
                new_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n" + tools_description
                })
                system_added = True
            else:
                new_messages.append(msg)
        
        # 如果没有 system message，添加一个
        if not system_added:
            new_messages.insert(0, {
                "role": "system",
                "content": tools_description
            })
        
        return new_messages
    
    def _parse_tool_calls(self, text: str) -> Dict[str, Any]:
        """解析工具调用"""
        # 尝试解析 JSON 格式的 function call
        import re
        
        # 查找 JSON 对象
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            tool_calls = []
            for match in matches:
                try:
                    call_data = json.loads(match)
                    tool_calls.append({
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data["arguments"])
                        }
                    })
                except json.JSONDecodeError:
                    continue
            
            if tool_calls:
                return {
                    "content": "",
                    "tool_calls": tool_calls
                }
        
        # 没有找到 tool calls
        return {"content": text}
    
    def _simple_chat_format(self, messages: List[Dict[str, str]]) -> str:
        """简单的消息格式化"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _prepare_sampling_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ):
        """准备采样参数"""
        from vllm import SamplingParams
        
        # 使用传入的参数或默认值
        config = self.sampling_config
        
        params = {
            "temperature": temperature if temperature is not None else config.temperature,
            "top_p": top_p if top_p is not None else config.top_p,
            "top_k": kwargs.get("top_k", config.top_k),
            "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
            "repetition_penalty": kwargs.get("repetition_penalty", config.repetition_penalty),
            "skip_special_tokens": kwargs.get("skip_special_tokens", config.skip_special_tokens),
        }
        
        # 处理边界值
        if params["top_p"] <= 0:
            params["top_p"] = 1.0
        if params["top_k"] <= 0:
            params["top_k"] = -1
        if params["repetition_penalty"] <= 0:
            params["repetition_penalty"] = 1.0
        
        # 添加 stop_token_ids
        if config.stop_token_ids is not None:
            params["stop_token_ids"] = config.stop_token_ids
        
        if config.seed is not None:
            params["seed"] = config.seed
        
        return SamplingParams(**params)
    
    def _get_usage_from_output(self, output) -> Dict[str, int]:
        """从输出中提取 token 使用信息"""
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = sum(len(out.token_ids) for out in output.outputs)
        
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    
    def get_api_info(self) -> Dict[str, str]:
        """获取 API 信息"""
        return {
            "api_type": self.api_type,
            "model_name": self.model_name,
            "model_path": self.model_path,
        }


class VLLMServerInference:
    """
    vLLM OpenAI 服务模式
    
    连接到 vLLM OpenAI 兼容服务器，适合多进程/多用户场景
    """
    
    def __init__(
        self,
        server_config: Optional[ServerConfig] = None,
        model_name: Optional[str] = None,
        sampling_config: Optional[SamplingConfig] = None,
        verbose: bool = True,
    ):
        """
        初始化 vLLM 服务客户端
        
        Args:
            server_config: 服务器配置
            model_name: 模型名称
            sampling_config: 采样配置
            verbose: 是否显示详细信息
        """
        try:
            import openai
        except ImportError:
            raise ImportError("请先安装 openai: pip install openai")
        
        self.verbose = verbose
        
        # 默认配置
        if server_config is None:
            server_config = ServerConfig()
        self.server_config = server_config
        
        if sampling_config is None:
            sampling_config = SamplingConfig()
        self.sampling_config = sampling_config
        
        # 创建 OpenAI 客户端
        if self.verbose:
            print(f"🔗 连接到 vLLM 服务器: {server_config.get_base_url()}")
        
        self.client = openai.OpenAI(
            api_key=server_config.api_key or "EMPTY",
            base_url=server_config.get_base_url(),
            timeout=server_config.timeout,
        )
        
        # 获取模型列表
        try:
            models = self.client.models.list()
            available_models = [model.id for model in models.data]
            
            if model_name is None:
                # 使用第一个可用模型
                self.model_name = available_models[0] if available_models else "unknown"
            else:
                self.model_name = model_name
            
            if self.verbose:
                print(f"✅ 连接成功，可用模型: {available_models}")
                print(f"   使用模型: {self.model_name}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  无法获取模型列表: {e}")
            self.model_name = model_name or "unknown"
        
        # API 信息
        self.api_type = "vllm_server"
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        return_usage: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        聊天补全接口（OpenAI 兼容）
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            top_p: Top-p 采样
            tools: 工具定义（Function Calling）
            return_usage: 是否返回使用信息
            **kwargs: 其他参数
        
        Returns:
            str 或 dict: 生成的内容或包含使用信息的字典
        """
        # 准备请求参数
        config = self.sampling_config
        
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
            "top_p": top_p if top_p is not None else config.top_p,
        }
        
        # 添加工具
        if tools is not None:
            request_params["tools"] = tools
        
        # 添加其他参数
        if "frequency_penalty" in kwargs:
            request_params["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            request_params["presence_penalty"] = kwargs["presence_penalty"]
        
        # 调用 API
        response = self.client.chat.completions.create(**request_params)
        
        # 提取结果
        message = response.choices[0].message
        
        # 检查是否有 tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                {
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in message.tool_calls
            ]
            
            if return_usage:
                return {
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                }
            
            return {
                "content": message.content or "",
                "tool_calls": tool_calls,
            }
        
        content = message.content
        
        if return_usage:
            return {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            }
        
        return content
    
    def check_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            import requests
            response = requests.get(
                self.server_config.get_health_url(),
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            if self.verbose:
                print(f"⚠️  健康检查失败: {e}")
            return False
    
    def get_api_info(self) -> Dict[str, str]:
        """获取 API 信息"""
        return {
            "api_type": self.api_type,
            "model_name": self.model_name,
            "server_url": self.server_config.get_base_url(),
        }

