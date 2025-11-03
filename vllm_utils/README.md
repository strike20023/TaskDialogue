# vLLM 推理工具库

为 AutoTOD 和 tau-bench 项目提供统一的 vLLM 推理部署功能。

## 功能特性

✨ **两种部署模式**
- **直接 Load 模式**: 使用 vLLM 的 `LLM` 类直接加载模型到 GPU 内存
- **OpenAI 服务模式**: 通过 OpenAI 客户端连接到 vLLM 服务器

🚀 **高性能推理**
- 支持张量并行 (Tensor Parallelism)
- 支持批量推理优化
- 灵活的 GPU 内存管理

🔧 **易于集成**
- OpenAI API 兼容接口
- 支持 Function/Tool Calling
- 统一的配置管理

## 安装依赖

```bash
# 安装 vLLM
pip install vllm

# 安装 OpenAI 客户端（用于服务模式）
pip install openai

# 安装 transformers（用于 tokenizer）
pip install transformers
```

## 快速开始

### 模式 1: 直接 Load 模式

```python
from vllm_utils import VLLMInference
from vllm_utils.config import VLLMConfig, SamplingConfig

# 配置
vllm_config = VLLMConfig(
    model_path="/path/to/model",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

sampling_config = SamplingConfig(
    temperature=0.01,
    max_tokens=512,
)

# 初始化推理引擎
inference = VLLMInference(
    model_path="/path/to/model",
    vllm_config=vllm_config,
    sampling_config=sampling_config,
)

# 进行推理
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好！"}
]

response = inference.chat_completion(messages)
print(response)
```

### 模式 2: OpenAI 服务模式

**步骤 1: 启动 vLLM 服务器**

```bash
# 使用提供的脚本启动
bash start_vllm_server.sh

# 或手动启动
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

**步骤 2: 使用客户端调用**

```python
from vllm_utils import VLLMServerInference
from vllm_utils.config import ServerConfig, SamplingConfig

# 配置
server_config = ServerConfig(
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
)

sampling_config = SamplingConfig(
    temperature=0.01,
    max_tokens=512,
)

# 初始化客户端
inference = VLLMServerInference(
    server_config=server_config,
    sampling_config=sampling_config,
)

# 进行推理
messages = [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好！"}
]

response = inference.chat_completion(messages)
print(response)
```

## Function Calling 支持

两种模式都支持 OpenAI 格式的 Function/Tool Calling：

```python
messages = [
    {"role": "system", "content": "你是一个旅行助手。"},
    {"role": "user", "content": "我想在剑桥找一家酒店。"}
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_hotel",
            "description": "搜索符合条件的酒店",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {"type": "string", "description": "酒店所在区域"},
                    "price": {"type": "string", "description": "价格范围"},
                },
                "required": ["area"]
            }
        }
    }
]

response = inference.chat_completion(messages, tools=tools)
```

## 测试

运行测试脚本验证两种模式：

```bash
# 测试直接模式
python test_vllm_inference.py --mode direct

# 测试服务模式（需先启动服务器）
python test_vllm_inference.py --mode server

# 测试两种模式
python test_vllm_inference.py --mode both
```

## 在 AutoTOD 中使用

在 AutoTOD 配置文件中添加 vLLM 配置：

```yaml
api:
  vllm:
    mode: direct  # 或 server
    model_path: /path/to/model
    server_url: http://localhost:8000/v1
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9

agent:
  model_type: vllm  # 或 vllm_server
  model: /path/to/model
```

然后正常运行推理：

```bash
python scripts/run_inference.py --config configs/vllm_config.yaml
```

## 在 tau-bench 中使用

在 tau-bench 配置文件中添加 vLLM 配置：

```yaml
agent:
  model_provider: vllm  # 或 vllm_server
  model: /path/to/model
  vllm_config:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
```

然后正常运行推理：

```bash
python scripts/run_inference.py --config configs/vllm_config.yaml
```

## 配置参数说明

### VLLMConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | 必需 | 模型路径 |
| `tensor_parallel_size` | int | 1 | 张量并行数（GPU 数量） |
| `pipeline_parallel_size` | int | 1 | 流水线并行数 |
| `gpu_memory_utilization` | float | 0.9 | GPU 内存使用率 (0-1) |
| `max_model_len` | int | None | 最大序列长度 |
| `dtype` | str | "auto" | 数据类型 (auto/float16/bfloat16/float32) |
| `trust_remote_code` | bool | True | 是否信任远程代码 |
| `enable_lora` | bool | False | 是否启用 LoRA |

### SamplingConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 0.01 | 采样温度 |
| `top_p` | float | 0.95 | Top-p 采样 |
| `top_k` | int | -1 | Top-k 采样 (-1 表示不使用) |
| `max_tokens` | int | 512 | 最大生成 token 数 |
| `repetition_penalty` | float | 1.0 | 重复惩罚 |
| `stop_token_ids` | List[int] | None | 停止 token IDs |
| `skip_special_tokens` | bool | True | 跳过特殊 token |
| `seed` | int | None | 随机种子 |

### ServerConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | "0.0.0.0" | 服务器主机地址 |
| `port` | int | 8000 | 服务器端口 |
| `api_base` | str | None | API Base URL |
| `api_key` | str | None | API Key（可选） |
| `timeout` | int | 120 | 请求超时时间（秒） |

## 两种模式对比

| 特性 | 直接 Load 模式 | OpenAI 服务模式 |
|------|---------------|-----------------|
| **部署复杂度** | 简单 | 需要启动服务器 |
| **性能** | 最高（无网络开销） | 略有网络开销 |
| **并发支持** | 需要手动管理 | 自动支持多并发 |
| **资源共享** | GPU 独占 | 多进程共享 |
| **适用场景** | 单机、批量推理 | 多进程、多用户 |
| **易用性** | 直接调用 | 需要服务管理 |

## 最佳实践

### 选择合适的模式

- **直接 Load 模式**: 适合单机推理、批量处理、追求最高性能
- **OpenAI 服务模式**: 适合多进程并发、多用户场景、易于扩展

### 内存管理

- 设置 `gpu_memory_utilization=0.9` 为默认值
- 如果 OOM，降低到 0.8 或 0.7
- 如果需要同时运行多个模型，相应降低该值

### 性能优化

- 使用 `tensor_parallel_size` 进行多 GPU 并行
- 批量推理时使用 `batch_chat_completion`
- 设置合适的 `max_model_len` 避免不必要的内存分配

## 故障排除

### 问题 1: CUDA Out of Memory

**解决方案**: 降低 `gpu_memory_utilization`

```python
vllm_config = VLLMConfig(
    model_path="/path/to/model",
    gpu_memory_utilization=0.7,  # 降低到 0.7
)
```

### 问题 2: 服务器无法连接

**解决方案**: 
1. 检查服务器是否启动: `curl http://localhost:8000/health`
2. 检查防火墙设置
3. 确认端口未被占用

### 问题 3: Function Calling 不生效

**解决方案**: 
1. 确保模型支持 Function Calling（如 Qwen2.5）
2. 检查工具定义格式是否正确
3. 查看模型输出是否包含 JSON 格式的函数调用

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目使用 Apache 2.0 许可证。

