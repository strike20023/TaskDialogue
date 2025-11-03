# 配置文件说明

## 📁 配置文件结构

```
configs/
├── multiwoz/
│   ├── default.yaml          # MultiWOZ 默认配置（Function Calling）
│   └── custom_format.yaml    # MultiWOZ Custom Format 配置（vLLM）
└── tau2/
    └── default.yaml          # Tau2 默认配置
```

---

## 🎯 使用场景

### MultiWOZ

#### 场景 1: 使用 API（OpenAI/DeepSeek）

**配置文件**: `multiwoz/default.yaml`

**关键配置**:
```yaml
model:
  agent:
    provider: "deepseek"  # 或 "openai"
    name: "deepseek-chat"

agent:
  type: "function_calling"  # 使用原生 function calling

api:
  deepseek:
    api_key: "your-api-key"
```

**运行**:
```bash
./run_multiwoz.sh -n 10
```

---

#### 场景 2: 使用 vLLM（开源模型）

**配置文件**: `multiwoz/custom_format.yaml`

**关键配置**:
```yaml
model:
  agent:
    provider: "vllm"
    name: "Qwen3-4B-Instruct-2507"

agent:
  type: "custom_format"  # 使用特殊 token

vllm:
  server:
    base_url: "http://localhost:8000"
  model_path: "/path/to/your/model"
  engine:
    enable_auto_tool_choice: false  # Custom format 不用原生
```

**运行**:
```bash
./run_multiwoz.sh -c configs/multiwoz/custom_format.yaml -n 10
```

---

### Tau2

#### 场景 3: Tau2 Benchmark

**配置文件**: `tau2/default.yaml`

**关键配置**:
```yaml
tau2:
  domain: airline  # airline, retail, telecom
  trials:
    num_tasks: 50

model:
  agent:
    provider: "openai"
    name: "gpt-4"
```

**运行**:
```bash
./run_tau2.sh -d airline -n 10 --use-native
```

---

## ⚙️ 配置参数详解

### 1. 模型配置

```yaml
model:
  agent:
    provider: "deepseek"     # 提供商: openai, deepseek, zhipuai, vllm
    name: "deepseek-chat"    # 模型名称
    temperature: 0.3         # 采样温度 (0-1，越低越确定)
    max_tokens: 2048         # 最大生成 token 数
    top_p: 0.95              # Nucleus sampling
```

### 2. Agent 配置

```yaml
agent:
  type: "function_calling"   # function_calling 或 custom_format
  max_turns: 30              # 最大对话轮数
  max_tool_iterations: 5     # 单轮最大工具调用次数
```

### 3. 评估配置

```yaml
evaluation:
  eval_model: "deepseek-chat"  # 评估使用的 LLM
  metrics:
    - inform    # 信息提供准确率
    - success   # 任务完成率
    - book      # 预订成功率
    - jga       # 联合目标准确率
    - f1        # F1 分数
  success_mode: "strict"  # strict 或 relaxed
```

### 4. vLLM 配置

```yaml
vllm:
  server:
    base_url: "http://localhost:8000"  # vLLM 服务器地址
    port: 8000
  model_path: "/path/to/model"  # 本地模型路径
  engine:
    tensor_parallel_size: 2  # GPU 并行数
    gpu_memory_utilization: 0.75  # 显存利用率
```

---

## 🔧 常见配置组合

### 配置 1: 快速测试（API）

```yaml
data:
  num_tasks: 5  # 只运行 5 个对话

model:
  agent:
    provider: "deepseek"
    name: "deepseek-chat"
```

### 配置 2: 完整评估（API）

```yaml
data:
  num_tasks: null  # 全部数据

evaluation:
  metrics:  # 完整指标
    - inform
    - success
    - book
    - jga
    - f1
```

### 配置 3: vLLM 本地推理

```yaml
model:
  agent:
    provider: "vllm"

agent:
  type: "custom_format"

vllm:
  model_path: "/path/to/model"
  engine:
    tensor_parallel_size: 2
```

---

## 📝 配置模板

### 创建新配置

```bash
# 复制默认配置
cp configs/multiwoz/default.yaml configs/multiwoz/my_config.yaml

# 编辑配置
vim configs/multiwoz/my_config.yaml

# 使用配置
./run_multiwoz.sh -c configs/multiwoz/my_config.yaml
```

---

## ⚠️ 注意事项

1. **API Key**: 直接在配置文件中填写（不使用环境变量）

2. **vLLM 模式**: 
   - `agent.type` 设为 `custom_format`
   - `vllm.engine.enable_auto_tool_choice` 设为 `false`

3. **Function Calling 模式**:
   - `agent.type` 设为 `function_calling`
   - `vllm.engine.enable_auto_tool_choice` 设为 `true`

4. **评估指标**: 
   - 完整版包含 5 个指标: inform, success, book, jga, f1
   - 可根据需要启用部分指标

---

## 📞 获取帮助

配置问题请参考：
- 主文档: `../README.md`
- 备份配置: `../.backup_v1/default.yaml`

