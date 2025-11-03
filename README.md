# TaskDialogue

> 统一的任务导向对话系统 Benchmark 框架

**项目路径**: `/mnt/ST8000/zhenhanbai/Reposi/TaskDialogue/`  
**版本**: v2.0  
**状态**: ✅ 完整实现，准备测试

---

## 🎯 支持的 Benchmarks

- **MultiWOZ**: 多领域任务导向对话（Restaurant, Hotel, Attraction, Train, Taxi）
- **Tau2**: 客服场景对话评测（Airline, Retail, Telecom）

---

## 📦 快速开始

### 安装

```bash
cd /mnt/ST8000/zhenhanbai/Reposi/TaskDialogue
pip install -e .
```

### 运行 MultiWOZ

```bash
# 快速测试（3个对话）
./run_multiwoz.sh -n 3

# 完整流程
./run_multiwoz.sh

# 使用 Python CLI
python -m taskdialogue multiwoz -c configs/multiwoz/default.yaml -n 10
```

### 运行 Tau2

```bash
# 运行 airline 领域
./run_tau2.sh -d airline -n 5 --use-native
```

---

## 💻 CLI 命令

```bash
# 查看帮助
python -m taskdialogue --help

# MultiWOZ
python -m taskdialogue multiwoz -c configs/multiwoz/default.yaml [选项]
  -m, --mode MODE       运行模式: full|inference|evaluation
  -n, --num-tasks N     限制任务数量
  --task-ids IDS        指定任务ID

# Tau2
python -m taskdialogue tau2 -c configs/tau2/default.yaml -d DOMAIN [选项]
  -d, --domain DOMAIN   领域: airline|retail|telecom
  -n, --num-tasks N     限制任务数量
  --use-native          使用原生实现
```

---

## 🏗️ 项目结构

```
TaskDialogue/
├── taskdialogue/            # 主代码
│   ├── core/               # 核心抽象层（通用组件）
│   │   ├── base/           # 基类（BaseAgent, BaseEvaluator等）
│   │   ├── schemas/        # 数据模型（Pydantic）
│   │   ├── models/         # 模型系统（OpenAI, vLLM等）
│   │   └── utils/          # 工具（Config, Logger等）
│   ├── benchmarks/          # Benchmark实现（完全独立）
│   │   ├── multiwoz/       # MultiWOZ（完整实现）
│   │   └── tau2/           # Tau2（完整实现）
│   └── cli/                # 统一CLI
├── configs/                # 配置文件
│   ├── multiwoz/
│   │   ├── default.yaml         # API模式
│   │   └── custom_format.yaml   # vLLM模式
│   └── tau2/
├── run_multiwoz.sh         # MultiWOZ脚本
├── run_tau2.sh             # Tau2脚本
└── README.md               # 本文档
```

---

## ⚙️ 配置

### API Key设置（直接在配置文件中）

编辑 `configs/multiwoz/default.yaml`:

```yaml
api:
  deepseek:
    api_key: "your-api-key-here"  # 直接填写
  openai:
    api_key: "sk-..."
```

### 模型配置

```yaml
model:
  agent:
    provider: "deepseek"  # openai, deepseek, vllm
    name: "deepseek-chat"
    temperature: 0.3
```

### Agent类型

```yaml
agent:
  type: "function_calling"  # 或 "custom_format" (vLLM)
```

---

## 📊 评估指标

### 6个完整指标

1. **inform** - 信息提供准确率（基础）
2. **success** - 任务完成率（基础）  
3. **combined_score** - 综合得分 = 0.5×inform + 0.5×success（主要指标）
4. **book** - 预订成功率（辅助）
5. **jga** - 联合目标准确率（槽位级）
6. **f1** - Slot F1分数（槽位级）

---

## 🎓 添加新Benchmark

### 1. 创建目录

```bash
mkdir -p taskdialogue/benchmarks/my_benchmark/{agents,evaluators,tools,data}
```

### 2. 实现组件

```python
# agents/agent.py
from taskdialogue.core.base.agent import BaseAgent

class MyAgent(BaseAgent):
    def generate_response(self, user_input, context):
        # 实现逻辑
        ...
    def reset(self):
        ...

# evaluators/evaluator.py  
from taskdialogue.core.base.evaluator import BaseEvaluator

class MyEvaluator(BaseEvaluator):
    def evaluate(self, inference_result, ground_truth):
        # 独立的评估逻辑（不共享）
        ...
        return EvaluationResult(...)

# pipeline.py
from taskdialogue.core.base.pipeline import BasePipeline

class MyPipeline(BasePipeline):
    def run_inference(self, task_ids):
        ...
    def run_evaluation(self, inference_results):
        ...
```

### 3. 注册CLI

```python
# taskdialogue/cli/main.py
@cli.command()
def my_benchmark(config):
    from taskdialogue.benchmarks.my_benchmark import MyPipeline
    pipeline = MyPipeline(config)
    pipeline.run_full_pipeline()
```

---

## 📁 输出格式

### 推理结果

```json
{
  "benchmark": "multiwoz",
  "task_id": "MUL0001",
  "num_turns": 15,
  "dialogue_history": [...],
  "timestamp": "2025-10-30T12:00:00Z"
}
```

### 评估结果

```json
{
  "benchmark": "multiwoz",
  "task_id": "MUL0001",
  "overall_score": 0.85,
  "metrics": [
    {"name": "inform", "value": 0.90},
    {"name": "success", "value": 0.80},
    {"name": "combined_score", "value": 0.85},
    {"name": "book", "value": 1.0},
    {"name": "jga", "value": 0.75},
    {"name": "f1", "value": 0.88}
  ]
}
```

---

## ✨ 架构特点

- ✅ **完全解耦** - 每个benchmark独立实现评估逻辑
- ✅ **统一接口** - 通过基类和统一数据格式通信
- ✅ **类型安全** - Pydantic数据验证
- ✅ **功能完整** - 保留所有旧功能

---

## 🔧 常见问题

### Q: 如何使用vLLM？

A: 使用 shell 脚本的 `--use-vllm --custom-format` 参数：

```bash
./run_multiwoz.sh --use-vllm \
  --vllm-model-path /path/to/model \
  --custom-format -n 10
```

### Q: 评估指标在哪里？

A: `results/multiwoz/evaluation/` 目录

### Q: 如何添加新模型？

A: 编辑配置文件的 `model.agent` 部分

---

## 📝 开发

详细架构和旧代码参考见 `.backup_v1/` 目录。

---

## 📜 License

MIT License

---

**TaskDialogue - 让任务导向对话系统开发更简单** 🚀
