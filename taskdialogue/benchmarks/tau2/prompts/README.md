# Tau2-Bench Prompts 模块

## 📋 概述

`prompts/` 模块集中管理 Tau2-bench 的所有 prompt 和 policy，完全内嵌在 Python 脚本中，不依赖外部 .md 文件。

## 📁 模块结构

```
prompts/
├── __init__.py          # 模块导出接口
├── prompts.py           # Agent prompts + User simulator prompts (245行)
├── policies.py          # Domain policies (388行)
└── README.md            # 本文档
```

**总计: 688 行代码**

---

## 🎯 完整的 Prompt 清单

### **一、Agent Prompts（2个）**

#### 1. `AGENT_INSTRUCTION`
- **文件**: `prompts.py` 
- **长度**: 306 字符
- **用途**: Agent 基础行为规则
- **内容**:
  - 你是客服Agent
  - 每轮只能：发消息 OR 调用工具
  - 遵循 policy
  - 生成有效的 JSON

#### 2. `AGENT_SYSTEM_PROMPT_TEMPLATE`
- **文件**: `prompts.py`
- **类型**: 模板
- **用途**: 组合 instruction + domain policy
- **格式**:
  ```xml
  <instructions>
  {agent_instruction}
  </instructions>
  <policy>
  {domain_policy}
  </policy>
  ```

---

### **二、User Simulator Prompts（3个）**

#### 3. `USER_SIMULATOR_GUIDELINES` (无工具版本)
- **文件**: `prompts.py`
- **长度**: 1,449 字符
- **行数**: ~20 行
- **用途**: 用户模拟器行为指南（不使用工具）
- **核心内容**:
  - **Core Principles (5条)**
  - **Task Completion (3个特殊token)**:
    - `###STOP###` - 任务完成
    - `###TRANSFER###` - 转接
    - `###OUT-OF-SCOPE###` - 超出范围

#### 4. `USER_SIMULATOR_GUIDELINES_TOOLS` (有工具版本)
- **文件**: `prompts.py`
- **长度**: 2,747 字符
- **行数**: ~30 行
- **用途**: 用户模拟器行为指南（可使用工具）
- **增强内容**:
  - 工具调用规则（10+条）
  - 每轮只能发消息或调用工具
  - 基于真实工具结果回复
  - 不编造工具调用结果

#### 5. `USER_SIMULATOR_SYSTEM_PROMPT_TEMPLATE`
- **文件**: `prompts.py`
- **类型**: 模板
- **用途**: 组合 guidelines + 场景指令
- **格式**:
  ```xml
  {global_user_sim_guidelines}
  
  <scenario>
  {instructions}
  </scenario>
  ```

---

### **三、Domain Policies（2个+）**

#### 6. `AIRLINE_POLICY`
- **文件**: `policies.py`
- **长度**: 7,676 字符
- **行数**: 167 行
- **领域**: 航空客服
- **覆盖功能**:
  - ✈️ Book flight (预订航班)
  - ✏️ Modify flight (修改预订)
  - ❌ Cancel flight (取消航班)
  - 💰 Refunds and Compensation (退款和补偿)
- **关键规则**:
  - 3种会员等级 (regular, silver, gold)
  - 3种舱位 (basic economy, economy, business)
  - 行李额度规则
  - 旅行保险
  - 补偿政策

#### 7. `RETAIL_POLICY`
- **文件**: `policies.py`
- **长度**: 6,698 字符
- **行数**: 136 行
- **领域**: 零售客服
- **覆盖功能**:
  - 📦 Cancel/Modify pending orders (取消/修改待处理订单)
  - 🔄 Return/Exchange delivered orders (退货/换货)
  - 📍 Modify user address (修改地址)
  - ℹ️ Provide information (提供信息)
- **关键规则**:
  - 用户身份验证
  - 订单状态管理 (pending, processed, delivered, cancelled)
  - 支付方式规则
  - 商品变体管理

#### 8. `TELECOM_POLICY` (待添加)
- **文件**: `policies.py`
- **状态**: 🚧 未实现
- **领域**: 电信客服

---

## 📊 统计信息

### 代码量统计

| 文件 | 行数 | 内容 |
|------|------|------|
| `__init__.py` | 55 | 导出接口 |
| `prompts.py` | 245 | Agent + User prompts |
| `policies.py` | 388 | Domain policies |
| **总计** | **688** | **完整的prompt系统** |

### Prompt 统计

| 类别 | 数量 | 总字符数 |
|------|------|----------|
| Agent prompts | 2 | ~300 |
| User simulator prompts | 3 | ~4,200 |
| Domain policies | 2 | ~14,400 |
| **总计** | **7** | **~19,000** |

---

## 🚀 使用方式

### 1. 获取 Agent Prompt

```python
from taskdialogue.benchmarks.tau2.prompts import get_agent_system_prompt, get_domain_policy

# 获取领域策略
policy = get_domain_policy("airline")

# 生成完整的 agent prompt
agent_prompt = get_agent_system_prompt(policy)

# 使用 prompt 初始化 agent
agent = LLMAgent(system_prompt=agent_prompt, tools=tools)
```

### 2. 获取 User Simulator Prompt

```python
from taskdialogue.benchmarks.tau2.prompts import get_user_simulator_system_prompt

# 获取任务指令
instructions = task.user_instructions

# 生成完整的 user simulator prompt
user_prompt = get_user_simulator_system_prompt(
    instructions=str(instructions),
    use_tools=False  # 或 True，根据是否需要工具
)

# 使用 prompt 初始化 user simulator
user = UserSimulator(...)
```

### 3. 直接访问常量

```python
from taskdialogue.benchmarks.tau2.prompts import (
    AGENT_INSTRUCTION,
    USER_SIMULATOR_GUIDELINES,
    AIRLINE_POLICY,
    RETAIL_POLICY,
)

# 直接使用常量
print(f"Agent instruction: {len(AGENT_INSTRUCTION)} chars")
print(f"Airline policy: {len(AIRLINE_POLICY)} chars")
```

### 4. 列出可用资源

```python
from taskdialogue.benchmarks.tau2.prompts import list_available_domains, get_policy_stats

# 列出所有域
domains = list_available_domains()
print(f"Available domains: {domains}")

# 获取policy统计
stats = get_policy_stats()
for domain, stat in stats.items():
    print(f"{domain}: {stat['lines']} lines, {stat['chars']} chars")
```

---

## ✨ 核心特性

### 1. **完全自包含** ✅
- 所有 prompt 和 policy 都在 Python 脚本中
- 不依赖外部 .md 文件
- 不需要文件 I/O 操作
- 部署更简单，没有文件路径问题

### 2. **分离关注点** ✅
- `prompts.py` - Agent 和 User simulator 的行为规则
- `policies.py` - 各领域的业务策略
- 清晰的职责划分

### 3. **易于扩展** ✅
```python
# 添加新的domain policy
TELECOM_POLICY = """..."""

POLICY_REGISTRY = {
    "airline": AIRLINE_POLICY,
    "retail": RETAIL_POLICY,
    "telecom": TELECOM_POLICY,  # 新增
}
```

### 4. **类型安全** ✅
- 完整的 docstring
- 类型提示
- IDE 友好的自动完成

### 5. **版本管理** ✅
- CHANGE LOG 记录变更历史
- 便于追踪 prompt 演化

---

## 🔄 迁移完成

### 之前（文件依赖）
```python
# 需要读取文件
with open(AIRLINE_POLICY_PATH, "r") as f:
    policy = f.read()

# 路径问题
# FileNotFoundError 风险
# 部署复杂
```

### 现在（内嵌常量）
```python
# 直接获取
policy = get_domain_policy("airline")

# 无文件依赖
# 无路径问题
# 部署简单
```

---

## 📚 Prompt 详细说明

### Agent Instruction (306 chars)
定义 Agent 的核心行为模式：
- 角色定位：客服代理
- 操作限制：每轮只能执行一个动作
- 遵循 policy
- JSON 格式要求

### User Simulator Guidelines
两个版本，差异在于是否支持工具调用：

**无工具版 (1,449 chars)**:
- 适用于纯对话场景
- 5条核心原则
- 3个结束token

**有工具版 (2,747 chars)**:
- 适用于需要用户执行操作的场景（如电信域）
- 10+条核心原则
- 详细的工具调用规则
- 同样的3个结束token

### Domain Policies

**Airline Policy (167 lines, 7,676 chars)**:
- 最复杂的策略
- 详细的预订、修改、取消规则
- 会员等级和权益
- 行李额度计算
- 补偿政策

**Retail Policy (136 lines, 6,698 chars)**:
- 订单管理流程
- 退换货规则
- 支付方式处理
- 用户身份验证

---

## 🛠️ 工具函数

### Prompts 相关
- `get_agent_instruction()` - 获取Agent指令
- `get_agent_system_prompt(policy)` - 生成Agent完整prompt
- `get_user_simulator_guidelines(use_tools)` - 获取用户指南
- `get_user_simulator_system_prompt(instructions, use_tools)` - 生成User完整prompt
- `validate_prompts()` - 验证prompts
- `list_available_prompts()` - 列出可用prompts

### Policies 相关
- `get_domain_policy(domain)` - 获取领域策略
- `list_available_domains()` - 列出可用领域
- `get_policy_stats()` - 获取统计信息

---

## 📖 最佳实践

1. **使用 getter 函数** - 不要直接访问常量（除非必要）
2. **验证域名** - 使用 `list_available_domains()` 检查
3. **缓存 policy** - policy 不变，可以缓存
4. **测试 prompt** - 定期运行 `validate_prompts()`

---

## 🔍 示例代码

```python
# 示例：完整的 Agent 初始化
from taskdialogue.benchmarks.tau2.prompts import get_agent_system_prompt, get_domain_policy
from taskdialogue.benchmarks.tau2.agent.llm_agent import LLMAgent

# 获取policy
policy = get_domain_policy("airline")

# 生成prompt
system_prompt = get_agent_system_prompt(policy)

# 初始化agent
agent = LLMAgent(
    tools=tools,
    domain_policy=policy,
    llm="gpt-4",
)
```

---

## 📌 重要提示

- ✅ 所有 prompt 已内嵌，不需要 .md 文件
- ✅ 支持 2 个域：airline, retail
- 🚧 Telecom 域待添加
- ✅ 完全从 YAML 配置控制
- ✅ 遵循 `taskdialogue.prompts` 设计模式

