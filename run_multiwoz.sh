#!/bin/bash
# TaskDialogue v2.0 - MultiWOZ 运行脚本
# 支持命令行参数覆盖 YAML 配置
# 支持 vLLM: 自动启动服务器 → 推理 → 评估 → 停止服务器

set -e

# 添加项目根目录到 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 添加 vllm_utils 父目录到 PYTHONPATH（vllm_utils 在 TaskDialogue 的父目录下）
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH}"

# ============================================================================
# 默认配置
# ============================================================================
CONFIG_FILE="configs/multiwoz/default.yaml"  # 配置文件路径
MODE="full"                                   # 运行模式: full, inference, evaluation
NUM_SAMPLES=""                                # 推理样本数量（空=使用配置文件或全部数据，用于覆盖 inference.num_samples）
TASK_IDS=""                                   # 指定任务ID（空=处理所有任务）
INPUT_FILE=""                                 # 输入文件（evaluation模式：指定要评估的predictions.jsonl）
OUTPUT_FILE=""                                # 输出文件基础名（不含扩展名和路径）

# ============================================================================
# 可覆盖的配置项（默认值，可通过命令行参数修改）
# ============================================================================

# 模型配置
PROVIDER="vllm"              # 模型提供商: openai, deepseek, zhipuai, vllm
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"       # 模型名称: deepseek-chat, gpt-4, gpt-4o, 等
TEMPERATURE="0.1"                # 采样温度 (0.1-0.5，越低越稳定)
MAX_TOKENS="512"                # 最大生成 token 数

# Agent 配置
AGENT_TYPE="function_calling"    # Agent 类型: function_calling (API), custom_format (vLLM)
MAX_TURNS="30"                   # 最大对话轮数（防止无限循环）
MAX_TOOL_ITERATIONS="5"          # 单轮最大工具调用次数（防止工具调用死循环）

# 并行配置
INFERENCE_WORKERS="1"           # 推理并发线程数 (推荐: 4-32，API 调用 I/O 密集)
EVALUATION_WORKERS="32"          # 评估并发进程数 (推荐: 4-32，CPU 密集)
SAVE_BATCH_SIZE="32"             # 批量保存大小 (每完成 N 个连续任务就保存，避免内存堆积)

# 评估配置
EVAL_MODEL="deepseek-chat"       # 评估用 LLM 模型
SUCCESS_MODE="strict"            # 评估模式: strict (严格), relaxed (宽松)

# vLLM 配置
USE_VLLM=1                       # 是否使用 vLLM (0=否, 1=是)
VLLM_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"  # vLLM 模型路径（本地模型目录，如: /path/to/Qwen3-4B-Instruct）
VLLM_SERVER_URL="http://localhost:8000"  # vLLM 服务器地址
VLLM_PORT=8000                   # vLLM 服务器端口
SKIP_VLLM_START=0                # 跳过启动 vLLM（1=服务器已运行，0=自动启动）

# 存储所有 override 参数
OVERRIDES=()

# 添加默认配置到 OVERRIDES（会被命令行参数覆盖）
OVERRIDES+=("model.agent.provider=$PROVIDER")
OVERRIDES+=("model.agent.name=$MODEL_NAME")
OVERRIDES+=("model.agent.temperature=$TEMPERATURE")
OVERRIDES+=("model.agent.max_tokens=$MAX_TOKENS")
OVERRIDES+=("agent.type=$AGENT_TYPE")
OVERRIDES+=("agent.max_turns=$MAX_TURNS")
OVERRIDES+=("agent.max_tool_iterations=$MAX_TOOL_ITERATIONS")
OVERRIDES+=("inference.num_workers=$INFERENCE_WORKERS")
OVERRIDES+=("evaluation.num_workers=$EVALUATION_WORKERS")
OVERRIDES+=("inference.save_batch_size=$SAVE_BATCH_SIZE")
OVERRIDES+=("evaluation.save_batch_size=$SAVE_BATCH_SIZE")
OVERRIDES+=("evaluation.eval_model=$EVAL_MODEL")
OVERRIDES+=("evaluation.success_mode=$SUCCESS_MODE")

# 根据 AGENT_TYPE 自动配置 vLLM 参数
if [ "$AGENT_TYPE" = "custom_format" ]; then
    # Custom Format 模式：禁用原生工具调用，使用自定义 token
    OVERRIDES+=("agent.custom_format.enabled=true")
    OVERRIDES+=("vllm.engine.enable_auto_tool_choice=false")
    OVERRIDES+=("vllm.engine.tool_call_parser=null")
else
    # Function Calling 模式：启用原生工具调用
    OVERRIDES+=("agent.custom_format.enabled=false")
    OVERRIDES+=("vllm.engine.enable_auto_tool_choice=true")
    # tool_call_parser 设为 null 让 vLLM 自动检测模型支持的格式
    OVERRIDES+=("vllm.engine.tool_call_parser=null")
fi

# ============================================================================
# 解析命令行参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --num-samples|-n)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --task-ids)
            TASK_IDS="$2"
            shift 2
            ;;
        --input-file|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        
        # ========== 模型配置 ==========
        --provider)
            PROVIDER="$2"
            OVERRIDES+=("model.agent.provider=$2")
            [ "$2" == "vllm" ] && USE_VLLM=1
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            OVERRIDES+=("model.agent.name=$2")
            shift 2
            ;;
        --temperature|-t)
            TEMPERATURE="$2"
            OVERRIDES+=("model.agent.temperature=$2")
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            OVERRIDES+=("model.agent.max_tokens=$2")
            shift 2
            ;;
        
        # ========== Agent 配置 ==========
        --agent-type)
            AGENT_TYPE="$2"
            OVERRIDES+=("agent.type=$2")
            shift 2
            ;;
        --custom-format)
            # 快捷方式：自动切换到 custom format 模式
            AGENT_TYPE="custom_format"
            OVERRIDES+=("agent.type=custom_format")
            OVERRIDES+=("agent.custom_format.enabled=true")
            shift 1
            ;;
        --max-turns)
            MAX_TURNS="$2"
            OVERRIDES+=("agent.max_turns=$2")
            shift 2
            ;;
        --max-tool-iterations)
            MAX_TOOL_ITERATIONS="$2"
            OVERRIDES+=("agent.max_tool_iterations=$2")
            shift 2
            ;;
        
        # ========== 并行配置 ==========
        --inference-workers)
            INFERENCE_WORKERS="$2"
            OVERRIDES+=("inference.num_workers=$2")
            shift 2
            ;;
        --evaluation-workers)
            EVALUATION_WORKERS="$2"
            OVERRIDES+=("evaluation.num_workers=$2")
            shift 2
            ;;
        --workers|-w)
            # 同时设置推理和评估的 workers
            INFERENCE_WORKERS="$2"
            EVALUATION_WORKERS="$2"
            OVERRIDES+=("inference.num_workers=$2")
            OVERRIDES+=("evaluation.num_workers=$2")
            shift 2
            ;;
        --save-batch-size)
            SAVE_BATCH_SIZE="$2"
            OVERRIDES+=("inference.save_batch_size=$2")
            OVERRIDES+=("evaluation.save_batch_size=$2")
            shift 2
            ;;
        
        # ========== 评估配置 ==========
        --eval-model)
            EVAL_MODEL="$2"
            OVERRIDES+=("evaluation.eval_model=$2")
            shift 2
            ;;
        --success-mode)
            SUCCESS_MODE="$2"
            OVERRIDES+=("evaluation.success_mode=$2")
            shift 2
            ;;
        
        # ========== vLLM 配置 ==========
        --use-vllm)
            USE_VLLM=1
            PROVIDER="vllm"
            OVERRIDES+=("model.agent.provider=vllm")
            shift 1
            ;;
        --vllm-model-path)
            VLLM_MODEL_PATH="$2"
            USE_VLLM=1
            OVERRIDES+=("vllm.model_path=$2")
            shift 2
            ;;
        --vllm-server)
            VLLM_SERVER_URL="$2"
            OVERRIDES+=("vllm.server.base_url=$2")
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            OVERRIDES+=("vllm.server.port=$2")
            shift 2
            ;;
        --skip-vllm-start)
            SKIP_VLLM_START=1
            shift 1
            ;;
        
        # ========== 通用覆盖 ==========
        --override|-o)
            OVERRIDES+=("$2")
            shift 2
            ;;
        
        --help|-h)
            echo "TaskDialogue v2.0 - MultiWOZ 运行脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "基本选项:"
            echo "  -c, --config FILE           配置文件路径 (默认: configs/multiwoz/default.yaml)"
            echo "  -m, --mode MODE             运行模式: full|inference|evaluation (默认: full)"
            echo "  -n, --num-samples N         推理样本数量（用于测试，覆盖 inference.num_samples）"
            echo "  --task-ids IDS              指定任务ID列表（逗号分隔）"
            echo "  -i, --input-file FILE       输入文件（evaluation模式：指定predictions.jsonl）"
            echo "  --output-file NAME          输出文件基础名（不含扩展名和路径）"
            echo ""
            echo "模型配置 (覆盖 YAML):"
            echo "  --provider PROVIDER         模型提供商: openai|deepseek|zhipuai|vllm"
            echo "  --model MODEL               模型名称 (如: deepseek-chat, gpt-4)"
            echo "  -t, --temperature T         采样温度 (推荐: 0.1-0.3)"
            echo "  --max-tokens N              最大生成 token 数"
            echo ""
            echo "Agent 配置 (覆盖 YAML):"
            echo "  --agent-type TYPE           Agent 类型: function_calling|custom_format"
            echo "  --custom-format             快捷方式：启用 custom format 模式"
            echo "  --max-turns N               最大对话轮数 (默认: 30)"
            echo "  --max-tool-iterations N     单轮最大工具调用次数 (默认: 5)"
            echo ""
            echo "并行配置 (覆盖 YAML):"
            echo "  -w, --workers N             并发数（同时设置推理和评估）"
            echo "  --inference-workers N       推理并发线程数 (推荐: 4-32)"
            echo "  --evaluation-workers N      评估并发进程数 (推荐: 4-32)"
            echo "  --save-batch-size N         批量保存大小 (默认: 32)"
            echo ""
            echo "评估配置 (覆盖 YAML):"
            echo "  --eval-model MODEL          评估用 LLM 模型"
            echo "  --success-mode MODE         评估模式: strict|relaxed"
            echo ""
            echo "vLLM 配置:"
            echo "  --use-vllm                  使用 vLLM（自动启动服务器）"
            echo "  --vllm-model-path PATH      vLLM 模型路径（本地模型目录）"
            echo "  --vllm-server URL           vLLM 服务器地址 (默认: http://localhost:8000)"
            echo "  --vllm-port PORT            vLLM 服务器端口 (默认: 8000)"
            echo "  --skip-vllm-start           跳过启动 vLLM（服务器已运行）"
            echo ""
            echo "高级选项:"
            echo "  -o, --override KEY=VALUE    覆盖任意 YAML 配置 (可多次使用)"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  # 快速测试（5个对话）"
            echo "  $0 -n 5"
            echo ""
            echo "  # 只运行推理"
            echo "  $0 --mode inference -n 32"
            echo ""
            echo "  # 只运行评估（自动加载最新的predictions）"
            echo "  $0 --mode evaluation"
            echo ""
            echo "  # 评估指定的predictions文件"
            echo "  $0 --mode evaluation -i results/multiwoz/inference/predictions_xxx.jsonl -o my_eval"
            echo ""
            echo "  # 使用 GPT-4，16个并发"
            echo "  $0 --provider openai --model gpt-4 -w 16"
            echo ""
            echo "  # 使用 vLLM + Custom Format（自动启动服务器）"
            echo "  $0 --use-vllm --vllm-model-path /path/to/model --custom-format"
            echo ""
            echo "  # vLLM 已运行，不启动服务器"
            echo "  $0 --use-vllm --skip-vllm-start --custom-format"
            echo ""
            echo "  # 宽松评估模式，8个并发"
            echo "  $0 --success-mode relaxed -w 8"
            echo ""
            echo "  # 低温度 + 多轮对话"
            echo "  $0 -t 0.1 --max-turns 20"
            echo ""
            echo "  # 高级：直接覆盖 YAML 路径"
            echo "  $0 -o agent.max_history=100 -o evaluation.per_domain_evaluation=false"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# ============================================================================
# 检查配置文件
# ============================================================================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "可用配置文件:"
    echo "  configs/multiwoz/default.yaml  - 统一配置（支持 API 和 vLLM）"
    echo ""
    echo "提示: 使用 --custom-format 切换到 vLLM 自定义格式模式"
    exit 1
fi

# ============================================================================
# 步骤 0: 启动 vLLM 服务器 (如果需要)
# ============================================================================
VLLM_PID=""
if [ $USE_VLLM -eq 1 ] && [ $SKIP_VLLM_START -eq 0 ]; then
    if [ -z "$VLLM_MODEL_PATH" ]; then
        echo "❌ 错误: 使用 vLLM 时必须指定 --vllm-model-path"
        echo "示例: $0 --use-vllm --vllm-model-path /path/to/model"
        exit 1
    fi
    
    echo "========================================================================"
    echo "🚀 步骤 0: 启动 vLLM 服务器"
    echo "========================================================================"
    
    # 检查并停止端口占用
    echo "检查端口 $VLLM_PORT 占用情况..."
    if lsof -ti:$VLLM_PORT > /dev/null 2>&1; then
        PORT_PIDS=$(lsof -ti:$VLLM_PORT)
        echo "⚠️  端口 $VLLM_PORT 已被占用，PID: $PORT_PIDS"
        echo "正在停止占用端口的进程..."
        kill -9 $PORT_PIDS 2>/dev/null || true
        sleep 3
        
        if lsof -ti:$VLLM_PORT > /dev/null 2>&1; then
            echo "❌ 错误: 无法释放端口 $VLLM_PORT"
            exit 1
        fi
        echo "✅ 端口已释放"
    else
        echo "✅ 端口 $VLLM_PORT 可用"
    fi
    
    # 清理旧的 PID 文件
    if [ -f ~/.vllm_server.pid ]; then
        OLD_PID=$(cat ~/.vllm_server.pid)
        if ps -p $OLD_PID > /dev/null 2>&1; then
            echo "停止记录的旧服务器进程 (PID: $OLD_PID)..."
            kill $OLD_PID 2>/dev/null || kill -9 $OLD_PID 2>/dev/null || true
            sleep 2
        fi
        rm -f ~/.vllm_server.pid
    fi
    
    # 从配置文件读取 vLLM 参数
    echo "读取配置文件参数: $CONFIG_FILE"
    VLLM_TENSOR_PARALLEL=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('vllm', {}).get('engine', {}).get('tensor_parallel_size', 2))")
    VLLM_GPU_MEMORY=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('vllm', {}).get('engine', {}).get('gpu_memory_utilization', 0.75))")
    VLLM_MAX_LEN=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('vllm', {}).get('engine', {}).get('max_model_len', 16384))")
    VLLM_MAX_NUM_SEQS=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('vllm', {}).get('engine', {}).get('max_num_seqs', 128))")
    VLLM_DTYPE=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('vllm', {}).get('engine', {}).get('dtype', 'auto'))")
    VLLM_TRUST_REMOTE=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(str(config.get('vllm', {}).get('engine', {}).get('trust_remote_code', True)).lower())")
    VLLM_CHAT_TEMPLATE=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); template=config.get('vllm', {}).get('engine', {}).get('chat_template'); print(template if template and template.lower() not in ['none', 'null'] else '')")
    VLLM_DISABLE_LOG=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(str(config.get('vllm', {}).get('engine', {}).get('disable_log_requests', True)).lower())")
    
    # 根据 AGENT_TYPE 配置工具调用参数
    if [ "$AGENT_TYPE" = "custom_format" ]; then
        # Custom Format：禁用原生工具调用
        VLLM_AUTO_TOOL="false"
        VLLM_TOOL_PARSER=""
        echo "检测到 Custom Format 模式：禁用 vLLM 原生工具调用"
    else
        # Function Calling：启用原生工具调用（从配置文件读取）
        VLLM_AUTO_TOOL=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(str(config.get('vllm', {}).get('engine', {}).get('enable_auto_tool_choice', True)).lower())")
        VLLM_TOOL_PARSER=$(python -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); parser=config.get('vllm', {}).get('engine', {}).get('tool_call_parser'); print(parser if parser and parser.lower() not in ['none', 'null'] else '')")
        echo "检测到 Function Calling 模式：使用配置文件的工具调用设置"
    fi
    
    # 启动新服务器
    mkdir -p logs
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/vllm_server_${TIMESTAMP}.log"
    echo "启动 vLLM 服务器..."
    echo "  模型: $VLLM_MODEL_PATH"
    echo "  端口: $VLLM_PORT"
    echo "  Agent模式: $AGENT_TYPE"
    echo "  Tensor并行: $VLLM_TENSOR_PARALLEL"
    echo "  GPU显存利用率: $VLLM_GPU_MEMORY"
    echo "  最大长度: $VLLM_MAX_LEN"
    echo "  最大并发序列: $VLLM_MAX_NUM_SEQS"
    echo "  数据类型: $VLLM_DTYPE"
    if [ "$AGENT_TYPE" = "custom_format" ]; then
        echo "  工具调用: 禁用（Custom Format 使用自定义 token）"
    else
        [ "$VLLM_AUTO_TOOL" = "true" ] && echo "  工具调用: 已启用"
        [ -n "$VLLM_TOOL_PARSER" ] && echo "  工具调用解析器: $VLLM_TOOL_PARSER"
    fi
    [ -n "$VLLM_CHAT_TEMPLATE" ] && echo "  Chat模板: $VLLM_CHAT_TEMPLATE"
    echo "  日志: $LOG_FILE"
    
    # 构建启动命令
    VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
        --model $VLLM_MODEL_PATH \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --served-model-name $(basename $VLLM_MODEL_PATH) \
        --tensor-parallel-size $VLLM_TENSOR_PARALLEL \
        --gpu-memory-utilization $VLLM_GPU_MEMORY \
        --max-model-len $VLLM_MAX_LEN \
        --max-num-seqs $VLLM_MAX_NUM_SEQS \
        --dtype $VLLM_DTYPE"
    
    # 添加布尔型参数
    [ "$VLLM_TRUST_REMOTE" = "true" ] && VLLM_CMD="$VLLM_CMD --trust-remote-code"
    [ "$VLLM_AUTO_TOOL" = "true" ] && VLLM_CMD="$VLLM_CMD --enable-auto-tool-choice"
    [ "$VLLM_DISABLE_LOG" = "true" ] && VLLM_CMD="$VLLM_CMD --disable-log-requests"
    
    # 添加工具调用相关参数
    [ -n "$VLLM_TOOL_PARSER" ] && VLLM_CMD="$VLLM_CMD --tool-call-parser $VLLM_TOOL_PARSER"
    [ -n "$VLLM_CHAT_TEMPLATE" ] && VLLM_CMD="$VLLM_CMD --chat-template $VLLM_CHAT_TEMPLATE"
    
    echo ""
    echo "启动命令: $VLLM_CMD"
    echo ""
    nohup $VLLM_CMD > "$LOG_FILE" 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > ~/.vllm_server.pid
    echo "vLLM 服务器已启动 (PID: $VLLM_PID)"
    
    # 等待服务器就绪
    echo "等待服务器就绪（最多 360 秒）..."
    MAX_WAIT=360
    WAITED=0
    SERVER_READY=false
    
    while [ $WAITED -lt $MAX_WAIT ]; do
        sleep 5
        WAITED=$((WAITED + 5))
        
        # 1. 检查进程是否还在运行
        if ! ps -p $VLLM_PID > /dev/null 2>&1; then
            echo "❌ 服务器进程已退出！"
            echo "   最后 20 行日志："
            tail -20 "$LOG_FILE" | sed 's/^/   /'
            exit 1
        fi
        
        # 2. 检查端口是否开始监听
        if ! lsof -ti:$VLLM_PORT > /dev/null 2>&1; then
            echo "   [$WAITED秒] 等待端口 $VLLM_PORT 监听..."
            continue
        fi
        
        # 3. 检查健康端点
        if curl -s "$VLLM_SERVER_URL/health" > /dev/null 2>&1; then
            # 4. 尝试获取模型列表验证服务器真正可用
            if curl -s "$VLLM_SERVER_URL/v1/models" 2>/dev/null | grep -q "data" 2>/dev/null; then
                echo "✅ 服务器已就绪！"
                SERVER_READY=true
                break
            fi
        fi
        
        echo "   [$WAITED秒] 等待服务器就绪..."
    done
    
    if [ "$SERVER_READY" = false ]; then
        echo "❌ 服务器启动超时（等待了 ${MAX_WAIT} 秒）"
        echo "   最后 30 行日志："
        tail -30 "$LOG_FILE" | sed 's/^/   /'
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    
    echo "========================================================================"
    echo ""
fi

# ============================================================================
# 构建命令
# ============================================================================
CMD="python -m taskdialogue multiwoz --config $CONFIG_FILE --mode $MODE"

# 添加运行时参数
[ -n "$NUM_SAMPLES" ] && CMD="$CMD --num-tasks $NUM_SAMPLES"
[ -n "$TASK_IDS" ] && CMD="$CMD --task-ids $TASK_IDS"
[ -n "$INPUT_FILE" ] && CMD="$CMD --input-file $INPUT_FILE"
[ -n "$OUTPUT_FILE" ] && CMD="$CMD --output-file $OUTPUT_FILE"

# 添加所有 override 参数
for override in "${OVERRIDES[@]}"; do
    CMD="$CMD --override $override"
done

# ============================================================================
# 打印配置信息
# ============================================================================
echo "========================================================================"
echo "🚀 TaskDialogue v2.0 - MultiWOZ Benchmark"
echo "========================================================================"
echo "配置文件: $CONFIG_FILE"
echo "运行模式: $MODE"
[ -n "$NUM_SAMPLES" ] && echo "样本数量: $NUM_SAMPLES"
[ -n "$TASK_IDS" ] && echo "指定任务: $TASK_IDS"
[ -n "$INPUT_FILE" ] && echo "输入文件: $INPUT_FILE"
[ -n "$OUTPUT_FILE" ] && echo "输出文件: $OUTPUT_FILE"
[ -n "$PROVIDER" ] && echo "模型提供商: $PROVIDER"
[ -n "$MODEL_NAME" ] && echo "模型名称: $MODEL_NAME"
[ -n "$AGENT_TYPE" ] && echo "Agent 类型: $AGENT_TYPE"
[ -n "$INFERENCE_WORKERS" ] && echo "推理并发: $INFERENCE_WORKERS"
[ -n "$EVALUATION_WORKERS" ] && echo "评估并发: $EVALUATION_WORKERS"

if [ ${#OVERRIDES[@]} -gt 0 ]; then
    echo ""
    echo "配置覆盖 (${#OVERRIDES[@]} 项):"
    for override in "${OVERRIDES[@]}"; do
        echo "  • $override"
    done
fi

echo "========================================================================"
echo ""
echo "执行命令: $CMD"
echo ""

# ============================================================================
# 运行
# ============================================================================
$CMD

EXIT_CODE=$?

# ============================================================================
# 清理 vLLM 服务器
# ============================================================================
if [ -n "$VLLM_PID" ]; then
    echo ""
    echo "========================================================================"
    echo "🛑 停止 vLLM 服务器"
    echo "========================================================================"
    
    if ps -p $VLLM_PID > /dev/null 2>&1; then
        echo "停止 vLLM 服务器 (PID: $VLLM_PID)..."
        kill $VLLM_PID 2>/dev/null || kill -9 $VLLM_PID 2>/dev/null || true
        sleep 2
    fi
    
    rm -f ~/.vllm_server.pid
    echo "✅ vLLM 服务器已停止"
    echo "========================================================================"
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
if [ $EXIT_CODE -eq 0 ]; then
echo ""
echo "========================================================================"
echo "✅ 完成！"
echo "========================================================================"
echo "查看结果:"
echo "  ls -lh results/multiwoz/inference/"
echo "  ls -lh results/multiwoz/evaluation/"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ 运行失败（退出码: $EXIT_CODE）"
echo "========================================================================"
fi

exit $EXIT_CODE
