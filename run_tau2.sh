#!/bin/bash
# TaskDialogue v2.0 - Tau2 运行脚本
# 支持命令行参数覆盖 YAML 配置
# 对齐 MultiWOZ 脚本风格

set -e

# 添加项目根目录到 PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 设置 Tau2 数据目录（如果未设置）
if [ -z "$TAU2_DATA_DIR" ]; then
    if [ -d "${SCRIPT_DIR}/data" ]; then
        export TAU2_DATA_DIR="${SCRIPT_DIR}/data"
    fi
fi

# ============================================================================
# 默认配置
# ============================================================================
CONFIG_FILE="configs/tau2/default.yaml"  # 配置文件路径
MODE="full"                               # 运行模式: full, inference, evaluation
NUM_TASKS="8"                              # 限制测试任务数量（前N个，用于快速测试）
INPUT_FILE=""                             # 输入文件（evaluation模式：指定要评估的predictions.json）
OUTPUT_FILE=""                            # 输出文件基础名（不含扩展名和路径）
ALL_DOMAINS=0                             # 是否运行所有domain (0=否, 1=是)

# ============================================================================
# 可覆盖的配置项（默认值，可通过命令行参数修改）
# ============================================================================

# Agent 模型配置
AGENT_PROVIDER="deepseek"                 # Agent 模型提供商: openai, deepseek, zhipuai, vllm
AGENT_MODEL="deepseek-chat"               # Agent 模型名称
AGENT_TEMPERATURE="0.1"                   # Agent 采样温度
AGENT_MAX_TOKENS="512"                   # Agent 最大生成 token 数

# User 模型配置（默认使用 deepseek）
USER_PROVIDER="deepseek"                  # User 模型提供商: openai, deepseek, zhipuai, vllm
USER_MODEL="deepseek-chat"                # User 模型名称
USER_TEMPERATURE="0.1"                    # User 采样温度
USER_MAX_TOKENS="512"                     # User 最大生成 token 数

# Tau2 配置
DOMAIN="airline"                          # 领域: airline, retail, telecom (all_domains=1时忽略)
NUM_TRIALS="1"                            # 每个任务的 trial 数
MAX_STEPS="50"                            # 最大步数
MAX_ERRORS="5"                            # 最大错误数
SEED="300"                                # 随机种子
PROFILE="hybrid"                        # 工具配置: original, sql, hybrid

# 并行配置
INFERENCE_WORKERS="8"                     # 推理并发线程数
EVALUATION_WORKERS="8"                    # 评估并发进程数

# vLLM 配置
USE_VLLM=0                                # 是否使用 vLLM (0=否, 1=是)
VLLM_MODEL_PATH=""                        # vLLM 模型路径（本地模型目录）
VLLM_SERVER_URL="http://localhost:8000"  # vLLM 服务器地址
VLLM_PORT=8000                            # vLLM 服务器端口
SKIP_VLLM_START=0                         # 跳过启动 vLLM（1=服务器已运行，0=自动启动）

# 覆盖项数组
OVERRIDES=()

# ============================================================================
# 解析命令行参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        # ========== 基本选项 ==========
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --mode|-m)
            MODE="$2"
            shift 2
            ;;
        --domain|-d)
            DOMAIN="$2"
            OVERRIDES+=("tau2.domain=$2")
            shift 2
            ;;
        --num-tasks|-n)
            NUM_TASKS="$2"
            OVERRIDES+=("tau2.trials.num_tasks=$2")
            shift 2
            ;;
        --all-domains)
            ALL_DOMAINS=1
            shift 1
            ;;
        --input-file|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        
        # ========== Agent 模型配置 ==========
        --provider|--agent-provider)
            AGENT_PROVIDER="$2"
            OVERRIDES+=("model.agent.provider=$2")
            [ "$2" == "vllm" ] && USE_VLLM=1
            shift 2
            ;;
        --model|--agent-model)
            AGENT_MODEL="$2"
            OVERRIDES+=("model.agent.name=$2")
            shift 2
            ;;
        --temperature|-t|--agent-temperature)
            AGENT_TEMPERATURE="$2"
            OVERRIDES+=("model.agent.temperature=$2")
            shift 2
            ;;
        --max-tokens|--agent-max-tokens)
            AGENT_MAX_TOKENS="$2"
            OVERRIDES+=("model.agent.max_tokens=$2")
            shift 2
            ;;
        
        # ========== User 模型配置 ==========
        --user-provider)
            USER_PROVIDER="$2"
            OVERRIDES+=("model.user.provider=$2")
            shift 2
            ;;
        --user-model)
            USER_MODEL="$2"
            OVERRIDES+=("model.user.name=$2")
            shift 2
            ;;
        --user-temperature)
            USER_TEMPERATURE="$2"
            OVERRIDES+=("model.user.temperature=$2")
            shift 2
            ;;
        --user-max-tokens)
            USER_MAX_TOKENS="$2"
            OVERRIDES+=("model.user.max_tokens=$2")
            shift 2
            ;;
        
        # ========== Tau2 配置 ==========
        --num-trials)
            NUM_TRIALS="$2"
            OVERRIDES+=("tau2.trials.num_trials=$2")
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            OVERRIDES+=("tau2.limits.max_steps=$2")
            shift 2
            ;;
        --max-errors)
            MAX_ERRORS="$2"
            OVERRIDES+=("tau2.limits.max_errors=$2")
            shift 2
            ;;
        --seed)
            SEED="$2"
            OVERRIDES+=("tau2.seed=$2")
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            if [[ ! "$PROFILE" =~ ^(original|sql|hybrid)$ ]]; then
                echo "❌ 错误: --profile 必须是 original, sql, 或 hybrid"
                exit 1
            fi
            OVERRIDES+=("tau2.tools.profile=$2")
            shift 2
            ;;
        
        # ========== 并行配置 ==========
        --workers|-w)
            INFERENCE_WORKERS="$2"
            EVALUATION_WORKERS="$2"
            OVERRIDES+=("inference.num_workers=$2")
            OVERRIDES+=("evaluation.num_workers=$2")
            shift 2
            ;;
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
        
        # ========== vLLM 配置 ==========
        --use-vllm)
            USE_VLLM=1
            AGENT_PROVIDER="vllm"  # vLLM 只影响 Agent
            OVERRIDES+=("model.agent.provider=vllm")
            shift 1
            ;;
        --vllm-model-path)
            VLLM_MODEL_PATH="$2"
            USE_VLLM=1
            AGENT_PROVIDER="vllm"  # vLLM 只影响 Agent
            OVERRIDES+=("model.agent.provider=vllm")
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
            echo "TaskDialogue v2.0 - Tau2 运行脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "基本选项:"
            echo "  -c, --config FILE           配置文件路径 (默认: configs/tau2/default.yaml)"
            echo "  -m, --mode MODE             运行模式: full|inference|evaluation (默认: full)"
            echo "  -d, --domain DOMAIN         运行的域: airline|retail|telecom (默认: airline)"
            echo "  -n, --num-tasks N           限制测试任务数量（前N个，用于快速测试）"
            echo "  --all-domains               运行所有domain (airline, retail, telecom)"
            echo "  -i, --input-file FILE       输入文件（evaluation模式：指定predictions.json）"
            echo "  --output-file NAME          输出文件基础名（不含扩展名和路径）"
            echo ""
            echo "Agent 模型配置 (覆盖 YAML):"
            echo "  --provider, --agent-provider PROVIDER"
            echo "                             Agent 模型提供商: openai|deepseek|zhipuai|vllm"
            echo "  --model, --agent-model MODEL"
            echo "                             Agent 模型名称 (如: deepseek-chat, gpt-4)"
            echo "  -t, --temperature, --agent-temperature T"
            echo "                             Agent 采样温度 (推荐: 0.1-0.3)"
            echo "  --max-tokens, --agent-max-tokens N"
            echo "                             Agent 最大生成 token 数"
            echo ""
            echo "User 模型配置 (覆盖 YAML，默认使用 deepseek):"
            echo "  --user-provider PROVIDER    User 模型提供商"
            echo "  --user-model MODEL          User 模型名称"
            echo "  --user-temperature T        User 采样温度"
            echo "  --user-max-tokens N         User 最大生成 token 数"
            echo ""
            echo "Tau2 配置 (覆盖 YAML):"
            echo "  --num-trials N              每个任务的 trial 数 (默认: 1)"
            echo "  --max-steps N               最大步数 (默认: 50)"
            echo "  --max-errors N              最大错误数 (默认: 5)"
            echo "  --seed N                    随机种子 (默认: 300)"
            echo "  --profile PROFILE           工具配置: original|sql|hybrid (默认: original)"
            echo ""
            echo "并行配置 (覆盖 YAML):"
            echo "  -w, --workers N             并发数（同时设置推理和评估）"
            echo "  --inference-workers N       推理并发线程数 (推荐: 4-16)"
            echo "  --evaluation-workers N      评估并发进程数 (推荐: 4-16)"
            echo ""
            echo "vLLM 配置 (⚠️  仅影响 Agent 模型):"
            echo "  --use-vllm                  使用 vLLM 作为 Agent 模型（自动启动服务器）"
            echo "  --vllm-model-path PATH      vLLM 模型路径（本地模型目录）"
            echo "  --vllm-server URL           vLLM 服务器地址 (默认: http://localhost:8000)"
            echo "  --vllm-port PORT            vLLM 服务器端口 (默认: 8000)"
            echo "  --skip-vllm-start           跳过启动 vLLM（服务器已运行）"
            echo "  注意: vLLM 配置只覆盖 Agent 模型，User 模型保持独立配置"
            echo ""
            echo "通用选项:"
            echo "  -o, --override KEY=VALUE    覆盖配置项（可多次使用）"
            echo "  -h, --help                  显示帮助信息"
            echo ""
            echo "示例:"
            echo "  # 快速测试：运行前5个任务"
            echo "  $0 -d airline -n 5"
            echo ""
            echo "  # 运行所有domain，每个domain前10个任务"
            echo "  $0 --all-domains -n 10"
            echo ""
            echo "  # 只运行推理"
            echo "  $0 -d airline -m inference -n 5"
            echo ""
            echo "  # 只运行评估（从已有推理结果）"
            echo "  $0 -d airline -m evaluation -i results/tau2/inference/airline/predictions_*.json"
            echo ""
            echo "  # 使用自定义 Agent 模型配置"
            echo "  $0 -d airline --provider deepseek --model deepseek-chat -t 0.2"
            echo ""
            echo "  # 分别配置 Agent 和 User 模型"
            echo "  $0 -d airline --provider openai --model gpt-4 --user-provider deepseek --user-model deepseek-chat"
            echo ""
            echo "  # 覆盖多个配置项"
            echo "  $0 -d airline -n 5 -o inference.num_workers=16 -o evaluation.num_workers=8"
            echo ""
            echo "  # 使用 vLLM + 自定义模型路径（自动启动服务器）"
            echo "  $0 -d airline --use-vllm --vllm-model-path /path/to/model"
            echo ""
            echo "  # vLLM 已运行，不启动服务器"
            echo "  $0 -d airline --use-vllm --skip-vllm-start"
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
# 检查必需参数
# ============================================================================
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "提示: 创建配置文件模板："
    echo "  python -m taskdialogue init-config -b tau2"
    exit 1
fi

# ============================================================================
# 构建命令
# ============================================================================
CMD="python -m taskdialogue tau2 --config $CONFIG_FILE --mode $MODE"

# 处理 domain 参数
if [ $ALL_DOMAINS -eq 1 ]; then
    # 运行所有domain时，不传递单个domain参数，由pipeline处理
    OVERRIDES+=("tau2.all_domains=true")
else
    CMD="$CMD --domain $DOMAIN"
fi

# 处理 num_tasks
[ -n "$NUM_TASKS" ] && CMD="$CMD --num-tasks $NUM_TASKS"

[ -n "$INPUT_FILE" ] && CMD="$CMD --input-file $INPUT_FILE"
[ -n "$OUTPUT_FILE" ] && CMD="$CMD --output-file $OUTPUT_FILE"

# 设置 Agent 模型配置（始终覆盖，确保使用脚本默认值）
# 检查是否已经有 model.agent.* 覆盖，如果没有则添加默认值
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^model\.agent\.provider="; then
    OVERRIDES+=("model.agent.provider=$AGENT_PROVIDER")
fi
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^model\.agent\.name="; then
    OVERRIDES+=("model.agent.name=$AGENT_MODEL")
fi
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^model\.agent\.temperature="; then
    OVERRIDES+=("model.agent.temperature=$AGENT_TEMPERATURE")
fi
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^model\.agent\.max_tokens="; then
    OVERRIDES+=("model.agent.max_tokens=$AGENT_MAX_TOKENS")
fi

# 设置 User 模型配置（始终覆盖，确保 User 使用独立配置）
OVERRIDES+=("model.user.provider=$USER_PROVIDER")
OVERRIDES+=("model.user.name=$USER_MODEL")
OVERRIDES+=("model.user.temperature=$USER_TEMPERATURE")
OVERRIDES+=("model.user.max_tokens=$USER_MAX_TOKENS")

# 设置默认并发数（确保脚本默认值覆盖 YAML 中的值）
# 检查是否已经有 inference.num_workers 覆盖，如果没有则添加默认值
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^inference\.num_workers="; then
    OVERRIDES+=("inference.num_workers=$INFERENCE_WORKERS")
fi
# 检查是否已经有 evaluation.num_workers 覆盖，如果没有则添加默认值
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^evaluation\.num_workers="; then
    OVERRIDES+=("evaluation.num_workers=$EVALUATION_WORKERS")
fi

# 设置 profile 配置（确保使用脚本默认值）
if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^tau2\.tools\.profile="; then
    OVERRIDES+=("tau2.tools.profile=$PROFILE")
fi

# 设置 vLLM server 配置（如果使用 vLLM，确保覆盖默认值）
if [ $USE_VLLM -eq 1 ]; then
    # 检查是否已经有 vllm.server.base_url 覆盖，如果没有则添加默认值
    if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^vllm\.server\.base_url="; then
        OVERRIDES+=("vllm.server.base_url=$VLLM_SERVER_URL")
    fi
    # 检查是否已经有 vllm.server.port 覆盖，如果没有则添加默认值
    if ! printf '%s\n' "${OVERRIDES[@]}" | grep -q "^vllm\.server\.port="; then
        OVERRIDES+=("vllm.server.port=$VLLM_PORT")
    fi
fi

# 添加覆盖参数
for override in "${OVERRIDES[@]}"; do
    CMD="$CMD --override $override"
done

# ============================================================================
# 打印配置信息
# ============================================================================
echo "========================================================================"
echo "🚀 TaskDialogue v2.0 - Tau2 Benchmark"
echo "========================================================================"
echo "配置文件: $CONFIG_FILE"
echo "运行模式: $MODE"
if [ $ALL_DOMAINS -eq 1 ]; then
    echo "领域: 全部 (airline, retail, telecom)"
else
    echo "领域: $DOMAIN"
fi
[ -n "$NUM_TASKS" ] && echo "测试任务数量: $NUM_TASKS"
echo "Agent 模型: $AGENT_PROVIDER / $AGENT_MODEL"
[ -n "$AGENT_TEMPERATURE" ] && echo "Agent 温度: $AGENT_TEMPERATURE"
[ -n "$AGENT_MAX_TOKENS" ] && echo "Agent 最大 tokens: $AGENT_MAX_TOKENS"
echo "User 模型: $USER_PROVIDER / $USER_MODEL"
[ -n "$USER_TEMPERATURE" ] && echo "User 温度: $USER_TEMPERATURE"
[ -n "$USER_MAX_TOKENS" ] && echo "User 最大 tokens: $USER_MAX_TOKENS"
echo "工具配置: $PROFILE"
echo "推理并发: $INFERENCE_WORKERS"
echo "评估并发: $EVALUATION_WORKERS"
[ $USE_VLLM -eq 1 ] && echo "⚠️  vLLM: 仅用于 Agent 模型（User 模型: $USER_PROVIDER）"
[ ${#OVERRIDES[@]} -gt 0 ] && echo "配置覆盖: ${#OVERRIDES[@]} 项"
if [ ${#OVERRIDES[@]} -gt 0 ] && [ ${#OVERRIDES[@]} -le 10 ]; then
    echo "覆盖项:"
    for override in "${OVERRIDES[@]}"; do
        echo "  • $override"
    done
fi
echo "========================================================================"
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
    echo "  推理结果: ls -lh results/tau2/inference/"
    echo "  评估结果: ls -lh results/tau2/evaluation/"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ 运行失败（退出码: $EXIT_CODE）"
    echo "========================================================================"
fi

exit $EXIT_CODE

