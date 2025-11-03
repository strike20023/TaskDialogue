#!/bin/bash
# vLLM OpenAI 服务器启动脚本

# 默认配置
MODEL_PATH="models/Qwen3-4B-Instruct-2507"  # 请修改为你的模型路径
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL_SIZE=2  # 使用 2 张 GPU
GPU_MEMORY_UTILIZATION=0.75
MAX_MODEL_LEN=32768  # 增加到 32K 支持更长对话
MAX_NUM_SEQS=64  # 减少并发以留出内存给长序列
DTYPE="auto"

# 日志文件
LOG_DIR="vllm_utils/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_server_$(date +%Y%m%d_%H%M%S).log"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --daemon)
            DAEMON=1
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model-path PATH              模型路径"
            echo "  --host HOST                    主机地址 (默认: 0.0.0.0)"
            echo "  --port PORT                    端口号 (默认: 8000)"
            echo "  --tensor-parallel-size SIZE    Tensor 并行数 (默认: 1)"
            echo "  --gpu-memory-utilization UTIL  GPU 内存使用率 (默认: 0.9)"
            echo "  --dtype DTYPE                  数据类型 (默认: auto)"
            echo "  --daemon                       后台运行"
            echo "  --help                         显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 从路径中提取模型名称（最后一个目录名，去掉结尾斜杠）
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# 打印完整配置
echo "================================================================"
echo "🚀 启动 vLLM OpenAI 服务器"
echo "================================================================"
echo "  • 模型路径: $MODEL_PATH"
echo "  • 模型名称: $MODEL_NAME (从路径提取)"
echo "  • 服务地址: http://$HOST:$PORT"
echo "  • Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  • GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  • Max Model Len: $MAX_MODEL_LEN"
echo "  • Max Num Seqs: $MAX_NUM_SEQS"
echo "  • DType: $DTYPE"
echo "  • 日志文件: $LOG_FILE"
echo "================================================================"
echo ""

# 构建命令
CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs $MAX_NUM_SEQS \
    --dtype $DTYPE \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json"

echo "执行命令:"
echo "$CMD"
echo ""

# 启动服务器
if [ -n "$DAEMON" ]; then
    # 后台运行
    echo "🔧 以守护进程模式启动..."
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    
    # 保存 PID
    echo $PID > ~/.vllm_server.pid
    
    echo "✅ vLLM 服务器已在后台启动 (PID: $PID)"
    echo "   日志文件: $LOG_FILE"
    echo "   使用以下命令停止: kill $PID"
    echo "   或查看 PID 文件: ~/.vllm_server.pid"
    echo ""
    
    # 等待几秒钟检查服务器是否启动成功
    echo "⏳ 等待服务器启动..."
    sleep 10
    
    # 检查服务器是否运行
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ 服务器健康检查通过"
        echo ""
        echo "🎉 服务器已成功启动！"
        echo "   API Base URL: http://$HOST:$PORT/v1"
        echo ""
    else
        echo "⚠️  服务器可能未成功启动，请查看日志文件: $LOG_FILE"
    fi
else
    # 前台运行
    echo "🔧 以前台模式启动..."
    echo "   按 Ctrl+C 停止服务器"
    echo ""
    
    $CMD 2>&1 | tee "$LOG_FILE"
fi

