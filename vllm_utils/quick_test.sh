#!/bin/bash
# vLLM 快速测试脚本

echo "================================"
echo "🧪 vLLM 推理工具快速测试"
echo "================================"
echo ""

# 检查 vLLM 是否安装
echo "📦 检查 vLLM 安装..."
python -c "import vllm; print(f'✅ vLLM 版本: {vllm.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ vLLM 未安装，请先运行: pip install vllm"
    exit 1
fi

# 检查模型路径（请修改为你的模型路径）
MODEL_PATH="models/Qwen3-4B-Instruct-2507"
echo ""
echo "📂 检查模型路径..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi
echo "✅ 模型路径存在"

# 测试选择
echo ""
echo "请选择测试模式:"
echo "  1) 测试直接 Load 模式"
echo "  2) 测试 OpenAI 服务模式"
echo "  3) 测试两种模式"
echo ""
read -p "请输入选项 (1/2/3, 默认 1): " choice
choice=${choice:-1}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

case $choice in
    1)
        echo ""
        echo "▶️  测试直接 Load 模式..."
        python "$SCRIPT_DIR/test_vllm_inference.py" --mode direct --model-path "$MODEL_PATH"
        ;;
    2)
        echo ""
        echo "▶️  测试 OpenAI 服务模式..."
        echo "📝 注意: 需要先启动 vLLM 服务器"
        read -p "是否现在启动服务器? (y/n, 默认 y): " start_server
        start_server=${start_server:-y}
        
        if [ "$start_server" = "y" ]; then
            echo ""
            echo "🚀 启动 vLLM 服务器..."
            bash "$SCRIPT_DIR/start_vllm_server.sh" --model-path "$MODEL_PATH" --daemon
            
            echo ""
            echo "⏳ 等待服务器启动 (20秒)..."
            sleep 20
            
            echo ""
            echo "🔍 检查服务器状态..."
            curl -s http://localhost:8000/health > /dev/null
            if [ $? -eq 0 ]; then
                echo "✅ 服务器已启动"
            else
                echo "⚠️  服务器可能未成功启动，请检查日志"
            fi
        fi
        
        echo ""
        echo "▶️  运行测试..."
        python "$SCRIPT_DIR/test_vllm_inference.py" --mode server
        
        if [ "$start_server" = "y" ]; then
            echo ""
            read -p "是否停止服务器? (y/n, 默认 n): " stop_server
            stop_server=${stop_server:-n}
            
            if [ "$stop_server" = "y" ]; then
                echo "⏹️  停止服务器..."
                PID=$(cat ~/.vllm_server.pid 2>/dev/null)
                if [ -n "$PID" ]; then
                    kill $PID
                    echo "✅ 服务器已停止"
                else
                    echo "⚠️  未找到服务器 PID"
                fi
            fi
        fi
        ;;
    3)
        echo ""
        echo "▶️  测试两种模式..."
        python "$SCRIPT_DIR/test_vllm_inference.py" --mode both --model-path "$MODEL_PATH"
        ;;
    *)
        echo "❌ 无效的选项"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "🎉 测试完成！"
echo "================================"

