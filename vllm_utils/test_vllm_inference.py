#!/usr/bin/env python
"""
vLLM 推理测试脚本

测试直接 Load 模式和 OpenAI 服务模式
"""

import sys
import json
import argparse
from pathlib import Path

# 添加父目录到 Python 路径（TaskDialogue 根目录）
taskdialogue_path = Path(__file__).parent.parent
if str(taskdialogue_path) not in sys.path:
    sys.path.insert(0, str(taskdialogue_path))

from vllm_utils import VLLMInference, VLLMServerInference
from vllm_utils.config import VLLMConfig, SamplingConfig, ServerConfig


def test_direct_mode(model_path: str):
    """测试直接 Load 模式"""
    print("\n" + "=" * 80)
    print("测试 1: 直接 Load 模式")
    print("=" * 80 + "\n")
    
    # 配置
    vllm_config = VLLMConfig(
        model_path=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
    )
    
    sampling_config = SamplingConfig(
        temperature=0.01,
        max_tokens=256,
        top_p=0.95,
    )
    
    # 初始化推理引擎
    inference = VLLMInference(
        model_path=model_path,
        vllm_config=vllm_config,
        sampling_config=sampling_config,
    )
    
    # 测试用例 1: 简单对话
    print("\n--- 测试用例 1: 简单对话 ---")
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请用一句话介绍一下剑桥大学。"}
    ]
    
    response = inference.chat_completion(messages, return_usage=True)
    print(f"问题: {messages[1]['content']}")
    print(f"回答: {response['content']}")
    print(f"Token 使用: {response['usage']}")
    
    # 测试用例 2: Function Calling
    print("\n--- 测试用例 2: Function Calling ---")
    messages = [
        {"role": "system", "content": "你是一个旅行助手，可以帮助用户查询和预订酒店。"},
        {"role": "user", "content": "我想在剑桥找一家中等价位的酒店，要有免费停车。"}
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
                        "price": {"type": "string", "description": "价格范围: cheap, moderate, expensive"},
                        "parking": {"type": "string", "description": "是否需要停车: yes, no"},
                    },
                    "required": ["area"]
                }
            }
        }
    ]
    
    response = inference.chat_completion(messages, tools=tools, return_usage=True)
    print(f"问题: {messages[1]['content']}")
    print(f"回答: {json.dumps(response, ensure_ascii=False, indent=2)}")
    
    print("\n✅ 直接 Load 模式测试完成")


def test_server_mode(server_url: str = "http://localhost:8000/v1"):
    """测试 OpenAI 服务模式"""
    print("\n" + "=" * 80)
    print("测试 2: OpenAI 服务模式")
    print("=" * 80 + "\n")
    
    # 配置
    server_config = ServerConfig(
        api_base=server_url,
        api_key="EMPTY",
        timeout=120,
    )
    
    sampling_config = SamplingConfig(
        temperature=0.01,
        max_tokens=256,
        top_p=0.95,
    )
    
    # 初始化客户端
    inference = VLLMServerInference(
        server_config=server_config,
        sampling_config=sampling_config,
    )
    
    # 检查服务器状态
    print("检查服务器健康状态...")
    if not inference.check_health():
        print("❌ 服务器未运行或不可访问")
        print(f"   请先启动服务器: bash start_vllm_server.sh")
        return
    
    # 测试用例 1: 简单对话
    print("\n--- 测试用例 1: 简单对话 ---")
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请用一句话介绍一下剑桥大学。"}
    ]
    
    response = inference.chat_completion(messages, return_usage=True)
    print(f"问题: {messages[1]['content']}")
    print(f"回答: {response['content']}")
    print(f"Token 使用: {response['usage']}")
    
    # 测试用例 2: Function Calling
    print("\n--- 测试用例 2: Function Calling ---")
    messages = [
        {"role": "system", "content": "你是一个旅行助手，可以帮助用户查询和预订酒店。"},
        {"role": "user", "content": "我想在剑桥找一家中等价位的酒店，要有免费停车。"}
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
                        "price": {"type": "string", "description": "价格范围: cheap, moderate, expensive"},
                        "parking": {"type": "string", "description": "是否需要停车: yes, no"},
                    },
                    "required": ["area"]
                }
            }
        }
    ]
    
    response = inference.chat_completion(messages, tools=tools, return_usage=True)
    print(f"问题: {messages[1]['content']}")
    print(f"回答: {json.dumps(response, ensure_ascii=False, indent=2)}")
    
    print("\n✅ OpenAI 服务模式测试完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="vLLM 推理测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "server", "both"],
        default="both",
        help="测试模式: direct (直接模式), server (服务模式), both (两者都测试)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Qwen3-4B-Instruct-2507",
        help="模型路径（用于直接模式）"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="服务器 URL（用于服务模式）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🧪 vLLM 推理测试")
    print("=" * 80)
    
    try:
        if args.mode in ["direct", "both"]:
            test_direct_mode(args.model_path)
        
        if args.mode in ["server", "both"]:
            test_server_mode(args.server_url)
        
        print("\n" + "=" * 80)
        print("🎉 所有测试完成！")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

