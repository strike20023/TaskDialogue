#!/usr/bin/env python
"""
vLLM 服务器管理脚本

用于启动、停止和管理 vLLM OpenAI 兼容服务器
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Optional

from .config import VLLMConfig, ServerConfig


def start_vllm_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    dtype: str = "auto",
    trust_remote_code: bool = True,
    enable_lora: bool = False,
    log_file: Optional[str] = None,
    daemon: bool = False,
) -> Optional[subprocess.Popen]:
    """
    启动 vLLM OpenAI 兼容服务器
    
    Args:
        model_path: 模型路径
        host: 主机地址
        port: 端口号
        tensor_parallel_size: Tensor 并行数
        gpu_memory_utilization: GPU 内存使用率
        max_model_len: 最大模型长度
        dtype: 数据类型
        trust_remote_code: 信任远程代码
        enable_lora: 启用 LoRA
        log_file: 日志文件路径
        daemon: 是否作为守护进程运行
    
    Returns:
        subprocess.Popen: 子进程对象（如果不是守护进程）
    """
    # 构建命令
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    if enable_lora:
        cmd.append("--enable-lora")
    
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    
    print("=" * 80)
    print("🚀 启动 vLLM OpenAI 服务器")
    print("=" * 80)
    print(f"  • 模型路径: {model_path}")
    print(f"  • 服务地址: http://{host}:{port}")
    print(f"  • Tensor Parallel Size: {tensor_parallel_size}")
    print(f"  • GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"  • DType: {dtype}")
    print("=" * 80)
    print(f"\n命令: {' '.join(cmd)}\n")
    
    # 打开日志文件
    if log_file:
        log_file_obj = open(log_file, "w")
        stdout = log_file_obj
        stderr = log_file_obj
        print(f"📝 日志输出到: {log_file}\n")
    else:
        stdout = None
        stderr = None
    
    # 启动服务器
    try:
        if daemon:
            # 守护进程模式
            process = subprocess.Popen(
                cmd,
                stdout=stdout,
                stderr=stderr,
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
            print(f"✅ vLLM 服务器已在后台启动 (PID: {process.pid})")
            print(f"   使用以下命令停止: kill {process.pid}")
            print(f"   或使用: python -m vllm_utils.vllm_server --stop --pid {process.pid}")
            
            # 保存 PID
            pid_file = Path.home() / ".vllm_server.pid"
            with open(pid_file, "w") as f:
                f.write(str(process.pid))
            print(f"   PID 已保存到: {pid_file}")
            
            return process
        else:
            # 前台模式
            process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
            print("✅ vLLM 服务器已启动")
            print("   按 Ctrl+C 停止服务器")
            
            # 等待进程结束
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n\n⏹️  收到停止信号，正在关闭服务器...")
                process.terminate()
                process.wait(timeout=10)
                print("✅ 服务器已停止")
            
            return None
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        if log_file:
            log_file_obj.close()
        raise


def stop_vllm_server(pid: Optional[int] = None):
    """
    停止 vLLM 服务器
    
    Args:
        pid: 进程 ID，如果为 None 则从 PID 文件读取
    """
    if pid is None:
        # 从 PID 文件读取
        pid_file = Path.home() / ".vllm_server.pid"
        if not pid_file.exists():
            print("❌ 未找到 PID 文件，无法停止服务器")
            print(f"   PID 文件路径: {pid_file}")
            return
        
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
    
    print(f"⏹️  正在停止 vLLM 服务器 (PID: {pid})...")
    
    try:
        # 发送 SIGTERM 信号
        os.kill(pid, signal.SIGTERM)
        
        # 等待进程结束
        for _ in range(20):  # 最多等待 10 秒
            time.sleep(0.5)
            try:
                os.kill(pid, 0)  # 检查进程是否存在
            except OSError:
                print("✅ 服务器已停止")
                
                # 删除 PID 文件
                pid_file = Path.home() / ".vllm_server.pid"
                if pid_file.exists():
                    pid_file.unlink()
                
                return
        
        # 如果还没停止，发送 SIGKILL
        print("⚠️  服务器未能优雅关闭，强制停止...")
        os.kill(pid, signal.SIGKILL)
        print("✅ 服务器已强制停止")
        
    except ProcessLookupError:
        print("⚠️  进程不存在，可能已经停止")
    except Exception as e:
        print(f"❌ 停止服务器失败: {e}")


def check_server_status(host: str = "localhost", port: int = 8000) -> bool:
    """
    检查服务器状态
    
    Args:
        host: 主机地址
        port: 端口号
    
    Returns:
        bool: 服务器是否运行
    """
    try:
        import requests
        response = requests.get(f"http://{host}:{port}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ vLLM 服务器正在运行 (http://{host}:{port})")
            return True
        else:
            print(f"⚠️  服务器响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务器 (http://{host}:{port})")
        return False
    except Exception as e:
        print(f"❌ 检查服务器状态失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="vLLM 服务器管理工具")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # start 命令
    start_parser = subparsers.add_parser("start", help="启动服务器")
    start_parser.add_argument("--model-path", type=str, required=True, help="模型路径")
    start_parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    start_parser.add_argument("--port", type=int, default=8000, help="端口号")
    start_parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor 并行数")
    start_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU 内存使用率")
    start_parser.add_argument("--max-model-len", type=int, help="最大模型长度")
    start_parser.add_argument("--dtype", type=str, default="auto", help="数据类型")
    start_parser.add_argument("--trust-remote-code", action="store_true", default=True, help="信任远程代码")
    start_parser.add_argument("--enable-lora", action="store_true", help="启用 LoRA")
    start_parser.add_argument("--log-file", type=str, help="日志文件路径")
    start_parser.add_argument("--daemon", action="store_true", help="后台运行")
    
    # stop 命令
    stop_parser = subparsers.add_parser("stop", help="停止服务器")
    stop_parser.add_argument("--pid", type=int, help="进程 ID")
    
    # status 命令
    status_parser = subparsers.add_parser("status", help="检查服务器状态")
    status_parser.add_argument("--host", type=str, default="localhost", help="主机地址")
    status_parser.add_argument("--port", type=int, default=8000, help="端口号")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start_vllm_server(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            enable_lora=args.enable_lora,
            log_file=args.log_file,
            daemon=args.daemon,
        )
    elif args.command == "stop":
        stop_vllm_server(pid=args.pid)
    elif args.command == "status":
        check_server_status(host=args.host, port=args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

