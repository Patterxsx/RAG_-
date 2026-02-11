#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用启动脚本 - 自动检测环境并启动
"""
import subprocess
import sys
import os
import webbrowser
import time

def check_python():
    """检查Python环境"""
    print("检查Python环境...")
    
    # 检查是否在conda环境中
    if sys.executable.endswith("python.exe") or sys.executable.endswith("python"):
        print(f"使用Python: {sys.executable}")
        return sys.executable
    
    return None

def install_deps():
    """安装依赖（如果没有）"""
    print("检查依赖...")
    try:
        import streamlit
        import langchain
        import faiss
        print("依赖已安装")
        return True
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("正在安装requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True

def main():
    print("="*50)
    print("红楼夜话 - 启动器")
    print("="*50)
    
    python_exe = check_python()
    if not python_exe:
        print("错误：未找到Python解释器")
        input("按回车退出...")
        return
    
    # 安装依赖
    if not install_deps():
        return
    
    # 检查数据文件
    if not os.path.exists("faiss_index/honglou_index/index.faiss"):
        print("警告：未找到向量索引，请先运行 step2_vectorize.py 构建知识库")
        input("按回车退出...")
        return
    
    # 启动Streamlit
    print("启动服务...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    process = subprocess.Popen(
        [python_exe, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        env=env
    )
    
    print("等待服务启动...")
    time.sleep(3)
    
    # 自动打开浏览器
    webbrowser.open("http://localhost:8501")
    
    print("服务已启动，请勿关闭此窗口")
    print("按 Ctrl+C 停止服务")
    
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
        process.terminate()

if __name__ == "__main__":
    main()