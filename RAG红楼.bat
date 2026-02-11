@echo off
echo 正在启动RAG红楼...
echo 请勿关闭此窗口，最小化即可

:: 直接调用环境内的 Python 运行 streamlit（无需 conda activate）
D:\miniconda\envs\tinyllm\python.exe -m streamlit run "E:\003 project\001 RAG_HongLou\app.py"

pause