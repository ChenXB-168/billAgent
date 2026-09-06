import asyncio
import sys

# Windows 下输出被重定向到文件时，Python 会退回系统 locale(GBK)，导致中文日志乱码。
# 显式把标准输出/错误流改为 UTF-8，保证任何启动方式下的日志编码一致。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from startup.chat_loop import run_chat


if __name__ == "__main__":
    asyncio.run(run_chat())