# ===================== 测试全局开关：LLM 一律走外部强模型 =====================
# 目的：测试验证的是 **harness 的可用性与正确性**，不是本地 1.8B 小模型的推理能力。
#       本地 CPU 推理单次可达 189s（e2e 4 条用例 36 分钟），且 JSON 输出不稳定易误判为缺陷。
# 作用范围：本进程 + 由本进程拉起的全部 MCP server 子进程（环境变量继承）。
# 配套硬开关 BILLAGENT_DISABLE_LOCAL_LLM=1：本地通道直接封死，
#   万一仍有调用走到 ollama_base_call 会**立即抛错**，便于定位路由遗漏，而不是静默变慢。
# 关闭方式：跑测试前显式设 BILLAGENT_FORCE_EXTERNAL=0 / BILLAGENT_DISABLE_LOCAL_LLM=0
# ⚠️ 必须在本文件 import 任何项目模块之前设置（`config.py` 在 import 期读取环境变量）。
import os
os.environ.setdefault("BILLAGENT_FORCE_EXTERNAL", "1")
os.environ.setdefault("BILLAGENT_DISABLE_LOCAL_LLM", "1")

import sys
import pytest
import subprocess
import time
import signal

import uuid

@pytest.fixture(scope="function")
def test_session_id():
    return f"test_{uuid.uuid4()}"

# 全局会话生命周期：所有测试执行前启动LLM MCP，全部跑完再关闭
@pytest.fixture(scope="session", autouse=True)
def spawn_llm_mcp_server():
    # 启动LLM MCP子进程，复用你原有启动脚本，不修改mcp客户端
    proc = subprocess.Popen(
        # 用当前解释器(venv)启动，避免系统 python 缺依赖导致 MCP 服务起不来
        [sys.executable, r"D:\MyCode\billAgent\mcpGateway\server_llm_base.py"],
        stdout=None,
        stderr=None,
        text=True
    )
    # 等待模型、MCP服务完全加载就绪
    time.sleep(4)
    yield proc
    # 测试全部结束后，终止子进程释放资源
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()