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
# ★与生产入口 `webUI/app.py::main()` 对齐：让 localhost 调用**绕过系统代理**。
#   生产里这两行是必需的（否则请求走注册表代理 127.0.0.1:7892 而全挂，见 `07` 踩坑记录），
#   但 pytest **不经过 main()** → 缺失 → httpx 连 MCP 服务时走 `proxy_request` 完成连接，
#   报 "Connection…/unhandled errors in a TaskGroup（httpcore._connect）"。
#   2026-09-17 实测：`tests/integration` 9 failed 的异常链中明确出现 `proxy_request`。
#   仅影响 127.0.0.1 / localhost，不改变任何对外请求行为（setdefault 不覆盖已有值）。
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

import pytest
import subprocess
import time

import uuid

@pytest.fixture(scope="function")
def test_session_id():
    """
    函数功能与逻辑描述：
        为每个用例提供互不相同的会话标识，隔离会话级记忆/草稿，避免用例之间互相污染；
        纯构造随机串，无前置条件、无清理逻辑。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        str：形如 "test_<uuid4>" 的唯一会话 ID。
    """
    return f"test_{uuid.uuid4()}"

# ===================== 全局会话生命周期：拉起全部 MCP 常驻服务 =====================
@pytest.fixture(scope="session", autouse=True)
def spawn_mcp_servers(request):
    """
    函数功能与逻辑描述：
        为整个测试会话提供**全部 3 个** MCP 常驻服务（sql_bill / llm_base / llm_finance）：
        session 作用域 + autouse，在首个用例前拉起，会话结束统一回收。
        实现要点：遍历 `mcpGateway.client.SERVER_CMD_MAP`（**单一来源**，路径已由 BASE_DIR
        解析为绝对路径，可跨机器 / CI / WSL 运行）拉起全部服务，并用**同步 subprocess.Popen**
        ——不涉及 asyncio，故不会与 pytest-asyncio 管理的事件循环冲突。这一点很关键：
        session 级 `asyncio.run(start_mcp_servers())` 会因新建并关闭 loop 而恶化成 14 errors
        （留档见 `tests/integration/conftest.py`）；同步 Popen 无此问题。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        list[subprocess.Popen]：yield 出的 3 个 MCP 子进程句柄（会话结束后被终止或强制杀死）。
    """
    # unit 层对全部外部依赖打桩（README §10「单元测试：零外部依赖」），不需要任何 MCP 常驻服务；
    # 仅当本次会话收集到 integration / e2e 用例时才拉起，既保住 L1 的"零依赖"语义，
    # 也省掉每次 L1 回归的启动等待。
    def _is_service_case(item) -> bool:
        # ⚠️ 跨平台要点：Windows 下 fspath 是反斜杠（`...\tests\integration\x.py`），
        #   直接匹配 "/integration/" 会全部落空 → fixture 误判为"无需服务" → 用例连不上。
        p = str(item.fspath).replace("\\", "/")
        return "/integration/" in p or "/e2e/" in p
    _has_service_tests = any(_is_service_case(it) for it in request.session.items)
    if not _has_service_tests:
        yield []
        return

    # 延迟导入：确保本文件顶部的环境变量（FORCE_EXTERNAL / NO_PROXY 等）先生效
    from mcpGateway.client import SERVER_CMD_MAP

    procs: list[subprocess.Popen] = []
    for _key, cmd in SERVER_CMD_MAP.items():
        # cmd 已含「当前解释器(venv) + -u + 绝对脚本路径」，直接照搬即可
        procs.append(subprocess.Popen(cmd, stdout=None, stderr=None, text=True))
    # 3 个服务并行加载，留足就绪窗口（M3 实测常驻就绪约 6s；首启含模型加载）
    time.sleep(8)

    yield procs

    # 会话结束：逐个终止，3s 未退出则强杀，确保子进程不残留
    for p in procs:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
