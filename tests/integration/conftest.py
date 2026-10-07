"""integration 测试的 MCP / 网络环境说明（★本文件刻意**不注册任何 fixture**）。

## 结论先说：integration 的失败根因不是"服务未拉起"，而是**缺 NO_PROXY**

2026-09-17 排查：`pytest tests/integration` 报 `9 failed, 5 passed`，其中
`test_mcp_gateway` 三条的异常链里出现 `handle_async_request(proxy_request)`
—— 即 **httpx 连接 127.0.0.1:8001/8002/8003 时走了系统代理**，而不是直连。

为什么偏偏测试环境缺这两行：生产入口 `webUI/app.py::main()` 里有

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

而 **pytest 不经过 `main()`** → 这两行缺失 → MCP 客户端连接全部走代理 → 失败。

✅ **修复**：已在顶层 `tests/conftest.py`（先于任何项目模块 import）补上同两行。
修复后实测：`pytest tests/integration` → **14 passed**（由 9 failed / 5 passed 转为全绿）。

## 为什么不在这里加 fixture（一次失败尝试的留档，勿重蹈）

曾在此加 session 级 sync fixture：

    @pytest.fixture(scope="session", autouse=True)
    def _mcp_servers():
        asyncio.run(start_mcp_servers())
        yield
        asyncio.run(stop_mcp_servers())

结果**从 `9 failed / 5 passed` 恶化成 `14 errors / 0 passed`** —— session 级
`asyncio.run(...)` 会创建并关闭一个事件循环，与 `pytest-asyncio` 为用例管理的 loop
**冲突**，连原本通过的 `test_session_memory.py`（3 条）都报 `RuntimeError`
（与 `设计_从零系列/07` 踩坑记录里 "attached to a different loop" 属同一类）。
`tests/e2e/conftest.py` 之所以能那样写，是因为它同时配有 `pytest_asyncio` 的 worker
fixture（`_start_agents_workers`）与相应 loop 约定，**此处不能照搬**。

## MCP 常驻服务由谁拉起

由 **顶层 `tests/conftest.py` 的 `spawn_mcp_servers`**（同步 `subprocess.Popen`，
无 loop 绑定）负责，本文件无需重复。

> 该 fixture 按 `mcpGateway.client.SERVER_CMD_MAP`（**单一来源**）拉起**全部 3 个**服务
> （sql_bill / llm_base / llm_finance），路径由 BASE_DIR 解析，可跨机器 / CI / WSL 运行。
> 使用**同步 Popen** 而非 `asyncio.run()`——后者会与 pytest-asyncio 的事件循环冲突，
> 即本文件开头记录的 14 errors 根因。
"""
