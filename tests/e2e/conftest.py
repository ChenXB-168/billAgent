"""e2e 测试专用 conftest：启动 MCP 常驻服务 + 4 个子 Agent 常驻 worker

关键：orchestrator 通过 A2A 内存队列派发子任务，若没有 worker 消费，
任务必然 240s 超时。与 tests/integration/test_memory_agent_integration.py
中的 _start_agents_workers fixture 保持一致（复用生产 bootstrap worker）。

★M3：HTTP 常驻模式下 `_call_server` 需连已拉起的 MCP 服务，但 e2e 直接调
graph（不经 bootstrap_all），故此处加 session 级 `_mcp_servers` fixture——
否则所有 MCP 调用连接失败走兜底回复，用例误判失败（`07` §3.2 To-Be 步骤3）。
stdio 回滚模式（BILLAGENT_MCP_STDIO=1）下 start_mcp_servers 为空操作，自动跳过。
"""
import asyncio

import pytest
import pytest_asyncio

from mcpGateway.a2a_queue import a2a_bus
from startup.bootstrap import (
    _bill_agent_worker,
    _finance_agent_worker,
    _price_agent_worker,
    _stat_agent_worker,
    start_mcp_servers,
    stop_mcp_servers,
)


@pytest.fixture(scope="session", autouse=True)
def _mcp_servers():
    """会话级：拉起 3 个 MCP 常驻服务（HTTP 常驻模式），会话结束关闭"""
    asyncio.run(start_mcp_servers())
    yield
    asyncio.run(stop_mcp_servers())


@pytest_asyncio.fixture(autouse=True)
async def _start_agents_workers():
    """每个用例前启动 4 个子 Agent 常驻 worker，用例结束取消"""
    a2a_bus.clear_all()
    workers = [
        asyncio.create_task(_bill_agent_worker()),
        asyncio.create_task(_stat_agent_worker()),
        asyncio.create_task(_price_agent_worker()),
        asyncio.create_task(_finance_agent_worker()),
    ]
    await asyncio.sleep(0.2)  # 等 worker 进入消费循环
    yield
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
