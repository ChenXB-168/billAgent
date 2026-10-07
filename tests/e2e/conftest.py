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
    """
    函数功能与逻辑描述：
        pytest fixture（session 作用域，autouse=True），为整个 e2e 会话提供 MCP 支撑：
        会话开始前调用 startup.bootstrap.start_mcp_servers 拉起 3 个 MCP 常驻服务
        （HTTP 常驻模式），会话结束（yield 之后）调用 stop_mcp_servers 关闭，避免遗留孤儿进程。
        stdio 回滚模式（环境变量 BILLAGENT_MCP_STDIO=1）下 start/stop 均为空操作，自动跳过。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无显式返回值（yield 仅用于分离 setup/teardown，不向用例提供数据）。
    """
    asyncio.run(start_mcp_servers())
    yield
    asyncio.run(stop_mcp_servers())


@pytest_asyncio.fixture(autouse=True)
async def _start_agents_workers():
    """
    函数功能与逻辑描述：
        pytest 异步 fixture（function 作用域，autouse=True），保证每个 e2e 用例都有子 Agent 消费方：
        用例前清空 a2a_bus，再创建 4 个常驻 worker 任务（bill/stat/price/finance，复用
        startup.bootstrap 的生产 worker），并 sleep 0.2s 等待 worker 进入消费循环；
        用例结束后 cancel 并 gather 回收，避免进程残留与跨用例串消息。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无显式返回值（yield 仅用于分离 setup/teardown，不向用例提供数据）。
    """
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
