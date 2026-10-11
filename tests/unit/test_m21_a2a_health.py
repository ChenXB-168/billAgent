# -*- coding: utf-8 -*-
"""M21 单元测试：A2A 结果泄漏防护（T-2）+ MCP 运行期心跳看门狗（T-4）。

覆盖：
  T-2 ① `wait_result` 超时后 `send_result` 的结果被**丢弃**（不再造成常驻泄漏）；
      ② 正常路径（有等待者）语义**零变更**——结果仍能正常取回；
      ③ `_abandoned` 标记**消费一次即移除**（不重复丢弃同一条）；
      ④ `_abandoned` 容量**有界**（达上限按插入序淘汰最旧）；
      ⑤ `clear_all()` 一并清空标记。
  T-4 ⑥ 间隔 <= 0 或 stdio 回退模式下看门狗**立即返回**（不挂起）；
      ⑦ 探活失败时**打 WARNING 告警**（可观测性核心产出）。

★隔离：每个用例使用**独立**的 `A2AMessageBus()` 实例，不触碰全局单例 `a2a_bus`；
  watchdog 用例全部 patch 掉配置与探活函数，不发真实 HTTP。
"""
import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, patch

import pytest

from mcpGateway import a2a_queue as aq
from mcpGateway.a2a_queue import A2AMessageBus
from startup import bootstrap as bs


# ============================ T-2：结果泄漏防护

@pytest.mark.asyncio
async def test_send_result_dropped_when_abandoned():
    """
    函数功能与逻辑描述：
        验证 T-2 的核心行为：`wait_result` 超时后（编排器已放弃），worker 事后调
        `send_result` **不得**再写入 `_result_store` / `_result_events`
        ——否则两条记录常驻内存、无人消费（改造前的泄漏形态）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    bus = A2AMessageBus()

    with pytest.raises(TimeoutError):
        await bus.wait_result("t1", timeout=0.01)

    assert "t1" in bus._abandoned                 # 超时已登记"已放弃"

    bus.send_result("t1", "late-result")

    assert "t1" not in bus._result_store          # ★丢弃，未写入
    assert "t1" not in bus._result_events         # ★未创建事件
    assert "t1" not in bus._abandoned             # 标记已消费（不重复生效）


@pytest.mark.asyncio
async def test_send_result_normal_path_unaffected():
    """
    函数功能与逻辑描述：
        验证**零回归**：正常路径（编排器正在 `wait_result` 等待）下，`send_result`
        照常写入结果并唤醒等待者，取值成功——T-2 只影响"已放弃"这一异常路径。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    bus = A2AMessageBus()

    async def _waiter():
        return await bus.wait_result("t2", timeout=1)

    task = asyncio.create_task(_waiter())
    await asyncio.sleep(0.05)                     # 让 wait_result 先建好 Event
    bus.send_result("t2", {"ok": True})

    assert await task == {"ok": True}
    assert "t2" not in bus._abandoned


@pytest.mark.asyncio
async def test_send_result_before_waiter_still_kept():
    """
    函数功能与逻辑描述：
        验证"结果先到、等待后至"的**既有语义不被破坏**（这是 `send_result` 原先
        "无条件先存入"的存在理由，也是不能改成"无等待者即丢弃"的原因——那会造成竞态）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    bus = A2AMessageBus()
    bus.send_result("t3", "early-result")         # 先回传
    assert await bus.wait_result("t3", timeout=1) == "early-result"


def test_abandoned_is_bounded():
    """
    函数功能与逻辑描述：
        验证 `_abandoned` 自身**有界**：连续登记超过 `_ABANDONED_MAX` 后长度不再增长
        （按插入序淘汰最旧），避免防泄漏机制反向泄漏。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    bus = A2AMessageBus()
    for i in range(aq._ABANDONED_MAX + 25):
        bus._remember_abandoned(f"t{i}")

    assert len(bus._abandoned) == aq._ABANDONED_MAX
    assert "t0" not in bus._abandoned             # 最旧的已被淘汰
    assert f"t{aq._ABANDONED_MAX + 24}" in bus._abandoned   # 最新的仍在


def test_clear_all_resets_abandoned():
    """
    函数功能与逻辑描述：
        验证 `clear_all()` 与结果缓存**同生命周期**地清空"已放弃"标记
        （否则跨用例/跨启动会残留脏标记，误丢新任务的结果）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    bus = A2AMessageBus()
    bus._remember_abandoned("t9")
    assert bus._abandoned

    bus.clear_all()

    assert bus._abandoned == {}


# ============================ T-4：运行期心跳看门狗

@pytest.mark.asyncio
async def test_watchdog_returns_immediately_when_disabled():
    """
    函数功能与逻辑描述：
        验证关闭开关：`MCP_HEALTH_INTERVAL <= 0` 时看门狗**立即返回**（不进入循环、
        不挂起），保证回滚开关可用。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    with patch.object(bs, "MCP_HEALTH_INTERVAL", 0):
        await asyncio.wait_for(bs._mcp_health_watchdog(), timeout=0.5)


@pytest.mark.asyncio
async def test_watchdog_returns_immediately_in_stdio_fallback():
    """
    函数功能与逻辑描述：
        验证 stdio 回退模式下无常驻服务可探，看门狗**立即返回**（不产生无谓探活）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    with patch.object(bs, "MCP_STDIO_FALLBACK", True):
        await asyncio.wait_for(bs._mcp_health_watchdog(), timeout=0.5)


@pytest.mark.asyncio
async def test_watchdog_warns_when_probe_fails():
    """
    函数功能与逻辑描述：
        验证 T-4 的核心产出：探活失败时**打 WARNING 告警**（这是把"240s 盲等"变成
        "可告警信号"的关键动作）。用极短间隔触发一轮，随后取消看门狗。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    warnings: list = []
    with patch.object(bs, "MCP_HEALTH_INTERVAL", 0.01), \
         patch.object(bs, "MCP_STDIO_FALLBACK", False), \
         patch.object(bs, "MCP_SERVER_PORTS", {"sql_bill": 8001}), \
         patch.object(bs, "_probe_mcp_server", new=AsyncMock(return_value=False)), \
         patch.object(bs.logger, "warning", side_effect=lambda m: warnings.append(str(m))):
        task = asyncio.create_task(bs._mcp_health_watchdog())
        await asyncio.sleep(0.06)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert any("探活失败" in m for m in warnings), f"未看到探活失败告警：{warnings}"
    assert any("sql_bill" in m for m in warnings)


@pytest.mark.asyncio
async def test_watchdog_reports_recovery():
    """
    函数功能与逻辑描述：
        验证"失败 → 恢复"的状态变化会被打出 INFO 日志（连续失败计数归零），
        使运维能确认服务已自愈。探活结果由 False 翻转为 True 来驱动。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    infos: list = []
    probe = AsyncMock(side_effect=[False, True, True, True, True, True])
    with patch.object(bs, "MCP_HEALTH_INTERVAL", 0.01), \
         patch.object(bs, "MCP_STDIO_FALLBACK", False), \
         patch.object(bs, "MCP_SERVER_PORTS", {"llm_base": 8002}), \
         patch.object(bs, "_probe_mcp_server", new=probe), \
         patch.object(bs.logger, "warning", side_effect=lambda m: None), \
         patch.object(bs.logger, "info", side_effect=lambda m: infos.append(str(m))):
        task = asyncio.create_task(bs._mcp_health_watchdog())
        await asyncio.sleep(0.08)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert any("已恢复" in m for m in infos), f"未看到恢复日志：{infos}"
