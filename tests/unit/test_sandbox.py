# -*- coding: utf-8 -*-
"""M4（D8 沙箱补强）单元测试。

覆盖（对齐 `11_开工顺序清单.md` M4 验收「构造死循环 / 爆内存工具 → 被拦住，
且常驻服务本身存活」）：

| 用例 | 验证点 | M4 映射 |
|---|---|---|
| `test_async_infinite_loop_tool_hit_timeout_and_loop_alive` | async 死循环工具 → 引擎超时拦截（TIMEOUT），且事件循环上其它任务照常完成 | 死循环被拦 + 服务存活 |
| `test_blocking_tool_hit_timeout_loop_alive` | 同步阻塞（`to_thread` 卸载）工具 → 超时拦截，事件循环不被冻结 | 同步调用被拦 + 服务存活 |
| `test_timeout_zero_disables_guard` | timeout=0（M4 回滚"可配置关闭"）→ 超时护栏关闭，慢工具正常返回 | 可配置关闭 |
| `test_memory_hog_child_is_killed` | 爆内存子进程（Windows Job Object / POSIX RLIMIT_AS）→ 被 OS 拦截 | 爆内存被拦 + 宿主进程不受影响 |
| `test_apply_process_limits_safe_branches` | 平台自适应分支不抛异常、关闭分支安全 | 沙箱是加固非故障点 |
| `test_sql_danger_whitelist_new_ops` | `ATTACH/DETACH/VACUUM/PRAGMA/REINDEX` 补入危险白名单 | 危险操作白名单 |

★不拉起真实 MCP 服务：死循环 / 超时用例走 `local` 协议（进程内注册坏工具），
内存用例 spawn 一次性子进程（sandbox 施加限制后被杀）——机制与常驻 server 完全一致。
"""
import asyncio
import os
import subprocess
import sys
import time

import pytest

from mcpGateway.adapters import LocalAdapter
from mcpGateway.executor import ExecutionEngine
from mcpGateway.registry import ToolRegistry
from mcpGateway.sandbox import apply_job_object_limits, apply_process_limits
from mcpGateway.sql_common.sql_validator import get_sql_operation
from mcpGateway.tool_model import ToolErrorKind, ToolSpec


def _build_registry_with(tool_fn, *, to_thread, timeout) -> ToolRegistry:
    """装配一个只含"故意坏工具"的独立 registry（不污染全局平台装配）"""
    reg = ToolRegistry()
    reg.register_adapter("local", LocalAdapter())
    reg.register(ToolSpec(
        name="t.bad", description="测试用坏工具（不进 list_tools）",
        params_schema={"type": "object", "properties": {}},
        protocol="local", binding={"func": tool_fn, "to_thread": to_thread},
        timeout=timeout, max_retry=0, hidden=True))
    reg.bind_engine(ExecutionEngine(reg))
    return reg


# ===================== 死循环 / 阻塞工具 → 超时拦截 + 服务存活 =====================
@pytest.mark.asyncio
async def test_async_infinite_loop_tool_hit_timeout_and_loop_alive():
    """async 死循环（`while True: await sleep`）→ wait_for 超时取消协程 → TIMEOUT"""
    async def _dead_loop(**kwargs):
        while True:
            await asyncio.sleep(1)

    reg = _build_registry_with(_dead_loop, to_thread=False, timeout=0.2)
    # 事件循环上并发一个"正常任务"，验证超时拦截不冻结服务（M4 验收：常驻服务本身存活）
    healthy = asyncio.create_task(asyncio.sleep(0.05))

    res = await reg.invoke("t.bad", {}, agent_id="bill_agent")

    assert res.ok is False
    assert res.error.kind is ToolErrorKind.TIMEOUT
    assert healthy.done() and healthy.result() is None   # 事件循环仍健康调度
    assert asyncio.get_running_loop().is_running()


@pytest.mark.asyncio
async def test_blocking_tool_hit_timeout_loop_alive():
    """同步阻塞工具（LocalAdapter 默认 to_thread 卸载）→ 超时返回，事件循环不被冻结"""
    def _blocking(**kwargs):
        time.sleep(0.5)
        return "done"

    reg = _build_registry_with(_blocking, to_thread=True, timeout=0.05)
    t0 = time.perf_counter()
    res = await reg.invoke("t.bad", {}, agent_id="bill_agent")
    elapsed = time.perf_counter() - t0

    assert res.ok is False
    assert res.error.kind is ToolErrorKind.TIMEOUT
    # 若同步调用未经 to_thread 卸载，事件循环被冻结，elapsed 会 ≥0.5s
    assert elapsed < 0.4


@pytest.mark.asyncio
async def test_timeout_zero_disables_guard():
    """M4 回滚要求：超时阈值可配置关闭（timeout=0 → 不做 wait_for，慢工具正常返回）"""
    async def _slow(**kwargs):
        await asyncio.sleep(0.15)
        return "ok"

    reg = _build_registry_with(_slow, to_thread=False, timeout=0)
    res = await reg.invoke("t.bad", {}, agent_id="bill_agent")
    assert res.ok is True
    assert res.data == "ok"


# ===================== 爆内存子进程 → OS 拦截 =====================
_ALLOC_CODE = (
    "import time\n"
    "blob = bytearray(1024 * 1024 * 256)   # 分配 256MB，远超 96MB 限制\n"
    "print('ALLOC_OK')\n"
    "time.sleep(0.2)\n"
)


def test_memory_hog_child_is_killed():
    """M4 验收核心：爆内存被拦 + 宿主（本 pytest 进程）不受影响。

    Windows：bootstrap 同款路径——子进程拉起后挂 Job Object（96MB 硬上限）；
    POSIX：无法事后约束子进程 → 复现 server 启动路径，用 preexec_fn 让子进程
    self-prison（RLIMIT_AS=96MB），与 `server_bill.py` 等 `__main__` 行为一致。
    """
    def _spawn():
        if os.name == "nt":
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", _ALLOC_CODE],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert apply_job_object_limits(proc, memory_mb=96) is True, "Job 挂载应成功"
            return proc
        # POSIX：preexec_fn 仅在 fork 后的子进程执行，设置 RLIMIT_AS
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _ALLOC_CODE],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=lambda: apply_process_limits(memory_mb=96))
        return proc

    proc = _spawn()
    out, _err = proc.communicate(timeout=60)

    assert "ALLOC_OK" not in out.decode(errors="replace"), \
        "子进程若成功分配 256MB 说明沙箱未生效"
    assert proc.returncode != 0, "超限子进程应被 OS 终止（非正常退出）"
    # 宿主进程存活：本断言能执行本身就是证明；再确认无残留子进程级联
    assert apply_process_limits(memory_mb=0) is False  # 宿主仍可正常调用沙箱（0=关闭分支）


# ===================== 平台自适应：沙箱是加固而非故障点 =====================
def test_apply_process_limits_safe_branches():
    """任何平台下 `memory_mb=0`（关闭）分支都应安全返回 False 且不抛异常"""
    assert apply_process_limits(memory_mb=0) is False       # 全平台安全
    assert apply_process_limits(memory_mb=-1) is False      # 负数同视为关闭
    if os.name != "posix":
        # Windows：self-prison no-op（限制由父进程 Job 承担）
        assert apply_process_limits(memory_mb=1024) is False
        # POSIX 上不调默认值——会把 pytest 自身进程 RLIMIT_AS 锁死


# ===================== 危险操作白名单补全 =====================
@pytest.mark.parametrize("sql,op", [
    ("ATTACH DATABASE 'x.db' AS x", "ATTACH"),     # M4 新增：挂外部库
    ("DETACH DATABASE x", "DETACH"),                # M4 新增
    ("VACUUM", "VACUUM"),                           # M4 新增：整库重写
    ("PRAGMA journal_mode=delete", "PRAGMA"),       # M4 新增：写类 pragma
    ("REINDEX idx_bill", "REINDEX"),                # M4 新增
    ("CREATE TABLE t (id INT)", "CREATE"),          # 原有 6 项回归
    ("DROP TABLE bill", "DROP"),
    ("GRANT ALL ON bill TO x", "GRANT"),
])
def test_sql_danger_whitelist_new_ops(sql, op):
    """危险操作白名单：DDL + 会话级/库级操作一律拒绝并归 danger 类"""
    kind, danger = get_sql_operation(sql)
    assert kind == "danger"
    assert danger == op
