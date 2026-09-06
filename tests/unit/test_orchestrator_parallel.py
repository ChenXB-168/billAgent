# -*- coding: utf-8 -*-
"""M6（P0-2 依赖动态化 + P1 批量派发/批量等待）单元测试。

覆盖 `03` §9.8.4 / §9.8.5 与 `11` M6 验收：

| 用例 | 验证点 |
|---|---|
| `test_collect_llm_deps_extracts_valid` | 提取 LLM `deps`（本批任务内的有效引用） |
| `test_collect_llm_deps_drops_invalid_refs` | 步骤④：丢弃不存在任务的引用 / 自依赖 / 非数组 |
| `test_merge_deps_union_and_prune` | 步骤①②：静态表 ∪ LLM deps，并裁剪到本批任务内 |
| `test_merge_deps_static_cannot_be_removed` | 🔴 **安全不变式**：LLM 只能**增加**依赖，不能删除静态表依赖 |
| `test_merge_deps_empty_llm_equals_static` | `llm_deps={}` 即回退纯静态表（环回退的实现基础） |
| `test_plan_node_ring_from_llm_deps_falls_back` | 🔴 **核心**：LLM deps 成环 → 回退静态表**继续执行**（非终止） |
| `test_dispatch_sends_all_ready_tasks` | 批量派发：`stat ∥ price` 同波发出，`finance` 未就绪不派 |
| `test_dispatch_dependency_chain_single_ready` | 依赖链场景只派 1 个 —— 行为与改造前**一致**（红线 1 基础） |
| `test_wait_result_collects_wave_concurrently` | 并发收集：耗时 ≈ `max(T)` 而非 `ΣT`；结果按任务归位 + 写 cache |
| `test_wait_result_round_limit_terminates` | 波次上限 `MAX_DISPATCH_ROUND`（8）→ 终止 |
| `test_wait_result_no_running_terminates` | 无 running 任务 → 终止（防 dispatch↔wait 死循环） |

★隔离：全部 patch 掉 A2A 总线、DB 查询、会话内存与 TaskManager 建单，不触达真实服务与数据库。
"""
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.config import MAX_DISPATCH_ROUND
from agents.orchestrator import nodes
from agents.orchestrator.nodes import _collect_llm_deps, _merge_deps


def _task(agent_id, deps, status="pending", result=None, task_id=None):
    """构造一个 task_plan 条目（字段与 `plan_node` 标准化后的结构一致）"""
    item = {
        "status": status,
        "deps": list(deps),
        "raw_segments": ["测试片段"],
        "operate_sub_type": "add" if agent_id == "bill_agent" else "query",
        "result": result,
        "agent_id": agent_id,
    }
    if task_id:
        item["task_id"] = task_id
    return item


def _state(task_plan, round_: int = 0) -> dict:
    return {
        "session_id": "test-m6-session",
        "user_input": "记午饭56元，查本月开销，看看广州贵不贵",
        "session_history": "",
        "task_plan": task_plan,
        "current_task": None,
        "all_task_results": [],
        "error_msg": None,
        "final_reply": "",
        "current_agent": "orchestrator_agent",
        "current_sub_agent_struct": None,
        "agent_result_cache": {},
        "dispatch_round": round_,
    }


# ===================== 依赖动态化：deps 提取与合并 =====================
def test_collect_llm_deps_extracts_valid():
    tasks = [
        {"name": "stat", "raw_segments": [], "operate_sub_type": "query", "deps": ["bill"]},
        {"name": "price", "raw_segments": [], "operate_sub_type": "query", "deps": ["bill", "stat"]},
    ]
    out = _collect_llm_deps(tasks, {"bill", "stat", "price"})
    assert out == {"stat": {"bill"}, "price": {"bill", "stat"}}


def test_collect_llm_deps_drops_invalid_refs():
    """步骤④：引用校验——不存在任务 / 自依赖 / 非数组一律丢弃"""
    tasks = [
        {"name": "stat", "deps": ["bill", "not_exist"]},     # not_exist 不在本批 → 丢
        {"name": "price", "deps": ["price"]},                # 自依赖 → 丢
        {"name": "bill", "deps": "stat"},                    # 非数组 → 忽略
        {"name": "finance", "deps": []},                     # 空数组 → 不记录
        "not-a-dict",                                        # 脏数据 → 忽略
    ]
    out = _collect_llm_deps(tasks, {"bill", "stat", "price", "finance"})
    assert out == {"stat": {"bill"}}


def test_merge_deps_union_and_prune():
    """步骤①②：并集 + 裁剪（只保留本批任务内存在的依赖）"""
    task_plan = {
        "bill": _task("bill_agent", []),
        "stat": _task("stat_agent", ["bill"]),
        "price": _task("price_agent", ["bill"]),
        "finance": _task("finance_agent", ["bill", "stat", "price"]),
    }
    # LLM 让 finance 额外等 price（本就不在静态表的 finance deps 里？静态表已含 → 演示并集去重）
    _, added = _merge_deps(task_plan, {"finance": {"stat"}, "stat": {"bill", "ghost"}})
    assert task_plan["finance"]["deps"] == ["bill", "price", "stat"]   # 静态 ∪ 动态（已含 stat → 无新增）
    assert task_plan["stat"]["deps"] == ["bill"]                       # ghost 被裁剪
    assert task_plan["bill"]["deps"] == []                             # 只问统计时不强制先记账
    assert added == {}                                                  # 无"静态表之外"的新增


def test_merge_deps_static_cannot_be_removed():
    """🔴 安全不变式：LLM **只能增加**依赖，不能删除静态表依赖（否则可能"本该串行却并行"）"""
    task_plan = {"bill": _task("bill_agent", []), "stat": _task("stat_agent", ["bill"])}
    # 恶意/错误 LLM 输出：声称 stat 不依赖 bill
    _merge_deps(task_plan, {"stat": set()})
    assert task_plan["stat"]["deps"] == ["bill"]   # 静态依赖仍在


def test_merge_deps_empty_llm_equals_static():
    """`llm_deps={}` = 回退纯静态表（环回退与开关关闭的实现基础）"""
    task_plan = {"bill": _task("bill_agent", []), "stat": _task("stat_agent", ["bill"])}
    _merge_deps(task_plan, {"stat": {"bill"}, "bill": {"stat"}})   # 先污染成环
    assert nodes._has_cycle(task_plan)
    _merge_deps(task_plan, {})                                     # 回退
    assert not nodes._has_cycle(task_plan)
    assert task_plan["bill"]["deps"] == []


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_ring_from_llm_deps_falls_back(mock_parse, monkeypatch):
    """🔴 核心验收：`03` §9.8.5 ③ —— LLM deps 成环时**回退静态表并继续执行**，而非终止。

    构造：静态表 `stat ← bill`；LLM 额外声明 `bill ← stat` ⇒ 并集成环。
    """
    monkeypatch.setattr(nodes, "DYNAMIC_DEPS_ENABLED", True)
    mock_parse.return_value = {
        "tasks": [
            {"name": "bill", "raw_segments": ["记午饭56元"], "operate_sub_type": "add",
             "deps": ["stat"]},                       # ← LLM 注入的成环依赖
            {"name": "stat", "raw_segments": ["查本月开销"], "operate_sub_type": "query",
             "deps": ["bill"]},
        ],
        "pre_check_pass": True,
        "block_tip": ""
    }
    state = _state({})
    state["user_input"] = "记午饭56元，查本月开销"

    cmd = await nodes.plan_node(state)

    # 未终止：进入 dispatch，且 error_msg 为空
    assert cmd.goto == "dispatch_node"
    assert cmd.update["error_msg"] is None
    # 依赖已回退为纯静态表：bill 无依赖、stat ← bill
    tp = cmd.update["task_plan"]
    assert tp["bill"]["deps"] == []
    assert tp["stat"]["deps"] == ["bill"]


# ===================== 批量派发 =====================
def _mock_session_mem():
    mem = MagicMock()
    mem.export_all.return_value = {}
    mem.add_msg = MagicMock()
    return mem


def _dispatch_ctx():
    """dispatch_node 的外部依赖隔离（不触达真实总线 / DB / 会话存储 / 状态库）"""
    return (
        patch.object(nodes.a2a_bus, "send_task"),
        patch.object(nodes, "get_user_target_config", new=AsyncMock(return_value={"city": "广州"})),
        patch.object(nodes, "get_session_memory", return_value=_mock_session_mem()),
        patch.object(nodes, "_load_recent_habits", new=AsyncMock(return_value=[])),
        patch("startup.task_manager.get_task_manager"),   # 建单为延迟导入 → patch 模块属性
    )


@pytest.mark.asyncio
async def test_dispatch_sends_all_ready_tasks():
    """`stat ∥ price` 同波批量派发；`finance` 依赖未满足 → 本波不派"""
    plan = {
        "bill": _task("bill_agent", [], status="done",
                      result={"success": True, "agent_type": "bill_agent", "data": {}}),
        "stat": _task("stat_agent", ["bill"]),
        "price": _task("price_agent", ["bill"]),
        "finance": _task("finance_agent", ["bill", "stat", "price"]),
    }
    p_send, _cfg, _mem, _habit, p_tm = _dispatch_ctx()
    with p_send as send, _cfg, _mem, _habit, p_tm as get_tm:
        cmd = await nodes.dispatch_node(_state(plan))

    assert send.call_count == 2                                   # 批量派发（改造前为 1）
    assert {c.kwargs["target_agent"] for c in send.call_args_list} == {"stat_agent", "price_agent"}
    tp = cmd.update["task_plan"]
    assert tp["stat"]["status"] == "running" and tp["stat"].get("task_id")
    assert tp["price"]["status"] == "running" and tp["price"].get("task_id")
    assert tp["finance"]["status"] == "pending"                   # 依赖未满足 → 本波不派
    assert get_tm.return_value.create_task.call_count == 2        # 每个任务各建一单（幂等键）


@pytest.mark.asyncio
async def test_dispatch_dependency_chain_single_ready():
    """依赖链场景（`bill → stat`）仍只派 1 个 —— 与改造前行为**一致**（红线 1 的基础）"""
    plan = {
        "bill": _task("bill_agent", []),
        "stat": _task("stat_agent", ["bill"]),
    }
    p_send, _cfg, _mem, _habit, p_tm = _dispatch_ctx()
    with p_send as send, _cfg, _mem, _habit, p_tm:
        cmd = await nodes.dispatch_node(_state(plan))

    assert send.call_count == 1
    assert send.call_args.kwargs["target_agent"] == "bill_agent"   # 只有 bill 就绪
    assert cmd.update["task_plan"]["stat"]["status"] == "pending"


# ===================== 批量等待 =====================
@pytest.mark.asyncio
async def test_wait_result_collects_wave_concurrently():
    """并发收集本波结果：耗时 ≈ `max(T)` 而非 `ΣT`，且结果按任务归位 + 写入 cache"""
    plan = {
        "stat": _task("stat_agent", [], status="running", task_id="t-stat"),
        "price": _task("price_agent", [], status="running", task_id="t-price"),
    }
    _T = 0.25   # 两个任务各自耗时 0.25s：串行 = 0.5s，并发 ≈ 0.25s

    async def _fake_wait(task_id, timeout=240):
        await asyncio.sleep(_T)
        agent = "stat_agent" if task_id == "t-stat" else "price_agent"
        return json.dumps({"success": True, "agent_type": agent, "msg": "ok",
                           "data": {}, "error": None}, ensure_ascii=False)

    with patch.object(nodes.a2a_bus, "wait_result", new=AsyncMock(side_effect=_fake_wait)):
        t0 = time.perf_counter()
        cmd = await nodes.wait_result_node(_state(plan))
        dt = time.perf_counter() - t0

    assert dt < _T * 1.7, f"整波耗时应≈max(T)={_T}s，实测 {dt:.2f}s（串行会≈{_T * 2}s）"
    tp = cmd.update["task_plan"]
    assert tp["stat"]["status"] == "done" and tp["price"]["status"] == "done"
    assert set(cmd.update["agent_result_cache"]) == {"stat_agent", "price_agent"}
    assert cmd.update["dispatch_round"] == 1            # 按**波次** +1（一次收齐两个）
    assert cmd.goto == "dispatch_node"
    # M9（D15）：习惯沉淀已下沉 bill_agent，orchestrator 收结果回路不再有沉淀调用（原断言点删除）


@pytest.mark.asyncio
async def test_wait_result_round_limit_terminates():
    """波次达 `MAX_DISPATCH_ROUND`（8）→ 终止，避免异常状态下无限调度"""
    plan = {"stat": _task("stat_agent", [], status="running", task_id="t-stat")}

    async def _fake_wait(task_id, timeout=240):
        return json.dumps({"success": True, "agent_type": "stat_agent", "msg": "ok",
                           "data": {}, "error": None})

    with patch.object(nodes.a2a_bus, "wait_result", new=AsyncMock(side_effect=_fake_wait)):
        cmd = await nodes.wait_result_node(_state(plan, round_=MAX_DISPATCH_ROUND))

    assert cmd.goto == "collect_node"
    assert f"调度波次超过{MAX_DISPATCH_ROUND}上限" in cmd.update["error_msg"]


@pytest.mark.asyncio
async def test_wait_result_no_running_terminates():
    """无 running 任务 → 终止（若回 dispatch 会造成 dispatch↔wait 死循环）"""
    plan = {"bill": _task("bill_agent", [], status="done")}
    cmd = await nodes.wait_result_node(_state(plan))
    assert cmd.goto == "collect_node"
    assert "未找到执行中的任务" in cmd.update["error_msg"]
