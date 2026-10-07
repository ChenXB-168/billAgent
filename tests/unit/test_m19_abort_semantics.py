# -*- coding: utf-8 -*-
"""M19（D20）请求级终止语义 + worker 任务级超时 —— 单元测试（`设计/17`）。

覆盖五条口径：
  ① `wait_result_node` 识别任务级失败（超时 / 执行异常）→ 置 `abort_reason` 并终止后续派发；
  ② `ABORT_ON_TASK_FAILURE=0` 时退回改造前行为（失败结果继续溜进汇总，不终止）；
  ③ `collect_node` 在终止分支**如实告知"已完成部分"**（防"账已记但用户以为没事"的认知错位）；
  ④ `track()` 异常分支补代传结果（编排器不再盲等 240s 才收口）；
  ⑤ `track()` 任务级超时 → 取消内层 runner 并代传失败结果（防挂死任务占住消费循环）。

★隔离：不连真实 DB、不触达真实 A2A 总线与子 Agent——`wait_result` / `_abort_inflight` /
  `TaskManager` 的 DB 方法全部打桩，只验控制流与状态流转。
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.orchestrator import nodes
from startup.task_manager import TaskManager


# ---------------------------------------------------------------- 构造工具

def _task(agent_id, deps, status="pending", result=None, task_id=None):
    """
    函数功能与逻辑描述：
        构造 `task_plan` 里单个任务的信息 dict（字段与 `plan_node` 产出口径一致），
        供 `wait_result_node` / `collect_node` 读取。
    入参说明：
        agent_id (str)：任务归属的 Agent 标识。
        deps (list)：前置依赖任务名。
        status (str)：任务状态，默认 pending。
        result (dict | None)：结构化结果载荷（status=done 时提供）。
        task_id (str | None)：任务单号（status=running 时必填，供 wait_result 定位）。
    返回值说明：
        dict：任务信息字典。
    """
    return {"status": status, "deps": deps, "raw_segments": ["记午饭30"],
            "operate_sub_type": "add", "result": result,
            "agent_id": agent_id, "task_id": task_id}


def _state(plan, round_=0, **extra):
    """
    函数功能与逻辑描述：
        构造 `wait_result_node` / `collect_node` 所需的编排器 state（含 M19 新增的
        `abort_reason` 字段），可经 extra 覆盖或追加任意字段。
    入参说明：
        plan (dict)：task_plan。
        round_ (int)：当前调度波次。
        **extra：需要覆盖/追加的 state 字段（如 cancelled=True、abort_reason="task_timeout"）。
    返回值说明：
        dict：编排器 state。
    """
    st = {
        "user_input": "记午饭30", "session_id": "s_m19", "session_history": "",
        "task_plan": plan, "current_task": None, "all_task_results": [],
        "error_msg": None, "final_reply": "", "current_agent": "orchestrator_agent",
        "current_sub_agent_struct": None, "agent_result_cache": {},
        "dispatch_round": round_, "cancelled": False, "abort_reason": None,
    }
    st.update(extra)
    return st


def _payload(success, agent_type, error=None, data=None):
    """
    函数功能与逻辑描述：
        构造与子 Agent 回传同构的结果 JSON 字符串（`wait_result` 的返回形态）。
    入参说明：
        success (bool)：业务是否成功。
        agent_type (str)：来源 Agent 标识。
        error (str | None)：失败原因文案。
        data (dict | None)：业务数据。
    返回值说明：
        str：JSON 字符串。
    """
    return json.dumps({"success": success, "agent_type": agent_type,
                       "msg": "ok" if success else "任务执行失败",
                       "data": data or {}, "error": error}, ensure_ascii=False)


TIMEOUT_PAYLOAD = _payload(False, "stat_agent", "任务[t1]执行超时")
ERROR_PAYLOAD = _payload(False, "stat_agent", "KeyError: 'amount'")
# ★bill 成功载荷必须带 category + amount + consume_date：`build_user_display_text`
#   的 add/edit 分支走固定契约句式（下游断言依赖），三者缺一会 KeyError。
OK_PAYLOAD = _payload(True, "bill_agent", None,
                      {"amount": 30.0, "category": "餐饮", "consume_date": "2026-10-09"})


# ================================ ① / ② wait_result_node 的终止识别

@pytest.mark.asyncio
async def test_wait_result_timeout_aborts_round():
    """
    函数功能与逻辑描述：
        验证任务级**超时**触发本轮终止：`goto=collect_node`、`cancelled=True`、
        `abort_reason="task_timeout"`，且**确实调用了 `_abort_inflight`** 去中断在途 runner
        （否则超时的 runner 会继续占住该 Agent 的单协程消费循环）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {"stat": _task("stat_agent", [], status="running", task_id="t1")}

    with patch.object(nodes.a2a_bus, "wait_result",
                      new=AsyncMock(return_value=TIMEOUT_PAYLOAD)), \
         patch.object(nodes, "_abort_inflight",
                      new=AsyncMock(return_value={})) as abort_mock:
        cmd = await nodes.wait_result_node(_state(plan))

    assert cmd.goto == "collect_node"
    assert cmd.update["cancelled"] is True
    assert cmd.update["abort_reason"] == "task_timeout"
    abort_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_result_error_aborts_round():
    """
    函数功能与逻辑描述：
        验证非超时的任务级**执行异常**（`success=False` 且 error 不含"超时"）同样终止本轮，
        但 `abort_reason="task_error"`（回执文案与超时区分）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {"stat": _task("stat_agent", [], status="running", task_id="t1")}

    with patch.object(nodes.a2a_bus, "wait_result",
                      new=AsyncMock(return_value=ERROR_PAYLOAD)), \
         patch.object(nodes, "_abort_inflight", new=AsyncMock(return_value={})):
        cmd = await nodes.wait_result_node(_state(plan))

    assert cmd.goto == "collect_node"
    assert cmd.update["abort_reason"] == "task_error"


@pytest.mark.asyncio
async def test_wait_result_abort_disabled_keeps_legacy_behavior():
    """
    函数功能与逻辑描述：
        验证回滚开关：`ABORT_ON_TASK_FAILURE=0` 时退回改造前行为——失败结果不触发终止，
        流程照常 `goto=dispatch_node`（继续下一波派发），且**不调用** `_abort_inflight`。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {"stat": _task("stat_agent", [], status="running", task_id="t1")}

    with patch.object(nodes, "ABORT_ON_TASK_FAILURE", False), \
         patch.object(nodes.a2a_bus, "wait_result",
                      new=AsyncMock(return_value=TIMEOUT_PAYLOAD)), \
         patch.object(nodes, "_abort_inflight",
                      new=AsyncMock(return_value={})) as abort_mock:
        cmd = await nodes.wait_result_node(_state(plan))

    assert cmd.goto == "dispatch_node"
    assert not cmd.update.get("cancelled")
    abort_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_result_success_keeps_dispatching():
    """
    函数功能与逻辑描述：
        验证零回归：**全部成功**时不受 M19 影响，照常 `goto=dispatch_node` 继续调度，
        且不写 `abort_reason`。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {"bill": _task("bill_agent", [], status="running", task_id="t1")}

    with patch.object(nodes.a2a_bus, "wait_result",
                      new=AsyncMock(return_value=OK_PAYLOAD)):
        cmd = await nodes.wait_result_node(_state(plan))

    assert cmd.goto == "dispatch_node"
    assert cmd.update.get("abort_reason") is None


@pytest.mark.asyncio
async def test_wait_result_business_rejection_does_not_abort():
    """
    函数功能与逻辑描述：
        验证 ★关键边界（`设计/17` §六 R-1）：子 Agent 的**业务性拒绝**——典型即追问
        `need_more_info`——虽然 `success=False`，但属**正常业务交互**，**不得**被判为任务级
        失败而终止本轮；否则"信息不全时追问"这一核心交互会被打成"执行失败"。
        判据之所以用 `msg` 白名单（`_TASK_FAILURE_MSGS`）而非 `success is False`，正是为此。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {"bill": _task("bill_agent", [], status="running", task_id="t1")}
    ask_payload = json.dumps({
        "success": False, "agent_type": "bill_agent", "msg": "需要补充信息",
        "data": {"prompt": "请问这笔消费多少钱？"}, "error": "need_more_info"},
        ensure_ascii=False)

    with patch.object(nodes.a2a_bus, "wait_result",
                      new=AsyncMock(return_value=ask_payload)), \
         patch.object(nodes, "_abort_inflight",
                      new=AsyncMock(return_value={})) as abort_mock:
        cmd = await nodes.wait_result_node(_state(plan))

    assert cmd.goto == "dispatch_node"               # 继续调度，未终止
    assert not cmd.update.get("cancelled")
    assert cmd.update.get("abort_reason") is None
    abort_mock.assert_not_awaited()


# ================================ ③ collect_node 的终止回执

@pytest.mark.asyncio
async def test_collect_abort_reports_partial_done():
    """
    函数功能与逻辑描述：
        验证任务级失败收口时**如实告知已完成部分**：bill 成功、stat 超时 → 回执含
        "超时"与"已完成部分"，且带上 bill 的渲染文本（防"账已记但用户以为没发生"）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {
        "bill": _task("bill_agent", [], status="done",
                      result=json.loads(OK_PAYLOAD), task_id="t1"),
        "stat": _task("stat_agent", [], status="done",
                      result=json.loads(TIMEOUT_PAYLOAD), task_id="t2"),
    }

    with patch.object(nodes, "get_session_memory", return_value=MagicMock()):
        cmd = await nodes.collect_node(
            _state(plan, cancelled=True, abort_reason="task_timeout"))

    reply = cmd.update["final_reply"]
    assert cmd.goto == nodes.END
    assert "超时" in reply
    assert "已完成部分" in reply
    assert "餐饮" in reply                      # bill 的成功结果被如实带出


@pytest.mark.asyncio
async def test_collect_user_cancel_keeps_original_reply():
    """
    函数功能与逻辑描述：
        验证 M16 用户取消回执**零回归**：`abort_reason="user_cancel"` 时仍只回一句
        "已按您的要求停止本次请求。"，不追加"已完成部分"（取消语义 = 别再继续，不是汇报进度）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    plan = {"bill": _task("bill_agent", [], status="done",
                          result=json.loads(OK_PAYLOAD), task_id="t1")}

    with patch.object(nodes, "get_session_memory", return_value=MagicMock()):
        cmd = await nodes.collect_node(
            _state(plan, cancelled=True, abort_reason="user_cancel"))

    assert cmd.update["final_reply"] == "已按您的要求停止本次请求。"


# ================================ ④ / ⑤ track 的失败补偿

def _bare_tm():
    """
    函数功能与逻辑描述：
        构造一个**跳过 `__init__`** 的 TaskManager（不连真实 DB），并把四个 DB 落状态方法
        与 `_agent_of` 换成 MagicMock，使 `track()` 的控制流可被独立验证。
    入参说明：
        无。
    返回值说明：
        TaskManager：已装好桩件的实例（`_running_tasks` 为空字典）。
    """
    tm = TaskManager.__new__(TaskManager)
    tm._running_tasks = {}
    tm.mark_running = MagicMock(return_value=True)
    tm.mark_completed = MagicMock()
    tm.mark_failed = MagicMock()
    tm.mark_interrupted = MagicMock()
    tm.is_cancelled = MagicMock(return_value=False)
    tm._agent_of = MagicMock(return_value="bill_agent")
    return tm


@pytest.mark.asyncio
async def test_track_notifies_result_on_exception():
    """
    函数功能与逻辑描述：
        验证 M19 补偿④：runner 抛异常时 `track()` 除 `mark_failed` 外**还必须代传失败结果**
        （改造前只落状态 → 编排器拿不到回传，只能盲等 240s 超时）；异常仍向上抛给 worker 循环。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    tm = _bare_tm()
    notified = []

    async def _boom():
        raise ValueError("boom")

    with patch.object(TaskManager, "_notify_result",
                      staticmethod(lambda tid, res: notified.append((tid, res)))):
        with pytest.raises(ValueError):
            await tm.track({"task_id": "t1"}, _boom)

    tm.mark_failed.assert_called_once()
    assert len(notified) == 1
    assert notified[0][0] == "t1"
    assert notified[0][1]["success"] is False
    assert "ValueError" in notified[0][1]["error"]


@pytest.mark.asyncio
async def test_track_task_timeout_cancels_and_notifies():
    """
    函数功能与逻辑描述：
        验证 M19 补偿⑤：runner 挂死超过 `TASK_TIMEOUT` → `track()` 取消内层 Task、
        落 `failed`、代传"任务执行超时"结果，并**返回**（吞掉异常）使 worker 消费循环得以继续
        ——这是"挂死任务不永久占住消费循环"的核心保障。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    tm = _bare_tm()
    notified = []

    async def _hang():
        await asyncio.sleep(3600)

    with patch("startup.task_manager.TASK_TIMEOUT", 0.05), \
         patch.object(TaskManager, "_notify_result",
                      staticmethod(lambda tid, res: notified.append(res))):
        res = await tm.track({"task_id": "t1"}, _hang)

    assert res["success"] is False
    assert "超时" in res["error"]
    tm.mark_failed.assert_called_once()
    assert len(notified) == 1 and notified[0]["success"] is False
    assert "t1" not in tm._running_tasks          # finally 已清理登记表
