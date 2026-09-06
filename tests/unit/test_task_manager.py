# -*- coding: utf-8 -*-
"""M5（D2 Durable / D5 追问治理）单元测试。

覆盖 `11` M5 验收项：

| 用例 | 验证点 |
|---|---|
| `test_state_machine_happy_path` | submitted → running → completed 状态流转与结果落库 |
| `test_create_task_is_idempotent` | 同一 `task_id` 重复建单不重复（幂等键防重投，`04` §3.2） |
| `test_retriable_default_by_agent` | 写操作（`bill_agent`）默认 `retriable=0`，只读任务为 1（`04` §3.3） |
| `test_recover_bill_with_record_marks_completed` | 🔴 **核心**：崩溃在"已记账未标记" → `completed` 而非 `failed`（`04` §3.1） |
| `test_recover_bill_without_record_marks_failed` | 未落库 + 不可重投 → `failed`（宁可告警也不重复记账） |
| `test_recover_readonly_task_resubmits` | 只读任务 → `submitted` 且 `retry_cnt+1` |
| `test_recover_respects_max_retry` | 重投达上限 → `failed` |
| `test_recover_scans_interrupted` | 优雅关闭遗留的 `interrupted` 同样参与恢复（`08` §6 Q7） |
| `test_resubmit_submitted_calls_sender` | 重投参数正确（agent/session/task_id/payload 原样回填） |
| `test_track_*` | worker 包装：成功→completed / 异常→failed 并继续抛出 / 取消→interrupted |
| `test_bill_insert_writes_task_id` | 记账 INSERT 携带 `task_id`（崩溃对账的依据由代码层保证） |
| `test_ask_round_limit_gives_up` | 追问 3 轮上限 + 放弃提示语 + 放弃后计数清零（`01` §3.4） |

★隔离：TaskManager 用 `tmp_path` 建临时库，不污染真实 `database/task_state.db`；
对账用的业务库以 `_FakeBillDB` 注入，不触碰 `bill.db`。
"""
import asyncio
import json

import pytest

from config.config import ASK_GIVEUP_TIP, ASK_ROUND_LIMIT
from startup.task_manager import TaskManager


class _FakeBillDB:
    """模拟业务库对账：`exists` 控制 `bill` 表是否已有该 `task_id` 的记录"""

    def __init__(self, exists: bool = False):
        self.exists = exists
        self.queries: list = []

    def query_sql(self, sql: str, params=None):
        self.queries.append((sql, params))
        return [{"1": 1}] if self.exists else []


@pytest.fixture
def tm(tmp_path):
    """每个用例一个独立临时库（TaskManager 支持注入 db_path）"""
    manager = TaskManager(tmp_path / "task_state.db")
    yield manager
    manager.close()


# ===================== 状态机基础流转 =====================
def test_state_machine_happy_path(tm):
    tm.create_task("t001", "s1", "bill_agent", {"amount": 56})
    assert tm.get_task("t001")["status"] == "submitted"

    tm.mark_running("t001")
    assert tm.get_task("t001")["status"] == "running"

    tm.mark_completed("t001", {"success": True})
    task = tm.get_task("t001")
    assert task["status"] == "completed"
    assert json.loads(task["result"]) == {"success": True}


def test_create_task_is_idempotent(tm):
    """同一 task_id 重复投递（崩溃重投场景）不应重复建单或覆盖首次 payload"""
    tm.create_task("t001", "s1", "bill_agent", {"amount": 56})
    tm.create_task("t001", "s1", "bill_agent", {"amount": 9999})   # 重复投递
    rows = tm._db.query_sql("SELECT payload FROM task_state WHERE task_id='t001'")
    assert len(rows) == 1
    assert json.loads(rows[0]["payload"]) == {"amount": 56}   # 保持首次（IGNORE 语义）


def test_retriable_default_by_agent(tm):
    """`04` §3.3：有副作用的写任务默认禁止自动重投"""
    tm.create_task("t-bill", "s1", "bill_agent", {})
    tm.create_task("t-stat", "s1", "stat_agent", {})
    assert tm.get_task("t-bill")["retriable"] == 0
    assert tm.get_task("t-stat")["retriable"] == 1


# ===================== 崩溃恢复（04 §3.1 修订版）=====================
def test_recover_bill_with_record_marks_completed(tm):
    """🔴 核心验收：崩溃发生在「子进程已记账」与「主进程标记 completed」之间 →
    恢复必须标记 **completed**，否则会告诉用户"记账失败"而账其实已入库。"""
    tm.create_task("t-bill-1", "s1", "bill_agent", {"amount": 56})
    tm.mark_running("t-bill-1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=True))

    assert tm.get_task("t-bill-1")["status"] == "completed"   # 不是 failed！
    assert stat == {"completed": 1, "resubmitted": 0, "failed": 0}


def test_recover_bill_without_record_marks_failed(tm):
    """未落库 + 写操作不可重投 → failed（宁可告警，也不自动重投导致重复记账）"""
    tm.create_task("t-bill-2", "s1", "bill_agent", {"amount": 56})
    tm.mark_running("t-bill-2")

    tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    assert tm.get_task("t-bill-2")["status"] == "failed"


def test_recover_readonly_task_resubmits(tm):
    """只读任务无落库副作用 → 直接重投，retry_cnt 自增"""
    tm.create_task("t-stat-1", "s1", "stat_agent", {"q": "本月几笔"})
    tm.mark_running("t-stat-1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    task = tm.get_task("t-stat-1")
    assert task["status"] == "submitted"
    assert task["retry_cnt"] == 1
    assert stat["resubmitted"] == 1


def test_recover_respects_max_retry(tm):
    tm.create_task("t-stat-2", "s1", "stat_agent", {})
    tm.mark_running("t-stat-2")
    tm._db.execute_sql("UPDATE task_state SET retry_cnt=3 WHERE task_id='t-stat-2'")

    tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    assert tm.get_task("t-stat-2")["status"] == "failed"


def test_recover_scans_interrupted(tm):
    """优雅关闭（worker 被 cancel）遗留的 interrupted 也要参与恢复（`08` §6 Q7）"""
    tm.create_task("t-int-1", "s1", "stat_agent", {})
    tm.mark_running("t-int-1")
    tm.mark_interrupted("t-int-1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    assert tm.get_task("t-int-1")["status"] == "submitted"
    assert stat["resubmitted"] == 1


def test_resubmit_submitted_calls_sender(tm):
    """DB 是唯一事实源：`submitted` 任务被重投时，派发参数必须原样回填"""
    payload = {"raw_segments": ["打车35元"], "operate_sub_type": "add"}
    tm.create_task("t-resubmit", "s1", "stat_agent", payload)

    sent: list = []
    n = tm.resubmit_submitted(
        lambda agent_id, session_id, task_id, data: sent.append((agent_id, session_id, task_id, data)))

    assert n == 1
    assert sent == [("stat_agent", "s1", "t-resubmit", payload)]


# ===================== worker 执行包装 =====================
@pytest.mark.asyncio
async def test_track_success_marks_completed(tm):
    tm.create_task("t-ok", "s1", "stat_agent", {})   # track 只负责流转，建单在派发前完成

    async def _ok():
        return {"result": "done"}

    await tm.track({"task_id": "t-ok"}, _ok)
    assert tm.get_task("t-ok")["status"] == "completed"


@pytest.mark.asyncio
async def test_track_exception_marks_failed_and_reraises(tm):
    """异常要继续抛出——worker 的 `except Exception` 依赖它保证循环不中断"""
    tm.create_task("t-boom", "s1", "stat_agent", {})

    async def _boom():
        raise RuntimeError("图执行崩溃")

    with pytest.raises(RuntimeError):
        await tm.track({"task_id": "t-boom"}, _boom)

    task = tm.get_task("t-boom")
    assert task["status"] == "failed"
    assert "RuntimeError" in json.loads(task["result"])["error"]


@pytest.mark.asyncio
async def test_track_cancelled_marks_interrupted(tm):
    """优雅关闭：CancelledError 必须标 interrupted 而非 failed——
    否则下次启动会因 retriable=0（写任务）把"已记账"误判为 failed（`04` §3.1）"""
    tm.create_task("t-cancel", "s1", "bill_agent", {})

    async def _cancelled():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await tm.track({"task_id": "t-cancel"}, _cancelled)

    assert tm.get_task("t-cancel")["status"] == "interrupted"


# ===================== D5：追问 3 轮治理 =====================
@pytest.mark.asyncio
async def test_ask_round_limit_gives_up():
    """`01` §3.4：追问最多 3 轮，第 3 轮仍不满足 → 固定提示语放弃，且放弃后计数清零"""
    from unittest.mock import AsyncMock, patch

    from memory.short_memory import get_session_memory
    from agents.orchestrator import nodes

    sid = "test-ask-round-session"
    prompt = "请问这笔消费的金额是多少？"
    struct = {"success": False, "agent_type": "bill_agent", "msg": "需补充",
              "data": {"prompt": prompt}, "error": "need_more_info"}
    state = {
        "session_id": sid,
        "user_input": "打车",
        "task_plan": {"bill": {"agent_id": "bill_agent", "status": "done",
                               "operate_sub_type": "add", "result": struct}},
        "final_reply": None,
        "error_msg": None,
        "agent_result_cache": {},
    }

    # 隔离副作用：collect_node 成功路径会查预算/预警事实并可能调汇总 LLM，
    # 这里只验证追问轮次治理，故全部 stub 掉（不触达真实 DB / 模型）
    with patch.object(nodes, "_calc_month_budget_status", new=AsyncMock(return_value={})), \
            patch.object(nodes, "_calc_alert_facts", new=AsyncMock(return_value={})), \
            patch.object(nodes, "get_user_target_config", new=AsyncMock(return_value={})), \
            patch.object(nodes, "EXTERNAL_LLM_ENABLED", False):
        finals = []
        for _ in range(ASK_ROUND_LIMIT):
            cmd = await nodes.collect_node(state)
            finals.append(cmd.update["final_reply"])

    assert finals[:ASK_ROUND_LIMIT - 1] == [prompt] * (ASK_ROUND_LIMIT - 1)  # 前 2 轮正常追问
    assert finals[-1] == ASK_GIVEUP_TIP                                     # 第 3 轮放弃
    assert prompt not in finals[-1]                                          # 放弃时不再追加追问

    # 放弃后计数已清零 → 下一次记账重新计数（不误伤）
    mem = get_session_memory(sid)
    assert mem.get_ask_round("bill_add") == 0


@pytest.mark.asyncio
async def test_ask_round_resets_on_success():
    """任务成功（非追问结果）后计数清零，避免历史追问累计影响下次记账"""
    from unittest.mock import AsyncMock, patch

    from memory.short_memory import get_session_memory
    from agents.orchestrator import nodes

    sid = "test-ask-reset-session"
    need_more = {"success": False, "agent_type": "bill_agent", "msg": "需补充",
                 "data": {"prompt": "金额？"}, "error": "need_more_info"}
    # 字段对齐真实记账成功载荷（`bill_agent/nodes.py`）：渲染需 category/amount/consume_date
    ok = {"success": True, "agent_type": "bill_agent", "msg": "记账成功",
          "data": {"amount": 56, "category": "交通", "consume_date": "2026-09-03",
                   "remark": "打车56元"}, "error": None}

    def _state(result):
        return {
            "session_id": sid,
            "user_input": "打车56元",
            "task_plan": {"bill": {"agent_id": "bill_agent", "status": "done",
                                   "operate_sub_type": "add", "result": result}},
            "final_reply": None,
            "error_msg": None,
            "agent_result_cache": {},
        }

    with patch.object(nodes, "_calc_month_budget_status", new=AsyncMock(return_value={})), \
            patch.object(nodes, "_calc_alert_facts", new=AsyncMock(return_value={})), \
            patch.object(nodes, "get_user_target_config", new=AsyncMock(return_value={})), \
            patch.object(nodes, "EXTERNAL_LLM_ENABLED", False):
        await nodes.collect_node(_state(need_more))
        mem = get_session_memory(sid)
        assert mem.get_ask_round("bill_add") == 1

        await nodes.collect_node(_state(ok))
    assert mem.get_ask_round("bill_add") == 0   # 成功即清零


# ===================== D2：记账 INSERT 携带 task_id =====================
@pytest.mark.asyncio
async def test_bill_insert_writes_task_id():
    """`04` §3.1：崩溃对账的依据是 `bill.task_id`——必须由**代码层**写入，不依赖 LLM 生成 SQL"""
    from unittest.mock import AsyncMock, patch

    from agents.bill_agent import nodes as bill_nodes

    task_id = "t1756857000123a1b2c3"
    state = {
        "session_id": "s-bill-task",
        "task_id": task_id,
        "raw_segments": ["打车56元"],
        "operate_sub_type": "add",
        "city": "广州",
        "month": "2026-09",
        "result": None,
    }
    parsed = {"need_more_info": False, "valid": True, "amount": 56.0,
              "category": "交通", "consume_date": "2026-09-03"}

    with patch.object(bill_nodes, "parse_json_output", new=AsyncMock(return_value=parsed)), \
            patch.object(bill_nodes.mcp_client, "call_bill_sql", new=AsyncMock(return_value="SQL执行成功")) as mock_sql:
        await bill_nodes.execute_node(state)

    assert mock_sql.await_count == 1
    _agent_id, sql, params = mock_sql.await_args.args
    assert "task_id" in sql                     # INSERT 必须包含 task_id 列
    assert params[-1] == task_id                # 值 = A2A 派发的 task_id
    assert len(params) == sql.count("?")        # 占位符与参数数量一致
