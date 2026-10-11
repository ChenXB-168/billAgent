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
| `test_ask_round_resets_on_success` | 记账成功后追问计数即时清零（成功即重置，避免历史追问累计） |

★隔离：TaskManager 用 `tmp_path` 建临时库，不污染真实 `database/task_state.db`；
对账用的业务库以 `_FakeBillDB` 注入，不触碰 `bill.db`。
"""
import asyncio
import json

import pytest

from config.config import ASK_GIVEUP_TIP, ASK_ROUND_LIMIT
from startup.task_manager import TaskManager


class _FakeBillDB:
    """
    函数/类功能与逻辑描述：
        模拟业务库对账的替身对象：崩溃恢复时按 task_id 反查 `bill` 表是否已有记账记录，
        避免单测触碰真实 `bill.db`；由 `exists` 开关控制查询返回"命中一行"还是"空结果"，
        并把每次调用记录到 `queries` 供用例断言。
    构造入参说明：
        exists (bool)：True = 模拟账单已落库（查询返回一行），False = 模拟未落库
            （返回空列表），默认 False。
    返回值说明：
        构造返回 _FakeBillDB 实例；业务方法见 `query_sql`，调用记录见 `self.queries`。
    """

    def __init__(self, exists: bool = False):
        """
        函数功能与逻辑描述：
            初始化替身：保存"账单是否存在"的开关，并准备调用记录列表，供用例断言对账
            查询是否按要求下发 SQL 与参数；无真实数据库连接、无副作用。
        入参说明：
            exists (bool)：是否模拟账单已落库，默认 False。
        返回值说明：
            无（仅初始化实例属性 exists 与 queries）。
        """
        self.exists = exists
        self.queries: list = []

    def query_sql(self, sql: str, params=None):
        """
        函数功能与逻辑描述：
            模拟 DatabaseCRUD.query_sql 的查询契约：把每次调用的 SQL 与参数追加到
            `self.queries` 供断言，并按 `self.exists` 返回命中行或空列表，
            使 `_bill_exists` 的对账分支可被确定性地驱动。
        入参说明：
            sql (str)：被调用的查询 SQL。
            params：SQL 参数（可选），默认 None。
        返回值说明：
            list：exists=True 时返回 [{"1": 1}]（模拟命中一条记录），否则返回 []（未命中）。
        """
        self.queries.append((sql, params))
        return [{"1": 1}] if self.exists else []


@pytest.fixture
def tm(tmp_path):
    """
    函数功能与逻辑描述：
        构造每个用例独享的 TaskManager：任务状态库文件落在 pytest 提供的临时目录
        `tmp_path` 下，从而隔离真实 `database/task_state.db`；fixture 作用域为函数级
        （默认），用例结束后自动 close 释放连接。
    入参说明：
        tmp_path：pytest 内置 fixture 注入的临时目录（pathlib.Path），提供一个隔离且
            可写的文件系统位置用于放置临时任务状态库。
    返回值说明：
        TaskManager：yield 出待测实例；fixture 在用例结束后调用 manager.close() 清理。
    """
    manager = TaskManager(tmp_path / "task_state.db")
    yield manager
    manager.close()


# ===================== 状态机基础流转 =====================
def test_state_machine_happy_path(tm):
    """
    函数功能与逻辑描述：
        验证任务状态机的基础快乐路径：create_task 建单后为 submitted，mark_running 后为
        running，mark_completed 后为 completed，且 result 以 JSON 序列化落库、可反序列化回
        原值。覆盖 submitted → running → completed 的完整流转与结果落库。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库），提供建单与状态流转能力。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t001", "s1", "bill_agent", {"amount": 56})
    assert tm.get_task("t001")["status"] == "submitted"

    tm.mark_running("t001")
    assert tm.get_task("t001")["status"] == "running"

    tm.mark_completed("t001", {"success": True})
    task = tm.get_task("t001")
    assert task["status"] == "completed"
    assert json.loads(task["result"]) == {"success": True}


def test_create_task_is_idempotent(tm):
    """
    函数功能与逻辑描述：
        验证 create_task 的幂等语义：同一 task_id 重复投递（崩溃重投场景）不会重复建单，
        也不会覆盖首次写入的 payload（INSERT OR IGNORE 语义）。重复投递后直接查 payload
        列，断言仅一行且值保持首次的 {"amount": 56}。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t001", "s1", "bill_agent", {"amount": 56})
    tm.create_task("t001", "s1", "bill_agent", {"amount": 9999})   # 重复投递
    rows = tm._db.query_sql("SELECT payload FROM task_state WHERE task_id='t001'")
    assert len(rows) == 1
    assert json.loads(rows[0]["payload"]) == {"amount": 56}   # 保持首次（IGNORE 语义）


def test_retriable_default_by_agent(tm):
    """
    函数功能与逻辑描述：
        验证 retriable 的默认推断规则（`04` §3.3）：有副作用的写任务 bill_agent 默认为
        0（禁止自动重投），无副作用的只读任务 stat_agent 默认为 1（允许重投）。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-bill", "s1", "bill_agent", {})
    tm.create_task("t-stat", "s1", "stat_agent", {})
    assert tm.get_task("t-bill")["retriable"] == 0
    assert tm.get_task("t-stat")["retriable"] == 1


# ===================== 崩溃恢复（04 §3.1 修订版）=====================
def test_recover_bill_with_record_marks_completed(tm):
    """
    函数功能与逻辑描述：
        🔴 核心验收：崩溃发生在「子进程已记账」与「主进程标记 completed」之间时，恢复必须
        标记 completed 而非 failed，否则会告诉用户"记账失败"而账其实已入库。构造 running
        状态的 bill_agent 任务，注入 _FakeBillDB(exists=True) 模拟账单已落库，调用
        recover_orphans 后断言状态为 completed 且统计为 {"completed": 1, "resubmitted": 0,
        "failed": 0}。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-bill-1", "s1", "bill_agent", {"amount": 56})
    tm.mark_running("t-bill-1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=True))

    assert tm.get_task("t-bill-1")["status"] == "completed"   # 不是 failed！
    assert stat == {"completed": 1, "resubmitted": 0, "failed": 0}


def test_recover_bill_without_record_marks_failed(tm):
    """
    函数功能与逻辑描述：
        验证对账的保守分支：账单未落库且写任务不可重投（retriable=0）→ 标记 failed，
        宁可告警也不自动重投导致重复记账。构造 running 的 bill_agent 任务，注入
        _FakeBillDB(exists=False) 模拟无记录，断言恢复后状态为 failed。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-bill-2", "s1", "bill_agent", {"amount": 56})
    tm.mark_running("t-bill-2")

    tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    assert tm.get_task("t-bill-2")["status"] == "failed"


def test_recover_readonly_task_resubmits(tm):
    """
    函数功能与逻辑描述：
        验证只读任务（无落库副作用）崩溃后直接重投：构造 running 的 stat_agent 任务，
        注入 _FakeBillDB(exists=False)，恢复后状态回到 submitted、retry_cnt 自增为 1，
        且统计 resubmitted=1。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-stat-1", "s1", "stat_agent", {"q": "本月几笔"})
    tm.mark_running("t-stat-1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    task = tm.get_task("t-stat-1")
    assert task["status"] == "submitted"
    assert task["retry_cnt"] == 1
    assert stat["resubmitted"] == 1


def test_recover_respects_max_retry(tm):
    """
    函数功能与逻辑描述：
        验证重投上限（MAX_RETRY=3）被遵守：先把只读任务的 retry_cnt 直接置为 3，再调用
        recover_orphans；因已达上限不再改回 submitted，而是标记 failed。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-stat-2", "s1", "stat_agent", {})
    tm.mark_running("t-stat-2")
    tm._db.execute_sql("UPDATE task_state SET retry_cnt=3 WHERE task_id='t-stat-2'")

    tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    assert tm.get_task("t-stat-2")["status"] == "failed"


def test_recover_scans_interrupted(tm):
    """
    函数功能与逻辑描述：
        验证优雅关闭（worker 被 cancel）遗留的 interrupted 任务同样参与恢复扫描
        （`08` §6 Q7）：只读任务经 mark_running + mark_interrupted 后调用 recover_orphans，
        断言状态回到 submitted 且统计 resubmitted=1。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-int-1", "s1", "stat_agent", {})
    tm.mark_running("t-int-1")
    tm.mark_interrupted("t-int-1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=False))

    assert tm.get_task("t-int-1")["status"] == "submitted"
    assert stat["resubmitted"] == 1


def test_resubmit_submitted_calls_sender(tm):
    """
    函数功能与逻辑描述：
        验证 DB 是唯一事实源：resubmit_submitted 遍历 status='submitted' 的任务，按
        (agent_id, session_id, task_id, payload) 原样回填派发参数。以 lambda 充当 send_fn
        收集调用，断言返回重投条数为 1 且参数与建单时的 payload 完全一致。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证 track 的成功分支：占位 runner 正常返回 → 状态流转为 completed。
        先 create_task 建单（track 只负责流转，建单在派发前完成），再 await track 传入
        返回 {"result": "done"} 的 runner，断言最终状态为 completed。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-ok", "s1", "stat_agent", {})   # track 只负责流转，建单在派发前完成

    async def _ok():
        """
        函数功能与逻辑描述：
            占位 runner：模拟子 Agent 图执行成功，直接返回固定结果 {"result": "done"}，
            用于驱动 track 的成功分支。
        入参说明：
            无（由 track 以无参形式 await 调用）。
        返回值说明：
            dict：固定成功结果 {"result": "done"}。
        """
        return {"result": "done"}

    await tm.track({"task_id": "t-ok"}, _ok)
    assert tm.get_task("t-ok")["status"] == "completed"


@pytest.mark.asyncio
async def test_track_exception_marks_failed_and_reraises(tm):
    """
    函数功能与逻辑描述：
        验证 track 的异常分支：runner 抛 RuntimeError → 标记 failed 并把异常类型与信息
        写入 result，同时异常必须继续向上抛出（worker 的 `except Exception` 依赖它保证
        消费循环不中断）。断言 raises(RuntimeError)、状态为 failed 且 result["error"]
        含 "RuntimeError"。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-boom", "s1", "stat_agent", {})

    async def _boom():
        """
        函数功能与逻辑描述：
            故障 runner：模拟子 Agent 图执行崩溃，无条件抛 RuntimeError("图执行崩溃")，
            用于驱动 track 的异常分支（先 mark_failed 记录异常类型，再原样向上抛）。
        入参说明：
            无（由 track 以无参形式 await 调用）。
        返回值说明：
            无（必定抛出 RuntimeError，不返回）。
        """
        raise RuntimeError("图执行崩溃")

    with pytest.raises(RuntimeError):
        await tm.track({"task_id": "t-boom"}, _boom)

    task = tm.get_task("t-boom")
    assert task["status"] == "failed"
    assert "RuntimeError" in json.loads(task["result"])["error"]


@pytest.mark.asyncio
async def test_track_cancelled_marks_interrupted(tm):
    """
    函数功能与逻辑描述：
        验证优雅关闭分支：runner 抛 asyncio.CancelledError → 必须标记 interrupted 而非
        failed，并继续向上抛出——否则下次启动会因 retriable=0（写任务）把"已记账"误判为
        failed（`04` §3.1）。断言 raises(asyncio.CancelledError) 且最终状态为 interrupted。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tm.create_task("t-cancel", "s1", "bill_agent", {})

    async def _cancelled():
        """
        函数功能与逻辑描述：
            取消 runner：模拟优雅关闭时 worker 被 cancel，无条件抛 asyncio.CancelledError，
            用于驱动 track 的取消分支（标记 interrupted 而非 failed，并原样向上抛）。
        入参说明：
            无（由 track 以无参形式 await 调用）。
        返回值说明：
            无（必定抛出 asyncio.CancelledError，不返回）。
        """
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await tm.track({"task_id": "t-cancel"}, _cancelled)

    assert tm.get_task("t-cancel")["status"] == "interrupted"


# ===================== D5：追问 3 轮治理 =====================
@pytest.mark.asyncio
async def test_ask_round_limit_gives_up():
    """
    函数功能与逻辑描述：
        验证 D5 追问治理（`01` §3.4）：信息不全最多追问 3 轮（ASK_ROUND_LIMIT），第 3 轮
        仍不满足 → 返回固定放弃提示语 ASK_GIVEUP_TIP、不再追加追问，且放弃后该记账动作的
        追问计数清零（下次记账重新计数）。构造 bill_agent 返回 need_more_info 的结果，
        循环调用 collect_node 三次；为隔离副作用，用 patch 把预算/预警/目标配置查询与
        外部 LLM 开关全部 stub 掉，不触达真实 DB 与模型。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证任务成功（非追问结果）后追问计数清零，避免历史追问累计影响下次记账。
        先以 need_more_info 结果调用 collect_node 使计数变为 1，再以对齐真实记账成功载荷
        的结果（含 category/amount/consume_date，供 bill_agent 渲染）调用 → 计数回到 0；
        同上用 patch 把预算/预警/目标配置查询与外部 LLM 开关 stub 掉以隔离副作用。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
        """
        函数功能与逻辑描述：
            构造 collect_node 的最小输入 state：固定 session_id 与 user_input，
            task_plan 内放一则 bill_agent 已完成的记账任务（operate_sub_type=add），
            result 由调用方注入（追问 / 成功载荷），用于驱动 collect_node 的计数分支。
        入参说明：
            result (dict)：该 bill 任务的子 agent 结构化结果，直接写入 task_plan 的 result 字段。
        返回值说明：
            dict：符合 collect_node 读取契约的 state 字典。
        """
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
    """
    函数功能与逻辑描述：
        验证 D2 崩溃对账依据：bill_agent.execute_node 生成的 INSERT 必须由**代码层**携带
        task_id 列（不依赖 LLM 生成 SQL），且参数最后一项等于 A2A 派发的 task_id、参数
        数量与 SQL 占位符数量一致。用 patch 替换解析 LLM 输出与 MCP SQL 调用（AsyncMock）
        后断言 call_bill_sql 收到的 (sql, params) 实参。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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


# ===================== M16：用户取消（cancelled 终态）=====================
def test_cancel_marks_cancelled_terminal(tm):
    """
    函数功能与逻辑描述：
        验证用户取消把在途任务置为 `cancelled`，且该状态属于 TERMINAL_STATUS
        （终态，不再参与恢复扫描）。M16 的基础语义。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from startup.task_manager import TERMINAL_STATUS

    tm.create_task("t-c1", "s1", "bill_agent", {})
    tm.mark_running("t-c1")
    tm.cancel_session("s1")

    assert tm.get_task("t-c1")["status"] == "cancelled"
    assert "cancelled" in TERMINAL_STATUS


def test_cancelled_not_resubmitted_after_restart(tm):
    """
    函数功能与逻辑描述：
        ★M16 最关键的一条：**用户取消的任务重启后不得被重投、不得再次触发回复**。
        验证 `recover_orphans` 的扫描条件（`status IN ('running','interrupted')`）
        不含 `cancelled`——即便它曾经是 `running`。
        并设对照组：未取消的 running 任务（同 agent、同 retriable）仍会被正常重投，
        证明"不重投"是取消状态带来的、而非因其它条件恰好不满足。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    # 被测组：只读任务（retriable=1，若为 interrupted 本应被重投）→ 取消
    tm.create_task("t-c2", "s1", "stat_agent", {})
    tm.mark_running("t-c2")
    tm.cancel_session("s1")

    stat = tm.recover_orphans(bill_db=_FakeBillDB(exists=False))
    assert stat["resubmitted"] == 0
    assert tm.get_task("t-c2")["status"] == "cancelled"

    # 对照组：同样条件但未取消 → 正常重投为 submitted
    tm.create_task("t-c3", "s1", "stat_agent", {})
    tm.mark_running("t-c3")
    stat2 = tm.recover_orphans(bill_db=_FakeBillDB(exists=False))
    assert stat2["resubmitted"] == 1
    assert tm.get_task("t-c3")["status"] == "submitted"


def test_cancel_does_not_touch_finished_tasks(tm):
    """
    函数功能与逻辑描述：
        验证取消只作用于**在途**任务（`submitted` / `running`），已完成任务不受影响——
        否则会把"已记账"改写成"已取消"，造成账实不符。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    tm.create_task("t-c4", "s1", "stat_agent", {})
    tm.mark_running("t-c4")
    tm.mark_completed("t-c4", {"ok": True})

    res = tm.cancel_session("s1")
    assert res["cancelled"] == []
    assert tm.get_task("t-c4")["status"] == "completed"


def test_mark_running_skips_cancelled_task(tm):
    """
    函数功能与逻辑描述：
        验证"取消一个还在队列里的任务"能生效：取消只改 DB 状态（队列中的消息无法安全剔除），
        故 worker 领取时必须靠 `mark_running` 的返回值做二次拦截——返回 False 表示已被取消、
        应跳过执行，且**不得把终态覆盖回 running**。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    tm.create_task("t-c5", "s1", "stat_agent", {})
    tm.cancel_session("s1")          # 此刻任务仍是 submitted（还躺在队列里）

    assert tm.mark_running("t-c5") is False
    assert tm.get_task("t-c5")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_running_interrupts_and_notifies(tm):
    """
    函数功能与逻辑描述：
        ★M16 核心：正在执行的子任务被中断，**且取消结果被回传到消息总线**。
        后者不可省略——runner（整张子图）被中断后 `reply_node` 不会执行，
        若不代传结果，编排器的 `wait_result(tid, timeout=240)` 会一直阻塞到 240 秒超时，
        用户点了"停止"却要等 4 分钟才有反应。
        断言三件事：track 返回取消结果、任务落在 `cancelled` 终态、结果已可从总线取出。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from mcpGateway.a2a_queue import a2a_bus

    a2a_bus.clear_all()
    started = asyncio.Event()

    async def _long_runner():
        """模拟长任务：置位 started 后长时间挂起，等待被取消。"""
        started.set()
        await asyncio.sleep(30)
        return {"should": "not reach"}

    tm.create_task("t-c6", "s1", "finance_agent", {})
    outer = asyncio.create_task(tm.track({"task_id": "t-c6"}, _long_runner))
    await started.wait()                          # 确保 runner 已真正开始
    assert tm.get_task("t-c6")["status"] == "running"

    tm.cancel_session("s1")                       # 用户取消
    result = await asyncio.wait_for(outer, timeout=5)   # 能及时返回 = 中断生效

    assert result["success"] is False
    assert result["error"] == "用户取消"
    assert result["agent_type"] == "finance_agent"      # 与常规结果同构，编排器无需特判
    assert tm.get_task("t-c6")["status"] == "cancelled"

    notified = a2a_bus._result_store.get("t-c6")        # ★结果已回传，编排器不会被挂住
    assert notified is not None
    assert json.loads(notified)["error"] == "用户取消"


@pytest.mark.asyncio
async def test_track_distinguishes_user_cancel_from_shutdown(tm):
    """
    函数功能与逻辑描述：
        验证 `track()` 对两种 `CancelledError` 的分流口径（M16 的关键区分）：
          - **用户取消**（DB 状态已是 `cancelled`）→ 吞掉异常、返回取消结果，
            worker 消费循环不中断；
          - **进程优雅关闭**（状态非 cancelled）→ 标 `interrupted` 并**继续向上抛**，
            由关停逻辑收敛（不能标 failed，否则重启后"账已记"却因 retriable=0 被判 failed）。
        两种情形在 Python 层都是同一个异常类型，故必须靠 DB 状态区分。
    入参说明：
        tm：pytest fixture 注入的隔离 TaskManager（临时库）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    async def _cancelled_runner():
        """runner 抛 CancelledError，模拟被 cancel。"""
        raise asyncio.CancelledError()

    # 场景 A：用户取消（状态已落 cancelled）→ 吞掉，不抛
    tm.create_task("t-c7", "s1", "stat_agent", {})
    tm.mark_cancelled("t-c7")
    res = await tm.track({"task_id": "t-c7"}, _cancelled_runner)
    assert res["error"] == "用户取消"
    assert tm.get_task("t-c7")["status"] == "cancelled"

    # 场景 B：进程优雅关闭（状态非 cancelled）→ 标 interrupted 并向上抛
    tm.create_task("t-c8", "s1", "stat_agent", {})
    with pytest.raises(asyncio.CancelledError):
        await tm.track({"task_id": "t-c8"}, _cancelled_runner)
    assert tm.get_task("t-c8")["status"] == "interrupted"
