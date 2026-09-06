# -*- coding: utf-8 -*-
"""
需求5 意图短板修复 端到端测试：真实 LLM 全链路验证三个意图场景。

M4 修复内容（agents/orchestrator/nodes.py + prompts 模板）：
1. 多意图参数抽取：'花了300买耳机，值吗' → bill+finance，且 finance 任务必须携带金额/品类参数
   （启发式 has_finance_review 把"记账上下文合理性疑问"映射为 finance 意图；金额上下文补全扩展到 finance）
2. 防误记账：'别记这笔' 等拒绝记账表述 → 即使模型规划 bill 也拦截，bill 表零新增
3. 单意图不脑补：'奶茶15块' → 仅 bill，不附带 stat/finance/price 多余任务

数据隔离（每个用例前后自动恢复，测试后数据零残留）：
- bill 表：MAX(id) 快照法，运行后删除 id > 快照 的新增记录
- user_config 表：全列备份恢复（场景1 finance 需预算）
- monthly_habit / pkl：测试前清空，避免历史习惯干扰 finance 推算
"""
import json
import uuid
from datetime import date

import pytest

from agents.orchestrator.graph import orchestrator_graph
from memory.long_memory import LONG_MEM_PATH, _load_mem, clear_all_long_mem
from utils.common import db

TODAY = date.today()

# 场景输入（带 [意图e2e] 标记便于排查，不影响语义）
MULTI_INTENT_INPUT = "花了300买耳机，值吗 [意图e2e]"
REFUSE_BILL_INPUT = "别记这笔，看看我最近吃火锅多不多 [意图e2e]"
SINGLE_BILL_INPUT = "奶茶15块 [意图e2e]"


async def _run_graph(user_input: str):
    """执行编排器全流程，返回 final state"""
    init_state = {
        "user_input": user_input,
        "session_id": f"intent_{uuid.uuid4().hex[:8]}",
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent",
    }
    return await orchestrator_graph.ainvoke(init_state)


def _max_id() -> int:
    rows = db.query_sql("SELECT MAX(id) AS m FROM bill")
    return int(rows[0]["m"] or 0) if rows else 0


def _new_bills(after_id: int) -> list:
    return db.query_sql("SELECT id, amount, category, remark FROM bill WHERE id > ? ORDER BY id", (after_id,))


def _backup_user_config() -> list:
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    db.execute_sql("DELETE FROM user_config")
    for r in backup:
        cols = list(r.keys())
        placeholders = ",".join(["?"] * len(cols))
        db.execute_sql(
            f"INSERT INTO user_config ({','.join(cols)}) VALUES ({placeholders})",
            [r[c] for c in cols],
        )


def _set_budget(amount: float):
    db.execute_sql("DELETE FROM user_config")
    db.add_user_config("广州", amount, "正常", "month")


def _backup_pkl() -> bytes:
    if __import__("os").path.exists(LONG_MEM_PATH):
        with open(LONG_MEM_PATH, "rb") as f:
            return f.read()
    return None


def _restore_pkl(blob: bytes):
    clear_all_long_mem()
    if blob is not None:
        with open(LONG_MEM_PATH, "wb") as f:
            f.write(blob)
        _load_mem()


def _collect_diagnostics(result: dict) -> str:
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(
            f"[{tname}] segs={tinfo.get('raw_segments')} status={tinfo.get('status')} "
            f"result={json.dumps(res, ensure_ascii=False, default=str)[:300]}"
        )
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _data_isolation():
    bill_before = _max_id()
    cfg_backup = _backup_user_config()
    pkl_backup = _backup_pkl()
    # 测试前清空历史习惯/长期记忆，避免干扰 finance 推算与账单预警
    db.execute_sql("DELETE FROM monthly_habit")
    clear_all_long_mem()
    yield
    db.execute_sql("DELETE FROM bill WHERE id > ?", (bill_before,))
    _restore_user_config(cfg_backup)
    _restore_pkl(pkl_backup)


@pytest.mark.asyncio
async def test_e2e_multi_intent_params():
    """M4: '花了300买耳机，值吗' → bill+finance 双任务，finance 参数携带金额/品类，bill 入库 300 元"""
    _set_budget(2000.0)
    bill_before = _max_id()

    result = await _run_graph(MULTI_INTENT_INPUT)
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① 双意图落地：bill + finance（finance 由启发式 has_finance_review 兜底补齐，不依赖模型是否规划）
    assert "bill" in tp, f"未触发记账任务: {tp.keys()}\n{diag}"
    assert "finance" in tp, f"未触发 finance 分析任务（合理性疑问应映射 finance）: {tp.keys()}\n{diag}"

    # ② finance 参数完整性：raw_segments 必须携带金额与消费品类（金额上下文补全扩展到 finance）
    fin_segs = " ".join(tp["finance"].get("raw_segments") or [])
    assert "300" in fin_segs, f"finance 参数缺失金额: {fin_segs}\n{diag}"
    assert "耳机" in fin_segs, f"finance 参数缺失品类: {fin_segs}\n{diag}"

    # ③ 任务成功执行且记账入库
    assert tp["bill"].get("status") == "done", diag
    assert tp["finance"].get("status") == "done", diag
    new_bills = _new_bills(bill_before)
    assert len(new_bills) == 1, f"应恰好入库 1 笔: {new_bills}\n{diag}"
    assert abs(float(new_bills[0]["amount"]) - 300.0) < 1e-6, f"入库金额应为300: {new_bills}\n{diag}"
    assert result.get("final_reply"), "final_reply 为空"


@pytest.mark.asyncio
async def test_e2e_refuse_bill_no_record():
    """M4: '别记这笔，看看我最近吃火锅多不多' → 拒绝记账拦截，bill 表零新增"""
    bill_before = _max_id()

    result = await _run_graph(REFUSE_BILL_INPUT)
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① 绝不派发 bill 任务（启发式 has_refuse_bill 拦截，即使模型规划了 bill）
    assert "bill" not in tp, f"拒绝记账场景仍派发 bill: {tp.keys()}\n{diag}"

    # ② bill 表零新增（双保险：任务层拦截 + 表级断言）
    new_bills = _new_bills(bill_before)
    assert new_bills == [], f"拒绝记账场景 bill 表不应新增: {new_bills}\n{diag}"

    # ③ 用户仍得到响应（统计/提示或未识别提示，均不得是记账）
    assert result.get("final_reply"), "final_reply 为空"
    assert "记账成功" not in (result.get("final_reply") or ""), "拒绝记账场景回复不应声称记账成功"


@pytest.mark.asyncio
async def test_e2e_single_bill_no_extra():
    """M4: '奶茶15块' → 仅 bill，不脑补 stat/finance/price 多余任务，且记账入库"""
    bill_before = _max_id()

    result = await _run_graph(SINGLE_BILL_INPUT)
    tp = result.get("task_plan") or {}
    diag = _collect_diagnostics(result)

    # ① 仅 bill：金额+消费实体词场景不附带分析/统计任务（force_bill 兜底，非记账意图任务被裁剪）
    assert list(tp.keys()) == ["bill"], f"单笔记账被脑补多余任务: {tp.keys()}\n{diag}"

    # ② 记账成功入库
    assert tp["bill"].get("status") == "done", diag
    new_bills = _new_bills(bill_before)
    assert len(new_bills) == 1, f"应恰好入库 1 笔: {new_bills}\n{diag}"
    assert result.get("final_reply"), "final_reply 为空"
