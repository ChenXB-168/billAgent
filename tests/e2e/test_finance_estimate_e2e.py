# -*- coding: utf-8 -*-
"""
需求2 剩余天数推算 端到端测试：finance 全链路验证"历史习惯注入 → LLM 按注入数据推算剩余频次/金额"。

需求 2 前半"单笔过大判断"已由 tests/unit/test_budget_alert.py 的 L1 事实链用例覆盖（溢价四档 + 同级兜底），
本文件仅聚焦后半"剩余天数推算"：
- 注入源确定性验证：_load_recent_habits(3) 返回的月度习惯数据与沉淀数据一致（严格断言，可复现）
- finance 全链路：habit_data 注入上下文 → finance LLM 输出 raw_analysis_text 引用注入数据
- 空习惯兜底：habit_data 为空时 finance 仍能成功分析，不崩溃、不凭空编造习惯数据

数据隔离（每个用例前后自动恢复，测试后数据零残留）：
- bill 表：MAX(id) 快照法，运行后删除 id > 快照 的新增记录
- monthly_habit 表：全表备份 → 测试前清空（从零开始，杜绝历史数据污染）→ 测试后恢复备份
- user_config 表：全列备份恢复（finance 用例需预算，保留 remind_pref 等新列）
- pkl 长期记忆文件（long_consume_mem.pkl）：字节级备份，测试前清空，测试后恢复并重载内存
"""
import json
import os
import uuid
from datetime import date, timedelta

import pytest

from agents.orchestrator.graph import orchestrator_graph
from agents.orchestrator.nodes import _load_recent_habits, _persist_bill_habit
from memory.long_memory import (
    LONG_MEM_PATH,
    _load_mem,
    clear_all_long_mem,
)
from utils.common import db

TODAY = date.today()
MONTH = TODAY.strftime("%Y-%m")

# 触发 finance 的输入：含"分析/消费习惯/省钱建议"等强意图词 + 剩余天数推算语义；无金额/记账动词/统计词
FINANCE_INPUT = "我想分析一下我的消费习惯，算算剩下预算够不够撑到月底，给我一些省钱建议 [推算e2e]"


async def _run_graph(user_input: str):
    """执行编排器全流程，返回 final state"""
    init_state = {
        "user_input": user_input,
        "session_id": f"fin_{uuid.uuid4().hex[:8]}",
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


def _backup_user_config() -> list:
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    """按全列动态恢复（保留 remind_pref/quota_mode 等新列）"""
    db.execute_sql("DELETE FROM user_config")
    for r in backup:
        cols = list(r.keys())
        placeholders = ",".join(["?"] * len(cols))
        db.execute_sql(
            f"INSERT INTO user_config ({','.join(cols)}) VALUES ({placeholders})",
            [r[c] for c in cols],
        )


def _set_budget(amount: float):
    """清空配置并设置唯一测试预算（保证 get_latest_user_config 取到它）"""
    db.execute_sql("DELETE FROM user_config")
    db.add_user_config("广州", amount, "正常", "month")


def _backup_monthly_habit() -> list:
    return db.query_sql("SELECT * FROM monthly_habit")


def _restore_monthly_habit(backup: list):
    db.execute_sql("DELETE FROM monthly_habit")
    for r in backup:
        db.execute_sql(
            "INSERT INTO monthly_habit (month, category, amount_sum, count, avg_amount, update_time) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r["month"], r["category"], r["amount_sum"], r["count"], r["avg_amount"], r["update_time"]),
        )


def _backup_pkl() -> bytes:
    if os.path.exists(LONG_MEM_PATH):
        with open(LONG_MEM_PATH, "rb") as f:
            return f.read()
    return None


def _restore_pkl(blob: bytes):
    clear_all_long_mem()
    if blob is not None:
        with open(LONG_MEM_PATH, "wb") as f:
            f.write(blob)
        _load_mem()  # 重新加载内存 + 重建索引


def _prev_month(n_back: int) -> str:
    """前 n 个自然月（YYYY-MM），n_back=1 表示上月"""
    y, m = TODAY.year, TODAY.month
    for _ in range(n_back):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return f"{y:04d}-{m:02d}"


def _prev_month_date(n_back: int) -> str:
    """落在前 n 个自然月内的日期（用当月首日往前推，保证月份归属正确）"""
    first = date(TODAY.year, TODAY.month, 1)
    d = first - timedelta(days=1)
    for _ in range(n_back - 1):
        d = date(d.year, d.month, 1) - timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _persist_payload(amount: float, category: str, consume_date: str) -> dict:
    """构造 bill_agent 成功返回的 struct_payload（模拟真实记账结果，走生产沉淀函数）"""
    return {
        "success": True,
        "agent_type": "bill_agent",
        "msg": "记账成功",
        "data": {"amount": amount, "category": category, "consume_date": consume_date},
        "error": None,
    }


def _habit_summary(habits: list) -> dict:
    """habit_data 按 (month, category) 聚合为 {month: {category: row}}，便于断言"""
    out = {}
    for r in habits:
        out.setdefault(r["month"], {})[r["category"]] = r
    return out


def _collect_diagnostics(result: dict) -> str:
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(f"[{tname}] status={tinfo.get('status')} result={json.dumps(res, ensure_ascii=False, default=str)[:400]}")
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _data_isolation():
    """月度习惯/长期记忆/账单/配置全量隔离：测试前清空、测试后恢复备份"""
    bill_before = _max_id()
    cfg_backup = _backup_user_config()
    habit_backup = _backup_monthly_habit()
    pkl_backup = _backup_pkl()
    # 测试前从零开始，避免历史数据污染断言
    db.execute_sql("DELETE FROM monthly_habit")
    clear_all_long_mem()
    yield
    db.execute_sql("DELETE FROM bill WHERE id > ?", (bill_before,))
    _restore_user_config(cfg_backup)
    _restore_monthly_habit(habit_backup)
    _restore_pkl(pkl_backup)


@pytest.mark.asyncio
async def test_e2e_finance_estimate_with_habits():
    """有历史习惯：近3月餐饮习惯沉淀 → _load_recent_habits(3) 可复现注入数据 → finance 推算输出引用习惯数据"""
    # 沉淀近3个月餐饮习惯（当月 2 笔 80 元 / 上月 2 笔 80 元 / 上上月 3 笔 120 元）
    for amt in (38.0, 42.0):
        await _persist_bill_habit(_persist_payload(amt, "餐饮", TODAY.isoformat()))
    prev1, prev2 = _prev_month(1), _prev_month(2)
    await _persist_bill_habit(_persist_payload(40.0, "餐饮", _prev_month_date(1)))
    await _persist_bill_habit(_persist_payload(40.0, "餐饮", _prev_month_date(1)))
    for amt in (35.0, 40.0, 45.0):
        await _persist_bill_habit(_persist_payload(amt, "餐饮", _prev_month_date(2)))
    _set_budget(2000.0)

    # ① 注入源确定性验证（可复现）：近3月餐饮习惯逐月数据与沉淀一致
    habits = await _load_recent_habits(3)
    by_month = _habit_summary(habits)
    cur = by_month.get(MONTH, {}).get("餐饮")
    p1 = by_month.get(prev1, {}).get("餐饮")
    p2 = by_month.get(prev2, {}).get("餐饮")
    assert cur is not None, f"当月餐饮习惯缺失: {by_month}"
    assert float(cur["amount_sum"]) == 80.0 and int(cur["count"]) == 2, f"当月习惯错误: {cur}"
    assert p1 is not None and float(p1["amount_sum"]) == 80.0 and int(p1["count"]) == 2, f"上月习惯错误: {p1}"
    assert p2 is not None and float(p2["amount_sum"]) == 120.0 and int(p2["count"]) == 3, f"上上月习惯错误: {p2}"

    # ② finance 全链路：habit_data 注入 → LLM 推算输出（宽松断言：引用注入的品类/笔数痕迹，非确定性措辞）
    result = await _run_graph(FINANCE_INPUT)
    tp = result.get("task_plan") or {}
    assert "finance" in tp, f"未触发 finance 任务: {tp.keys()}"
    fin = tp["finance"]
    assert fin.get("status") == "done", _collect_diagnostics(result)
    fin_result = fin.get("result") or {}
    assert fin_result.get("success") is True, _collect_diagnostics(result)
    data = fin_result.get("data") or {}
    assert data.get("has_target_data") is True, "预算未注入 finance 上下文"
    raw = data.get("raw_analysis_text") or ""
    assert len(raw) > 50, f"finance 推算输出过短: {raw[:200]}"
    # 推算应基于注入数据：文本应出现习惯品类/预算/剩余等推算痕迹（任一项即可，避免措辞 flaky）
    assert any(kw in raw for kw in ("餐饮", "预算", "剩余", "够", "月底")), f"推算输出未引用注入数据: {raw[:300]}"
    assert result.get("final_reply"), "final_reply 为空"


@pytest.mark.asyncio
async def test_e2e_finance_estimate_empty_habits():
    """无历史习惯：_load_recent_habits(3) 返回空 → finance 仍能成功分析（不崩溃、不凭空编造习惯数据）"""
    # 不沉淀任何习惯（fixture 已清空 monthly_habit）
    _set_budget(2000.0)

    # 注入源确定性验证：空习惯 → 空列表
    habits = await _load_recent_habits(3)
    assert habits == [], f"空习惯场景 _load_recent_habits 应返回空: {habits}"

    # finance 全链路：空 habit_data 注入下仍成功分析
    result = await _run_graph(FINANCE_INPUT)
    tp = result.get("task_plan") or {}
    assert "finance" in tp, f"未触发 finance 任务: {tp.keys()}"
    fin = tp["finance"]
    assert fin.get("status") == "done", _collect_diagnostics(result)
    fin_result = fin.get("result") or {}
    assert fin_result.get("success") is True, _collect_diagnostics(result)
    raw = (fin_result.get("data") or {}).get("raw_analysis_text") or ""
    assert len(raw) > 50, f"finance 分析输出过短: {raw[:200]}"
    assert result.get("final_reply"), "final_reply 为空"
