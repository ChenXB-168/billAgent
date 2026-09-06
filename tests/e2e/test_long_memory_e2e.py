# -*- coding: utf-8 -*-
"""
需求3 长期记忆闭环 端到端测试：记账成功 → monthly_habit 事实表聚合（同月累加/跨月独立）
→ pkl 语义层同步 → finance 全链路注入 habit_data → 12 个月滚动清理。

数据隔离（每个用例前后自动恢复，测试后数据零残留）：
- bill 表：MAX(id) 快照法，运行后删除 id > 快照 的新增记录
- monthly_habit 表：全表备份 → 测试前清空（从零开始，杜绝历史数据污染）→ 测试后恢复备份
- user_config 表：全表备份恢复（finance 用例需预算）
- pkl 长期记忆文件（long_consume_mem.pkl）：字节级备份，测试前清空，测试后恢复并重载内存
"""
import json
import os
import pickle
import uuid
from datetime import date, timedelta

import pytest

from agents.orchestrator.graph import orchestrator_graph
from agents.orchestrator.nodes import _load_recent_habits, _persist_bill_habit
from memory.long_memory import (
    LONG_MEM_PATH,
    _keep_recent_months_boundary,
    _load_mem,
    clear_all_long_mem,
    upsert_month_habit_memory,
)
from utils.common import db

TODAY = date.today()
MONTH = TODAY.strftime("%Y-%m")


async def _run_graph(user_input: str):
    """执行编排器全流程，返回 final state"""
    init_state = {
        "user_input": user_input,
        "session_id": f"mem_{uuid.uuid4().hex[:8]}",
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
    db.execute_sql("DELETE FROM user_config")
    for r in backup:
        db.execute_sql(
            "INSERT INTO user_config (city, month_budget, consume_mode, budget_type) VALUES (?, ?, ?, ?)",
            (r["city"], r["month_budget"], r["consume_mode"], r["budget_type"]),
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


def _month_habit_rows(month: str) -> dict:
    """该月全部品类聚合行 → {category: row}"""
    rows = db.get_monthly_habits([month])
    return {r["category"]: r for r in rows}


def _collect_diagnostics(result: dict) -> str:
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(f"[{tname}] status={tinfo.get('status')} result={json.dumps(res, ensure_ascii=False, default=str)[:400]}")
    return "\n".join(lines)


def _persist_payload(amount: float, category: str, consume_date: str) -> dict:
    """构造 bill_agent 成功返回的 struct_payload（模拟真实记账结果）"""
    return {
        "success": True,
        "agent_type": "bill_agent",
        "msg": "记账成功",
        "data": {"amount": amount, "category": category, "consume_date": consume_date},
        "error": None,
    }


@pytest.fixture(autouse=True)
def _data_isolation():
    """月度习惯/长期记忆相关数据全量隔离：测试前清空、测试后恢复备份"""
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
async def test_e2e_habit_same_month_accumulate():
    """同月多笔记账沉淀 → monthly_habit 该月该品类金额求和/笔数+1/均额重算正确"""
    # 一笔走真实 LLM 记账链路（验证 记账→沉淀 端到端），其余走生产沉淀函数补齐累加
    result = await _run_graph("晚饭38元记一下 [记忆e2e]")
    # ★2026-08-31 断言调整：final_reply 为外部LLM自然表达（summarize.md 要求记账
    #   确认必含品类），"记账成功"固定短语仅存在于本地确定性渲染；本用例核心是
    #   monthly_habit 聚合（下方 L157-161 断言），此处仅验证记账链路已返回品类。
    assert "餐饮" in result["final_reply"], _collect_diagnostics(result)

    await _persist_bill_habit(_persist_payload(42.0, "餐饮", TODAY.isoformat()))
    await _persist_bill_habit(_persist_payload(20.0, "餐饮", TODAY.isoformat()))

    habit = _month_habit_rows(MONTH).get("餐饮")
    assert habit is not None, f"monthly_habit 缺少本月餐饮行: {_month_habit_rows(MONTH)}"
    assert float(habit["amount_sum"]) == 100.0, f"金额求和错误: {habit}"
    assert int(habit["count"]) == 3, f"笔数累加错误: {habit}"
    assert round(float(habit["avg_amount"]), 2) == 33.33, f"均额重算错误: {habit}"


@pytest.mark.asyncio
async def test_e2e_habit_cross_month_write():
    """跨月账单（consume_date 为上月）→ monthly_habit 独立月份行，与本月互不污染"""
    prev = TODAY - timedelta(days=TODAY.day)  # 上月末（一定在上月内）
    prev_month = prev.strftime("%Y-%m")
    prev_date = prev.strftime("%Y-%m-%d")

    await _persist_bill_habit(_persist_payload(35.0, "交通", prev_date))
    await _persist_bill_habit(_persist_payload(38.0, "餐饮", TODAY.isoformat()))

    prev_habit = _month_habit_rows(prev_month).get("交通")
    cur_habit = _month_habit_rows(MONTH).get("餐饮")
    assert prev_habit is not None, f"上月交通行缺失: {_month_habit_rows(prev_month)}"
    assert float(prev_habit["amount_sum"]) == 35.0, f"上月金额错误: {prev_habit}"
    assert int(prev_habit["count"]) == 1, f"上月笔数错误: {prev_habit}"
    assert cur_habit is not None and float(cur_habit["amount_sum"]) == 38.0, f"本月餐饮行错误: {cur_habit}"
    # 月份行相互独立：上月无餐饮、本月无交通
    assert "餐饮" not in _month_habit_rows(prev_month), "上月被本月品类污染"
    assert "交通" not in _month_habit_rows(MONTH), "本月被上月品类污染"


@pytest.mark.asyncio
async def test_e2e_finance_uses_habit_data():
    """finance 全链路：记账沉淀习惯 → 理财请求命中 FINANCE_HINT → 注入 habit_data 并成功分析"""
    # 先沉淀本月餐饮习惯（模拟历史数据，走生产沉淀函数）
    await _persist_bill_habit(_persist_payload(38.0, "餐饮", TODAY.isoformat()))
    await _persist_bill_habit(_persist_payload(42.0, "餐饮", TODAY.isoformat()))
    _set_budget(2000.0)  # 提供预算，避免 finance 走 need_more_info 追问

    # 注入源验证：生产函数 _load_recent_habits 能取到沉淀数据
    habits = await _load_recent_habits(3)
    cur_food = [h for h in habits if h["month"] == MONTH and h["category"] == "餐饮"]
    assert cur_food, f"_load_recent_habits 未返回本月餐饮习惯: {habits}"
    assert float(cur_food[0]["amount_sum"]) == 80.0 and int(cur_food[0]["count"]) == 2, f"习惯数据错误: {cur_food}"

    # finance 全链路：含 FINANCE_HINT 关键词且无金额 → 只触发 finance 任务
    result = await _run_graph("我想分析一下我的消费习惯，给我一些省钱建议 [记忆e2e]")
    tp = result.get("task_plan") or {}
    assert "finance" in tp, f"未触发 finance 任务: {tp.keys()}"
    fin = tp["finance"]
    assert fin.get("status") == "done", _collect_diagnostics(result)
    fin_result = fin.get("result") or {}
    assert fin_result.get("success") is True, _collect_diagnostics(result)
    data = fin_result.get("data") or {}
    assert data.get("has_target_data") is True, "预算未注入 finance 上下文"
    assert result.get("final_reply"), "final_reply 为空"


@pytest.mark.asyncio
async def test_e2e_habit_rollup_cleanup():
    """12 个月滚动清理：DB 层 cleanup_monthly_habit 删除超期行；语义层 upsert 时裁剪超期月份"""
    # DB 层：写入 13 个月前的月份（早于边界），cleanup 后应被删除
    boundary = _keep_recent_months_boundary(12)
    y, m = int(boundary[:4]), int(boundary[5:7])
    m -= 1
    if m == 0:
        m = 12
        y -= 1
    old_month = f"{y:04d}-{m:02d}"
    db.upsert_monthly_habit(old_month, "餐饮", 100.0)
    db.upsert_monthly_habit(MONTH, "餐饮", 50.0)
    assert old_month in {r["month"] for r in db.get_monthly_habits()}
    db.cleanup_monthly_habit(12)
    remaining = {r["month"] for r in db.get_monthly_habits()}
    assert old_month not in remaining, f"超期月份 {old_month} 未被清理"
    assert MONTH in remaining, "当月数据被误删"

    # 语义层：先写入超期月份摘要，再写入当月摘要触发滚动裁剪
    upsert_month_habit_memory(old_month, [{"category": "餐饮", "amount_sum": 100.0, "count": 1, "avg_amount": 100.0}])
    upsert_month_habit_memory(MONTH, [{"category": "餐饮", "amount_sum": 50.0, "count": 1, "avg_amount": 50.0}])
    assert os.path.exists(LONG_MEM_PATH), "pkl 语义层未落盘"
    with open(LONG_MEM_PATH, "rb") as f:
        texts = pickle.load(f)
    joined = "\n".join(texts)
    assert old_month not in joined, f"语义层未裁剪超期月份: {joined}"
    assert MONTH in joined, f"语义层缺少当月摘要: {joined}"
