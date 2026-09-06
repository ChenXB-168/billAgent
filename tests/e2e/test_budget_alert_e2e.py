# -*- coding: utf-8 -*-
"""
需求1/4 自动预警链 端到端测试：从用户端输入走完整 orchestrator 全链路，
验证 collect_node 在"纯记账成功且无追问"时由外部 LLM 按汇总提示词生成最终回复。

汇总节点已改为 LLM 表达层（不再硬编码文案），测试对 collect_node 的汇总 LLM
调用做 monkeypatch：仅拦截"最终回复生成器"（汇总提示词）调用，按注入的
alert_facts 确定性拼回复，从而端到端验证确定性计算层（L3 总量事实）正确产出；
规划节点等其他 LLM 调用透传真实外部模型。

数据隔离：
- bill 表：MAX(id) 快照法，运行后删除 id > 快照 的新增记录（含测试种子）
- user_config 表：运行前备份全部行 → 全删 → 插入测试预算 → 运行 → 恢复备份（全列）
- 预算按"种子后当月总额 + 预期新增 38 元"动态计算，保证 used_ratio 稳定落在目标档位
"""
import json
import re
import uuid
from datetime import date

import pytest

from agents.orchestrator.graph import orchestrator_graph
from mcpGateway.client import mcp_client
from utils.common import db

TODAY = date.today()
MONTH = TODAY.strftime("%Y-%m")


async def _run_graph(user_input: str):
    """执行编排器全流程，返回 final state"""
    init_state = {
        "user_input": user_input,
        "session_id": f"alert_{uuid.uuid4().hex[:8]}",
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


def _month_spent() -> float:
    """当月 bill 金额总和（预警链 _sum_month_bills 同口径）"""
    rows = db.query_sql(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM bill WHERE substr(consume_time, 1, 7) = ?",
        (MONTH,),
    )
    return float(rows[0]["s"] or 0) if rows else 0.0


def _backup_user_config() -> list:
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    """恢复 user_config 到测试前状态（全删后按全列重插备份行，保留 remind_pref/quota_mode 等新列）"""
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


def _seed_month_bill(amount: float, category: str):
    db.execute_sql(
        "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
        (amount, category, TODAY.isoformat(), "预警e2e种子"),
    )


def _collect_diagnostics(result: dict) -> str:
    """失败时输出关键 state 便于定位"""
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(f"[{tname}] status={tinfo.get('status')} result={json.dumps(res, ensure_ascii=False, default=str)[:300]}")
    return "\n".join(lines)


# ---------------- 汇总 LLM 劫持（表达层确定性模拟） ----------------
_FACTS_BLOCK_RE = re.compile(r"预警事实（JSON，可为空）：\n(\{.*?\})\n\n用户提醒偏好", re.S)


def _extract_alert_facts(user_text: str) -> dict:
    """从汇总提示词 user 模板中解析注入的 alert_facts JSON"""
    m = _FACTS_BLOCK_RE.search(user_text or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _fake_summarize_reply(facts: dict) -> str:
    """按注入的预警事实确定性生成回复文案（验证 L3 事实链路）"""
    l3 = (facts or {}).get("l3") or {}
    level = l3.get("level", "ok")
    if level == "warn":
        daily = l3.get("daily_limit")
        return f"记账成功。本月预算预警：剩余预算紧张，每日建议控制在{daily}元以内。"
    if level == "over":
        cats = l3.get("over_categories") or []
        main = cats[0]["category"] if cats else ""
        return f"记账成功。本月预算已超支，主因品类：{main}。"
    if level == "reward":
        return "记账成功。预算还很充裕，可以适当犒劳一下自己。"
    return "记账成功。"


def _install_summarize_fake(monkeypatch):
    """仅劫持汇总节点 LLM 调用；规划/解析等其余调用透传真实外部模型"""
    original = mcp_client.call_llm_base

    async def _fake_call_llm_base(sys_prompt, user_text, agent_tag="unknown"):
        if "最终回复生成器" in sys_prompt:
            facts = _extract_alert_facts(user_text)
            return _fake_summarize_reply(facts)
        return await original(sys_prompt, user_text, agent_tag=agent_tag)

    monkeypatch.setattr(mcp_client, "call_llm_base", _fake_call_llm_base)


@pytest.mark.asyncio
async def test_e2e_budget_warn_alert(monkeypatch):
    """纯记账 → 预算接近红线（warn 档）→ 汇总 LLM 注入 L3 事实 → 回复含'预算预警 + 每日建议 + 数字'"""
    _install_summarize_fake(monkeypatch)
    cfg_backup = _backup_user_config()
    before_id = _max_id()
    try:
        _seed_month_bill(38.0, "餐饮")
        spent = _month_spent()          # 种子后当月总额
        _set_budget((spent + 38) / 0.9)  # 记账后 used_ratio ≈ 0.9 → warn 档
        result = await _run_graph("晚饭38元记一下 [预警e2e]")
        final = result["final_reply"] or ""
        assert "记账成功" in final, _collect_diagnostics(result)
        assert "预算预警" in final, _collect_diagnostics(result)
        assert "每日建议控制在" in final, _collect_diagnostics(result)
        assert re.search(r"[\d.]+元", final), _collect_diagnostics(result)
    finally:
        db.execute_sql("DELETE FROM bill WHERE id > ?", (before_id,))
        _restore_user_config(cfg_backup)


@pytest.mark.asyncio
async def test_e2e_budget_reward_alert(monkeypatch):
    """纯记账 → 预算充裕（reward 档）→ 汇总 LLM 注入 L3 事实 → 回复含犒劳提醒"""
    _install_summarize_fake(monkeypatch)
    cfg_backup = _backup_user_config()
    before_id = _max_id()
    try:
        _seed_month_bill(38.0, "餐饮")
        spent = _month_spent()
        _set_budget((spent + 38) / 0.25)  # 记账后 used_ratio ≈ 0.25 → reward 档
        result = await _run_graph("晚饭38元记一下 [预警e2e]")
        final = result["final_reply"] or ""
        assert "记账成功" in final, _collect_diagnostics(result)
        assert "预算还很充裕" in final, _collect_diagnostics(result)
        assert "犒劳" in final, _collect_diagnostics(result)
    finally:
        db.execute_sql("DELETE FROM bill WHERE id > ?", (before_id,))
        _restore_user_config(cfg_backup)
