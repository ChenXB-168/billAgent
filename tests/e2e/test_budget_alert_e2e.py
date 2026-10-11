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
    """
    函数功能与逻辑描述：
        辅助协程：以随机 session_id（前缀 alert_）构造 orchestrator 初始 state
        （空历史、空任务计划、无错误），直接 await 执行 orchestrator_graph.ainvoke，
        返回终态 state 供用例断言 task_plan / final_reply 等字段。
        运行时依赖真实外部 LLM（规划与解析环节）与本地账单库（bill / user_config）。
    入参说明：
        user_input (str)：用户本轮自然语言输入，作为 state["user_input"]。
    返回值说明：
        dict：orchestrator 执行完成后的 final state（含 task_plan / final_reply / error_msg）。
    """
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
    """
    函数功能与逻辑描述：
        测试数据隔离辅助：取 bill 表当前最大 id 作为快照基线，用例结束按此基线
        删除 id > 基线的全部新增记录（含本次种子数据），避免污染真实账单库。
    入参说明：
        无。
    返回值说明：
        int：bill 表当前 MAX(id)；表为空或查询返回空列表时返回 0。
    """
    rows = db.query_sql("SELECT MAX(id) AS m FROM bill")
    return int(rows[0]["m"] or 0) if rows else 0


def _month_spent() -> float:
    """
    函数功能与逻辑描述：
        直接以 SQL 汇总当月（MONTH = 由 TODAY 派生的 "YYYY-MM"）bill 表金额总和，
        与预警链 _sum_month_bills（走 bill.sum_by_month 事实工具）保持同口径，
        用于在设置预算前取得"种子后当月总额"基线。
    入参说明：
        无（月份取模块级 MONTH 常量）。
    返回值说明：
        float：当月账单金额合计；无数据或查询失败时返回 0.0。
    """
    rows = db.query_sql(
        "SELECT COALESCE(SUM(amount), 0) AS s FROM bill WHERE substr(consume_time, 1, 7) = ?",
        (MONTH,),
    )
    return float(rows[0]["s"] or 0) if rows else 0.0


def _backup_user_config() -> list:
    """
    函数功能与逻辑描述：
        备份 user_config 表全部行，供用例在 finally 中恢复，防止测试预算配置
        残留污染真实用户配置（与 _restore_user_config 配对使用）。
    入参说明：
        无。
    返回值说明：
        list[dict]：user_config 全表行（列名 → 值）；表为空时返回空列表。
    """
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    """
    函数功能与逻辑描述：
        恢复 user_config 到测试前状态：先全表 DELETE，再按备份行的全部列动态拼接
        INSERT 重插，从而兼容并保留 remind_pref / quota_mode / category_budget 等新增列。
    入参说明：
        backup (list)：_backup_user_config() 备份的行列表；为空列表时仅执行清空。
    返回值说明：
        无（副作用：覆盖 user_config 表内容；单条 INSERT 失败由 execute_sql 返回 False，不抛异常）。
    """
    db.execute_sql("DELETE FROM user_config")
    for r in backup:
        cols = list(r.keys())
        placeholders = ",".join(["?"] * len(cols))
        db.execute_sql(
            f"INSERT INTO user_config ({','.join(cols)}) VALUES ({placeholders})",
            [r[c] for c in cols],
        )


def _set_budget(amount: float):
    """
    函数功能与逻辑描述：
        清空 user_config 后写入唯一一条测试预算（广州 / 消费模式"正常" / budget_type "month"），
        保证 get_latest_user_config（按 create_time 倒序取首行）读到的就是本用例设定的预算。
    入参说明：
        amount (float)：月度总预算，单位元。
    返回值说明：
        无（副作用：改写 user_config 表内容）。
    """
    db.execute_sql("DELETE FROM user_config")
    db.add_user_config("广州", amount, "正常", "month")


def _seed_month_bill(amount: float, category: str):
    """
    函数功能与逻辑描述：
        向 bill 表插入一条"当天"的种子账单（remark 固定为"预警e2e种子"），用于把
        当月已花金额抬到目标档位；用例在 finally 中按 id 快照统一删除该行。
    入参说明：
        amount (float)：种子金额，单位元。
        category (str)：消费品类（如 "餐饮"）。
    返回值说明：
        无（副作用：向 bill 表写入 1 行）。
    """
    db.execute_sql(
        "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
        (amount, category, TODAY.isoformat(), "预警e2e种子"),
    )


def _collect_diagnostics(result: dict) -> str:
    """
    函数功能与逻辑描述：
        断言失败时的诊断文本拼装：先输出 final_reply，再遍历 task_plan 逐任务打印
        status 与 result（JSON 序列化后截断 300 字符），便于定位是哪个环节未产出预期事实。
    入参说明：
        result (dict)：orchestrator 执行后的 final state。
    返回值说明：
        str：多行诊断文本（1 行 final_reply + 每个任务 1 行）。
    """
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(f"[{tname}] status={tinfo.get('status')} result={json.dumps(res, ensure_ascii=False, default=str)[:300]}")
    return "\n".join(lines)


# ---------------- 汇总 LLM 劫持（表达层确定性模拟） ----------------
_FACTS_BLOCK_RE = re.compile(r"预警事实（JSON，可为空）：\n(\{.*?\})\n\n用户提醒偏好", re.S)


def _extract_alert_facts(user_text: str) -> dict:
    """
    函数功能与逻辑描述：
        从汇总提示词 user 侧文本（summarize_user.j2 渲染结果）中，用 _FACTS_BLOCK_RE
        抠出"预警事实（JSON，可为空）："与"用户提醒偏好"两段之间夹注的 JSON 字符串并
        反序列化，使被劫持的表达层能拿到确定性计算层产出的 alert_facts。
        边界：正则不匹配、JSON 非法、或顶层不是 dict 时统一返回空 dict，不抛异常。
    入参说明：
        user_text (str)：render_summarize_user 渲染出的用户提示词全文。
    返回值说明：
        dict：解析出的预警事实（含 l1/l2/l3）；无法解析时返回 {}。
    """
    m = _FACTS_BLOCK_RE.search(user_text or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _fake_summarize_reply(facts: dict) -> str:
    """
    函数功能与逻辑描述：
        按注入的预警事实确定性生成最终回复文案，替代真实外部 LLM 的汇总调用，
        从而端到端验证 L3 事实链路：l3.level 为 warn → 输出含"预算预警 + 每日建议控制在
        N 元以内"；over → 输出超支并点名首个超支品类；reward → 输出"预算还很充裕…犒劳"；
        其余（ok 或缺失）→ 仅输出"记账成功。"。
    入参说明：
        facts (dict)：_extract_alert_facts 解析出的预警事实；None/空 dict 时按 ok 档处理。
    返回值说明：
        str：确定性的中文最终回复文本（数字取自 l3.daily_limit / l3.over_categories）。
    """
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
    """
    函数功能与逻辑描述：
        安装表达层桩：用 monkeypatch 替换 mcp_client.call_llm_base，仅当 sys_prompt 命中
        汇总系统提示词（summarize.md 的角色行"最终回复生成器"）时返回 _fake_summarize_reply
        的确定性文案；其余 LLM 调用（规划节点、bill 解析等）透传原始 call_llm_base 走真实外部模型。
    入参说明：
        monkeypatch：pytest 内置 fixture，提供 setattr 以在用例内替换属性并在用例结束后自动还原。
    返回值说明：
        无（副作用：用例生命周期内替换 mcp_client.call_llm_base）。
    """
    original = mcp_client.call_llm_base

    async def _fake_call_llm_base(sys_prompt, user_text, agent_tag="unknown"):
        """
        函数功能与逻辑描述：
            替换 mcp_client.call_llm_base 的内层闭包：当 sys_prompt 命中汇总系统提示词
            （summarize.md 角色行"最终回复生成器"）时，用 _extract_alert_facts 解析出注入的
            预警事实并交由 _fake_summarize_reply 生成确定性文案返回；其余提示词（规划/解析等）
            原样透传给闭包捕获的原始 call_llm_base，走真实模型。
        入参说明：
            sys_prompt (str)：系统提示词，用于判定是否为汇总表达层调用。
            user_text (str)：用户侧提示词文本（汇总调用时含注入的预警事实 JSON）。
            agent_tag (str)：鉴权主体 / 差异化推理参数标签，默认 "unknown"；透传时原样转给原生实现。
        返回值说明：
            str：汇总调用返回确定性文案；其余调用返回原生 call_llm_base 的模型输出文本。
        """
        if "最终回复生成器" in sys_prompt:
            facts = _extract_alert_facts(user_text)
            return _fake_summarize_reply(facts)
        return await original(sys_prompt, user_text, agent_tag=agent_tag)

    monkeypatch.setattr(mcp_client, "call_llm_base", _fake_call_llm_base)


@pytest.mark.asyncio
async def test_e2e_budget_warn_alert(monkeypatch):
    """
    函数功能与逻辑描述：
        端到端验证「纯记账 → 预算接近红线（warn 档）」预警链：先备份 user_config 并记录
        bill 的 MAX(id) 快照，插入 38 元餐饮种子后按 (spent + 38) / 0.9 设置月预算，使本轮
        记账落库后 used_ratio ≈ 0.9（≥ warn_ratio 0.8）；再劫持汇总 LLM，断言最终回复含
        "记账成功""预算预警""每日建议控制在"及金额数字（正则 [\d.]+元）。
        运行时前置依赖：真实外部 LLM（规划与解析）、本地 bill / user_config 库、常驻子 Agent
        worker（由 conftest 拉起）；测试结束在 finally 中按 id 快照删除新增账单并恢复配置备份。
    入参说明：
        monkeypatch：pytest 内置 fixture，用于替换 mcp_client.call_llm_base 做表达层确定性模拟。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        端到端验证「纯记账 → 预算充裕（reward 档）」预警链：备份 user_config 并记录 bill
        的 MAX(id) 快照，插入 38 元餐饮种子后按 (spent + 38) / 0.25 设置月预算，使记账后
        used_ratio ≈ 0.25（0 < ratio ≤ reward_ratio 0.5）；再劫持汇总 LLM，断言最终回复含
        "记账成功""预算还很充裕""犒劳"。
        运行时前置依赖：真实外部 LLM（规划与解析）、本地 bill / user_config 库、常驻子 Agent
        worker（由 conftest 拉起）；测试结束在 finally 中按 id 快照删除新增账单并恢复配置备份。
    入参说明：
        monkeypatch：pytest 内置 fixture，用于替换 mcp_client.call_llm_base 做表达层确定性模拟。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
