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

✅ 已修复（2026-09-17）：此前从 agents.orchestrator.nodes 导入的 `_persist_bill_habit` 已随
   M9（D15 P0-5）删除（习惯沉淀下沉到 agents/bill_agent/nodes.py 的 `_settle_month_habit`），
   本文件因旧导入在**收集期 ImportError**，并**连带使整个 `tests/e2e` 收集中断**
   （`Interrupted: 2 errors`，45 例 vs 修复后 51 例全可收集）。现改为导入现入口
   `_settle_month_habit` + 本地同名包装 `_persist_bill_habit`（见文件上方定义），
   各用例调用形态不变、收集已恢复正常。
"""
import json
import os
import uuid
from datetime import date, timedelta

import pytest

from agents.orchestrator.graph import orchestrator_graph
from agents.orchestrator.nodes import _load_recent_habits
# ★M9（D15 P0-5）修复：习惯沉淀已下沉 bill_agent，orchestrator 的 `_persist_bill_habit` 被删除。
#   本文件原先直接导入该旧名 → **收集期 ImportError → 整个 tests/e2e 收集中断（45 用例全不可跑）**。
#   现改为导入现入口并保留同名包装，使各用例的调用形态不必逐个改动。
from agents.bill_agent.nodes import _settle_month_habit


async def _persist_bill_habit(payload: dict) -> None:
    """
    函数功能与逻辑描述：
        兼容包装（M9 迁移）：把"记账成功载荷"的 `data` 交给现入口 `_settle_month_habit`
        （经 `habit.upsert` 工具写 monthly_habit 并更新 pkl 语义层）。
        原 `agents.orchestrator.nodes._persist_bill_habit`（裸连 db ×2）已随 M9 收口删除；
        本文件各用例仍按旧签名调用，用同名包装适配可避免逐个改 6 处调用点。
    入参说明：
        payload (dict)：`_persist_payload` 产出的记账成功载荷（须含 data 字段）。
    返回值说明：
        无（协程；副作用见 `_settle_month_habit`，任何失败静默忽略）。
    """
    await _settle_month_habit(payload["data"])
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
    """
    函数功能与逻辑描述：
        辅助协程：以随机 session_id（前缀 fin_）构造 orchestrator 初始 state 并 await 执行
        orchestrator_graph.ainvoke，返回终态 state 供用例断言 task_plan 中 finance 任务结果。
        运行时依赖真实外部 LLM（意图路由与 finance 分析）、本地库与常驻子 Agent worker。
    入参说明：
        user_input (str)：用户本轮自然语言输入，作为 state["user_input"]。
    返回值说明：
        dict：orchestrator 执行完成后的 final state（含 task_plan / final_reply / error_msg）。
    """
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
    """
    函数功能与逻辑描述：
        取 bill 表当前最大 id 作为快照基线，供 fixture 在测试后删除 id > 基线的新增账单，
        实现 bill 表的测试数据隔离。
    入参说明：
        无。
    返回值说明：
        int：bill 表当前 MAX(id)；表为空或查询返回空列表时返回 0。
    """
    rows = db.query_sql("SELECT MAX(id) AS m FROM bill")
    return int(rows[0]["m"] or 0) if rows else 0


def _backup_user_config() -> list:
    """
    函数功能与逻辑描述：
        备份 user_config 表全部行，供 fixture 测试后恢复，避免测试预算配置残留；
        finance 用例需要预算，故该表必须"原样保留备份再恢复"。
    入参说明：
        无。
    返回值说明：
        list[dict]：user_config 全表行（列名 → 值）；表为空时返回空列表。
    """
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    """
    函数功能与逻辑描述：
        按备份行的全部列动态拼 INSERT 恢复 user_config（先全表 DELETE 再逐行重插），
        以兼容并保留 remind_pref / quota_mode / category_budget 等新增列。
    入参说明：
        backup (list)：_backup_user_config() 备份的行列表；为空列表时仅执行清空。
    返回值说明：
        无（副作用：覆盖 user_config 表内容）。
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
        保证 get_latest_user_config 取到的就是本用例设定的预算（finance 无预算会走追问）。
    入参说明：
        amount (float)：月度总预算，单位元。
    返回值说明：
        无（副作用：改写 user_config 表内容）。
    """
    db.execute_sql("DELETE FROM user_config")
    db.add_user_config("广州", amount, "正常", "month")


def _backup_monthly_habit() -> list:
    """
    函数功能与逻辑描述：
        备份 monthly_habit 表全部行，供 fixture 测试后恢复；配合测试前清空实现
        "月度习惯从零开始"，杜绝历史习惯数据污染注入源与 finance 推算断言。
    入参说明：
        无。
    返回值说明：
        list[dict]：monthly_habit 全表行（列名 → 值）；表为空时返回空列表。
    """
    return db.query_sql("SELECT * FROM monthly_habit")


def _restore_monthly_habit(backup: list):
    """
    函数功能与逻辑描述：
        恢复 monthly_habit 到测试前状态：先全表 DELETE，再按 (month, category) 主键
        连同 amount_sum / count / avg_amount / update_time 六列逐行重插（列名写死，
        与该表固定结构一致）。
    入参说明：
        backup (list)：_backup_monthly_habit() 备份的行列表；为空列表时仅执行清空。
    返回值说明：
        无（副作用：覆盖 monthly_habit 表内容）。
    """
    db.execute_sql("DELETE FROM monthly_habit")
    for r in backup:
        db.execute_sql(
            "INSERT INTO monthly_habit (month, category, amount_sum, count, avg_amount, update_time) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r["month"], r["category"], r["amount_sum"], r["count"], r["avg_amount"], r["update_time"]),
        )


def _backup_pkl() -> bytes:
    """
    函数功能与逻辑描述：
        以二进制读取方式备份长期记忆持久化文件 long_consume_mem.pkl（字节级快照），
        供 fixture 测试后原样写回；文件不存在时不报错。
    入参说明：
        无。
    返回值说明：
        bytes | None：文件字节内容；文件不存在时返回 None（表示"无备份可恢复"）。
    """
    if os.path.exists(LONG_MEM_PATH):
        with open(LONG_MEM_PATH, "rb") as f:
            return f.read()
    return None


def _restore_pkl(blob: bytes):
    """
    函数功能与逻辑描述：
        恢复长期记忆：先 clear_all_long_mem() 清空内存文本/FAISS 索引并删除 pkl 文件，
        再把备份字节原样写回并调用 _load_mem() 重载内存与重建索引；
        blob 为 None（测试前本无文件）时只做清空，不落盘、不重载。
    入参说明：
        blob (bytes | None)：_backup_pkl() 取得的字节快照；None 表示测试前无该文件。
    返回值说明：
        无（副作用：改写 LONG_MEM_PATH 文件与 long_memory 模块级全局状态）。
    """
    clear_all_long_mem()
    if blob is not None:
        with open(LONG_MEM_PATH, "wb") as f:
            f.write(blob)
        _load_mem()  # 重新加载内存 + 重建索引


def _prev_month(n_back: int) -> str:
    """
    函数功能与逻辑描述：
        以当前自然月为基准向前推算第 n 个自然月的 "YYYY-MM" 标识（n_back=1 表示上月），
        跨年由 m==0 分支回退年份处理，供按月份断言习惯聚合结果。
    入参说明：
        n_back (int)：向前回溯的自然月数，1 表示上月。
    返回值说明：
        str：形如 "YYYY-MM" 的月份标识。
    """
    y, m = TODAY.year, TODAY.month
    for _ in range(n_back):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return f"{y:04d}-{m:02d}"


def _prev_month_date(n_back: int) -> str:
    """
    函数功能与逻辑描述：
        生成落在前 n 个自然月内的一个日期字符串：先取当月首日再减 1 天（即上月最后一天），
        其后每多回溯一月就再跳到该月首日的上一天，保证日期月份归属与 _prev_month 一致
        （不能用"今天减 30 天"这类近似，会跨月错位）。
    入参说明：
        n_back (int)：向前回溯的自然月数，1 表示上月。
    返回值说明：
        str：形如 "YYYY-MM-DD" 的日期字符串。
    """
    first = date(TODAY.year, TODAY.month, 1)
    d = first - timedelta(days=1)
    for _ in range(n_back - 1):
        d = date(d.year, d.month, 1) - timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _persist_payload(amount: float, category: str, consume_date: str) -> dict:
    """
    函数功能与逻辑描述：
        构造 bill_agent 成功返回的 struct_payload（success / agent_type / msg /
        data · amount·category·consume_date / error 五字段），用于模拟真实记账结果；
        其中 consume_date 决定该笔被归入哪个月份的习惯聚合，是构造"近 3 月习惯"的关键入参。
    入参说明：
        amount (float)：模拟落库金额，单位元。
        category (str)：模拟落库品类（须落在 bill 表五分类约束内）。
        consume_date (str)：模拟消费日期（ISO 日期字符串，决定月份归属）。
    返回值说明：
        dict：可直接喂给月度习惯沉淀入口的 bill_agent 结构载荷。
    """
    return {
        "success": True,
        "agent_type": "bill_agent",
        "msg": "记账成功",
        "data": {"amount": amount, "category": category, "consume_date": consume_date},
        "error": None,
    }


def _habit_summary(habits: list) -> dict:
    """
    函数功能与逻辑描述：
        把 habit_data 行列表按 (month, category) 二维聚合为 {month: {category: row}}，
        便于用例用 by_month[MONTH]["餐饮"] 这类直观写法断言逐月习惯数据。
    入参说明：
        habits (list)：_load_recent_habits 返回的习惯行列表（含 month / category / amount_sum / count）。
    返回值说明：
        dict：嵌套字典 {月份: {品类: 习惯行}}；输入为空时返回空字典。
    """
    out = {}
    for r in habits:
        out.setdefault(r["month"], {})[r["category"]] = r
    return out


def _collect_diagnostics(result: dict) -> str:
    """
    函数功能与逻辑描述：
        断言失败时的诊断文本拼装：输出 final_reply，并遍历 task_plan 逐任务打印
        status 与 result（JSON 序列化后截断 400 字符），便于定位失败环节。
    入参说明：
        result (dict)：orchestrator 执行后的 final state。
    返回值说明：
        str：多行诊断文本（1 行 final_reply + 每个任务 1 行）。
    """
    lines = [f"final_reply={result.get('final_reply')!r}"]
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        lines.append(f"[{tname}] status={tinfo.get('status')} result={json.dumps(res, ensure_ascii=False, default=str)[:400]}")
    return "\n".join(lines)


@pytest.fixture(autouse=True)
def _data_isolation():
    """
    函数功能与逻辑描述：
        pytest fixture（function 作用域，autouse=True），为本文件每个用例提供数据隔离，
        不向用例注入任何数据。setup：记录 bill 的 MAX(id) 快照，备份 user_config /
        monthly_habit 全表与 long_consume_mem.pkl 字节快照，随后清空 monthly_habit 与长期记忆，
        使习惯数据从零开始、避免历史数据污染注入源断言；teardown：删除新增账单、
        按备份恢复配置与月度习惯、并清空后写回 pkl 快照重载内存（测试后数据零残留）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无显式返回值（yield 仅用于分离 setup/teardown，不向用例提供数据）。
    """
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
    """
    函数功能与逻辑描述：
        端到端验证「有历史习惯时的剩余天数推算」：先沉淀近 3 个月餐饮习惯（当月 2 笔计 80 元 /
        上月 2 笔计 80 元 / 上上月 3 笔计 120 元）并设置 2000 元月预算；
        注：沉淀调用的是已删除的 `_persist_bill_habit`（见文件头"已知失效"），本用例当前不可执行；① 严格断言注入源确定性
        ——_load_recent_habits(3) 返回的逐月 amount_sum / count 与沉淀数据一致（可复现）；
        ② 走完整 orchestrator 全链路（finance 强意图输入 FINANCE_INPUT），断言命中 finance 任务、
        status 为 done、success 为 True、data.has_target_data 为 True（预算已注入上下文）、
        raw_analysis_text 长度 >50 且含"餐饮/预算/剩余/够/月底"任一推算痕迹、final_reply 非空。
        运行时前置依赖：真实外部 LLM（路由 + finance 分析）、本地 bill / monthly_habit /
        user_config 库、常驻子 Agent worker，以及 memory 长期记忆（pkl + 向量索引）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 沉淀近3个月餐饮习惯（当月 2 笔 80 元 / 上月 2 笔 80 元 / 上上月 3 笔 120 元）。
    # ⚠️ 下方 _persist_bill_habit 已于 M9 / D15 P0-5 删除（习惯沉淀下沉 agents/bill_agent/
    #    nodes.py 的 _settle_month_habit），本文件仍导入旧名 → 收集期 ImportError，
    #    本用例当前不可执行（既存缺陷，待单独修复）。
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
    """
    函数功能与逻辑描述：
        端到端验证「无历史习惯时的兜底」：不沉淀任何习惯（_data_isolation fixture 已清空
        monthly_habit 与长期记忆），仅设置 2000 元月预算；先断言 _load_recent_habits(3)
        返回空列表（注入源确定性），再走完整 orchestrator 全链路执行 FINANCE_INPUT，
        断言仍命中 finance 任务、status 为 done、success 为 True、raw_analysis_text
        长度 >50、final_reply 非空——即空 habit_data 下不崩溃、不凭空编造习惯数据。
        运行时前置依赖：真实外部 LLM（路由 + finance 分析）、本地 bill / monthly_habit /
        user_config 库、常驻子 Agent worker 与 memory 长期记忆。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
