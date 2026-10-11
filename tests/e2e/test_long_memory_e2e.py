# -*- coding: utf-8 -*-
"""
需求3 长期记忆闭环 端到端测试：记账成功 → monthly_habit 事实表聚合（同月累加/跨月独立）
→ pkl 语义层同步 → finance 全链路注入 habit_data → 12 个月滚动清理。

数据隔离（每个用例前后自动恢复，测试后数据零残留）：
- bill 表：MAX(id) 快照法，运行后删除 id > 快照 的新增记录
- monthly_habit 表：全表备份 → 测试前清空（从零开始，杜绝历史数据污染）→ 测试后恢复备份
- user_config 表：全表备份，测试后按 city / month_budget / consume_mode / budget_type
  四列重插恢复（finance 用例需预算；不走全列恢复，category_budget / quota_mode /
  remind_pref / create_time 不还原）
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
import pickle
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
        本文件各用例仍按旧签名调用，用同名包装适配可避免逐个改 7 处调用点，
        并把"为什么改"集中在一处，便于后人核对。
    入参说明：
        payload (dict)：`_persist_payload` 产出的记账成功载荷（须含 data 字段）。
    返回值说明：
        无（协程；副作用见 `_settle_month_habit`，任何失败静默忽略）。
    """
    await _settle_month_habit(payload["data"])
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
    """
    函数功能与逻辑描述：
        辅助协程：以随机 session_id（前缀 mem_）构造 orchestrator 初始 state 并 await 执行
        orchestrator_graph.ainvoke，返回终态 state 供用例断言 final_reply 等字段
        （习惯聚合断言另经 DB 查询，不依赖 state）。
        运行时依赖真实外部 LLM 与本地账单库；习惯沉淀链路已下沉 bill_agent，由记账方收尾。
    入参说明：
        user_input (str)：用户本轮自然语言输入，作为 state["user_input"]。
    返回值说明：
        dict：orchestrator 执行完成后的 final state（含 task_plan / final_reply / error_msg）。
    """
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
        备份 user_config 表全部行，供 fixture 测试后恢复（finance 用例需要预算存在），
        避免测试预算配置残留污染真实用户配置。
    入参说明：
        无。
    返回值说明：
        list[dict]：user_config 全表行（列名 → 值）；表为空时返回空列表。
    """
    return db.query_sql("SELECT * FROM user_config")


def _restore_user_config(backup: list):
    """
    函数功能与逻辑描述：
        恢复 user_config：先全表 DELETE，再按备份行重插 city / month_budget /
        consume_mode / budget_type 四列（列名写死，不还原 category_budget、quota_mode、
        remind_pref、create_time 等列，故为"部分列恢复"而非全表还原）。
    入参说明：
        backup (list)：_backup_user_config() 备份的行列表；为空列表时仅执行清空。
    返回值说明：
        无（副作用：覆盖 user_config 表内容，且新增列取默认值/NULL）。
    """
    db.execute_sql("DELETE FROM user_config")
    for r in backup:
        db.execute_sql(
            "INSERT INTO user_config (city, month_budget, consume_mode, budget_type) VALUES (?, ?, ?, ?)",
            (r["city"], r["month_budget"], r["consume_mode"], r["budget_type"]),
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
        "月度习惯从零开始"，杜绝历史习惯数据污染本轮断言。
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


def _month_habit_rows(month: str) -> dict:
    """
    函数功能与逻辑描述：
        读取指定月份的全部月度习惯聚合行（走 db.get_monthly_habits([month])），
        并按品类转成 {category: row} 便于用例直接按品类取行断言金额/笔数/均额。
    入参说明：
        month (str)：自然月标识，格式 "YYYY-MM"。
    返回值说明：
        dict：{品类: 习惯行}；该月无数据时返回空字典。
    """
    rows = db.get_monthly_habits([month])
    return {r["category"]: r for r in rows}


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


def _persist_payload(amount: float, category: str, consume_date: str) -> dict:
    """
    函数功能与逻辑描述：
        构造 bill_agent 成功返回的 struct_payload（success / agent_type / msg /
        data · amount·category·consume_date / error 五字段），用于模拟真实记账结果；
        consume_date 决定该笔被归入哪个月份的习惯聚合（跨月用例据此构造上月账单）。
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


@pytest.fixture(autouse=True)
def _data_isolation():
    """
    函数功能与逻辑描述：
        pytest fixture（function 作用域，autouse=True），为本文件每个用例提供数据隔离，
        不向用例注入任何数据。setup：记录 bill 的 MAX(id) 快照，备份 user_config /
        monthly_habit 全表与 long_consume_mem.pkl 字节快照，随后清空 monthly_habit 与长期记忆，
        使习惯与语义层数据从零开始、避免历史数据污染断言；teardown：删除新增账单、
        恢复 user_config（四列）与 monthly_habit 备份、并清空后写回 pkl 快照重载内存。
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
async def test_e2e_habit_same_month_accumulate():
    """
    函数功能与逻辑描述：
        验证「同月多笔记账 → monthly_habit 累加」：第一笔走真实记账链路
        （晚饭38元记一下 [记忆e2e]，端到端验证 记账→习惯沉淀），断言回复含品类"餐饮"；
        随后再补沉淀 42 元与 20 元两笔，断言本月餐饮行 amount_sum 求和为 100.0
        （38+42+20）、count 累加为 3、avg_amount 重算为 33.33（两位小数）。
        运行时前置依赖：真实外部 LLM 与本地 bill / monthly_habit 库、常驻子 Agent worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 一笔走真实 LLM 记账链路（验证 记账→沉淀 端到端），其余两笔经习惯沉淀入口补齐累加。
    # ⚠️ 该入口即 `_persist_bill_habit`（已于 M9 / D15 P0-5 从 orchestrator 删除，习惯沉淀
    #    下沉 agents/bill_agent/nodes.py 的 _settle_month_habit）；本文件仍导入旧名，
    #    pytest 收集期即 ImportError，本用例当前不可执行（既存缺陷，待单独修复）。
    result = await _run_graph("晚饭38元记一下 [记忆e2e]")
    # ★2026-08-31 断言调整：final_reply 为外部LLM自然表达（summarize.md 要求记账
    #   确认必含品类），"记账成功"固定短语仅存在于本地确定性渲染；本用例核心是
    #   monthly_habit 聚合（下方三条 amount_sum / count / avg_amount 断言），
    #   此处仅验证记账链路已返回品类。
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
    """
    函数功能与逻辑描述：
        验证「跨月账单写入独立月份行」：以 consume_date 落在上月的方式沉淀 35 元交通，
        再沉淀本月 38 元餐饮，断言上月交通行 amount_sum 为 35.0 且 count 为 1、本月餐饮行
        amount_sum 为 38.0；并反向断言月份间互不污染（上月无餐饮行、本月无交通行）。
        运行时前置依赖：本地 bill / monthly_habit 库（不走 LLM，全部经习惯沉淀入口写入——
        该入口 `_persist_bill_habit` 已于 M9 / D15 P0-5 删除，见文件头"已知失效"，
        本用例当前因收集期 ImportError 不可执行）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证「习惯数据注入 finance」端到端：先沉淀本月餐饮 38 元 + 42 元两笔并设置 2000 元预算；
        ① 注入源校验——_load_recent_habits(3) 能取到本月餐饮行且 amount_sum 为 80.0、count 为 2；
        ② 以含 FINANCE_HINT 关键词且无金额的输入（“我想分析一下我的消费习惯，给我一些省钱建议
        [记忆e2e]”）走完整 orchestrator 全链路，断言命中 finance、status 为 done、success 为 True、
        data.has_target_data 为 True（预算与习惯已注入上下文）、final_reply 非空。
        运行时前置依赖：真实外部 LLM（路由 + finance 分析）、本地 bill / monthly_habit /
        user_config 库、常驻子 Agent worker 与 memory 长期记忆。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 先沉淀本月餐饮习惯（模拟历史数据，经习惯沉淀入口写入）。
    # ⚠️ 入口 `_persist_bill_habit` 已于 M9 / D15 P0-5 删除（下沉 bill_agent 的
    #    _settle_month_habit），本文件旧导入 → 收集期 ImportError，用例当前不可执行。
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
    """
    函数功能与逻辑描述：
        验证「12 个月滚动清理」双链路：① DB 层——以 _keep_recent_months_boundary(12) 取边界月
        再往前推一月得到超期月份，写入该月与当月的习惯后调用 db.cleanup_monthly_habit(12)，
        断言超期月份被删除、当月数据保留；② 语义层——分别 upsert 超期月份与当月的习惯摘要，
        断言 pkl（LONG_MEM_PATH）已落盘、文件中不再含超期月份摘要、仍含当月摘要。
        运行时前置依赖：本地 monthly_habit 库与长期记忆 pkl/FAISS（不涉及 LLM 与子 Agent）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；数据隔离由 _data_isolation fixture 提供）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
