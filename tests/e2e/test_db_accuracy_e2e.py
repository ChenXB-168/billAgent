# -*- coding: utf-8 -*-
"""
端到端【数据准确性】测试：从用户端输入走完整 orchestrator 全链路，落库硬校验。

定位策略：MAX(id) 快照法 —— 不依赖 remark 标记（LLM 会剥离方括号标记），
每次用例运行前记录 max_id，运行后查询 id>max_id 的新增记录逐字段校验。
"""
import json
import uuid
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from agents.orchestrator.graph import orchestrator_graph
from utils.common import db

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


async def _run_graph(user_input: str, session_id: str | None = None):
    """
    函数功能与逻辑描述：
        辅助协程：构造 orchestrator 初始 state（空历史、空任务计划、无错误）并 await 执行
        orchestrator_graph.ainvoke，返回终态 state 供用例做落库/统计硬校验。
        运行时依赖真实外部 LLM（规划与解析环节）、本地账单库与常驻子 Agent worker。
    入参说明：
        user_input (str)：用户本轮自然语言输入，作为 state["user_input"]。
        session_id (str | None)：会话号；缺省（None）时随机生成 acc_ + 8 位十六进制，
            多轮连续性测试需显式传入同一 session 以跨轮复用会话内存与同一会话的账单归属。
    返回值说明：
        dict：orchestrator 执行完成后的 final state（含 task_plan / final_reply / error_msg）。
    """
    init_state = {
        "user_input": user_input,
        "session_id": session_id or f"acc_{uuid.uuid4().hex[:8]}",
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
        取 bill 表当前最大 id 作为快照基线（不依赖 remark 标记——LLM 会剥离方括号标记），
        用例据此在运行后仅捞出 id > 基线的新增记录做逐字段校验。
    入参说明：
        无。
    返回值说明：
        int：bill 表当前 MAX(id)；表为空或查询返回空列表时返回 0。
    """
    rows = db.query_sql("SELECT MAX(id) AS m FROM bill")
    return int(rows[0]["m"] or 0) if rows else 0


def _new_records(after_id: int):
    """
    函数功能与逻辑描述：
        查询 id 大于快照基线的全部账单记录（按 id 升序），返回字段含
        id / amount / category / consume_time / remark，供用例校验落库条数与字段正确性。
    入参说明：
        after_id (int)：快照基线，即 _max_id() 在用例开始前取得的值。
    返回值说明：
        list[dict]：新增账单行（列名 → 值）；无新增时返回空列表。
    """
    return db.query_sql(
        "SELECT id, amount, category, consume_time, remark FROM bill WHERE id > ? ORDER BY id ASC",
        (after_id,),
    )


def _clean_after(after_id: int):
    """
    函数功能与逻辑描述：
        测试数据隔离：删除 id 大于基线的全部账单行，把 bill 表还原到用例开始前的状态；
        删除失败被吞掉（pass），不因清理异常影响用例断言结果。
    入参说明：
        after_id (int)：快照基线，即 _max_id() 在用例开始前取得的值。
    返回值说明：
        无（副作用：删除 bill 表新增行；异常时静默跳过）。
    """
    try:
        db.execute_sql("DELETE FROM bill WHERE id > ?", (after_id,))
    except Exception:
        pass


def _collect_diagnostics(result: dict) -> str:
    """
    函数功能与逻辑描述：
        断言失败时的诊断文本拼装：输出 error_msg，并遍历 task_plan 逐任务打印
        agent_type / success / msg / error / data（data 截断 200 字符），
        非 dict 结果则直接打印原值与 status，便于定位失败环节。
    入参说明：
        result (dict)：orchestrator 执行后的 final state。
    返回值说明：
        str：多行诊断文本（error_msg + 各任务详情 + final_reply）。
    """
    lines = []
    lines.append(f"error_msg={result.get('error_msg')!r}")
    tp = result.get("task_plan") or {}
    for tname, tinfo in tp.items():
        res = tinfo.get("result")
        if isinstance(res, dict):
            lines.append(
                f"[{tname}] agent={res.get('agent_type')} success={res.get('success')} "
                f"msg={res.get('msg')} error={res.get('error')}"
            )
            if res.get("data") is not None:
                lines.append(f"[{tname}] data={json.dumps(res['data'], ensure_ascii=False)[:200]}")
        else:
            lines.append(f"[{tname}] result={res!r} status={tinfo.get('status')}")
    lines.append(f"final_reply={result.get('final_reply')!r}")
    return "\n".join(lines)


# ============================================================
# 一、记账准确性（从用户端输入 -> 全链路 -> 查库硬校验）
# ============================================================
# (id, user_input, 期望金额, 期望类别, 期望日期或None=今天)
# ★2026-09-17 用例修正（实测驱动）：本组输入原先**多数不含记账指令**（如"晚饭38元""奶茶18块"），
#   在"金额+消费实体即记账"的旧规则下可入库；但现行规则（`system.md` 2026-09-05）要求
#   **必须有显式记账指令**（"记/记账/记一下/帮我记"等）才记账，无指令的消费陈述一律归**比价咨询**。
#   实测：ba_10/11/12/14/15 因此稳定失败，而 ba_02/04/05… 只是**偶发通过**（LLM 随机规划了 bill）
#   —— 同属不稳定的错误期望。现统一为每条补上自然记账指令，**保留原验证目标**
#   （金额 / 品类 / 日期 三项抽取准确性），使用例与规则一致且不再 flaky。
BILL_ACC_CASES = [
    ("ba_01", "打车35元帮我记一下", 35.0, "交通", TODAY.isoformat()),
    ("ba_02", "晚饭38元记一下", 38.0, "餐饮", TODAY.isoformat()),
    ("ba_03", "买了一件衣服199元记个账", 199.0, "购物", TODAY.isoformat()),
    ("ba_04", "住酒店一晚388元记一下", 388.0, "住宿", TODAY.isoformat()),
    ("ba_05", "看电影花了45元记一下", 45.0, "娱乐", TODAY.isoformat()),
    ("ba_06", "中午外卖25.5元记一下", 25.5, "餐饮", TODAY.isoformat()),
    ("ba_07", "地铁充值50元记一下", 50.0, "交通", TODAY.isoformat()),
    ("ba_08", "买书花了59.9元记一下", 59.9, "购物", TODAY.isoformat()),
    ("ba_09", "今天打车到机场85元记一下", 85.0, "交通", TODAY.isoformat()),
    ("ba_10", "奶茶18块记一下", 18.0, "餐饮", TODAY.isoformat()),
    ("ba_11", f"昨天打车花了35元记一下", 35.0, "交通", YESTERDAY.isoformat()),
    ("ba_12", f"昨天晚饭吃了62元记一下", 62.0, "餐饮", YESTERDAY.isoformat()),
    ("ba_13", "KTV唱歌180元记一下", 180.0, "娱乐", TODAY.isoformat()),
    ("ba_14", "打车到公司28.5元记一下", 28.5, "交通", TODAY.isoformat()),
    ("ba_15", "超市买日用品128元记一下", 128.0, "购物", TODAY.isoformat()),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id,user_input,exp_amount,exp_category,exp_date",
    BILL_ACC_CASES,
    ids=[c[0] for c in BILL_ACC_CASES],
)
async def test_bill_record_accuracy(case_id, user_input, exp_amount, exp_category, exp_date):
    """
    函数功能与逻辑描述：
        记账准确性参数化用例：对每条输入走完整 orchestrator 全链路（真实外部 LLM 抽取），
        运行前后以 MAX(id) 快照圈定新增记录，硬校验「新增恰好 1 条」且其
        amount / category / consume_time 与 BILL_ACC_CASES 期望值完全一致（金额容差 0.001）。
        覆盖边界：今日与昨日两种日期（昨天打车/昨天晚饭）；未落库、多落库、字段不符
        三种失败形态均给出结构化 detail 与全链路诊断。结束后 _clean_after 清理新增记录。
    入参说明：
        case_id (str)：参数组合标识（ba_01~ba_15），pytest 用其作为用例 id。
        user_input (str)：该组合的用户自然语言输入。
        exp_amount (float)：期望落库金额（如 35.0 / 25.5）。
        exp_category (str)：期望落库品类（交通/餐饮/购物/住宿/娱乐）。
        exp_date (str)：期望 consume_time（ISO 日期字符串；昨天场景为 YESTERDAY）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    before_id = _max_id()
    result = await _run_graph(user_input)
    rows = _new_records(before_id)
    ok = True
    detail = ""
    if not rows:
        ok = False
        detail = "数据库未新增任何记录（未落库）"
    elif len(rows) > 1:
        ok = False
        detail = f"新增 {len(rows)} 条记录（期望1条）: {json.dumps(rows, ensure_ascii=False, default=str)[:200]}"
    else:
        r = rows[0]
        amount_ok = abs(float(r["amount"]) - exp_amount) < 0.001
        cat_ok = r["category"] == exp_category
        date_ok = r["consume_time"] == exp_date
        if not (amount_ok and cat_ok and date_ok):
            ok = False
            detail = (
                f"落库数据不符 期望(amount={exp_amount}, category={exp_category}, date={exp_date}) "
                f"实际(amount={r['amount']}, category={r['category']}, date={r['consume_time']}, remark={r['remark']!r}) "
                f"金额{'OK' if amount_ok else 'FAIL'} 类别{'OK' if cat_ok else 'FAIL'} 日期{'OK' if date_ok else 'FAIL'}"
            )
    _clean_after(before_id)
    assert ok, f"{case_id} 输入={user_input!r}\n{detail}\n{_collect_diagnostics(result)}"


# (id, user_input) —— 预期"不落库"（收入/退款类非支出内容、负数非法金额）
BILL_NO_RECORD_CASES = [
    ("bn_01", "这个月工资发了8000元"),
    ("bn_02", "打车-30元记一下"),
    ("bn_03", "收到了退款100元"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,user_input", BILL_NO_RECORD_CASES, ids=[c[0] for c in BILL_NO_RECORD_CASES])
async def test_bill_no_record(case_id, user_input):
    """
    函数功能与逻辑描述：
        不落库反向校验参数化用例：对每条输入走完整 orchestrator 全链路，以 MAX(id) 快照
        圈定新增记录，断言 bill 表「零新增」——覆盖工资收入（bn_01）、负数金额
        「打车-30元」（bn_02）、退款（bn_03）三类不得入账的场景；失败时打印实际写入行
        与全链路诊断。无论断言结果如何均执行 _clean_after 清理。
    入参说明：
        case_id (str)：参数组合标识（bn_01~bn_03），pytest 用其作为用例 id。
        user_input (str)：该组合的用户自然语言输入（预期不产生账单记录）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    before_id = _max_id()
    result = await _run_graph(user_input)
    rows = _new_records(before_id)
    detail = ""
    if rows:
        detail = (
            f"应当不落库，但实际写入 {len(rows)} 条: "
            f"{json.dumps(rows, ensure_ascii=False, default=str)[:200]}"
        )
    _clean_after(before_id)
    assert not rows, f"{case_id} 输入={user_input!r}\n{detail}\n{_collect_diagnostics(result)}"


# ============================================================
# 二、统计准确性（从用户端输入 -> 全链路 -> 与 bill 表真实值比对）
# ============================================================
def _month_range():
    """
    函数功能与逻辑描述：
        返回本月统计区间（本月首日 ~ 今天）的两个 ISO 日期字符串，作为 seed 前基线聚合
        与期望值推导共用的 consume_time BETWEEN 边界。
    入参说明：
        无（首日由 TODAY.replace(day=1) 推导，末值为 TODAY）。
    返回值说明：
        tuple[str, str]：(本月首日, 今天) 的 ISO 日期字符串元组。
    """
    first = TODAY.replace(day=1).isoformat()
    return first, TODAY.isoformat()


def _seed_bill(amount: float, category: str, consume_time: str, remark: str):
    """
    函数功能与逻辑描述：
        直接向 bill 表插入一条可控的种子账单（金额/品类/消费时间/备注全量指定），
        把统计口径的基线抬到已知值；用例在 finally 中按 id 快照统一清理。
    入参说明：
        amount (float)：种子金额，单位元。
        category (str)：消费品类。
        consume_time (str)：消费时间（ISO 日期字符串，决定是否落入本月统计区间）。
        remark (str)：备注；明细类用例据此在返回结果中定位该条记录。
    返回值说明：
        无（副作用：向 bill 表写入 1 行）。
    """
    db.execute_sql(
        "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
        (amount, category, consume_time, remark),
    )


@pytest.mark.asyncio
async def test_stat_total_amount_accuracy():
    """
    函数功能与逻辑描述：
        统计准确性（本月总支出）：先取本月真实 SUM 作为基线，再插入 10/20/30 三笔种子，
        执行"这个月总共花了多少钱"，断言 stat 任务成功且 data.total_amount 等于
        「DB 基线 + 60.0」（容差 0.01），并断言最终回复含标准术语"总支出"。
        期望值由库内真实聚合推导，兼容库中已有历史记录、不依赖空库环境。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ts, te = _month_range()
    base = float(db.query_sql("SELECT SUM(amount) AS s FROM bill WHERE consume_time BETWEEN ? AND ?", (ts, te))[0]["s"] or 0)
    # seed 3 笔已知数据（本月）
    seeds = [(10.0, "餐饮"), (20.0, "交通"), (30.0, "购物")]
    for amt, cat in seeds:
        _seed_bill(amt, cat, TODAY.isoformat(), "统计种子数据")
    before_id = _max_id()
    try:
        result = await _run_graph("这个月总共花了多少钱")
        stat_res = (result.get("task_plan") or {}).get("stat", {}).get("result")
        assert stat_res, f"stat 任务无结果\n{_collect_diagnostics(result)}"
        assert stat_res.get("success"), f"stat 失败: {_collect_diagnostics(result)}"
        total = float(stat_res["data"].get("total_amount", 0))
        expect = base + 60.0
        assert abs(total - expect) < 0.01, (
            f"本月总支出不符 期望={expect} 实际={total} "
            f"(seed 10+20+30, 基线={base})\n{_collect_diagnostics(result)}"
        )
        reply = result.get("final_reply") or ""
        assert "总支出" in reply, f"回复缺少'总支出'字段: {_collect_diagnostics(result)}"
    finally:
        _clean_after(before_id)


@pytest.mark.asyncio
async def test_stat_category_accuracy():
    """
    函数功能与逻辑描述：
        统计准确性（分类聚合）：插入餐饮 15.5 + 餐饮 24.5 + 交通 99.0 三笔种子，执行
        "这个月餐饮总共花了多少钱"，断言 stat 任务成功且 data.category_summary["餐饮"]
        （键缺失时退化取 data.total_amount）等于 DB 按 category='餐饮' 的真实 SUM（容差 0.01）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ts, te = _month_range()
    _seed_bill(15.5, "餐饮", TODAY.isoformat(), "统计种子-餐饮1")
    _seed_bill(24.5, "餐饮", TODAY.isoformat(), "统计种子-餐饮2")
    _seed_bill(99.0, "交通", TODAY.isoformat(), "统计种子-交通")
    before_id = _max_id()
    try:
        result = await _run_graph("这个月餐饮总共花了多少钱")
        stat_res = (result.get("task_plan") or {}).get("stat", {}).get("result")
        assert stat_res, f"stat 任务无结果\n{_collect_diagnostics(result)}"
        assert stat_res.get("success"), f"stat 失败: {_collect_diagnostics(result)}"
        data = stat_res["data"]
        cat_summary = data.get("category_summary") or {}
        expect_cat = float(db.query_sql(
            "SELECT SUM(amount) AS s FROM bill WHERE consume_time BETWEEN ? AND ? AND category='餐饮'",
            (ts, te),
        )[0]["s"] or 0)
        actual = float(cat_summary.get("餐饮", 0)) if "餐饮" in cat_summary else float(data.get("total_amount", 0))
        assert abs(actual - expect_cat) < 0.01, (
            f"餐饮分类统计不符 期望={expect_cat} 实际={actual} data={json.dumps(data, ensure_ascii=False)}\n"
            f"{_collect_diagnostics(result)}"
        )
    finally:
        _clean_after(before_id)


@pytest.mark.asyncio
async def test_stat_count_accuracy():
    """
    函数功能与逻辑描述：
        统计准确性（账单笔数）：先取本月真实 COUNT 作为基线，再插入 2 笔种子，执行
        "这个月一共有几笔消费"，断言 stat 任务成功且 data.total_count 恰等于
        「DB 基线 + 2」（整数精确相等，无容差）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ts, te = _month_range()
    base_cnt = int(db.query_sql(
        "SELECT COUNT(*) AS c FROM bill WHERE consume_time BETWEEN ? AND ?", (ts, te)
    )[0]["c"])
    _seed_bill(1.0, "餐饮", TODAY.isoformat(), "统计种子-计数1")
    _seed_bill(2.0, "交通", TODAY.isoformat(), "统计种子-计数2")
    before_id = _max_id()
    try:
        result = await _run_graph("这个月一共有几笔消费")
        stat_res = (result.get("task_plan") or {}).get("stat", {}).get("result")
        assert stat_res, f"stat 任务无结果\n{_collect_diagnostics(result)}"
        assert stat_res.get("success"), f"stat 失败: {_collect_diagnostics(result)}"
        actual_cnt = int(stat_res["data"].get("total_count", 0))
        expect_cnt = base_cnt + 2
        assert actual_cnt == expect_cnt, (
            f"账单笔数不符 期望={expect_cnt} 实际={actual_cnt} "
            f"(seed 2 笔, 基线={base_cnt})\n{_collect_diagnostics(result)}"
        )
    finally:
        _clean_after(before_id)


@pytest.mark.asyncio
async def test_stat_detail_accuracy():
    """
    函数功能与逻辑描述：
        统计准确性（明细查询）：插入 1 笔备注为"统计种子-明细66.6"的 66.6 元餐饮记录，
        执行"这个月的消费明细"，断言 stat 成功、data.detail_list 中能按 remark 定位到
        该条记录，且其 amount 为 66.6（容差 0.001）、category 为"餐饮"。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ts, te = _month_range()
    _seed_bill(66.6, "餐饮", TODAY.isoformat(), "统计种子-明细66.6")
    before_id = _max_id()
    try:
        result = await _run_graph("这个月的消费明细")
        stat_res = (result.get("task_plan") or {}).get("stat", {}).get("result")
        assert stat_res, f"stat 任务无结果\n{_collect_diagnostics(result)}"
        assert stat_res.get("success"), f"stat 失败: {_collect_diagnostics(result)}"
        detail = stat_res["data"].get("detail_list") or []
        found = [d for d in detail if "统计种子-明细66.6" in (d.get("remark") or "")]
        assert found, (
            f"明细中未找到 seed 记录 detail={json.dumps(detail, ensure_ascii=False)}\n"
            f"{_collect_diagnostics(result)}"
        )
        assert abs(float(found[0]["amount"]) - 66.6) < 0.001, f"明细金额不符: {found[0]}"
        assert found[0]["category"] == "餐饮", f"明细类别不符: {found[0]}"
    finally:
        _clean_after(before_id)


@pytest.mark.asyncio
async def test_stat_avg_max_min_accuracy():
    """
    函数功能与逻辑描述：
        统计准确性（平均/最大/最小）：先取本月真实 COUNT/SUM/MAX/MIN 作为基线，再插入
        10/20/30 三笔购物种子，执行"这个月消费的平均值、最大单笔和最小单笔分别是多少"，
        断言 stat 成功且 data.avg_amount、max_record.amount、min_record.amount 分别等于
        (base_sum + 60) / (base_cnt + 3)、max(base_max, 30.0)、min(base_min, 10.0)（容差 0.01）。
        期望值全部由 DB 真实聚合推导，兼容库中已有历史记录，不依赖空库环境。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ts, te = _month_range()
    base_agg = db.query_sql(
        "SELECT COUNT(*) AS c, COALESCE(SUM(amount),0) AS s, "
        "MAX(amount) AS mx, MIN(amount) AS mn FROM bill "
        "WHERE consume_time BETWEEN ? AND ?",
        (ts, te),
    )[0]
    base_cnt = int(base_agg["c"])
    base_sum = float(base_agg["s"] or 0)
    base_max = float(base_agg["mx"]) if base_agg["mx"] is not None else None
    base_min = float(base_agg["mn"]) if base_agg["mn"] is not None else None
    seed_sum = 10.0 + 20.0 + 30.0
    for amt in [10.0, 20.0, 30.0]:
        _seed_bill(amt, "购物", TODAY.isoformat(), "统计种子-极值")
    before_id = _max_id()
    try:
        result = await _run_graph("这个月消费的平均值、最大单笔和最小单笔分别是多少")
        stat_res = (result.get("task_plan") or {}).get("stat", {}).get("result")
        assert stat_res, f"stat 任务无结果\n{_collect_diagnostics(result)}"
        assert stat_res.get("success"), f"stat 失败: {_collect_diagnostics(result)}"
        data = stat_res["data"]
        # 期望值 = 库里全月真实聚合（seed后）
        expect_avg = (base_sum + seed_sum) / (base_cnt + 3)
        expect_max = max(base_max or 0, 30.0)
        expect_min = min(base_min if base_min is not None else 1e9, 10.0)
        avg = float(data.get("avg_amount", 0))
        mx = float((data.get("max_record") or {}).get("amount", 0))
        mn = float((data.get("min_record") or {}).get("amount", 0))
        assert abs(avg - expect_avg) < 0.01, (
            f"平均不符 期望={expect_avg:.4f} 实际={avg} "
            f"(基线 cnt={base_cnt} sum={base_sum})\n{_collect_diagnostics(result)}"
        )
        assert abs(mx - expect_max) < 0.01, (
            f"最大不符 期望={expect_max} 实际={mx}\n{_collect_diagnostics(result)}"
        )
        assert abs(mn - expect_min) < 0.01, (
            f"最小不符 期望={expect_min} 实际={mn}\n{_collect_diagnostics(result)}"
        )
    finally:
        _clean_after(before_id)


# ============================================================
# 三、多轮连续性：追问 → 补充 → 成功（同一会话连续两轮）
# ============================================================
def _bill_plan(text: str) -> dict:
    """
    函数功能与逻辑描述：
        构造确定性注入的主 Agent（orchestrator）规划结果——仅含单个 bill 任务，
        raw_segments 为传入文本、operate_sub_type 为 add，并置 pre_check_pass=True、
        block_tip=""，用于多轮连续性用例中对规划 LLM 打桩，剔除模型不确定性。
    入参说明：
        text (str)：账单原文，作为该 bill 任务的 raw_segments 唯一元素。
    返回值说明：
        dict：形如 {"tasks": [...], "pre_check_pass": True, "block_tip": ""} 的规划结果。
    """
    return {
        "tasks": [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}],
        "pre_check_pass": True,
        "block_tip": "",
    }


@pytest.mark.asyncio
async def test_e2e_followup_then_success():
    """
    函数功能与逻辑描述：
        多轮连续性 e2e：同一 session_id 连续执行「追问 → 补充 → 成功记账」两轮。
        第 1 轮 bill 解析被桩为 need_more_info=True（缺日期）→ 断言零落库且 final_reply 非空；
        第 2 轮同会话补充为完整信息（35 元 / 交通 / 今天）→ 断言恰好落库 1 条、金额 35.0、
        品类"交通"，验证追问轮不会打乱同会话后续记账（会话内存 / A2A 派发 / DAG 跨轮连续）。
        打桩范围：仅 patch agents.orchestrator.nodes.parse_json_output（注入 _bill_plan 单 bill 任务）
        与 agents.bill_agent.nodes.parse_json_output（注入解析结果）；其余环节（常驻 worker、
        A2A 派发、会话内存、真实库写入）全走生产链路。结束时 _clean_after 清理新增账单。
        运行时前置依赖：本地 bill 库与 conftest 拉起的常驻子 Agent worker（真实外部 LLM 被桩绕过）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = f"acc_followup_{uuid.uuid4().hex[:8]}"
    before_id = _max_id()
    today = TODAY.isoformat()

    # —— 第 1 轮：bill 解析判定缺信息（缺日期）→ 追问，不得落库 ——
    need_more = {"need_more_info": True, "valid": False,
                 "prompt": "请补充消费的具体日期，例如：2026-09-13"}
    with patch("agents.orchestrator.nodes.parse_json_output",
               new=AsyncMock(return_value=_bill_plan("记打车35元"))), \
         patch("agents.bill_agent.nodes.parse_json_output",
               new=AsyncMock(return_value=need_more)):
        r1 = await _run_graph("记打车35元", session_id=sid)
        assert not _new_records(before_id), f"追问轮不得落库\n{_collect_diagnostics(r1)}"
        assert r1.get("final_reply"), "追问轮应有回复"

    # —— 第 2 轮：同会话补充完整 → 成功落库 35 元交通 ——
    ok = {"need_more_info": False, "valid": True, "amount": 35.0,
          "category": "交通", "consume_date": today}
    with patch("agents.orchestrator.nodes.parse_json_output",
               new=AsyncMock(return_value=_bill_plan("记打车35元，就是今天"))), \
         patch("agents.bill_agent.nodes.parse_json_output",
               new=AsyncMock(return_value=ok)):
        r2 = await _run_graph("记打车35元，就是今天", session_id=sid)
        rows = _new_records(before_id)
        assert len(rows) == 1, f"补充轮应落库 1 条\n{_collect_diagnostics(r2)}"
        assert abs(float(rows[0]["amount"]) - 35.0) < 0.001, rows[0]
        assert rows[0]["category"] == "交通", rows[0]

    _clean_after(before_id)


if __name__ == "__main__":
    print("请通过 pytest 运行本测试文件")
