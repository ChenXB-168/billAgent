# -*- coding: utf-8 -*-
"""
端到端【数据准确性】测试：从用户端输入走完整 orchestrator 全链路，落库硬校验。

定位策略：MAX(id) 快照法 —— 不依赖 remark 标记（LLM 会剥离方括号标记），
每次用例运行前记录 max_id，运行后查询 id>max_id 的新增记录逐字段校验。
"""
import json
import uuid
from datetime import date, timedelta

import pytest

from agents.orchestrator.graph import orchestrator_graph
from utils.common import db

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


async def _run_graph(user_input: str):
    """执行编排器全流程，返回 final state"""
    init_state = {
        "user_input": user_input,
        "session_id": f"acc_{uuid.uuid4().hex[:8]}",
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


def _new_records(after_id: int):
    return db.query_sql(
        "SELECT id, amount, category, consume_time, remark FROM bill WHERE id > ? ORDER BY id ASC",
        (after_id,),
    )


def _clean_after(after_id: int):
    try:
        db.execute_sql("DELETE FROM bill WHERE id > ?", (after_id,))
    except Exception:
        pass


def _collect_diagnostics(result: dict) -> str:
    """从 final state 提取出错结点信息，用于失败时定位问题环节"""
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
BILL_ACC_CASES = [
    ("ba_01", "打车35元帮我记一下", 35.0, "交通", TODAY.isoformat()),
    ("ba_02", "晚饭38元", 38.0, "餐饮", TODAY.isoformat()),
    ("ba_03", "买了一件衣服199元记个账", 199.0, "购物", TODAY.isoformat()),
    ("ba_04", "住酒店一晚388元", 388.0, "住宿", TODAY.isoformat()),
    ("ba_05", "看电影花了45元", 45.0, "娱乐", TODAY.isoformat()),
    ("ba_06", "中午外卖25.5元", 25.5, "餐饮", TODAY.isoformat()),
    ("ba_07", "地铁充值50元", 50.0, "交通", TODAY.isoformat()),
    ("ba_08", "买书花了59.9元", 59.9, "购物", TODAY.isoformat()),
    ("ba_09", "今天打车到机场85元", 85.0, "交通", TODAY.isoformat()),
    ("ba_10", "奶茶18块", 18.0, "餐饮", TODAY.isoformat()),
    ("ba_11", f"昨天打车花了35元", 35.0, "交通", YESTERDAY.isoformat()),
    ("ba_12", f"昨天晚饭吃了62元", 62.0, "餐饮", YESTERDAY.isoformat()),
    ("ba_13", "KTV唱歌180元记一下", 180.0, "娱乐", TODAY.isoformat()),
    ("ba_14", "打车到公司28.5元", 28.5, "交通", TODAY.isoformat()),
    ("ba_15", "超市买日用品128元", 128.0, "购物", TODAY.isoformat()),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id,user_input,exp_amount,exp_category,exp_date",
    BILL_ACC_CASES,
    ids=[c[0] for c in BILL_ACC_CASES],
)
async def test_bill_record_accuracy(case_id, user_input, exp_amount, exp_category, exp_date):
    """记账准确性：LLM抽取 -> 落库 -> 查库硬校验 金额/类别/日期"""
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


# (id, user_input) —— 预期"不落库"（收入不入账/非法金额/追问）
BILL_NO_RECORD_CASES = [
    ("bn_01", "这个月工资发了8000元"),
    ("bn_02", "打车-30元记一下"),
    ("bn_03", "收到了退款100元"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,user_input", BILL_NO_RECORD_CASES, ids=[c[0] for c in BILL_NO_RECORD_CASES])
async def test_bill_no_record(case_id, user_input):
    """不落库校验：收入/退款/负数金额 不得写入 bill 表"""
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
    first = TODAY.replace(day=1).isoformat()
    return first, TODAY.isoformat()


def _seed_bill(amount: float, category: str, consume_time: str, remark: str):
    db.execute_sql(
        "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
        (amount, category, consume_time, remark),
    )


@pytest.mark.asyncio
async def test_stat_total_amount_accuracy():
    """统计准确性：本月总支出 必须等于 bill 表真实 SUM"""
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
    """统计准确性：分类统计 必须等于 bill 表真实分类聚合"""
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
    """统计准确性：账单笔数 必须等于 bill 表真实 COUNT"""
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
    """统计准确性：明细查询 必须与 bill 表真实记录一致"""
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
    """统计准确性：平均/最大/最小 与 bill 表真实聚合一致（期望值从DB真实聚合推导，
    兼容库中已存在的历史记录，避免依赖空库环境）"""
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


if __name__ == "__main__":
    print("请通过 pytest 运行本测试文件")
