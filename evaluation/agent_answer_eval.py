# -*- coding: utf-8 -*-
"""Agent 回答质量评测（L3 层）—— 按《评测体系构建.md》§5 规范实现

职责：
  1. 基于真实编排链路（orchestrator_graph → A2A → 子Agent）运行评测用例
  2. 五维 rubric 规则打分：意图正确性 / 参数抽取 / 业务语义 / 格式完整 / 无幻觉（每维 0-2，满分 10）
  3. 可选 LLM-as-a-Judge 交叉验证（--with-judge）
  4. 产出 eval_reports/eval_report_YYYYMMDD.json + 控制台汇总 + 定位清单

设计原则（与文档一致）：
  - 确定性优先：能用正则/关键词判的分绝不用 LLM 判
  - 正负双向断言：must_contain 锁"该有的有"，must_not_contain 锁"不该有的没有"
  - 每个用例独立 session_id，避免多轮历史污染
"""
import sys
import os
import re
import ast
import json
import asyncio
import subprocess
import time
import uuid
from datetime import date
from pathlib import Path

# Windows 控制台中文编码根治
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mcpGateway.client import mcp_client
from mcpGateway.a2a_queue import a2a_bus
from startup.bootstrap import (
    _bill_agent_worker,
    _stat_agent_worker,
    _price_agent_worker,
    _finance_agent_worker,
)
from agents.orchestrator.graph import orchestrator_graph

# =====================================================================
# 一、评测集（Golden Set）
# =====================================================================
# category: 单任务 / 多任务 / 边界 / 对抗
# expected: intent 期望意图；params 期望参数；must_contain 正向关键词；
#           must_not_contain 负向红线；no_hallucination 是否做金额幻觉严格检查
EVAL_CASES = [
    # ---------------- 单任务 ----------------
    {
        "case_id": "single_bill_001",
        "category": "单任务",
        "user_input": "晚饭38元帮我记一下 [测试标记]",
        "expected": {
            "intent": "bill",
            "params": {"amount": 38.0, "category": "餐饮"},
            "must_contain": ["记账成功"],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": True,
            "db_verify": {"type": "record", "amount": 38.0, "category": "餐饮", "marker": "测试标记"},
        },
    },
    {
        "case_id": "single_stat_002",
        "category": "单任务",
        "user_input": "看看这个月花了多少钱",
        "expected": {
            "intent": "stat",
            "params": {},
            "must_contain": ["月度消费统计", "总支出"],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": False,
            # stat 数字一致性为软校验：DB 共享、历史数据不可控，回复数字与 DB 本月 SUM 尽力比对
            "db_verify": {"type": "stat_sum"},
        },
    },
    {
        "case_id": "single_price_003",
        "category": "单任务",
        "user_input": "吃火锅花了80，贵不贵",
        "expected": {
            "intent": "price",
            "params": {"amount": 80.0},
            "must_contain": ["物价对比", "溢价率"],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": False,
        },
    },
    # 注：price 回复必含溢价率等派生数字，无幻觉强检查只保留给纯记账类用例
    {
        "case_id": "single_finance_004",
        "category": "单任务",
        "user_input": "给我点本月省钱建议",
        "expected": {
            "intent": "finance",
            "params": {},
            "must_contain": [],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": False,
        },
    },
    # ---------------- 多任务 ----------------
    {
        "case_id": "bill_plus_stat_005",
        "category": "多任务",
        "user_input": "买零食花了45元记一下 [测试标记]，查本月总开销",
        "expected": {
            "intent": ["bill", "stat"],
            "params": {"amount": 45.0, "category": "餐饮"},
            "must_contain": ["记账成功", "月度消费统计"],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": False,  # 统计文本含 DB 真实总支出，非幻觉
            "db_verify": {"type": "record", "amount": 45.0, "category": "餐饮", "marker": "测试标记"},
        },
    },
    {
        "case_id": "bill_plus_finance_006",
        "category": "多任务",
        "user_input": "打车25元存账单 [测试标记]，给我理财建议",
        "expected": {
            "intent": ["bill", "finance"],
            "params": {"amount": 25.0, "category": "交通"},
            "must_contain": ["记账成功"],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": False,  # 理财分析文本含大量估算数字
            "db_verify": {"type": "record", "amount": 25.0, "category": "交通", "marker": "测试标记"},
        },
    },
    {
        "case_id": "full_four_tasks_007",
        "category": "多任务",
        "user_input": "午饭32元记一下 [测试标记]，查7月开销，对比餐饮物价，给省钱建议",
        "expected": {
            "intent": ["bill", "stat", "price", "finance"],
            "params": {"amount": 32.0, "category": "餐饮"},
            "must_contain": ["记账成功", "月度消费统计", "物价对比"],
            "must_not_contain": ["失败", "FINANCE_ERROR"],
            "no_hallucination": False,  # 统计/理财/物价派生数字多
            "db_verify": {"type": "record", "amount": 32.0, "category": "餐饮", "marker": "测试标记"},
        },
    },
    # ---------------- 边界 ----------------
    {
        "case_id": "boundary_no_record_cmd_008",
        "category": "边界",
        "user_input": "今天喝奶茶花了18，是不是花多了",
        "expected": {
            "intent": ["stat", "price"],
            "params": {"amount": 18.0},
            "must_contain": [],
            "must_not_contain": ["记账成功", "失败"],
            "no_hallucination": False,
            # 该用例无记账指令，预期不写库：快照 MAX(id) 对比验证无新增记录
            "db_verify": {"type": "no_record"},
        },
    },
    {
        "case_id": "boundary_income_009",
        "category": "对抗",
        "user_input": "发奖金5000元，帮我记上",
        "expected": {
            "intent": "income_block",
            "params": {},
            # 实际文案：收入类内容不予入账，本系统仅支持支出记账
            "must_contain": ["不予入账"],
            "must_not_contain": ["记账成功"],
            "no_hallucination": False,
            "db_verify": {"type": "no_record"},
        },
    },
    {
        "case_id": "boundary_invalid_010",
        "category": "边界",
        "user_input": "哈哈哈随便聊聊",
        "expected": {
            "intent": "no_business",
            "params": {},
            "must_contain": [],
            "must_not_contain": [],
            "no_hallucination": False,
            "db_verify": {"type": "no_record"},
        },
    },
    {
        "case_id": "boundary_missing_amount_011",
        "category": "边界",
        "user_input": "我昨天去超市买了东西",
        "expected": {
            "intent": "need_more_info",
            "params": {},
            "must_contain": [],
            "must_not_contain": ["记账成功", "FINANCE_ERROR"],
            "no_hallucination": False,
            "db_verify": {"type": "no_record"},
        },
    },
    # ---------------- 对抗（金额幻觉） ----------------
    {
        "case_id": "anti_hallucination_012",
        "category": "对抗",
        "user_input": "打车花了30元 [测试标记]",
        "expected": {
            "intent": "bill",
            "params": {"amount": 30.0, "category": "交通"},
            "must_contain": ["记账成功"],
            "must_not_contain": ["失败"],
            "no_hallucination": True,
            # 金额幻觉双验证：回复只能含 30，且落库金额也必须恰好 30
            "db_verify": {"type": "record", "amount": 30.0, "category": "交通", "marker": "测试标记"},
        },
    },
]

_AMOUNT_RE = re.compile(r"-?\d+(?:\.\d+)?")
# 日期整体剔除（2026-07-14 / 2026/07/14），避免月日数字被误判为金额
_DATE_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")


# =====================================================================
# 二、五维 rubric 规则打分
# =====================================================================
def _extract_amounts(text: str) -> list:
    """抽取文本中的金额数字（带符号）；日期整体剔除，行首序号（如 "1. " "2）"）剔除，避免月/日/序号被误判为金额"""
    text = _DATE_RE.sub("", text)
    # 剔除行首序号（"1. " "2）" "3、" 等），防止确定性渲染的列表序号被误判为幻觉金额
    text = re.sub(r"(?m)^\s*\d+[.)、]\s*", "", text)
    return [float(m.group(0)) for m in _AMOUNT_RE.finditer(text) if m.group(0) != "0"]


def _redline_violated(reply: str, case: dict) -> bool:
    """红线是否被突破：must_not_contain 任一命中"""
    exp = case["expected"]
    return any(k in reply for k in exp.get("must_not_contain", []))


def score_intent(reply: str, case: dict) -> int:
    """意图正确性：负向红线优先，红线突破记 0；其次正向关键词"""
    exp = case["expected"]
    must_contain = exp.get("must_contain", [])
    if _redline_violated(reply, case):
        return 0
    if not must_contain:
        return 2 if reply.strip() else 0
    hit = sum(1 for k in must_contain if k in reply)
    if hit == len(must_contain):
        return 2
    if hit > 0:
        return 1
    return 0


def score_params(reply: str, case: dict) -> int:
    """参数抽取：期望金额 + 期望类别是否出现在回复中"""
    exp = case["expected"]
    params = exp.get("params", {})
    amount = params.get("amount")
    category = params.get("category")

    if amount is None and category is None:
        # 无参数期望：回复非空且无明显报错即满分
        return 2 if reply.strip() and "FINANCE_ERROR" not in reply else 0

    found_amount = any(abs(a - amount) < 0.01 for a in _extract_amounts(reply)) if amount is not None else True
    if not found_amount:
        # 金额缺失/错误
        return 1 if ("记账成功" in reply or "元" in reply) else 0
    # 金额命中，检查类别
    if category is not None and category not in reply:
        return 1  # 金额对但类别错
    return 2


def score_semantics(reply: str, case: dict) -> int:
    """业务语义：金额真实（来自原文），不凭空捏造"""
    exp = case["expected"]
    amount = exp.get("params", {}).get("amount")
    if amount is None:
        return 2 if not _redline_violated(reply, case) else 0
    found = [a for a in _extract_amounts(reply) if abs(a - amount) < 0.01]
    return 2 if found else 0


def score_format(reply: str, case: dict) -> int:
    """格式完整：非空、无异常前缀、长度达标"""
    if not reply or not reply.strip():
        return 0
    if "FINANCE_ERROR" in reply or "Traceback" in reply:
        return 0
    if len(reply) < 4:
        return 1
    return 2


def score_no_hallucination(reply: str, case: dict) -> int:
    """无幻觉：回复中的金额 ⊆ 原文数字集合 ∪ 期望金额（仅对强检查用例）

    放行确定性预警数据：记账提醒（单笔溢价/品类进度/预算档位）中的均价、溢价率、
    额度、预算等数字来自系统确定性查询（price_agent / _calc_alert_facts），非 LLM 幻觉。
    识别包含"提醒/均价/溢价/额度/预算/剩余/还剩/累计/建议/控制在"等固定句式的行并跳过其数字。
    """
    exp = case["expected"]
    if not exp.get("no_hallucination"):
        # 弱检查：统计/理财文本必然含大量派生数字（DB真实值/估算），不判幻觉
        return 2 if "FINANCE_ERROR" not in reply else 0
    src_nums = set(_extract_amounts(case["user_input"]))
    exp_amount = exp.get("params", {}).get("amount")
    if exp_amount is not None:
        src_nums.add(exp_amount)
    reply_amounts = set(_extract_amounts(reply))
    _SYS_ALERT_KW = ("提醒", "建议", "控制在", "额度", "预算", "均价", "溢价", "剩余", "还剩", "累计", "合理")
    allowed = set()
    for line in reply.splitlines():
        if any(k in line for k in _SYS_ALERT_KW):
            allowed |= set(_extract_amounts(line))
    hallucinated = {a for a in reply_amounts if a not in src_nums and a not in allowed}
    if not hallucinated:
        return 2
    return 1 if len(hallucinated) <= 1 else 0


def evaluate_case(reply: str, case: dict) -> dict:
    """对单个用例执行五维评分，返回明细。
    pass 判定：总分 ≥8 且红线未突破（must_not_contain 一票否决）
    """
    dims = {
        "intent": score_intent(reply, case),
        "params": score_params(reply, case),
        "semantics": score_semantics(reply, case),
        "format": score_format(reply, case),
        "no_hallucination": score_no_hallucination(reply, case),
    }
    total = sum(dims.values())
    redline = _redline_violated(reply, case)
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "user_input": case["user_input"],
        "reply": reply,
        "dims": dims,
        "total": total,
        "redline_violated": redline,
        "pass": total >= 8 and not redline,  # 红线一票否决
    }


# =====================================================================
# 2.5 数据正确性硬校验（db_accuracy）
# 回复文本对 ≠ 数据库对：直查 bill.db 验证"数据真相"
# =====================================================================
async def _db_query(sql: str, params: list = None) -> list:
    """直查 bill.db（走 MCP sql_bill，bill_agent 有读写权限），返回字典列表"""
    try:
        raw = await mcp_client.call_bill_sql("bill_agent", sql, params or [])
    except Exception:
        return []
    try:
        data = ast.literal_eval(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


async def _db_max_id() -> int:
    rows = await _db_query("SELECT MAX(id) AS max_id FROM bill")
    if rows and rows[0].get("max_id") is not None:
        return int(rows[0]["max_id"])
    return 0


async def verify_db(case: dict, snapshot_max: int = None, reply: str = "") -> dict:
    """数据正确性硬校验（0/1/2 分）。
    - record   : 按标记查库，金额/类别与期望一致（硬校验，不过则 FAIL 阻断）
    - no_record: 快照 MAX(id) 对比，预期不写库却写入（硬校验，不过则 FAIL 阻断）
    - stat_sum : 回复中的总支出数字与 DB 本月 SUM 一致性（软校验，仅记录不阻断）
    """
    dbv = case.get("expected", {}).get("db_verify")
    if not dbv:
        return {"checked": False, "soft": False, "score": 2, "detail": "无 db_verify 期望，跳过"}
    vtype = dbv["type"]
    try:
        if vtype == "record":
            # 快照模式：查用例执行期间新增的记录（id > snapshot_max）并匹配金额/类别。
            # 不依赖 remark 含"[测试标记]"——orchestrator 规划时可能剥掉用户输入中的标记。
            if snapshot_max is None:
                snapshot_max = await _db_max_id()
            rows = await _db_query(
                "SELECT id, amount, category FROM bill WHERE id > ? ORDER BY id ASC",
                [snapshot_max],
            )
            if not rows:
                return {"checked": True, "soft": False, "score": 0, "detail": "用例执行期间无新增账单记录（写入缺失）"}
            best = None
            for r in rows:
                if abs(float(r["amount"]) - dbv["amount"]) < 0.01 and r["category"] == dbv["category"]:
                    best = r
                    break
            if best is not None:
                return {"checked": True, "soft": False, "score": 2, "detail": f"记录准确，DB写入: amount={best['amount']}, category={best['category']}（id={best['id']}）"}
            row = rows[-1]
            ok_amount = abs(float(row["amount"]) - dbv["amount"]) < 0.01
            ok_cat = row["category"] == dbv["category"]
            detail = f"DB实际: amount={row['amount']}, category={row['category']}；期望: {dbv['amount']}/{dbv['category']}"
            if ok_amount or ok_cat:
                return {"checked": True, "soft": False, "score": 1, "detail": f"部分偏差，{detail}"}
            return {"checked": True, "soft": False, "score": 0, "detail": f"金额与类别均不符，{detail}"}
        if vtype == "no_record":
            after = await _db_max_id()
            if snapshot_max is not None and after > snapshot_max:
                return {"checked": True, "soft": False, "score": 0, "detail": f"预期不写库，却新增记录 id={after}（快照 {snapshot_max}）"}
            return {"checked": True, "soft": False, "score": 2, "detail": "预期不写库，DB 无新增记录"}
        if vtype == "stat_sum":
            rows = await _db_query(
                "SELECT SUM(amount) AS total FROM bill "
                "WHERE strftime('%Y-%m', consume_time) = strftime('%Y-%m', 'now')"
            )
            db_total = round(float(rows[0]["total"] or 0), 2) if rows else 0.0
            exp_amount = case.get("expected", {}).get("params", {}).get("amount")
            candidates = [a for a in _extract_amounts(reply) if exp_amount is None or abs(a - exp_amount) >= 0.01]
            if not candidates:
                return {"checked": True, "soft": True, "score": 1, "detail": f"回复无总支出数字可比（DB本月SUM={db_total}）"}
            best = min(candidates, key=lambda a: abs(a - db_total))
            if abs(best - db_total) < 0.5:
                return {"checked": True, "soft": True, "score": 2, "detail": f"统计一致：回复={best} ≈ DB本月SUM={db_total}"}
            return {"checked": True, "soft": True, "score": 1, "detail": f"统计偏差：回复={best}，DB本月SUM={db_total}"}
    except Exception as e:
        return {"checked": True, "soft": False, "score": 0, "detail": f"DB校验异常: {e}"}
    return {"checked": False, "soft": False, "score": 2, "detail": "未知校验类型，跳过"}


def _db_ok(db_res: dict) -> bool:
    """pass 判定的 DB 硬门槛：无期望/软校验不阻断；硬校验必须满分"""
    if not db_res.get("checked"):
        return True
    if db_res.get("soft"):
        return True
    return db_res.get("score") == 2


# =====================================================================
# 三、LLM-as-a-Judge（可选交叉验证，默认关闭）
# =====================================================================
JUDGE_SYSTEM = """你是记账理财多Agent系统的质量评审员。请对助手回答按五个维度各打0-2分：
intent(意图正确性), params(参数抽取), semantics(业务语义), format(格式完整), no_hallucination(无幻觉)。
只输出JSON：{"intent":n,"params":n,"semantics":n,"format":n,"no_hallucination":n}"""


def judge_with_llm(reply: str, case: dict) -> dict:
    """调用大模型按 rubric 打分（受模型能力限制，仅供交叉验证参考）"""
    from modelService.llm_loader import ollama_base_call

    user_content = json.dumps(
        {
            "用户输入": case["user_input"],
            "期望": case["expected"],
            "助手回答": reply,
        },
        ensure_ascii=False,
    )
    try:
        raw = ollama_base_call(JUDGE_SYSTEM, user_content, timeout=120)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {"judge_raw": raw, "judge_error": "judge 输出非JSON"}
        return json.loads(m.group(0))
    except Exception as e:
        return {"judge_raw": "", "judge_error": str(e)}


# =====================================================================
# 四、运行链路：环境自举 + 全量用例
# =====================================================================
def _init_state(user_input: str) -> dict:
    return {
        "user_input": user_input,
        "session_id": f"eval_{uuid.uuid4().hex[:10]}",
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent",
    }


async def _run_case(case: dict, with_judge: bool) -> dict:
    """运行单个评测用例：回复评分 + 数据正确性硬校验（查真实 bill.db）"""
    dbv = case.get("expected", {}).get("db_verify")
    snapshot_max = await _db_max_id() if dbv else None
    run_id = None
    try:
        # M7（D3）：评测入口同生产入口接入 run 上下文（每用例独立 run_id）——
        #   失败用例凭 run_id 用 `python utils/trace_view.py <run_id>` 导出 span 树定位环节
        from utils.tracer import run_scope, span
        state = _init_state(case["user_input"])

        async def _trace_run():
            nonlocal run_id
            with run_scope(session_id=state["session_id"]) as rid:
                run_id = rid
                with span("orchestrator.run", kind="orchestrator",
                          agent_id="orchestrator_agent"):
                    return await orchestrator_graph.ainvoke(state)

        result = await _trace_run()
        reply = result.get("final_reply") or ""
    except Exception as e:
        reply = f"RUN_ERROR: {e}"

    eval_res = evaluate_case(reply, case)
    eval_res["run_id"] = run_id  # M7：失败定位用（见 run_all FAIL 分支提示）
    db_res = await verify_db(case, snapshot_max, reply)
    eval_res["db_verify"] = db_res
    eval_res["db_accuracy"] = db_res["score"]  # 独立硬校验，不并入 10 分制
    eval_res["pass"] = eval_res["pass"] and _db_ok(db_res)  # 回复对但库没写对 → FAIL
    if with_judge:
        eval_res["judge"] = judge_with_llm(reply, case)
    return eval_res


async def run_all(with_judge: bool = False, only: str = None) -> list:
    """启动 4 个 A2A worker 并运行全部（或指定）用例
    only: 逗号分隔的 case_id 子串，匹配任一即运行
    """
    a2a_bus.clear_all()
    all_start_max = await _db_max_id()  # 评测前快照，用于 finally 清理本次评测产生的全部数据
    workers = [
        asyncio.create_task(_bill_agent_worker()),
        asyncio.create_task(_stat_agent_worker()),
        asyncio.create_task(_price_agent_worker()),
        asyncio.create_task(_finance_agent_worker()),
    ]
    await asyncio.sleep(0.2)  # 等 worker 进入消费循环

    if only:
        keys = [k.strip() for k in only.split(",") if k.strip()]
        cases = [c for c in EVAL_CASES if any(k in c["case_id"] for k in keys)]
    else:
        cases = list(EVAL_CASES)
    results = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"[EVAL] {i}/{len(cases)} {case['case_id']} 开始: {case['user_input'][:24]}...", flush=True)
            t0 = time.time()
            res = await _run_case(case, with_judge)
            res["elapsed"] = round(time.time() - t0, 1)
            results.append(res)
            db_mark = res.get("db_accuracy", "-")
            db_fail = " DB-硬校验未过" if not _db_ok(res.get("db_verify", {})) else ""
            print(
                f"[EVAL] {case['case_id']} 得分={res['total']}/10 db={db_mark}/2 "
                f"{'PASS' if res['pass'] else 'FAIL'}{db_fail} 耗时={res['elapsed']}s",
                flush=True,
            )
            print(f"  回复: {res['reply'][:160]!r}", flush=True)
            if res.get("db_verify", {}).get("detail"):
                print(f"  DB: {res['db_verify']['detail']}", flush=True)
            # 每跑完一条立即增量落盘，中断不丢已跑结果
            save_report(results)
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        # 清理评测期间写入的全部数据（含 orchestrator 剥掉"[测试标记]"的无标记记录），避免污染统计口径与长时记忆
        try:
            await mcp_client.call_bill_sql("bill_agent", "DELETE FROM bill WHERE remark LIKE ?", ["%测试标记%"])
            if all_start_max is not None:
                await mcp_client.call_bill_sql("bill_agent", "DELETE FROM bill WHERE id > ?", [all_start_max])
            print("[EVAL] 已清理评测写入的测试数据", flush=True)
        except Exception as e:
            print(f"[EVAL] 测试数据清理失败: {e}", flush=True)
    return results


# =====================================================================
# 五、报告输出
# =====================================================================
def _summary(results: list) -> dict:
    """汇总：通过率 + 各维平均分 + 最弱维度"""
    n = len(results)
    passed = sum(1 for r in results if r["pass"])
    dims = ["intent", "params", "semantics", "format", "no_hallucination"]
    avg = {d: round(sum(r["dims"][d] for r in results) / n, 2) for d in dims}
    weakest = sorted(avg.items(), key=lambda kv: kv[1])[0]
    db_checked = [r for r in results if r.get("db_verify", {}).get("checked")]
    db_failed = [r for r in db_checked if not _db_ok(r.get("db_verify", {}))]
    db_avg = round(sum(r.get("db_accuracy", 2) for r in results) / n, 2)
    return {
        "date": date.today().isoformat(),
        "total_cases": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": round(passed / n * 100, 1),
        "avg_score": round(sum(r["total"] for r in results) / n, 2),
        "avg_by_dim": avg,
        "weakest_dim": {"dim": weakest[0], "avg": weakest[1]},
        "db_checked": len(db_checked),
        "db_failed": len(db_failed),
        "db_avg": db_avg,
        "fail_cases": [r["case_id"] for r in results if not r["pass"]],
    }


def _print_report(results: list):
    """控制台汇总 + 定位清单"""
    s = _summary(results)
    print("\n" + "=" * 60)
    print("Agent 回答质量评测报告（L3）")
    print("=" * 60)
    print(f"日期: {s['date']}  用例数: {s['total_cases']}")
    print(f"通过: {s['passed']}  失败: {s['failed']}  通过率: {s['pass_rate']}%")
    print(f"平均总分: {s['avg_score']}/10   最弱维度: {s['weakest_dim']['dim']} ({s['weakest_dim']['avg']}/2)")
    print(f"数据正确性: 校验 {s['db_checked']}/{s['total_cases']} 条, 硬校验未过 {s['db_failed']} 条, db_accuracy 均分 {s['db_avg']}/2")
    print("-" * 60)
    print("各维平均分: " + "  ".join(f"{k}={v}" for k, v in s["avg_by_dim"].items()))
    print("-" * 60)
    for r in results:
        mark = "PASS" if r["pass"] else "FAIL"
        dims = " ".join(f"{k}:{v}" for k, v in r["dims"].items())
        db_d = r.get("db_accuracy", "-")
        db_flag = "!" if not _db_ok(r.get("db_verify", {})) else ""
        print(f"  [{mark}] {r['case_id']:<24} total={r['total']}/10 db={db_d}/2{db_flag}  {dims}")
    print("-" * 60)
    if s["fail_cases"]:
        print("定位清单（重点关注，按失败维度排查）：")
        for r in results:
            if not r["pass"]:
                low = [k for k, v in r["dims"].items() if v <= 1]
                if not _db_ok(r.get("db_verify", {})):
                    low.append(f"db_accuracy={r.get('db_accuracy', '-')}")
                print(f"  - {r['case_id']}: 低分维度 {low}")
                print(f"     输入: {r['user_input'][:40]}")
                print(f"     回复: {r['reply'][:80]!r}")
                if r.get("db_verify", {}).get("detail"):
                    print(f"     DB: {r['db_verify']['detail']}")
    else:
        print("全部用例通过，无定位项。")
    print("=" * 60)


def save_report(results: list, merge: bool = True) -> str:
    """存档 JSON 报告（L4 趋势对比的数据源）。
    merge=True 时合并当日已有报告：同 case_id 覆盖，其余保留，实现增量续跑。
    """
    report_dir = ROOT / "eval_reports"
    report_dir.mkdir(exist_ok=True)
    path = report_dir / f"eval_report_{date.today().strftime('%Y%m%d')}.json"

    all_cases = []
    if merge and path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            all_cases = [c for c in old.get("cases", []) if c.get("case_id") not in {r["case_id"] for r in results}]
        except (json.JSONDecodeError, KeyError):
            all_cases = []
    all_cases += results

    payload = {"summary": _summary(all_cases), "cases": all_cases}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def _spawn_llm_mcp() -> subprocess.Popen:
    """启动 LLM MCP 子进程（与 tests/conftest.py 一致）"""
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "mcpGateway" / "server_llm_base.py")],
        stdout=None, stderr=None, text=True,
    )
    time.sleep(4)
    return proc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agent 回答质量评测（L3）")
    parser.add_argument("--with-judge", action="store_true", help="启用 LLM-as-a-Judge 交叉验证")
    parser.add_argument("--only", default=None, help="只跑指定 case_id 子串（如 single_bill_001）")
    parser.add_argument("--no-mcp", action="store_true", help="跳过自启 LLM MCP（外部已启动时使用）")
    args = parser.parse_args()

    proc = None
    if not args.no_mcp:
        print("[EVAL] 启动 LLM MCP server ...", flush=True)
        proc = _spawn_llm_mcp()

    try:
        results = asyncio.run(run_all(with_judge=args.with_judge, only=args.only))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    _print_report(results)
    path = save_report(results)
    print(f"\n报告已存档: {path}")


if __name__ == "__main__":
    main()
