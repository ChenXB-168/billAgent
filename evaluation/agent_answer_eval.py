# -*- coding: utf-8 -*-
"""Agent 回答质量评测（L3 层）—— 按《09_评测体系设计.md》§5 规范实现

职责：
  1. 基于真实编排链路（orchestrator_graph → A2A → 子Agent）运行评测用例（EVAL_CASES 共 12 条）
  2. 五维 rubric 规则打分：意图正确性 / 参数抽取 / 业务语义 / 格式完整 / 无幻觉（每维 0-2，满分 10）
  3. 数据正确性硬校验 db_accuracy（直查真实 bill.db，独立于 10 分制，硬校验不过则 FAIL）
  4. 可选 LLM-as-a-Judge 交叉验证（--with-judge）
  5. 产出 eval_reports/eval_report_YYYYMMDD.json + 控制台汇总 + 定位清单

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
    """
    函数功能与逻辑描述：
        金额抽取基础算子：用 `_AMOUNT_RE` 从文本中提取全部数字（负号保留），是"回复是否含幻觉金额"
        与"回复总支出是否与 DB 一致"两类判定的共用前置步骤；纯函数，不改动入参。
        剔除规则：先整体删除日期串（`_DATE_RE`，如 2026-07-14 / 2026/07/14），再删除行首列表序号
        （如 "1. " "2）" "3、" 等确定性渲染的序号），最后过滤掉数值 0，避免月/日/序号/占位 0 被误判为金额。
    入参说明：
        text (str)：待抽取文本，通常为 LLM 的 final_reply，也可为用例 user_input；无长度与编码约束。
    返回值说明：
        list：float 列表，按出现顺序排列，仅保留非零数值（负号保留）；无匹配时返回 []。
    """
    text = _DATE_RE.sub("", text)
    # 剔除行首序号（"1. " "2）" "3、" 等），防止确定性渲染的列表序号被误判为幻觉金额
    text = re.sub(r"(?m)^\s*\d+[.)、]\s*", "", text)
    return [float(m.group(0)) for m in _AMOUNT_RE.finditer(text) if m.group(0) != "0"]


def _redline_violated(reply: str, case: dict) -> bool:
    """
    函数功能与逻辑描述：
        红线判定（一票否决语义的判定算子）：只要 `must_not_contain` 中任一关键词出现在回复里，
        即认为红线被突破。被 `score_intent` 用作前置短路条件（突破直接 0 分），
        并在 `evaluate_case` 中单独落为 redline_violated 字段参与 pass 判定。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本。
        case (dict)：单条评测用例；读 case["expected"]["must_not_contain"]，该键缺失时按空列表处理（等价于无红线）。
    返回值说明：
        bool：True=红线被突破（命中至少一个禁止词）；False=未突破。
    """
    exp = case["expected"]
    return any(k in reply for k in exp.get("must_not_contain", []))


def score_intent(reply: str, case: dict) -> int:
    """
    函数功能与逻辑描述：
        维度一「意图正确性」规则打分（0-2）：先做负向红线短路——`must_not_contain` 命中直接 0 分；
        否则按 `must_contain` 正向关键词命中比例给分（全中 2 分、部分命中 1 分、全不中 0 分）。
        当用例未设置正向期望（must_contain 为空）时退化为"回复非空即满分"，
        用于"无需回答业务/闲聊"类用例（如 boundary_invalid_010）。
        评测口径：纯规则（关键词子串包含匹配），不调用 LLM，不依赖数据库。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本。
        case (dict)：单条评测用例；读 case["expected"] 下的 must_contain / must_not_contain。
    返回值说明：
        int：0=红线突破或正向词全部未命中（或无正向期望但回复为空白）；
             1=正向词部分命中；2=正向词全部命中（或无正向期望且回复非空）。
    """
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
    """
    函数功能与逻辑描述：
        维度二「参数抽取」规则打分（0-2）：核对期望参数（amount / category）是否出现在回复中。
        金额用 `_extract_amounts` 抽取后以 |a - amount| < 0.01 容差比对，类别用子串包含判断；
        两者均无期望时退化为"回复非空且不含 FINANCE_ERROR 即满分"。
        注：金额未命中时仅凭"记账成功"或"元"字样给 1 分，用于区分"记了但金额报错"与"完全没记"。
        评测口径：纯规则（正则抽数 + 字符串包含），不调用 LLM，不依赖数据库。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本。
        case (dict)：单条评测用例；读 case["expected"]["params"]，其中 amount / category 可分别缺失。
    返回值说明：
        int：0=金额未命中且无任何记账迹象（或无参数期望但回复为空/含 FINANCE_ERROR）；
             1=金额未命中但有"记账成功"或"元"字样，或金额命中而类别未命中；
             2=金额与类别均命中（或无参数期望且回复非空且无报错）。
    """
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
    """
    函数功能与逻辑描述：
        维度三「业务语义」规则打分（0-2）：核对期望金额是否真实出现在回复中，识别
        "记了账但关键金额没报出来"这类语义缺失（与 `score_params` 的区别：本维不看类别，只看金额是否落地）。
        无金额期望时退化为红线检查（未突破红线即满分），用于统计/理财等无固定金额口径的用例。
        评测口径：纯规则；本维不做"回复金额是否源自原文"的来源追溯，
        来源正确性（幻觉金额）由 `score_no_hallucination` 覆盖，避免两维重复。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本。
        case (dict)：单条评测用例；读 case["expected"]["params"]["amount"]（可缺失）。
    返回值说明：
        int：2=期望金额命中（容差 0.01）或无金额期望且未突破红线；0=期望金额未出现，或突破红线。
    """
    exp = case["expected"]
    amount = exp.get("params", {}).get("amount")
    if amount is None:
        return 2 if not _redline_violated(reply, case) else 0
    found = [a for a in _extract_amounts(reply) if abs(a - amount) < 0.01]
    return 2 if found else 0


def score_format(reply: str, case: dict) -> int:
    """
    函数功能与逻辑描述：
        维度四「格式完整」规则打分（0-2）：只做基础可用性检查——回复非空、不含 `FINANCE_ERROR`
        异常前缀、不含 `Traceback` 堆栈，且长度不小于 4 字符（过滤"好的""嗯"类无信息回复）。
        本维不解析业务字段，故不校验金额/分类等关键项是否齐全（由意图维与参数维覆盖）。
        评测口径：纯规则（字符串检查），不调用 LLM，不依赖数据库。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本；None 或纯空白串均判 0 分。
        case (dict)：为保持五个维度函数签名统一而保留，本维未使用其内容。
    返回值说明：
        int：0=空回复或含报错/堆栈；1=有内容但长度不足 4 字符；2=格式正常。
    """
    if not reply or not reply.strip():
        return 0
    if "FINANCE_ERROR" in reply or "Traceback" in reply:
        return 0
    if len(reply) < 4:
        return 1
    return 2


def score_no_hallucination(reply: str, case: dict) -> int:
    """
    函数功能与逻辑描述：
        维度五「无幻觉」规则打分（0-2）：仅对 `expected.no_hallucination=True` 的强检查用例生效，
        其余用例走弱检查（不含 FINANCE_ERROR 即满分），因为统计/理财回复必然包含大量 DB 真实值或估算数字。
        强检查口径：回复中的金额集合须 ⊆ 原文数字 ∪ 期望金额 ∪ 系统确定性预警行数字；
        无法归属的金额计为幻觉，≤1 个记 1 分、≥2 个记 0 分。
        放行白名单：命中 `_SYS_ALERT_KW`（提醒 / 建议 / 控制在 / 额度 / 预算 / 均价 / 溢价 / 剩余 / 还剩 / 累计 / 合理）
        的行整行跳过——其数字来自 price_agent 与 `_calc_alert_facts` 的确定性查询，非 LLM 编造。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本；按 splitlines() 逐行判断是否为预警行。
        case (dict)：单条评测用例；读 case["expected"] 的 no_hallucination 开关与 params.amount，并以 case["user_input"] 作为原文数字来源。
    返回值说明：
        int：2=无幻觉（或弱检查且无 FINANCE_ERROR）；1=弱检查含 FINANCE_ERROR，或强检查下仅 1 个无法归属的金额；0=强检查下 ≥2 个无法归属的金额。
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
    """
    函数功能与逻辑描述：
        对单条用例执行五维 rubric 评分并汇总：dims 为五项各 0-2 的字典，total 为五项之和（满分 10）。
        pass 判定口径：total >= 8 且未突破红线（redline_violated 一票否决）；
        此处仅为文本层判定，`_run_case` 中还会以 `and _db_ok(...)` 叠加 DB 硬校验门槛
        （即"回复满分但落库错误"仍判 FAIL）。
        评测口径：五维全部为确定性规则打分，不调用 LLM、不访问数据库。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本；链路异常时形如 "RUN_ERROR: ..."，会被格式维判 0。
        case (dict)：单条评测用例，须含 case_id / category / user_input / expected 四个键。
    返回值说明：
        dict：评分明细，字段为 case_id(str)、category(str)、user_input(str)、reply(str)、
              dims(五维分数字典)、total(int 0-10)、redline_violated(bool)、pass(bool)；
              不含 elapsed / db_verify / run_id 等由上游追加的字段。
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
    """
    函数功能与逻辑描述：
        账单库查询基础算子：经 `mcp_client.call_bill_sql` 以 bill_agent 身份执行 SQL
        （SELECT 开头走只读工具 sql.read、其余走写工具 sql.write，权限与审计由工具平台引擎统一校验），
        再把返回文本用 `ast.literal_eval` 解析为结构化行。
        容错口径：调用与解析统一 try/except 兜底，任一环节异常均返回空列表而非抛出，
        避免校验器自身故障把用例误判为通过。
    入参说明：
        sql (str)：待执行 SQL 文本；本文件仅传入 SELECT 语句。
        params (list | None)：SQL 占位符参数列表，默认 None，调用前归一为 []。
    返回值说明：
        list：字典列表（每行一个 dict，键为列名）；调用失败、返回非 list 或解析失败时返回 []。
    """
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
    """
    函数功能与逻辑描述：
        取当前 bill 表最大主键 id，作为"用例期间是否新增记录"的快照基准，
        供 `verify_db` 的 no_record（预期不写库）与 record（写库快照）两类校验使用。
    入参说明：
        无。
    返回值说明：
        int：bill 表 MAX(id)；表为空、查询失败或结果为 NULL 时返回 0。
    """
    rows = await _db_query("SELECT MAX(id) AS max_id FROM bill")
    if rows and rows[0].get("max_id") is not None:
        return int(rows[0]["max_id"])
    return 0


async def verify_db(case: dict, snapshot_max: int = None, reply: str = "") -> dict:
    """
    函数功能与逻辑描述：
        数据正确性硬校验（db_accuracy，0/1/2 分，独立于 10 分制）：直查真实 bill.db，
        在"回复文本对"之外验证"数据真相"。按 case["expected"]["db_verify"]["type"] 分三种口径：
        record（在 id > snapshot_max 的新增记录中匹配金额容差 0.01 且类别一致 → 2 分；
        仅金额或类别一项符合 → 1 分；两项均不符或期间无新增 → 0 分。不依赖 remark 标记，
        因 orchestrator 规划时可能剥掉用户输入中的"[测试标记]"）、
        no_record（对比执行前后 MAX(id)，预期不写库却新增 → 0 分，否则 2 分）、
        stat_sum（回复中的候选总支出与 DB 本月 SUM 比对，容差 0.5，软校验只记录不阻断）。
        前置依赖：需真实可写的 bill.db 与 MCP 连接；校验器自身异常兜底为 0 分并在 detail 中说明。
    入参说明：
        case (dict)：单条评测用例；读 case["expected"]["db_verify"]，该键缺失时直接跳过校验并记满分。
        snapshot_max (int | None)：用例执行前快照的 MAX(id)；record 类型为 None 时现场补取，no_record 类型为 None 时跳过新增判定。
        reply (str)：被测回复文本，仅 stat_sum 使用（作为抽取候选总支出数字的来源）。
    返回值说明：
        dict：{checked(bool 是否执行了校验), soft(bool 是否软校验不阻断), score(int 0-2), detail(str 可读结论)}；
              未配置 db_verify 时返回 checked=False、score=2；未知校验类型同样返回 checked=False、score=2。
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
    """
    函数功能与逻辑描述：
        pass 判定的 DB 硬门槛：把 `verify_db` 的结果归一为"是否放行"。未执行校验（checked=False）
        与软校验（soft=True）一律放行；硬校验必须满分（score == 2）才放行，否则该用例整体 FAIL。
    入参说明：
        db_res (dict)：`verify_db` 的返回包；允许缺键，内部用 .get 兜底。
    返回值说明：
        bool：True=放行（不阻断 pass）；False=硬校验未过（用例判 FAIL）。
    """
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
    """
    函数功能与逻辑描述：
        可选交叉验证项（仅 --with-judge 时被调用）：把「用户输入 + 期望标注 + 助手回答」连同
        `JUDGE_SYSTEM` 五维 rubric 提示词交给基座模型，要求仅输出五维 JSON 分数，
        用于与规则打分互校；因 Judge 本身是 LLM、会引入二次不确定性，故只作参考、不参与 pass 判定。
        前置依赖：需本地真实模型可用（modelService.llm_loader.ollama_base_call），单次超时 120s。
    入参说明：
        reply (str)：被测 Agent 的最终回复文本。
        case (dict)：单条评测用例，取 user_input 与 expected 拼装判题上下文（JSON、ensure_ascii=False）。
    返回值说明：
        dict：成功且输出可解析时为五维分数字典（键 intent / params / semantics / format / no_hallucination）；
              输出非 JSON 时返回 {"judge_raw": 原始输出, "judge_error": "judge 输出非JSON"}；
              调用抛异常时返回 {"judge_raw": "", "judge_error": 异常字符串}。
    """
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
    """
    函数功能与逻辑描述：
        构造单次评测用例运行所需的编排图初始 state。每个用例独立生成 `eval_<10位随机hex>`
        形式的 session_id，避免多轮对话历史与长时记忆在用例间串味。
    入参说明：
        user_input (str)：该用例的用户输入原文（取自 EVAL_CASES[i]["user_input"]）。
    返回值说明：
        dict：orchestrator 入参 state，含 user_input、session_id、空 session_history、task_plan={}、
              current_task=None、all_task_results=[]、final_reply=None、error_msg=None、
              current_agent="orchestrator_agent"（回复与错误由编排链路回填）。
    """
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
    """
    函数功能与逻辑描述：
        运行单条评测用例的完整链路：按 db_verify 需要先快照 MAX(id) → 在 tracer 的 run_scope 内
        以独立 run_id 调用真实编排链路（orchestrator_graph.ainvoke）→ 做五维文本评分 →
        做 DB 正确性硬校验，并以 `_db_ok` 二次收紧 pass（回复对但库没写对 → FAIL）。
        异常口径：链路抛错时把回复置为 "RUN_ERROR: {e}"（格式维判 0），run_id 若已生成则保留用于定位。
        评测口径：样本为该条 case；需真实模型/MCP 与可写 bill.db（不可写时 DB 校验会记 0 分）。
    入参说明：
        case (dict)：单条评测用例，含 case_id / category / user_input / expected（db_verify 可选）。
        with_judge (bool)：是否追加 LLM-as-a-Judge 交叉验证；True 时每条用例额外调用一次基座模型。
    返回值说明：
        dict：`evaluate_case` 明细的扩展——新增 run_id(str|None)、db_verify(dict)、db_accuracy(int 0-2)、
              被 DB 硬校验收紧后的 pass；with_judge=True 时另含 judge(dict)；
              elapsed 由 `run_all` 回填，本函数不输出。
    """
    dbv = case.get("expected", {}).get("db_verify")
    snapshot_max = await _db_max_id() if dbv else None
    run_id = None
    try:
        # M7（D3）：评测入口同生产入口接入 run 上下文（每用例独立 run_id）——
        #   失败用例凭 run_id 用 `python utils/trace_view.py <run_id>` 导出 span 树定位环节
        from utils.tracer import run_scope, span
        state = _init_state(case["user_input"])

        async def _trace_run():
            """
            函数功能与逻辑描述：
                在 tracer 的 run_scope 内执行编排图调用，把本次用例运行绑定到一个全新 run_id，
                并以 span("orchestrator.run", kind="orchestrator") 标注编排环节，便于失败用例
                凭 run_id 导出 span 树定位问题环节。
                前置依赖：utils.tracer 的 run_scope / span 上下文管理器（落库到本地 trace_span 表）。
            入参说明：
                无（闭包读取外层 state，并将 run_id 写回外层 nonlocal 变量）。
            返回值说明：
                dict：orchestrator_graph.ainvoke 的最终 state；调用方只取 final_reply 字段。
            """
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
    """
    函数功能与逻辑描述：
        评测主流程：清空 A2A 总线 → 并发启动 4 个 A2A worker（bill / stat / price / finance，
        起后 sleep 0.2s 等消费循环就绪）→ 评测前快照 MAX(id) → 逐条串行执行用例
        （每条计时、打印得分与 PASS/FAIL、打印回复与 DB 结论，并立即增量落盘，中断不丢已跑结果）
        → finally 中取消 worker，并清理本次评测写入的数据：先删 remark LIKE '%测试标记%'，
        再删 id > 评测前快照的全部新增记录（含标记被 orchestrator 剥掉的无标记记录），
        防止污染统计口径与长时记忆。
        前置依赖：需真实模型、MCP 与可写 bill.db。
    入参说明：
        with_judge (bool)：是否对每条用例追加 LLM-as-a-Judge 交叉验证，默认 False。
        only (str | None)：逗号分隔的 case_id 子串过滤条件，命中任一子串即运行；默认 None 表示跑全量 EVAL_CASES（12 条）。
    返回值说明：
        list：逐条结果列表（`_run_case` 返回包 + elapsed 秒数，保留 1 位小数）；无匹配用例时返回空列表。
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
    """
    函数功能与逻辑描述：
        由逐条结果汇总统计：用例总数、通过/失败数、通过率、总均分、五维均分、最弱维度，
        以及 db_accuracy 的已校验条数、硬校验未过条数与均分，并列出全部未 pass 的 case_id。
        注：未对 results 为空做保护（各分母均用 len(results)），调用方需保证非空。
    入参说明：
        results (list)：`run_all` 收集的结果列表，可为全量或仅本次增量用例（条数不必等于评测集条数）。
    返回值说明：
        dict：报告与存档用汇总，含 date(YYYY-MM-DD)、total_cases、passed、failed、
              pass_rate(百分数，1 位小数)、avg_score、avg_by_dim(五维均分)、
              weakest_dim{dim,avg}、db_checked、db_failed、db_avg、fail_cases(case_id 列表)。
    """
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
    """
    函数功能与逻辑描述：
        只读型报告输出：先打印汇总头（日期 / 用例数 / 通过失败与通过率 / 平均总分 / 最弱维度 /
        数据正确性统计）与各维平均分，再逐条打印 [PASS|FAIL] 行（total、db_accuracy、五维明细），
        最后对失败用例输出定位清单（低分维度即 ≤1 的维度、DB 未过标记、输入前 40 字、回复前 80 字、
        DB detail）；全部通过时打印"全部用例通过，无定位项。"。
    入参说明：
        results (list)：`run_all` 收集的结果列表；空列表会因 `_summary` 除零而异常，调用方需保证非空。
    返回值说明：
        无（仅打印控制台报告，不写文件、不修改入参）。
    """
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
    """
    函数功能与逻辑描述：
        把评测结果存档为 JSON（L4 趋势对比的数据源），路径固定为
        <ROOT>/eval_reports/eval_report_YYYYMMDD.json（目录不存在则创建）。
        merge=True 时先读当日已有报告，用本次结果按 case_id 覆盖同名旧记录、其余保留，实现增量续跑；
        旧文件损坏（JSONDecodeError / KeyError）时按空历史处理，不阻断本次落盘。
        注：summary 基于合并后的全量 cases 重算，故续跑后报告汇总为"当日累计"口径。
    入参说明：
        results (list)：本次运行的结果列表（可为增量，条数不必等于评测集条数）。
        merge (bool)：是否与当日已有报告合并，默认 True；False 时整份覆盖写入。
    返回值说明：
        str：写入的报告文件绝对路径（str 形式）。
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


def _start_mcp_servers() -> None:
    """
    函数功能与逻辑描述：
        拉起**全部 3 个** MCP 常驻服务（sql_bill / llm_base / llm_finance）并**等待健康检查通过**。
        必须包含 sql_bill：本评测集含记账类用例，其 `db_verify` 需直查 `bill.db`、
        链路需 `sql.write`，缺则记账链路连不上、评测结果失真（原文档因此要求"先手工启动系统"）。
        复用 `startup.bootstrap.start_mcp_servers()`：按 `SERVER_CMD_MAP`（单一来源）
        拉起全部服务、**逐个 MCP 握手探活**（就绪才返回，120s 超时兜底）、失败即抛错并打日志。
        本脚本是**独立进程、无 pytest-asyncio 参与**，故 `asyncio.run()` 不会引发事件循环冲突
        （与 `tests/integration` 的约束不同，见该目录 conftest 的失败留档）。
    入参说明：
        无。
    返回值说明：
        无（全部服务就绪后返回；任一服务探活失败抛 RuntimeError）。
    """
    from startup.bootstrap import start_mcp_servers

    asyncio.run(start_mcp_servers())


def main():
    """
    函数功能与逻辑描述：
        命令行入口：解析 --with-judge / --only / --no-mcp 三个参数；默认先拉起**全部 3 个**
        MCP 常驻服务（详见 `_start_mcp_servers`），
        再以 asyncio.run 执行 `run_all` 全流程；无论成功与否都在 finally 中回收子进程
        （先 terminate，3 秒内未退出则 kill），最后打印控制台报告并将 JSON 报告落盘到 eval_reports。
    入参说明：
        无（参数取自 sys.argv：--with-judge 开启 Judge 交叉验证；--only 传 case_id 子串过滤；
        --no-mcp 表示外部已启动 MCP，跳过自启子进程）。
    返回值说明：
        无（仅打印报告路径与写文件；异常沿调用栈向上抛出，不额外捕获）。
    """
    import argparse

    parser = argparse.ArgumentParser(description="Agent 回答质量评测（L3）")
    parser.add_argument("--with-judge", action="store_true", help="启用 LLM-as-a-Judge 交叉验证")
    parser.add_argument("--only", default=None, help="只跑指定 case_id 子串（如 single_bill_001）")
    parser.add_argument("--no-mcp", action="store_true", help="跳过自启 MCP 服务（外部已启动时使用）")
    args = parser.parse_args()

    if not args.no_mcp:
        print("[EVAL] 启动 MCP 常驻服务（sql_bill / llm_base / llm_finance）并等待就绪 ...", flush=True)
        _start_mcp_servers()

    try:
        results = asyncio.run(run_all(with_judge=args.with_judge, only=args.only))
    finally:
        if not args.no_mcp:
            from startup.bootstrap import stop_mcp_servers
            asyncio.run(stop_mcp_servers())

    _print_report(results)
    path = save_report(results)
    print(f"\n报告已存档: {path}")


if __name__ == "__main__":
    main()
