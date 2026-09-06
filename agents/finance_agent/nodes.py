import json
import re
import traceback
import time
from typing import List, Dict, Optional, Any
from langgraph.types import Command
from langgraph.graph import END
from agents.finance_agent.state import FinanceState
from agentCore.parsers.json_parser import parse_json_output
from mcpGateway.client import mcp_client
from mcpGateway.a2a_queue import a2a_bus
from .prompts.prompt_loader import load_finance_sys, render_finance_user
# D16（M11）：数字白名单共享工具（与 orchestrator collect_node 同源，见 utils/number_whitelist.py）
from utils.number_whitelist import extract_numbers

# 固定标识，和TASK_AGENT_MAP对应
FINANCE_AGENT_ID = "finance_agent"
# 统一载荷key（全局标准，和Orch完全对齐）
KEY_SUCCESS = "success"
KEY_AGENT_TYPE = "agent_type"
KEY_MSG = "msg"
KEY_DATA = "data"
KEY_ERROR = "error"


# ===================== D16 防幻觉（M11）工具 =====================
# M11.1（2026-09-04 用户拍板）：建议性数字豁免——白名单只约束**事实断言**；
# 模式 B/C 的节约幅度 / 预算分配 / 剩余天数推算（system.md）属建议性/推算性数字，
# 可保留但必须以建议词框定、基于注入数据推算，不得冒充事实（语义核查下沉到 verify）。
# 限定词出现后的子句才豁免（限定词之前的数字仍按事实断言拦截，防"本月支出3300元，建议…"式规避）。
_ADVICE_MARKERS_RE = re.compile(
    r"建议|推荐|可考虑|争取|预计|预估|推算|还需|下调|上调|压缩|缩减|预留|"
    r"节省|节约|省下|可省|分配|留出|降至|升至|降到|提到|设在|设置|控制在"
)
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


def _suggestion_exempt_numbers(text: str) -> set:
    """建议性数字豁免集合：句子含建议/推算限定词时，限定词**之后**子句内的数字豁免白名单。

    - 豁免前提（system.md 模式 B/C）：节约幅度/预算下调/剩余天数推算本就不是确定性事实，
      硬拦会逼死表达力（M11 遗留边界 → M11.1 拍板放开）。
    - 限定词之前仍严格：如"本月支出3300元，建议压缩"→ 3300 在"建议"之前，照常拦截（事实断言）。
    - 残余风险（设计内）：整句"建议：本月支出3300元"式规避交由 verify 语义层核查（checklist ①③）。
    """
    exempt: set = set()
    for sent in _SENTENCE_SPLIT_RE.split(text):
        m = _ADVICE_MARKERS_RE.search(sent)
        if m:
            exempt |= extract_numbers(sent[m.end():])
    return exempt


def _violating_numbers(text: str, whitelist: set) -> set:
    """事实断言违规数字：白名单外的数字 减去 建议性豁免数字"""
    return {
        v for v in extract_numbers(text)
        if not any(abs(v - w) < 0.5 for w in whitelist)
    } - _suggestion_exempt_numbers(text)


def _build_fact_sheet(target_data: dict, stat_data: dict, price_data: dict) -> dict:
    """确定性规则事实底稿：只含预算/当月支出/余额/超标百分比等确定数值（白名单唯一来源）。

    预算兼容现役 target_control（`budget_type=month` + `budget_amount`）与旧格式 `month_budget`；
    支出合计兼容 stat 输出键 `total_amount`（现役）与 `month_total_spend`（历史/测试）。
    无数据的字段不入底稿（LLM 不得编造缺失项）。
    """
    sheet = {}
    budget = None
    if target_data:
        if str(target_data.get("budget_type", "")).lower() in ("month", "月", ""):
            budget = target_data.get("budget_amount") or target_data.get("month_budget")
        # 储蓄目标（小数 0.3 → 口语 30%）
        tsr = target_data.get("target_save_rate")
        if tsr is not None:
            sheet["target_save_rate_percent"] = round(float(tsr) * 100)
    spend = None
    if stat_data:
        spend = stat_data.get("total_amount") or stat_data.get("month_total_spend")
        # 储蓄率（小数 0.2 → 口语 20%）
        sr = stat_data.get("save_rate")
        if sr is not None:
            sheet["save_rate_percent"] = round(float(sr) * 100)
        # 类目占比（小数 0.42 → 口语 42%）
        cr = stat_data.get("category_ratio")
        if isinstance(cr, dict):
            sheet["category_ratio_percent"] = {k: round(float(v) * 100) for k, v in cr.items()}
    if budget is not None:
        sheet["monthly_budget"] = round(float(budget), 2)
    if spend is not None:
        sheet["month_spend"] = round(float(spend), 2)
    if "monthly_budget" in sheet and "month_spend" in sheet and sheet["monthly_budget"] > 0:
        b, s = sheet["monthly_budget"], sheet["month_spend"]
        sheet["balance"] = round(b - s, 2)                 # 预算余额（负=超支）
        sheet["usage_percent"] = round(s / b * 100)        # 支出占预算 %
        sheet["is_over_budget"] = s > b
        if s > b:
            sheet["over_by"] = round(s - b, 2)             # 超出预算额
            sheet["over_percent"] = round((s - b) / b * 100)  # 超出预算 %
    # 单笔消费城市对标（price_agent 基准，确定性数据库值）允许引用
    if price_data:
        for k in ("avg_price", "user_amount", "premium_rate"):
            if price_data.get(k) is not None:
                sheet[k] = price_data[k]
    return sheet


def _build_finance_whitelist(state: FinanceState) -> set:
    """D16 白名单：用户原文 ∪ 确定性注入数据 ∪ 事实底稿（与 orchestrator collect_node 同源）。

    确定性注入数据 = 目标配置 / 统计结果 / 物价对标 / 历史习惯（`monthly_habit` 表）——
    均为 DB/计算层确定性产出且已注入 prompt（`03` §9.9.3 / T8：只有确定性层产出的事实
    才可被白名单校验），LLM 引用其数字属合法引证，逐字可溯源，故一并纳入。

    用户笔记（user_docs）数字**不入白名单**（`03` §9.9.3 规则 2：用户资料非确定性来源，
    文本承载数字 ≠ 系统事实；冲突时确定性优先并向用户说明）——LLM 引用笔记数字会被拦下并修正/回退。
    """
    pool: set = set()
    sources = [state.get("user_query") or ""]
    for key in ("fact_sheet", "target_control"):
        raw = state.get(key)
        if isinstance(raw, dict) and raw:
            sources.append(json.dumps(raw, ensure_ascii=False))
    # 统计/物价结果取成功载荷的 data 部分（与 prompt 注入口径一致）
    for rkey in ("stat_result", "price_result"):
        res = state.get(rkey)
        if isinstance(res, dict) and res.get(KEY_SUCCESS) and isinstance(res.get(KEY_DATA), dict):
            sources.append(json.dumps(res[KEY_DATA], ensure_ascii=False))
    for h in state.get("habit_data") or []:
        sources.append(json.dumps(h, ensure_ascii=False) if isinstance(h, dict) else str(h))
    for s in sources:
        pool |= extract_numbers(s)
    return pool


def _render_fact_fallback(fact_sheet: dict, user_query: str) -> str:
    """白名单修正仍不过时的确定性兜底：只渲染底稿确定数字，不推算、不编造"""
    b = fact_sheet.get("monthly_budget")
    s = fact_sheet.get("month_spend")
    lines = ["【消费诊断 · 数据底稿】"]
    if s is not None:
        lines.append(f"当月支出合计 {s:g} 元。")
    if b is not None:
        lines.append(f"月度预算 {b:g} 元。")
    if b is not None and s is not None and b > 0:
        if s > b:
            lines.append(
                f"已超出预算 {fact_sheet['over_by']:g} 元（达预算的 {fact_sheet['usage_percent']}%）。"
                f"建议：优先压缩超支品类的非必要开销；如需分品类优化建议，请提供品类明细后再分析。"
            )
        else:
            lines.append(
                f"预算内剩余 {fact_sheet['balance']:g} 元（当前支出为预算的 {fact_sheet['usage_percent']}%）。"
                f"如需更具体的分品类优化建议，请告诉我各品类支出情况。"
            )
    return "\n".join(lines)


def assemble_analysis_prompt(state: FinanceState) -> str:
    """纯动态组装完整提示词，全部数据注入，由LLM自主判断输出长短/结构"""
    # 合并用户输入文本
    user_query = "\n".join(state["raw_segments"])
    state["user_query"] = user_query

    # 填充四类上下文数据，空数据传空字典
    target_data = state["target_control"] if state["target_control"] else {}

    stat_data = state["stat_result"][KEY_DATA] if (state["stat_result"] and state["stat_result"].get(KEY_SUCCESS)) else {}

    price_data = state["price_result"][KEY_DATA] if (state["price_result"] and state["price_result"].get(KEY_SUCCESS)) else {}

    # 历史消费习惯（近N月聚合数据，主Agent从 monthly_habit 表注入）
    habit_data = state["habit_data"] if state.get("habit_data") else []

    # M8：用户资料（主Agent下发 `memory/user_docs.py` 检索结果，标注「用户资料」来源）
    user_docs_data = state["user_docs"] if state.get("user_docs") else []

    # D16 防幻觉（M11）：确定性规则事实底稿（预算/当月支出/余额，白名单唯一来源）
    fact_sheet = _build_fact_sheet(target_data, stat_data, price_data)
    state["fact_sheet"] = fact_sheet

    # 动态渲染用户侧prompt
    user_prompt = render_finance_user(
        user_query=user_query,
        target_data=target_data,
        stat_data=stat_data,
        price_data=price_data,
        habit_data=habit_data,
        user_docs_data=user_docs_data,
        fact_sheet=fact_sheet
    )
    sys_prompt = load_finance_sys()
    full_prompt = f"{sys_prompt}\n\n{user_prompt}"
    state["llm_input_prompt"] = full_prompt
    return full_prompt


async def build_prompt_node(state: FinanceState):
    """节点1：整合所有前置数据，动态生成完整理财模型提示词，全局异常捕获"""
    # 【兜底初始化】无论什么情况，先给final_struct赋值，杜绝None
    state["final_struct"] = {
        KEY_SUCCESS: False,
        KEY_AGENT_TYPE: FINANCE_AGENT_ID,
        KEY_MSG: "流程未执行",
        KEY_DATA: None,
        KEY_ERROR: "未捕获异常"
    }
    try:
        # 校验核心必备数据：无预算且无账单统计，无法分析，直接返回追问
        if not state["target_control"] and not state["stat_result"]:
            payload = {
                KEY_SUCCESS: False,
                KEY_AGENT_TYPE: FINANCE_AGENT_ID,
                KEY_MSG: "缺少预算与账单统计数据，无法开展理财分析",
                KEY_DATA: {"prompt": "我需要获取你的月度预算和消费账单数据才能为你做理财分析，请先记录账单或设置月度预算目标。"},
                KEY_ERROR: "need_more_info"
            }
            state["final_struct"] = payload
            # 追问分支：无需再走 execute（llm_input_prompt 不存在会 KeyError），直接回传
            return Command(update=state, goto="reply_node")

        # 动态拼接完整提示词（渲染报错会被外层try捕获）
        assemble_analysis_prompt(state)

    except Exception as e:
        # 任何异常：数据校验、模板渲染、文件读取全部兜底
        stack = traceback.format_exc()
        state["error_stack"] = stack
        payload = {
            KEY_SUCCESS: False,
            KEY_AGENT_TYPE: FINANCE_AGENT_ID,
            KEY_MSG: "理财分析上下文组装失败",
            KEY_DATA: None,
            KEY_ERROR: str(e)
        }
        state["final_struct"] = payload
        # 组装已失败，llm_input_prompt 不存在，直接回传，避免 execute 二次 KeyError 覆盖错误信息
        return Command(update=state, goto="reply_node")

    # 无异常正常流转
    return Command(update=state, goto="execute_finance_node")


async def execute_finance_node(state: FinanceState):
    """节点2：调用理财 LLM + D16 数字白名单闸门（修正 1 次 → 回退底稿渲染）

    白名单 = 用户原文 ∪ 确定性注入数据（目标/统计/物价/习惯/事实底稿），
    与 orchestrator `collect_node` 同源（`utils/number_whitelist.py`）；user_docs 数字不入。
    **事实断言数字**（白名单外、且未按建议性豁免）出现 → 注入清单重生成 1 次；
    仍不过 → `_render_fact_fallback` 确定性兜底。
    M11.1：**建议性数字豁免**（`_suggestion_exempt_numbers`）——句含建议/推算限定词且位于
    限定词之后的数字（节约幅度/预算下调/预计推算，system.md 模式 B/C）不强制在白名单内，
    真实性由 verify 语义层核查（checklist ①③）。底稿为空不下闸门（避免误伤无数据纯建议）。
    """
    print(f"[FIN] {time.strftime('%H:%M:%S')} execute_finance_node 开始", flush=True)
    fact_sheet = state.get("fact_sheet") or {}
    whitelist = _build_finance_whitelist(state)
    fallback_used = False
    try:
        # 拆分完整prompt为 system + user 两部分
        full_prompt = state["llm_input_prompt"]
        sys_part, user_part = full_prompt.split("\n\n", 1)
        # D9 Reflection（M11）：上轮语义校验未过时注入反馈修正
        verify_feedback = state.get("verify_feedback")
        if verify_feedback:
            user_part = user_part + f"\n\n【上轮语义校验未通过，必须据此修正】\n{verify_feedback}"
        # 固定agent_id为当前finance标识
        raw_answer = (await mcp_client.call_llm_finance(FINANCE_AGENT_ID, sys_part, user_part)).strip()

        # D16 白名单闸门：仅当存在确定性底稿时强制；只拦"事实断言"违规数字（建议性豁免见 M11.1）
        violations = _violating_numbers(raw_answer, whitelist) if fact_sheet else set()
        if violations:
            print(f"[FIN] {time.strftime('%H:%M:%S')} 白名单拦截（事实断言含底稿外数字 {sorted(violations)}），修正重试1次", flush=True)
            wl_desc = "、".join(f"{v:g}" for v in sorted(whitelist))
            fix_part = (
                f"\n\n【数字修正——必须遵守】\n上轮输出把白名单之外的数字当**事实**陈述（如实际支出/余额/超支），"
                f"可引用的事实数字仅限：{wl_desc}。\n"
                f"（事实底稿与历史习惯中的数字可直接引用；用户笔记/资料中的数字**不可**作为事实金额引用）\n"
                f"**建议性/推算性数字例外**：节约幅度、预算下调、预计值等可用『建议/预计/下调至/控制在』等词"
                f"明确框定为建议后保留，但必须基于上述事实数字推算，不得写成实际发生额。\n"
                f"请重写：事实数字一律改用白名单数值，建议数字显式加建议词。"
            )
            retry = (await mcp_client.call_llm_finance(FINANCE_AGENT_ID, sys_part, user_part + fix_part)).strip()
            if not _violating_numbers(retry, whitelist):  # 本分支已在 fact_sheet 非空前提下进入
                raw_answer = retry
            else:
                print(f"[FIN] {time.strftime('%H:%M:%S')} 修正仍含事实断言违规数字 → 回退确定性底稿渲染", flush=True)
                raw_answer = _render_fact_fallback(fact_sheet, state.get("user_query") or "")
                fallback_used = True
        state["llm_raw_answer"] = raw_answer

        # 标准成功载荷
        payload = {
            KEY_SUCCESS: True,
            KEY_AGENT_TYPE: FINANCE_AGENT_ID,
            KEY_MSG: "理财分析完成",
            KEY_DATA: {
                "raw_analysis_text": raw_answer,
                # D16（M11）：底稿随结果下发（供 orchestrator 确定性渲染/审计）
                "fact_sheet": fact_sheet or None,
                # D16（M11）：True = 白名单修正失败已回退确定性底稿渲染
                "fallback_used": fallback_used,
                "has_price_data": state["price_result"] is not None,
                "has_stat_data": state["stat_result"] is not None,
                "has_target_data": state["target_control"] is not None
            },
            KEY_ERROR: None
        }
        state["final_struct"] = payload

    except Exception as e:
        stack = traceback.format_exc()
        state["error_stack"] = stack
        payload = {
            KEY_SUCCESS: False,
            KEY_AGENT_TYPE: FINANCE_AGENT_ID,
            KEY_MSG: "理财模型调用失败",
            KEY_DATA: None,
            KEY_ERROR: f"理财分析服务异常：{str(e)}"
        }
        state["final_struct"] = payload

    # D9 Reflection（M11）：成功结果先进 verify 语义自检（失败/追问载荷由 verify 判 success=False 直发）
    return Command(update=state, goto="verify_query_node")


# ========== 节点3（D9 Reflection · M11，复用 M10 stat_agent 模式）：语义自检 ==========
MAX_VERIFY_ATTEMPTS = 2

VERIFY_SYSTEM_PROMPT = """你是理财分析质检员，用 checklist 核对「用户问题 ↔ 确定性事实底稿 ↔ 分析文本」三者一致性。只输出 JSON，不要任何解释。

检查项：
① 数字可溯：**事实断言**——描述本月实际支出/余额/超支额/占比/单笔金额等的数字，必须在事实底稿、用户原文或历史习惯中可溯（execute 白名单已强制，本层复核有无漏网）；**建议性数字豁免**——句含『建议/推荐/预计/下调至/控制在』等建议/推算词框定的数字不要求在白名单内，但必须以建议/预计口吻呈现，不得冒充"本月实际/已支出/已超支"等事实；
② 结论与底稿一致：底稿未超支不说超支、不夸大程度；超支时差额与百分比须与底稿一致；
③ 无越界断言：底稿与输入数据未提供的数字/预测不得当作事实陈述（如历史习惯为空不得编造日均/月均推算）；**建议性数字必须有数据基础**——节约幅度/预算下调/预计推算应可由预算余额、超支额、类目支出、历史均额等推出，凭空给出的建议数字（如无储蓄计划却建议固定储蓄999）判违规；
④ 回答对题：文本覆盖用户问题——单笔值不值 → 是否正面回应；整体收支/预算 → 是否覆盖预算达成与超支判断。

全部通过 → {"pass": true, "issues": [], "feedback": ""}；
任一不通过 → {"pass": false, "issues": ["缺陷1", ...], "feedback": "写给分析者的修正指令，指明问题与改法"}。"""


def _summarize_fact_sheet(fact_sheet: dict) -> str:
    return json.dumps(fact_sheet or {}, ensure_ascii=False)[:800]


async def _verify_llm_call(sys_prompt: str, user_prompt: str, agent_tag: str = ""):
    """finance verify 的 LLM 通道：走理财专用模型（`LLM_FINANCE`）。

    RBAC 权威声明（`06` §RBAC 现状表 / `10` §5.2.5）：finance_agent 权限 = `BILL_READ` + `LLM_FINANCE`，
    **刻意无 `LLM_BASE`**（只读 + 专用模型）。stat_agent verify 走 `call_llm_base`（stat 有 LLM_BASE）；
    finance 复用同一 verify 节点模式但**改走自身 LLM_FINANCE 通道**——否则 `llm.chat`
    （required_perm = `LLM_BASE`，`client.py` L209）会因 finance 无 base 权限被引擎拒绝，
    verify 将在生产环境恒被 PermissionError 跳过（等于语义自检失效）。
    适配 parse_json_output 回调契约 `(sys_prompt, user_prompt, agent_tag=...)` → 转调
    `call_llm_finance(agent_id, sys_prompt, user_text)`（`client.py` L229）。
    """
    return await mcp_client.call_llm_finance(FINANCE_AGENT_ID, sys_prompt, user_prompt)


async def verify_query_node(state: FinanceState):
    """D9 Reflection（M11）：数字白名单（硬校验）之后的**语义层**自检，复用 M10 stat_agent 模式。

    通道说明：finance 无 `LLM_BASE` 权限（RBAC 权威声明 `06`/`10` §5.2.5），verify 经
    `_verify_llm_call` 走自身 `LLM_FINANCE` 通道，RBAC 最小权限保持不变（`03` §9.9.3 verify 行）。

    路由：pass → reply；fail 且重试 <MAX → 反馈回 execute_finance_node 重生成；
    fail 达上限 / 校验调用异常 → reply（兜底：文本已过白名单，数据可溯）。
    """
    fs = state.get("final_struct")
    if not isinstance(fs, dict) or not fs.get("success"):
        # execute 失败/追问载荷不参与语义自检，直发
        return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")
    attempt = int(state.get("verify_attempt") or 0)
    data = fs.get("data") or {}
    text = str(data.get("raw_analysis_text") or "")[:1500]
    verify_usr = (
        f"用户问题：{state.get('user_query') or ''}\n"
        f"事实底稿：{_summarize_fact_sheet(state.get('fact_sheet'))}\n"
        f"分析文本：{text}\n"
        f"请按 system 的 checklist 输出校验 JSON。"
    )
    schema_example = json.dumps({"pass": True, "issues": [], "feedback": ""}, ensure_ascii=False)
    try:
        verdict = await parse_json_output(
            _verify_llm_call, VERIFY_SYSTEM_PROMPT, verify_usr, schema_example,
            ignore_missing_keys=["issues"],
        )
    except Exception as e:
        # verify 是增强项：异常不阻断（文本已过数字白名单，兜底可溯）
        print(f"[FIN] {time.strftime('%H:%M:%S')} verify 调用异常，跳过校验直发: {str(e)[:80]}", flush=True)
        return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")

    if verdict.get("pass") is True:
        print(f"[FIN] {time.strftime('%H:%M:%S')} verify 通过，发送结果", flush=True)
        return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")

    issues = verdict.get("issues") or []
    feedback = str(verdict.get("feedback") or "")
    if not feedback and issues:
        feedback = "；".join(str(i) for i in issues)
    if not feedback:
        feedback = "分析文本与事实底稿/用户问题不一致，请重新撰写"
    if attempt < MAX_VERIFY_ATTEMPTS:
        print(f"[FIN] {time.strftime('%H:%M:%S')} verify 不过（{attempt + 1}/{MAX_VERIFY_ATTEMPTS}），反馈重生成: {feedback[:80]}", flush=True)
        return Command(
            update={
                "verify_decision": "retry",
                "verify_feedback": feedback,
                "verify_attempt": attempt + 1,
            },
            goto="execute_finance_node",
        )
    print(f"[FIN] {time.strftime('%H:%M:%S')} verify 重试达上限，直发当前结果", flush=True)
    return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")


async def reply_node(state: FinanceState):
    """节点4：统一A2A返回结果给Orchestrator，格式完全匹配主Agent接收规范"""
    output_json = json.dumps(state["final_struct"], ensure_ascii=False)
    a2a_bus.send_result(
        task_id=state["task_id"],
        result=output_json
    )
    return Command(update={}, goto=END)


# 导出列表移除recv_task_node
__all__ = [
    "build_prompt_node",
    "execute_finance_node",
    "verify_query_node",
    "reply_node",
    "FINANCE_AGENT_ID",
    "MAX_VERIFY_ATTEMPTS",
    "_build_fact_sheet",
    "_build_finance_whitelist",
    "_render_fact_fallback",
    "_suggestion_exempt_numbers",
    "_violating_numbers"
]