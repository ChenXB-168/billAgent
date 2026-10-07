# ==============================================
# finance_agent 节点实现 - 理财分析生成 + D16 数字白名单 + D9 语义自检
# 说明：四段链路「组装提示词 → 调专用模型 → 白名单闸门 → 语义自检」，
#       数字真实性由「确定性事实底稿 + 白名单硬校验 + verify 语义复核」三层共同保证。
# ==============================================
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
    """
    函数功能与逻辑描述：
        计算「建议性数字豁免集合」：把文本按句末标点（。！？；及换行）切句，
        对含建议/推算限定词（_ADVICE_MARKERS_RE）的句子，只把限定词**之后**子句内出现的数字
        纳入豁免集合。用于让节约幅度、预算下调、剩余天数推算这类非确定性数字不被白名单硬拦
        （M11 遗留边界 → M11.1 拍板放开，否则会逼死模型表达力）。
        限定词之前的数字**不豁免**——如"本月支出3300元，建议压缩"中的 3300 属事实断言，照常拦截。
        残余风险（设计内已确认）：整句"建议：本月支出3300元"这类把事实伪装进建议句的规避手法，
        本函数无法识别，交由 verify_query_node 的语义层 checklist ①③ 核查。
    入参说明：
        text (str)：待扫描的分析文本。
    返回值说明：
        set：应豁免的数字绝对值集合；无建议句或无数字时返回空集合。
    """
    exempt: set = set()
    for sent in _SENTENCE_SPLIT_RE.split(text):
        m = _ADVICE_MARKERS_RE.search(sent)
        if m:
            exempt |= extract_numbers(sent[m.end():])
    return exempt


def _violating_numbers(text: str, whitelist: set) -> set:
    """
    函数功能与逻辑描述：
        D16 白名单闸门的核心判定：找出文本中**被当作事实断言**却无法溯源的数字。
        计算方式为两步差集：先从文本抽取的全部数字中剔除能在白名单内找到近似值的部分
        （近似判据为绝对差 < 0.5，容忍浮点与四舍五入误差），再减去建议性豁免数字。
        结果非空即表示存在幻觉风险，由 execute_finance_node 触发修正重试或回退底稿渲染。
        本函数纯计算、不打印、不修改入参。
    入参说明：
        text (str)：待校验的模型输出文本。
        whitelist (set)：可信数字白名单（由 _build_finance_whitelist 构造）。
    返回值说明：
        set：违规数字集合（白名单外的**事实断言**数字）；全部可溯源或均可豁免时返回空集合。
    """
    return {
        v for v in extract_numbers(text)
        if not any(abs(v - w) < 0.5 for w in whitelist)
    } - _suggestion_exempt_numbers(text)


def _build_fact_sheet(target_data: dict, stat_data: dict, price_data: dict) -> dict:
    """
    函数功能与逻辑描述：
        构建「确定性规则事实底稿」：只收录预算、当月支出、余额、超标额与百分比等确定数值，
        它是白名单的**唯一事实来源**，也是白名单校验失败时的兜底渲染素材。
        兼容性处理：预算兼容现役 target_control（`budget_type=month` + `budget_amount`）
        与旧格式 `month_budget`；支出合计兼容 stat 输出的 `total_amount`（现役）与
        `month_total_spend`（历史/测试）；仅在 budget_type 为 month/月/空时读取预算。
        比例类字段统一由小数换算为百分数口径（如 0.3 → 30），避免模型把 0.3 当 30% 误读。
        关键约束：数据缺失的字段**完全不写入底稿**（而非写 0），使下游能区分"确实为 0"与"无数据"，
        并保证模型不得编造缺失项。派生字段（余额/占比/超标）仅当预算 > 0 时才计算。
    入参说明：
        target_data (dict)：目标配置数据（预算类型、预算金额、目标储蓄率）。
        stat_data (dict)：统计数据（当月支出合计、储蓄率、类目占比）。
        price_data (dict)：单笔消费城市对标数据（avg_price / user_amount / premium_rate）。
    返回值说明：
        dict：事实底稿。可能包含的键：target_save_rate_percent、save_rate_percent、
            category_ratio_percent、monthly_budget、month_spend、balance、usage_percent、
            is_over_budget、over_by、over_percent、avg_price、user_amount、premium_rate；
            三项入参均为空/无效时返回空字典 {}（此时下游不下白名单闸门）。
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
    """
    函数功能与逻辑描述：
        构造 D16 数字白名单，来源为「用户原文 ∪ 确定性注入数据 ∪ 事实底稿」，
        与 orchestrator 的 collect_node 同源（共用 utils.number_whitelist.extract_numbers）。
        确定性注入数据 = 目标配置 / 统计结果 / 物价对标 / 历史习惯（monthly_habit 表）——
        均为 DB 或计算层的确定性产出且已注入 prompt（`03` §9.9.3 / T8：只有确定性层产出的事实
        才可被白名单校验），LLM 引用其数字属合法引证、逐字可溯源，故一并纳入。
        统计/物价结果只取成功载荷的 data 部分（与 prompt 注入口径一致），失败载荷的数字不入白名单。
        ★关键排除：用户笔记（user_docs）的数字**不入白名单**（`03` §9.9.3 规则 2）——
        用户资料非确定性来源，文本承载数字 ≠ 系统事实；冲突时确定性优先并向用户说明。
        因此 LLM 引用笔记中的数字会被闸门拦下并触发修正或回退。
    入参说明：
        state (FinanceState)：理财状态字典，读取 user_query、fact_sheet、target_control、
            stat_result、price_result、habit_data。
    返回值说明：
        set：可信数字绝对值集合（已去重、已剔除 0）；无任何来源数字时返回空集合
            （此时 execute 侧因 fact_sheet 为空也不会启用闸门）。
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
    """
    函数功能与逻辑描述：
        白名单修正仍不通过时的**确定性兜底渲染**：只把事实底稿里的确定数字拼成一段中文说明，
        不推算、不编造任何未提供的数值。输出以「【消费诊断 · 数据底稿】」开头，
        依次追加当月支出、月度预算，再按是否超支给出两句不同结论：
        超支时给出超出额与达预算百分比，并建议压缩超支品类的非必要开销；
        预算内时给出剩余额与当前百分比，并提示可提供品类明细以获取更具体建议。
        底稿缺少预算或支出时只输出已有条目，不会拼出残缺结论句（派生字段仅在有预算时存在）。
        入参中的 user_query 当前未被使用，仅为调用方接口兼容而保留。
    入参说明：
        fact_sheet (dict)：事实底稿（由 _build_fact_sheet 产出）。
        user_query (str)：用户原始问题（当前未参与渲染内容）。
    返回值说明：
        str：确定性兜底文本，至少包含标题行；底稿为空时返回仅含标题行的文本。
    """
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
    # 底稿不足以支撑结论时，**必须说明缺什么 + 给出下一步**：
    #   只堆已有数字（如仅有预算时只输出"【消费诊断 · 数据底稿】+ 月度预算 X 元"）会让用户
    #   既不知道为何没有分析、也不知道该做什么——信息不足时应"补足信息"，而非静默省略。
    if s is None or b is None:
        lacking = [n for n, v in (("当月支出", s), ("月度预算", b)) if v is None]
        lines.append(f"暂缺{'、'.join(lacking)}数据，因此无法给出支出结构分析与规划结论。")
        if b is None:
            lines.append("设置月度预算后，我就能按预算给出花费进度、超支判断与优化建议。")
        elif s is None:
            lines.append("记账或查询当月开销后，我就能据此给出具体的规划建议。")
    return "\n".join(lines)


def assemble_analysis_prompt(state: FinanceState) -> str:
    """
    函数功能与逻辑描述：
        纯动态组装完整提示词：把用户输入与四类上下文数据（目标配置 / 统计数据 / 物价对标 /
        历史习惯 / 用户资料）全部注入，由 LLM 自主判断输出长短与结构。
        副作用（写回 state）：合并后的原文写入 state["user_query"]，
        事实底稿写入 state["fact_sheet"]（供 execute 闸门与 verify 复核复用），
        最终拼接结果写入 state["llm_input_prompt"]。
        数据取值口径：统计与物价只取成功载荷的 data 部分，失败载荷按空字典处理；
        目标配置、历史习惯、用户资料为空时统一传空值（空字典/空列表），
        由 Jinja 模板渲染成"无数据"文案而非报错。
        最终提示词格式为「系统提示词 + 空行 + 用户提示词」，空行分隔是 execute 侧
        split("\\n\\n", 1) 拆分 system/user 的约定，修改拼接格式须同步修改那里。
    入参说明：
        state (FinanceState)：理财状态字典，读取 raw_segments、target_control、stat_result、
            price_result、habit_data、user_docs；并写回 user_query、fact_sheet、llm_input_prompt。
    返回值说明：
        str：拼接完成的完整提示词（同时已写入 state["llm_input_prompt"]）。
    异常说明：
        不捕获异常：模板渲染或文件读取失败会向上抛出，由 build_prompt_node 的 try 兜底转错误载荷。
    """
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
    """
    函数功能与逻辑描述：
        理财链路第 1 个节点：整合所有前置数据并动态生成完整理财模型提示词，带全局异常捕获。
        先做**兜底初始化**——无论后续发生什么，先把 state["final_struct"] 置为"流程未执行"的
        失败载荷，杜绝下游拿到 None。随后两条关键分支：
        ① 无预算且无账单统计时无法分析，直接返回追问载荷并 goto="reply_node"
        ——此处必须直接回传，因为 llm_input_prompt 尚未生成，继续走 execute 会 KeyError；
        ② 组装成功则 goto="execute_finance_node"。
        任何异常（数据校验、模板渲染、文件读取）都被捕获，记录 error_stack 与失败载荷后
        同样直接回传，避免 execute 二次异常覆盖真实错误信息。
    入参说明：
        state (FinanceState)：理财状态字典，读取 target_control、stat_result 等；
            写回 final_struct、error_stack 与（成功时）llm_input_prompt 等字段。
    返回值说明：
        Command：Command(update=state, goto=...)。goto 为 "reply_node"（追问分支或异常分支，
            此时 final_struct 为 success=False）或 "execute_finance_node"（正常流转）。
            注意本节点整体回写 state，与其它 agent 只回写部分字段的风格不同。
    """
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
    """
    函数功能与逻辑描述：
        理财链路第 2 个节点：调用理财专用 LLM，并执行 D16 数字白名单闸门。
        白名单 = 用户原文 ∪ 确定性注入数据（目标/统计/物价/习惯/事实底稿），
        与 orchestrator 的 collect_node 同源（`utils/number_whitelist.py`）；user_docs 数字不入。
        闸门逻辑：**事实断言数字**（白名单外且未按建议性豁免）出现时，注入白名单清单与修正指令
        重新生成 1 次；若修正后仍违规，改由 _render_fact_fallback 做确定性底稿渲染并把
        fallback_used 置为 True 随载荷下发。**底稿为空时不下闸门**（避免误伤无数据的纯建议场景）。
        M11.1：建议性数字豁免见 _suggestion_exempt_numbers——含建议/推算限定词且位于限定词之后的数字
        （节约幅度/预算下调/预计推算，system.md 模式 B/C）不强制在白名单内，真实性由 verify 语义层核查。
        其它要点：提示词按首个空行拆为 system/user 两部分；若 state 带回 verify_feedback
        （上轮语义校验未过），会以【上轮语义校验未通过，必须据此修正】追加到 user 部分实现反馈再生成。
        成功载荷的 data 除分析文本外还含 fact_sheet、fallback_used 与三个 has_* 数据可用性标记。
        本节点异常统一转失败载荷，随后**一律**进 verify_query_node（失败/追问载荷由 verify 直接放行）。
    入参说明：
        state (FinanceState)：理财状态字典，读取 llm_input_prompt、fact_sheet、verify_feedback、
            user_query、price_result、stat_result、target_control；
            写回 llm_raw_answer、final_struct、error_stack。
    返回值说明：
        Command：Command(update=state, goto="verify_query_node")。final_struct 为
            success=True 的"理财分析完成"载荷（data 含 raw_analysis_text、fact_sheet、
            fallback_used、has_price_data、has_stat_data、has_target_data），
            或 success=False 的"理财模型调用失败"载荷。
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
    """
    函数功能与逻辑描述：
        把事实底稿转为供 verify 提示词引用的紧凑 JSON 文本，并截断至 800 字符，
        防止底稿（尤其含 category_ratio_percent 明细时）撑大 verify 的输入预算。
        入参为 None 或空字典时输出 "{}"。截断可能把 JSON 截成不完整片段，
        这是可接受的——verify 仅将其作为校对参考，不要求机器可解析。
    入参说明：
        fact_sheet (dict)：事实底稿；None 与空字典等价处理。
    返回值说明：
        str：JSON 文本（ensure_ascii=False 保留中文），最长 800 字符。
    """
    return json.dumps(fact_sheet or {}, ensure_ascii=False)[:800]


async def _verify_llm_call(sys_prompt: str, user_prompt: str, agent_tag: str = ""):
    """
    函数功能与逻辑描述：
        理财 verify 的 LLM 通道：走理财专用模型（`LLM_FINANCE`）。
        RBAC 权威声明（`06` §RBAC 现状表 / `10` §5.2.5）：finance_agent 权限 = `BILL_READ` + `LLM_FINANCE`，
        **刻意无 `LLM_BASE`**（只读 + 专用模型）。stat_agent 的 verify 走 `call_llm_base`（stat 有 LLM_BASE）；
        finance 复用同一 verify 节点模式但**改走自身 LLM_FINANCE 通道**——否则 `llm.chat`
        （required_perm = `LLM_BASE`）会因 finance 无 base 权限被引擎拒绝，
        导致 verify 在生产环境恒被 PermissionError 跳过，等于语义自检失效。
        本函数同时是 parse_json_output 的 llm_call_func 回调，需满足其契约
        `(sys_prompt, user_prompt, agent_tag=...)` → 内部转调
        `call_llm_finance(agent_id, sys_prompt, user_text)`。
    入参说明：
        sys_prompt (str)：verify 系统提示词（质检 checklist）。
        user_prompt (str)：verify 用户提示词（用户问题 + 事实底稿 + 分析文本）。
        agent_tag (str)：调用方标识；本函数固定使用 FINANCE_AGENT_ID，该参数仅为满足回调契约而保留。
    返回值说明：
        str：verify 模型的原始输出文本（预期为校验 JSON 字符串）。
    异常说明：
        不捕获异常：调用失败向上抛出，由 verify_query_node 捕获并按"校验增强项失败"放行处理。
    """
    return await mcp_client.call_llm_finance(FINANCE_AGENT_ID, sys_prompt, user_prompt)


async def verify_query_node(state: FinanceState):
    """
    函数功能与逻辑描述：
        D9 Reflection（M11）语义自检节点：在数字白名单硬校验之后再做**语义层**复核
        （数字可溯性口径、结论与底稿一致性、越界断言、回答对题四项），复用 M10 stat_agent 模式。
        通道说明：finance 无 `LLM_BASE` 权限（RBAC 权威声明 `06` / `10` §5.2.5），
        verify 经 _verify_llm_call 走自身 `LLM_FINANCE` 通道，RBAC 最小权限保持不变（`03` §9.9.3 verify 行）。
        路由规则：pass → goto reply；fail 且尝试次数 < MAX_VERIFY_ATTEMPTS(2) → 带 feedback
        回 execute_finance_node 重生成（并递增 verify_attempt）；fail 达上限、
        或校验调用本身异常 → goto reply（兜底放行：文本已过白名单，数据可溯）。
        两类载荷不参与自检、直接放行：final_struct 非字典或 success 为假（execute 失败/追问载荷）。
        feedback 缺失时先用 issues 拼接，仍为空则使用固定兜底文案"分析文本与事实底稿/用户问题不一致，请重新撰写"，
        保证回传给 execute 的修正指令永不为空。
    入参说明：
        state (FinanceState)：理财状态字典，读取 final_struct、verify_attempt、user_query、fact_sheet；
            写回 verify_decision（"reply"/"retry"）与 verify_feedback、verify_attempt。
    返回值说明：
        Command：Command(update={...}, goto=...)。
            - 放行：update 为 {"verify_decision": "reply", "verify_feedback": None}，goto="reply_node"。
            - 重生成：update 为 {"verify_decision": "retry", "verify_feedback": <反馈>, 
              "verify_attempt": attempt+1}，goto="execute_finance_node"。
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
    """
    函数功能与逻辑描述：
        理财链路的收尾节点（第 4 步）：把 state["final_struct"] 序列化为 JSON 字符串
        并经 A2A 总线按 task_id 回传给编排器，格式完全匹配主 Agent 的接收规范。
        同步投递、不 await；不做结果加工（序列化即全部工作）；执行后无条件 goto=END。
    入参说明：
        state (FinanceState)：理财状态字典，需含 final_struct（标准五键载荷）与 task_id。
    返回值说明：
        Command：Command(update={}, goto=END)，不更新 state（结果已走 A2A 旁路回传）。
    """
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
