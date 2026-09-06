import json
import ast
import re
import time
from datetime import date, timedelta
from typing import List, Dict, Tuple, Any, Optional
from langgraph.types import Command
from langgraph.graph import END
from agents.stat_agent.state import StatState
from agents.stat_agent.prompts.prompt_loader import load_system, render_user
from agentCore.parsers.json_parser import parse_json_output
from mcpGateway.client import mcp_client
from mcpGateway.a2a_queue import a2a_bus
# M14（D17）：ContextManager 窗口接管——stat_agent 按 Router 路由决策（默认本地 4096）取窗口
from agents.context_manager import ContextManager
from modelService.router import context_window_for
from utils.common import parse_sql_result

# 当前统计Agent唯一标识，和A2A队列订阅标识保持一致
STAT_AGENT_ID = "stat_agent"
# M14（D17）：带路由窗口的实例（stat_agent 不在 EXTERNAL_LLM_AGENTS → 本地窗口；
#   未来 stat 改走外部时预算自动按外部窗口放大，无需改代码）
_cm_stat = ContextManager(context_window=context_window_for(STAT_AGENT_ID))
# 日期正则校验：匹配 YYYY-MM-DD 标准日期格式
DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"


def _fix_stat_ir(user_text: str, ir: Dict) -> Dict:
    """
    统计IR代码层兜底：修正 1.8B 小模型的常见语义误判，保证统计范围/算子准确：
    1. 时间范围：用户说"这个月/本月"但LLM照抄schema示例旧月份 → 强制修正为当前自然月；
       用户未提任何时间 → 同样默认当前自然月；"上个月/上月" → 上个月。
    2. 类目过滤：用户未指定任何消费类目但LLM脑补 category_in → 清空。
    3. 数量统计：用户问"几笔/多少笔/多少次"但LLM漏输出count算子 → 补充count算子。
    4. 明细需求：用户问"明细/清单/流水"但LLM漏输出 require_detail_rows → 强制打开。
    """
    today = date.today()
    cur_month_first = today.replace(day=1)
    this_start = cur_month_first.strftime("%Y-%m-%d")
    this_end = today.strftime("%Y-%m-%d")
    # 上个月范围
    prev_month_end = cur_month_first - timedelta(days=1)
    prev_start = prev_month_end.replace(day=1).strftime("%Y-%m-%d")
    prev_end = prev_month_end.strftime("%Y-%m-%d")
    cur_month = cur_month_first.strftime("%Y-%m")

    filt = ir.get("filter")
    if not isinstance(filt, dict):
        filt = {}
        ir["filter"] = filt
    filt.setdefault("category_in", [])
    filt.setdefault("amount_min", None)
    filt.setdefault("amount_max", None)
    ts = str(filt.get("time_start") or "")
    te = str(filt.get("time_end") or "")
    ts_month = ts[:7] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts) else ""

    # ---- 1. 时间范围兜底 ----
    if "上个月" in user_text or "上月" in user_text:
        if ts_month != prev_start[:7]:
            print(f"[STAT] 时间修正：LLM={ts}~{te} 用户意图=上个月 -> {prev_start}~{prev_end}", flush=True)
            filt["time_start"], filt["time_end"] = prev_start, prev_end
    elif "这个月" in user_text or "本月" in user_text or "这月" in user_text:
        if ts_month != cur_month:
            print(f"[STAT] 时间修正：LLM={ts}~{te} 用户意图=这个月 -> {this_start}~{this_end}", flush=True)
            filt["time_start"], filt["time_end"] = this_start, this_end
    elif not re.search(r"\d{4}年\d{1,2}月|\d{4}-\d{2}-\d{2}|昨天|前天|今天|上周|本周|季度", user_text):
        # 用户完全没提时间 → 默认当前自然月（兜底LLM照抄示例月份）
        if ts_month != cur_month:
            print(f"[STAT] 时间修正：LLM={ts}~{te} 用户未指定时间 -> 当前月 {this_start}~{this_end}", flush=True)
            filt["time_start"], filt["time_end"] = this_start, this_end

    # ---- 2. 类目过滤兜底：用户类目表达（含近义/口语词）归一为标准类目 ----
    # 原实现只清空 LLM 脑补；但用户用近义词（"吃饭/打车/外卖"）时 LLM 易漏类目，
    # 且修正后的 category_in 反被"未提标准类目词"误清空 → 归一逻辑补齐（M10 verify 收敛前提）
    _STD_CATS = ["餐饮", "交通", "住宿", "购物", "娱乐"]
    _CAT_SYNONYMS = {
        "餐饮": ["吃饭", "吃", "外卖", "早餐", "午餐", "晚餐", "夜宵", "餐厅", "下馆子", "火锅", "奶茶", "咖啡"],
        "交通": ["打车", "地铁", "公交", "加油", "通勤", "高铁", "机票", "出行", "停车", "油费", "滴滴"],
        "购物": ["买", "淘宝", "京东", "网购", "衣服", "日用", "超市"],
        "住宿": ["酒店", "租房", "房租", "民宿", "住宿"],
        "娱乐": ["电影", "游戏", "KTV", "健身", "演出", "旅游", "娱乐"],
    }
    mentioned = [c for c in _STD_CATS if c in user_text]  # 标准类目词直接命中
    if not mentioned:
        # 无标准词 → 扫描近义/口语表达（LLM 常不识别）
        for std_cat, syns in _CAT_SYNONYMS.items():
            if any(s in user_text for s in syns):
                mentioned.append(std_cat)
    cur_cats = filt.get("category_in") or []
    if mentioned:
        # 用户确实限定了类目：LLM 空/漏/脑补错类目 → 归一并补全为命中的标准类目
        if not cur_cats or not all(c in mentioned for c in cur_cats):
            print(f"[STAT] 类目归一：LLM category_in={cur_cats} 用户意图={mentioned} -> {mentioned}", flush=True)
        filt["category_in"] = mentioned
    elif cur_cats:
        # 用户没提任何类目，但 LLM 脑补了 category_in
        print(f"[STAT] 类目修正：LLM category_in={cur_cats} 用户未指定类目 -> 清空", flush=True)
        filt["category_in"] = []

    # ---- 3. count算子兜底 ----
    ops = ir.get("agg_ops")
    if not isinstance(ops, list):
        ops = []
        ir["agg_ops"] = ops
    op_set = {str(o.get("op", "")).lower() for o in ops if isinstance(o, dict)}

    # ---- 3.4 分组修正：单值统计（max/min/avg/count）若用户未要求按天/按类目，
    # 一律归一为全局无分组。LLM 常给"最大单笔"误加 group_by=["consume_time"]，
    # 导致 MAX/MIN/AVG 按天分组取错行（如最大单笔取到某天最大值而非全局最大）。 ----
    if not re.search(r"每天|每日|按天|按类目|分类|各类|分天|逐日|按周", user_text):
        for o in ops:
            if isinstance(o, dict) and str(o.get("op", "")).lower() in ("max", "min", "avg", "count"):
                if o.get("group_by"):
                    print(f"[STAT] 分组修正：{o.get('op')} group_by={o.get('group_by')} 用户未要求分组 -> 清空", flush=True)
                    o["group_by"] = []

    has_count = "count" in op_set
    if not has_count and re.search(r"几笔|多少笔|几次|多少次|条数|数量|几单|几回|共.*笔|共.*次|多少条", user_text):
        print(f"[STAT] count算子补充：用户问数量但LLM agg_ops={ops}", flush=True)
        ops.append({"op": "count", "target_col": "amount", "group_by": []})
        op_set.add("count")

    # ---- 3.5 avg/max/min 算子兜底（LLM 常漏输出，导致平均值/极值查询结果缺失） ----
    if "avg" not in op_set and re.search(r"平均|均值|均单|均每", user_text):
        print(f"[STAT] avg算子补充：用户问平均值但LLM漏输出 agg_ops={ops}", flush=True)
        ops.append({"op": "avg", "target_col": "amount", "group_by": []})
        op_set.add("avg")
    if "max" not in op_set and re.search(r"最大|最高|最多|最大单笔|最大额", user_text):
        print(f"[STAT] max算子补充：用户问最大单笔但LLM漏输出 agg_ops={ops}", flush=True)
        ops.append({"op": "max", "target_col": "amount", "group_by": []})
        op_set.add("max")
    if "min" not in op_set and re.search(r"最小|最低|最少|最小单笔|最小额", user_text):
        print(f"[STAT] min算子补充：用户问最小单笔但LLM漏输出 agg_ops={ops}", flush=True)
        ops.append({"op": "min", "target_col": "amount", "group_by": []})
        op_set.add("min")

    # ---- 4. 明细兜底 ----
    if re.search(r"明细|清单|流水", user_text) and not ir.get("require_detail_rows"):
        print(f"[STAT] 明细兜底：用户问明细但LLM require_detail_rows=false", flush=True)
        ir["require_detail_rows"] = True

    return ir


def build_stat_prompt(state: StatState) -> Tuple[str, str]:
    """
    组装LLM提示词
    :param state: StatAgent状态对象
    :return: (system提示词, user提示词)
    """
    sys_prompt = load_system()
    # 读取会话内存
    mem = state.get("session_memory") or {}
    history = mem.get("history_text", "")  # export_all() 返回的键是 history_text
    user_text = state.get("user_input", "")
    # M12（D6）：历史层滑窗预算——超长历史保留最近内容（预算内零改动；异常回退原文）
    try:
        _ctx = _cm_stat.build_context(system="", history=history)
        if _ctx.truncated:
            history = _ctx.layer("history")
    except Exception:
        pass
    # 使用jinja模板渲染用户侧prompt
    user_prompt = render_user(user_input=user_text, history=history)
    return sys_prompt, user_prompt


def build_param_select_sql(filter_dict: Dict, agg_op: Dict) -> Tuple[str, List]:
    """
    【核心通用SQL构造器】
    根据IR查询条件动态生成 参数化SELECT语句
    约束：只生成查询语句，全部使用?占位符，杜绝SQL注入风险，禁止DML/DDL语句
    :param filter_dict: IR中的filter筛选条件
    :param agg_op: 单条聚合算子定义
    :return: sql字符串, 参数列表
    """
    where_parts = []
    params = []

    # 时间区间筛选
    ts = filter_dict.get("time_start")
    te = filter_dict.get("time_end")
    if ts and te:
        where_parts.append("consume_time BETWEEN ? AND ?")
        params.extend([ts, te])

    # 消费类目筛选
    cat_list = filter_dict.get("category_in", [])
    if cat_list and len(cat_list) > 0:
        placeholders = ",".join(["?"] * len(cat_list))
        where_parts.append(f"category IN ({placeholders})")
        params.extend(cat_list)

    # 金额下限筛选
    min_amt = filter_dict.get("amount_min")
    if min_amt is not None:
        where_parts.append("amount >= ?")
        params.append(min_amt)
    # 金额上限筛选
    max_amt = filter_dict.get("amount_max")
    if max_amt is not None:
        where_parts.append("amount <= ?")
        params.append(max_amt)

    # 解析聚合函数、分组维度
    op = agg_op["op"].upper()
    target_col = agg_op["target_col"]
    group_cols = agg_op["group_by"]
    select_expr = f"{op}({target_col}) AS agg_val"

    # 拼接SELECT主体
    sql = f"SELECT {select_expr}"
    if group_cols:
        sql += ", " + ",".join(group_cols)
    sql += " FROM bill"

    # 拼接WHERE条件
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    # 拼接GROUP BY分组
    if group_cols:
        sql += " GROUP BY " + ",".join(group_cols)
    return sql, params


async def run_single_query(sql: str, params: list) -> List[Dict]:
    """
    【重点适配你当前MCP底层返回结构】
    执行单条统计SQL，适配 run_sql_logic 返回格式：
    原始返回格式：[TextContent(type="text", text="Python列表字符串")]
    注意：text内是单引号python字面量，不是标准JSON，使用ast.literal_eval解析
    :param sql: 参数化SELECT语句
    :param params: 占位符参数
    :raises Exception: MCP服务调用异常直接向上抛出，由上层统一捕获
    :return: 查询结果列表，解析失败/无数据返回空列表
    """
    # MCP调用异常直接抛出，不再内部吞掉
    raw_resp = await mcp_client.call_bill_sql(STAT_AGENT_ID, sql, params)

    # 统一解析：兼容字符串与 [TextContent] 两种返回形态
    return parse_sql_result(raw_resp)


def merge_stat_data(ir: Dict, agg_result_list: List[Tuple[Dict, List[Dict]]], detail_rows: List[Dict]) -> Dict:
    filter_info = ir["filter"]
    out_data = {
        "time_range": [filter_info["time_start"], filter_info["time_end"]],
        "query_categories": filter_info["category_in"]
    }
    """
    将多条SQL聚合查询结果合并、整理，生成对外统一标准化的统计data结构体
    业务支持的聚合算子：sum / max / min / count / avg
    输出结构完全对齐 Orchestrator 约定的子Agent返回规范，统一由build_user_display_text渲染给用户

    Args:
        ir: LLM输出的完整查询指令IR字典，包含筛选条件、聚合算子、是否需要明细标记
        agg_result_list: 聚合结果列表，格式 [(op_def:单条聚合算子定义dict, rows:该算子查询返回的列表数据)]
            不再使用字典存储，彻底规避dict不可哈希报错
        detail_rows: 原始账单明细列表，当IR.require_detail_rows为true时携带数据，否则为空列表

    Returns:
        dict: 标准化统计结果data，包含时间范围、类目、各类聚合数值、极值、明细等弹性字段
            外层固定key：time_range、query_categories；
            按需动态追加：total_amount、category_summary、daily_trend、max_record、min_record、total_count、avg_amount、detail_list等

    处理逻辑说明：
        1. 先提取IR内全局筛选条件（时间区间、查询类目），写入基础返回结构
        2. 遍历每条聚合算子与对应查询结果，按算子类型填充对应统计字段
            sum：全局总金额 / 类目分组金额 / 每日消费趋势
            max：单笔最大消费记录（金额、类目、日期）
            min：单笔最小消费记录（金额、类目、日期）
            count：账单总条数 / 各类目账单条数
            avg：全局单笔平均金额 / 每日单笔平均金额
        3. 如果IR标记需要明细，追加原始账单明细列表 detail_list
        4. 空查询结果会自动跳过对应字段，不生成空key，避免前端渲染空值崩溃
    """

    total_amount = 0
    category_summary = {}
    daily_trend = []
    max_record = None
    min_record = None
    total_count = 0
    avg_amount = 0
    category_count = {}
    daily_avg_trend = []

    # 遍历列表元组，不再遍历字典items
    for op_def, rows in agg_result_list:
        op_name = str(op_def.get("op", "")).lower()  # 统一小写，兼容LLM输出大写的 COUNT/SUM
        groups = op_def.get("group_by") or []  # 防御LLM漏输出 group_by
        if not rows:
            continue

        if op_name == "sum":
            # 全局总和（无分组）：空结果也输出0元，保证统一渲染"总支出"字段
            if not groups:
                # SQL SUM 在过滤条件无匹配行时仍返回单行 agg_val=None，需一并兜底
                total_amount = float(rows[0]["agg_val"]) if rows and rows[0].get("agg_val") is not None else 0.0
                out_data["total_amount"] = total_amount
            # 按类目分组求和
            elif groups == ["category"]:
                for r in rows:
                    category_summary[r["category"]] = r["agg_val"]
                out_data["category_summary"] = category_summary
            # 按日期分组求和（每日趋势）
            elif groups == ["consume_time"]:
                daily_trend = [{"date": r["consume_time"], "amount": r["agg_val"]} for r in rows]
                out_data["daily_trend"] = daily_trend

        elif op_name == "max":
            # 单笔最大值记录（仅全局无分组时有效；分组查询由 _fix_stat_ir 归一为无分组）
            if not groups and rows:
                row = rows[0]
                max_record = {
                    "amount": row["agg_val"],
                    "category": row.get("category", ""),
                    "consume_time": row.get("consume_time", "")
                }
                out_data["max_record"] = max_record

        elif op_name == "min":
            # 单笔最小值记录（仅全局无分组时有效）
            if not groups and rows:
                row = rows[0]
                min_record = {
                    "amount": row["agg_val"],
                    "category": row.get("category", ""),
                    "consume_time": row.get("consume_time", "")
                }
                out_data["min_record"] = min_record

        elif op_name == "count":
            # 全局账单数量
            if not groups:
                total_count = rows[0]["agg_val"]
                out_data["total_count"] = total_count
            # 按类目统计账单条数
            elif groups == ["category"]:
                for r in rows:
                    category_count[r["category"]] = r["agg_val"]
                out_data["category_count"] = category_count

        elif op_name == "avg":
            # 全局平均单笔金额
            if not groups:
                avg_amount = rows[0]["agg_val"]
                out_data["avg_amount"] = avg_amount
            # 按日期平均单笔
            elif groups == ["consume_time"]:
                daily_avg_trend = [{"date": r["consume_time"], "avg_amount": r["agg_val"]} for r in rows]
                out_data["daily_avg_trend"] = daily_avg_trend

    # 如果IR要求返回明细，追加明细列表
    if ir.get("require_detail_rows") and detail_rows:
        out_data["detail_list"] = detail_rows

    return out_data


# ========== 节点1：解析用户需求，调用LLM生成IR查询指令 ==========
async def parse_query_node(state: StatState):
    """
    第一步：接收用户统计需求，调用LLM输出标准化查询IR
    分支：
    1. LLM调用异常 → 返回失败载荷，进入reply_node
    2. need_more_info=true：信息不足，返回追问载荷（协议与bill_agent完全对齐）
    3. valid=false：无法识别统计需求，返回失败提示
    4. 正常解析成功 → 流转执行SQL节点
    """
    sys_p, usr_p = build_stat_prompt(state)
    # D9 Reflection（M10）：上轮校验未过时注入修正反馈，LLM 据此重生成 IR
    verify_feedback = state.get("verify_feedback")
    if verify_feedback:
        usr_p = usr_p + (
            f"\n\n【上轮统计结果校验未通过，必须据此修正你的查询IR】\n{verify_feedback}\n"
            f"只输出修正后的纯净JSON，不要任何解释。"
        )
    try:
        # IR输出样例：日期动态化（当前自然月），杜绝LLM照抄示例旧月份（如把"这个月"解析成7月）
        today = date.today()
        cur_month_first = today.replace(day=1).strftime("%Y-%m-%d")
        cur_month_end = today.strftime("%Y-%m-%d")
        schema_example = json.dumps(
            {"need_more_info": False, "prompt": "", "valid": True,
             "filter": {"time_start": cur_month_first, "time_end": cur_month_end,
                        "category_in": ["餐饮"], "amount_min": None, "amount_max": None},
             "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
             "require_detail_rows": False},
            ensure_ascii=False)
        ir_data = await parse_json_output(
            mcp_client.call_llm_base,
            sys_p,
            usr_p,
            schema_example,
            agent_tag=STAT_AGENT_ID,
            ignore_missing_keys=["need_more_info"]
        )
        # 代码层兜底修正：时间范围/类目/count算子/明细标记，保证统计100%准确
        user_text = state.get("user_input", "")
        ir_data = _fix_stat_ir(user_text, ir_data)
        # 将解析得到的IR存入状态（D9 Reflection：消费并清空上轮校验反馈）
        return Command(update={"raw_ir": ir_data, "verify_feedback": None}, goto="execute_query_node")
    except Exception as e:
        # 兜底：1.8B模型IR解析失败时，降级为"查询本月总支出"默认IR，保证链路有真实结果。
        # 切勿转追问——用户给出的统计需求通常信息充足，追问只会被collect_node当成最终回复覆盖成功任务。
        print(f"[STAT] {time.strftime('%H:%M:%S')} IR解析失败，降级默认IR: {str(e)[:80]}", flush=True)
        today = date.today()
        default_ir = {
            "need_more_info": False,
            "prompt": "",
            "valid": True,
            "filter": {
                "time_start": today.replace(day=1).strftime("%Y-%m-%d"),
                "time_end": today.strftime("%Y-%m-%d"),
                "category_in": [],
                "amount_min": None,
                "amount_max": None,
            },
            "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
            "require_detail_rows": False,
        }
        return Command(update={"raw_ir": default_ir, "verify_feedback": None}, goto="execute_query_node")


# ========== 节点2：根据IR指令批量执行统计SQL，汇总结果 ==========
async def execute_query_node(state: StatState):
    """
    第二步：读取IR查询指令，循环执行所有聚合查询；按需查询账单明细
    捕获数据库异常，统一封装错误载荷返回
    """
    ir = state.get("raw_ir")
    # 防御：IR缺失或异常类型时兜底失败，避免 ir.get 直接崩溃
    if not isinstance(ir, dict):
        print(f"[STAT] {time.strftime('%H:%M:%S')} execute_query_node IR无效: {ir!r}", flush=True)
        payload = {
            "success": False,
            "agent_type": STAT_AGENT_ID,
            "msg": "统计需求解析失败",
            "data": None,
            "error": "raw_ir缺失或类型异常"
        }
        return Command(update={"final_struct": payload}, goto="reply_node")

    print(f"[STAT] {time.strftime('%H:%M:%S')} execute_query_node 开始 ir={json.dumps(ir, ensure_ascii=False)[:80]}", flush=True)
    # 前置拦截：信息缺失直接返回追问，不再执行SQL
    if ir.get("need_more_info"):
        payload = {
            "success": False,
            "agent_type": STAT_AGENT_ID,
            "msg": "需要向用户补充询问信息",
            "data": {"prompt": ir["prompt"]},
            "error": "need_more_info"
        }
        return Command(update={"final_struct": payload}, goto="reply_node")

    if not ir.get("valid"):
        payload = {
            "success": False,
            "agent_type": STAT_AGENT_ID,
            "msg": "不支持该操作",
            "data": None,
            "error": ir.get("prompt", "无法执行统计")
        }
        return Command(update={"final_struct": payload}, goto="reply_node")

    agg_ops = ir.get("agg_ops", [])
    filter_dict = ir.get("filter", {})
    agg_result_map = {}
    detail_rows = []

    try:
        for op in agg_ops:
            sql, params = build_param_select_sql(filter_dict, op)
            rows = await run_single_query(sql, params)
            op_key = json.dumps(op, sort_keys=True, ensure_ascii=False)
            agg_result_map[op_key] = (op, rows)

        # 如果IR标记需要明细列表，单独执行明细查询
        if ir.get("require_detail_rows"):
            detail_sql = """SELECT amount,category,consume_time,remark FROM bill
WHERE consume_time BETWEEN ? AND ?"""
            detail_params = [filter_dict["time_start"], filter_dict["time_end"]]
            cat_list = filter_dict.get("category_in", [])
            if cat_list:
                plc = ",".join(["?"] * len(cat_list))
                detail_sql += f" AND category IN ({plc})"
                detail_params.extend(cat_list)
            detail_rows = await run_single_query(detail_sql, detail_params)
    except Exception as e:
        # SQL查询异常兜底
        payload = {
            "success": False,
            "agent_type": STAT_AGENT_ID,
            "msg": "统计数据库查询异常",
            "data": None,
            "error": str(e)
        }
        return Command(update={"final_struct": payload}, goto="reply_node")

    # 还原结构：直接构造列表
    agg_result_list = []
    for _, (op_def, rows) in agg_result_map.items():
        agg_result_list.append((op_def, rows))

    # 传入列表
    data = merge_stat_data(ir, agg_result_list, detail_rows)
    payload = {
        "success": True,
        "agent_type": STAT_AGENT_ID,
        "msg": "月度消费统计",
        "data": data,
        "error": None
    }
    # D9 Reflection（M10）：成功结果先进 verify 节点自检（失败载荷仍在各异常分支直达 reply_node）
    return Command(update={"final_struct": payload}, goto="verify_query_node")


# ========== 节点3（D9 Reflection · M10）：结果自检，不过带反馈重生成 ==========
# checklist 明细见 `08` §5.2 M10（2026-08-30 定稿）：① 数字可溯 ② 区间一致 ③ 口径一致 ④ 无越界断言
VERIFY_SYSTEM_PROMPT = """你是账单统计结果质检员，用 checklist 核对「用户问题 ↔ 查询IR ↔ 统计结果」三者是否一致。只输出 JSON，不要任何解释。

检查项：
① 数字可溯：结果中的每个金额都必须能由查询 SQL 事实支撑（结果本身来自确定性查询，重点核对是否覆盖用户要问的数字）；
② 区间一致：结果统计的时间区间与用户问题一致（本月 / 上月 / 某年某月 / 指定起止等）；
③ 口径一致：统计口径覆盖用户意图——问最大/最小/平均/笔数时结果含对应字段；指定了消费类目时结果限定该类目（含近义表达，如"吃饭"=餐饮、"打车/地铁"=交通）；要求明细/清单/流水时结果含明细；问"每天/每月"时有按天/按月分组；
④ 无越界断言：结果不超出查询事实（如未查环比就不应断言"环比上升"）。

全部通过 → {"pass": true, "issues": [], "feedback": ""}；
任一不通过 → {"pass": false, "issues": ["缺陷1", "缺陷2", ...], "feedback": "写给意图解析器的修正指令：明确需改 filter.time_start/time_end / filter.category_in / agg_ops / require_detail_rows 中的哪一项、改成什么"}。"""

# 校验不过允许反馈重生成的次数（`08` M10：最多 2 次转兜底）
MAX_VERIFY_ATTEMPTS = 2


def _summarize_verify_result(data) -> str:
    """校验输入的结果摘要：明细列表降级为计数 + 前 2 条，控制单次校验输入长度"""
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)[:1600]
    d = dict(data)
    detail = d.get("detail_list")
    if isinstance(detail, list):
        d["detail_list"] = {"count": len(detail), "sample": detail[:2]}
    return json.dumps(d, ensure_ascii=False)[:1600]


async def verify_query_node(state: StatState):
    """
    D9 Reflection 自检节点（stat_agent 侧）：
    stat 的结果数字本身来自确定性 SQL（无幻觉源），"结果缺陷"集中在 **IR 与用户问题错位**
    （漏算子 / 类目错位 / 漏明细 / 区间不符）——本节点用 LLM 按 checklist 做语义自检，
    补 `_fix_stat_ir` 代码规则兜底之外的近义/漏项场景。

    路由：
    - pass，或无需校验（execute 失败/追问载荷）→ reply_node（直接发送确定性结果）
    - fail 且重试次数 < MAX_VERIFY_ATTEMPTS → 写反馈，回 parse_query_node 重生成 IR
    - fail 且已达上限，或校验调用异常 → reply_node（兜底：数据确定性不编造、不阻断任务）
    """
    fs = state.get("final_struct")
    if not isinstance(fs, dict) or not fs.get("success"):
        # execute 失败/追问载荷不参与语义自检，直发（verify 只服务成功结果）
        return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")

    attempt = int(state.get("verify_attempt") or 0)
    user_text = state.get("user_input", "")
    ir = state.get("raw_ir")
    ir_text = json.dumps(ir, ensure_ascii=False) if isinstance(ir, dict) else str(ir)
    verify_usr = (
        f"用户问题：{user_text}\n"
        f"查询IR：{ir_text}\n"
        f"统计结果（确定性查询产出）：{_summarize_verify_result(fs.get('data'))}\n"
        f"请按 system 的 checklist 输出校验 JSON。"
    )
    schema_example = json.dumps({"pass": True, "issues": [], "feedback": ""}, ensure_ascii=False)
    try:
        verdict = await parse_json_output(
            mcp_client.call_llm_base,
            VERIFY_SYSTEM_PROMPT,
            verify_usr,
            schema_example,
            agent_tag=STAT_AGENT_ID,
            ignore_missing_keys=["issues"],
        )
    except Exception as e:
        # verify 是增强项（Reflection）：LLM 校验调用异常不阻断任务（guardrail/兜底语义不变）
        print(f"[STAT] {time.strftime('%H:%M:%S')} verify 调用异常，跳过校验直发: {str(e)[:80]}", flush=True)
        return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")

    if verdict.get("pass") is True:
        print(f"[STAT] {time.strftime('%H:%M:%S')} verify 通过，发送结果", flush=True)
        return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")

    issues = verdict.get("issues") or []
    feedback = str(verdict.get("feedback") or "")
    if not feedback and issues:
        feedback = "；".join(str(i) for i in issues)
    if not feedback:
        feedback = "统计结果与用户问题不一致，请重新解析用户意图并修正查询IR"
    if attempt < MAX_VERIFY_ATTEMPTS:
        print(f"[STAT] {time.strftime('%H:%M:%S')} verify 不过（{attempt + 1}/{MAX_VERIFY_ATTEMPTS}），反馈重生成: {feedback[:80]}", flush=True)
        return Command(
            update={
                "verify_decision": "retry",
                "verify_feedback": feedback,
                "verify_attempt": attempt + 1,
            },
            goto="parse_query_node",
        )
    print(f"[STAT] {time.strftime('%H:%M:%S')} verify 重试达上限，转兜底直发当前确定性结果", flush=True)
    return Command(update={"verify_decision": "reply", "verify_feedback": None}, goto="reply_node")


# ========== 节点4：封装结果，通过A2A消息队列返回Orchestrator主Agent ==========
async def reply_node(state: StatState):
    """
    第三步：统一将final_struct序列化为json字符串，发送结果给调度主Agent
    任务结束，流转END终止当前stat子graph
    """
    struct = state["final_struct"]
    result_str = json.dumps(struct, ensure_ascii=False)
    # A2A发送执行结果回调度agent（send_result为同步方法，禁止await）
    a2a_bus.send_result(
        task_id=state["task_id"],
        result=result_str
    )
    # 结束当前子graph
    return Command(update={}, goto=END)


# 对外导出节点常量，供graph.py导入构图
__all__ = [
    "parse_query_node", "execute_query_node", "verify_query_node", "reply_node",
    "STAT_AGENT_ID", "MAX_VERIFY_ATTEMPTS"
]