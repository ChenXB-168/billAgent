# ==============================================
# 统计Agent（stat_agent）LangGraph 节点实现
# 说明：承接用户自然语言统计需求 → LLM 生成查询IR → _fix_stat_ir 代码层兜底 →
#       参数化 SQL 批量执行 → merge_stat_data 归一化 → verify 自检 → A2A 回传 Orchestrator。
# ==============================================
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
    函数功能与逻辑描述：
        统计IR代码层兜底：修正 1.8B 小模型的常见语义误判，保证统计范围/算子准确；
        纯内存原地改写并返回同一 dict，不访问数据库。
        IR 结构为 **`groups` 数组**（一个「查询组」= 一个时间范围 + 组内算子），本函数负责：
        ① **归一**：把顶层的 `filter`/`agg_ops`/`require_detail_rows`（单组简写，模型可能按
           旧格式输出）**展开为 `groups=[{...}]`**；若已有非空 `groups` 则原样沿用。
        ② **逐组兜底**：四类兜底（时间范围 / 类目归一 / 算子补齐+分组修正 / 明细标记）**按组执行**，
           逻辑见 `_fix_one_group`。整句级的"明细"意图作用于所有组。
        ③ **回写兼容**：仅有一组时把该组结果回写顶层旧字段，使只读旧字段的消费方仍可读。
        边界：filter / agg_ops 缺失或类型异常时按空结构重建，不抛异常。
    入参说明：
        user_text (str)：用户原始自然语言问句，用于关键词/正则识别真实统计意图。
        ir (Dict)：LLM 解析得到的查询IR字典，可含 groups 或（兼容）顶层 filter/agg_ops。
    返回值说明：
        Dict：修正后的同一 IR 字典（原地改写并返回）；保证 `groups` 为非空列表，
            且每组含标准化的 filter（time_start/time_end/category_in/amount_min/amount_max）、
            列表型 agg_ops 与布尔型 require_detail_rows；仅一组时会同时回写顶层同名字段。
    """
    # ---- 0. 归一为 groups 列表（顶层单组简写 → 展开）----
    groups = ir.get("groups")
    if not isinstance(groups, list) or not groups:
        raw_filter = ir.get("filter") if isinstance(ir.get("filter"), dict) else {}
        raw_ops = ir.get("agg_ops") if isinstance(ir.get("agg_ops"), list) else []
        groups = [{
            "filter": raw_filter,
            "agg_ops": raw_ops,
            "require_detail_rows": bool(ir.get("require_detail_rows")),
        }]
        ir["groups"] = groups
    groups = [g for g in groups if isinstance(g, dict)]
    if not groups:  # 极端：groups 里全是非 dict → 回落单组空结构
        groups = [{"filter": {}, "agg_ops": [], "require_detail_rows": False}]
        ir["groups"] = groups
    is_multi = len(groups) > 1

    # ---- 1~3. 逐组兜底（时间/类目/算子）----
    for g in groups:
        _fix_one_group(user_text, g, is_multi)

    # ---- 4. 明细兜底（整句级意图 → 作用于**所有组**）----
    if re.search(r"明细|清单|流水", user_text):
        for g in groups:
            if not g.get("require_detail_rows"):
                print("[STAT] 明细兜底：用户问明细但LLM require_detail_rows=false", flush=True)
                g["require_detail_rows"] = True

    # ---- 5. 回写兼容：仅一组时同步顶层旧字段（未改造的消费方仍可读）----
    if not is_multi:
        ir["filter"] = groups[0]["filter"]
        ir["agg_ops"] = groups[0]["agg_ops"]
        ir["require_detail_rows"] = groups[0]["require_detail_rows"]

    return ir


def _fix_one_group(user_text: str, g: Dict, is_multi: bool) -> None:
    """
    函数功能与逻辑描述：
        对**单个查询组**执行三类代码层兜底（原地改写 `g`，无返回值）：
        ① 时间范围：用户说"这个月/本月"但 LLM 照抄 schema 示例旧月份 → 修正为当前自然月；
           用户完全未提时间 → 默认当前自然月；"上个月/上月" → 上个月。
           ★**多组时不做整句意图改写**（见下）；② 类目过滤：命中标准类目词或近义/口语词时
           归一为 5 个标准类目并补全，用户未指定却脑补 category_in → 清空；
        ③ 算子兜底：用户问数量/平均/最大/最小但漏输出 → 补齐 count/avg/max/min，
           且用户未要求按天/按类目分组时把单值算子的 group_by 归一清空。
    入参说明：
        user_text (str)：用户原始自然语言问句。
        g (Dict)：单个查询组 {filter, agg_ops, require_detail_rows}，被原地改写。
        is_multi (bool)：IR 是否为多组（决定时间兜底策略，见下）。
    返回值说明：
        无（原地改写 g）。
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

    filt = g.get("filter")
    if not isinstance(filt, dict):
        filt = {}
        g["filter"] = filt
    filt.setdefault("category_in", [])
    filt.setdefault("amount_min", None)
    filt.setdefault("amount_max", None)
    ts = str(filt.get("time_start") or "")
    te = str(filt.get("time_end") or "")
    ts_month = ts[:7] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ts) else ""

    # ---- 1. 时间范围兜底 ----
    if is_multi:
        # 多组时**只在该组时间缺失时**兜底为当前月，**绝不按整句意图改写**：
        #   整句可能同时含"这个月"和"上个月"（"这个月跟上个月比"），若按整句关键词统一改写，
        #   会把**两组都改成同一个范围** → 对比退化为"同一区间算两遍"。
        #   多组场景下各范围由 LLM 分别填写，代码只补空缺。
        if not (ts and te):
            print(f"[STAT] 时间兜底(多组)：该组时间为空 -> 当前月 {this_start}~{this_end}", flush=True)
            filt["time_start"], filt["time_end"] = this_start, this_end
    elif "上个月" in user_text or "上月" in user_text:
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
    # 不能只清空 LLM 脑补：用户用近义词（"吃饭/打车/外卖"）时 LLM 易漏类目，
    # 且修正后的 category_in 反被"未提标准类目词"误清空 → 需按近义词归一补齐（verify 收敛前提）
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

    # ---- 3. 算子兜底 ----
    ops = g.get("agg_ops")
    if not isinstance(ops, list):
        ops = []
        g["agg_ops"] = ops
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


def build_stat_prompt(state: StatState) -> Tuple[str, str]:
    """
    函数功能与逻辑描述：
        组装统计意图解析所需的 LLM 提示词：加载 system 模板，从会话内存取出历史文本，
        经 ContextManager 历史层滑窗裁剪（超预算才截断，异常时回退原文）后，用 jinja 模板
        渲染用户侧 prompt，返回 (system, user) 二元组供 parse_query_node 调用。
    入参说明：
        state (StatState)：stat_agent 图状态，读取 session_memory.history_text 与 user_input。
    返回值说明：
        Tuple[str, str]：长度固定为 2 的元组，依次为 system 提示词、渲染后的 user 提示词。
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
    函数功能与逻辑描述：
        核心通用 SQL 构造器：根据 IR 的 filter 条件与单条聚合算子在内存中拼装参数化 SELECT
        语句，固定查询 bill 表。时间区间/类目/金额上下限按需拼入 WHERE，聚合函数与分组维度
        决定 SELECT 与 GROUP BY，纯字符串拼接、不访问数据库。
        约束：只生成查询语句，全部条件使用 ? 占位符（参数与 SQL 分离）杜绝注入风险，禁止 DML/DDL。
    入参说明：
        filter_dict (Dict)：IR.filter 筛选条件，键含 time_start/time_end/category_in/
            amount_min/amount_max；缺失键视为不参与过滤，category_in 为空列表时不拼接类目条件。
        agg_op (Dict)：单条聚合算子定义，须含 op（聚合函数名）、target_col（目标列）、
            group_by（分组列列表，为空则全局聚合）。
    返回值说明：
        Tuple[str, List]：参数化 SQL 字符串、与 ? 占位符顺序一一对应的参数列表。
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
    函数功能与逻辑描述：
        执行单条统计 SQL：经 MCP 客户端 call_bill_sql 向账单服务发起调用，返回结果统一交给
        parse_sql_result 解析（兼容字符串与 [TextContent] 两种返回形态，上层无需关心底层格式）。
        MCP 调用异常不在此吞掉，直接向上抛出，由 execute_query_node 统一捕获并封装失败载荷。
    入参说明：
        sql (str)：参数化 SELECT 语句，条件部分使用 ? 占位符。
        params (list)：与占位符顺序一一对应的参数列表。
    返回值说明：
        List[Dict]：查询结果行字典列表；解析失败或无数据时返回空列表。
    """
    # MCP调用异常直接抛出，不再内部吞掉
    raw_resp = await mcp_client.call_bill_sql(STAT_AGENT_ID, sql, params)

    # 统一解析：兼容字符串与 [TextContent] 两种返回形态
    return parse_sql_result(raw_resp)


def merge_stat_data(ir: Dict, agg_result_list: List[Tuple[Dict, List[Dict]]], detail_rows: List[Dict],
                    group_results: List[Dict] = None) -> Dict:
    """
    函数功能与逻辑描述：
        将多条 SQL 聚合查询结果合并、整理，生成对外统一标准化的统计 data 结构体，输出结构对齐
        Orchestrator 约定的子 Agent 返回规范，统一由 build_user_display_text 渲染给用户。
        本函数按"按组渲染 + 向后兼容"两层工作：
        · **单组**（或未传 `group_results`）→ 返回单组结构（旧调用方无需改动）；
        · **多组** → 返回 `{"groups": [<单组结构>, ...]}`，**原样保留每组各自的区间与统计**，
          不做"对比语义"归并（把解释权留给渲染层，减少语义耦合）。
        单组结构的内容由 `_merge_group_block` 产出。
    入参说明：
        ir (Dict)：LLM 输出的完整查询指令 IR 字典（含 groups 或兼容的顶层 filter/agg_ops）。
        agg_result_list (List[Tuple[Dict, List[Dict]]])：**扁平**聚合结果列表，元素为
            (op_def, rows)；多组时由 execute_query_node 汇总所有组的算子，供兼容路径使用。
        detail_rows (List[Dict])：**扁平**明细列表（仅单组兼容路径使用）。
        group_results (List[Dict], 可选)：按组组织的结果，元素形如
            {"filter": {...}, "agg_ops": [(op_def, rows), ...], "detail_rows": [...]}。
            传入多组时走 `groups[]` 渲染；仅一组时等价于单组。
    返回值说明：
        Dict：单组结构（time_range / query_categories / 各统计字段），
            或 `{"groups": [单组结构, ...]}`（多组时）。
    """
    if group_results and len(group_results) > 1:
        return {
            "groups": [
                _merge_group_block(gr.get("filter") or {}, gr.get("agg_ops") or [],
                                   gr.get("detail_rows") or [])
                for gr in group_results
            ]
        }
    if group_results:
        gr = group_results[0]
        return _merge_group_block(gr.get("filter") or {}, gr.get("agg_ops") or [],
                                  gr.get("detail_rows") or [])
    # 兼容路径：未传 group_results → 完全沿用旧行为（读 ir 顶层 filter）
    return _merge_group_block(ir["filter"], agg_result_list, detail_rows)


def _merge_group_block(filter_info: Dict, agg_result_list: List[Tuple[Dict, List[Dict]]],
                       detail_rows: List[Dict]) -> Dict:
    """
    函数功能与逻辑描述：
        【单个查询组】的合并渲染：把该组的筛选条件与算子结果整理成一份标准化 data 结构
        （即 `merge_stat_data` 改造前的原有逻辑，此处 1:1 搬移以便多组复用）。
        先写入时间区间与查询类目，再按算子类型填充：sum → 总金额/类目分组/每日趋势；
        max → 单笔最大；min → 单笔最小；count → 总条数/类目条数；avg → 全局均值/每日均值。
        空结果自动跳过对应字段（不生成空 key），避免前端渲染空值崩溃；
        `require_detail_rows` 由**调用方**决定是否填充 detail_rows（本函数只在 detail_rows 非空时追加）。
    入参说明：
        filter_info (Dict)：该组的筛选条件（time_start/time_end/category_in/...）。
        agg_result_list (List[Tuple[Dict, List[Dict]]])：该组的 (算子, 行列表) 集合。
        detail_rows (List[Dict])：该组的明细行（无明细时为空列表）。
    返回值说明：
        Dict：该组的标准化统计结果（time_range / query_categories / 各统计字段）。
    """
    out_data = {
        "time_range": [filter_info.get("time_start"), filter_info.get("time_end")],
        "query_categories": filter_info.get("category_in") or []
    }

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

    # 明细：由**调用方**决定是否传入（单组兼容路径下，等价于原
    # `ir.require_detail_rows and detail_rows` —— 因为 execute_query_node 只在
    # require_detail_rows 为真时才查询明细，故 detail_rows 非空即代表需要追加）
    if detail_rows:
        out_data["detail_list"] = detail_rows

    return out_data


# ========== 节点1：解析用户需求，调用LLM生成IR查询指令 ==========
async def parse_query_node(state: StatState):
    """
    函数功能与逻辑描述：
        统计链路第一步：接收用户统计需求，构造日期动态化的 IR schema 样例，调用 LLM 输出标准化
        查询IR；随后经 _fix_stat_ir 做代码层兜底修正，并把上轮 D9 Reflection 的 verify_feedback
        追加进 user prompt 驱动重生成（消费后清空该反馈）。
        分支：LLM 解析成功 → 写 raw_ir 并流转 execute_query_node；解析异常 → 降级为
        「查询本月总支出」默认 IR 仍流转 execute_query_node（切勿转追问——用户统计需求通常信息
        充足，追问会被 collect_node 当成最终回复覆盖成功任务）。
        注意：need_more_info / valid=false 的拦截统一在 execute_query_node 内完成，本节点不做拦截。
    入参说明：
        state (StatState)：stat_agent 图状态，读取 user_input 与 verify_feedback（可选）。
    返回值说明：
        Command：更新 raw_ir（修正后的 IR）并清空 verify_feedback，goto 固定为 "execute_query_node"。
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
        # schema 顶层为 **`groups` 数组**（一个「查询组」= 一个时间范围 + 该范围内的 agg_ops）：
        #   单一统计只需一个元素；"这个月跟上个月比"这类**多时间窗**才用多个元素。
        #   本示例同时是**必需字段的来源**（`parse_json_output` 会正则抽键名做校验），
        #   故 filter / agg_ops / require_detail_rows 等内层键名必须保留在示例中。
        schema_example = json.dumps(
            {"need_more_info": False, "prompt": "", "valid": True,
             "groups": [
                 {"filter": {"time_start": cur_month_first, "time_end": cur_month_end,
                             "category_in": ["餐饮"], "amount_min": None, "amount_max": None},
                  "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
                  "require_detail_rows": False}
             ]},
            ensure_ascii=False)
        ir_data = await parse_json_output(
            mcp_client.call_llm_base,
            sys_p,
            usr_p,
            schema_example,
            agent_tag=STAT_AGENT_ID,
            # `groups` 需豁免：schema 顶层已改为 `groups` 数组，但**顶层 `filter`/`agg_ops`
            #   的单组简写仍受代码侧兼容**（`_fix_stat_ir` 会把它展开成 groups）。若不豁免，
            #   模型按旧格式输出时会因"缺 groups"被反复重试，最终**降级为默认 IR**。
            ignore_missing_keys=["need_more_info", "groups"]
        )
        # 代码层兜底修正：时间范围/类目归一/count·avg·max·min算子补齐/单值算子分组归一/明细标记
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
    函数功能与逻辑描述：
        统计链路第二步：读取 IR 并做前置校验（IR 类型异常 / need_more_info / valid=false 三类情况
        直接封装失败或追问载荷直达 reply_node），否则循环执行每条聚合算子的参数化 SQL，按需拼接
        并列条件查询账单明细；任一 SQL 异常统一封装为失败载荷直达 reply_node。
        成功时调用 merge_stat_data 归一化数据，并流转 verify_query_node 做 D9 Reflection 自检。
    入参说明：
        state (StatState)：stat_agent 图状态，读取 raw_ir。
    返回值说明：
        Command：更新 final_struct（标准载荷 {success, agent_type, msg, data, error}）；
            成功时 goto="verify_query_node"，各类失败/追问时 goto="reply_node"。
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

    # **双层循环**（组 × 组内算子）。
    #   兼容：若 IR 无 groups（模型按旧格式输出且未进入 _fix_stat_ir）→ 按**顶层单组**处理，
    #   使本节点对两种形态都成立。
    groups = ir.get("groups")
    if not isinstance(groups, list) or not groups:
        groups = [{"filter": ir.get("filter", {}) or {},
                   "agg_ops": ir.get("agg_ops", []) or [],
                   "require_detail_rows": ir.get("require_detail_rows", False)}]

    group_results = []
    try:
        for g in groups:
            filter_dict = g.get("filter", {}) or {}
            agg_result_map = {}
            for op in (g.get("agg_ops") or []):
                sql, params = build_param_select_sql(filter_dict, op)
                rows = await run_single_query(sql, params)
                op_key = json.dumps(op, sort_keys=True, ensure_ascii=False)
                agg_result_map[op_key] = (op, rows)

            # 该组若标记需要明细列表，单独执行明细查询（使用**本组**的 filter）
            group_detail_rows = []
            if g.get("require_detail_rows"):
                detail_sql = """SELECT amount,category,consume_time,remark FROM bill
WHERE consume_time BETWEEN ? AND ?"""
                detail_params = [filter_dict["time_start"], filter_dict["time_end"]]
                cat_list = filter_dict.get("category_in", [])
                if cat_list:
                    plc = ",".join(["?"] * len(cat_list))
                    detail_sql += f" AND category IN ({plc})"
                    detail_params.extend(cat_list)
                group_detail_rows = await run_single_query(detail_sql, detail_params)

            group_results.append({
                "filter": filter_dict,
                "agg_ops": list(agg_result_map.values()),
                "detail_rows": group_detail_rows,
            })
    except Exception as e:
        # SQL查询异常兜底：**任一组失败即整体失败**，避免给出"半份统计"误导用户
        payload = {
            "success": False,
            "agent_type": STAT_AGENT_ID,
            "msg": "统计数据库查询异常",
            "data": None,
            "error": str(e)
        }
        return Command(update={"final_struct": payload}, goto="reply_node")

    # 兼容：同时汇总出**扁平**列表（供未改造的消费方与单组降级路径使用）
    agg_result_list = [x for gr in group_results for x in gr["agg_ops"]]
    detail_rows = [x for gr in group_results for x in gr["detail_rows"]]

    # 传入按组结果：单组时等价于旧行为，多组时输出 {"groups": [...]}（决策 D-3 = 原始 groups[]）
    data = merge_stat_data(ir, agg_result_list, detail_rows, group_results=group_results)
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
    """
    函数功能与逻辑描述：
        生成供 verify 校验使用的结果摘要：输入为 dict 时把 detail_list 降级为
        {"count": 条数, "sample": 前 2 条}，整体 JSON 序列化后截断到 1600 字符，控制单次校验
        输入长度；输入非 dict 时直接序列化并截断。
    入参说明：
        data (Any)：待摘要的统计结果，通常为 final_struct["data"]。
    返回值说明：
        str：JSON 字符串，最长 1600 字符；detail_list 已降级为计数 + 前 2 条样本。
    """
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)[:1600]
    d = dict(data)
    detail = d.get("detail_list")
    if isinstance(detail, list):
        d["detail_list"] = {"count": len(detail), "sample": detail[:2]}
    return json.dumps(d, ensure_ascii=False)[:1600]


async def verify_query_node(state: StatState):
    """
    函数功能与逻辑描述：
        D9 Reflection 自检节点（stat_agent 侧，即节点3）：stat 的结果数字本身来自确定性 SQL
        （无幻觉源），结果缺陷集中在 IR 与用户问题错位（漏算子 / 类目错位 / 漏明细 / 区间不符），
        本节点用 LLM 按 VERIFY_SYSTEM_PROMPT 的 checklist 做语义自检，覆盖 _fix_stat_ir
        代码规则兜底之外的近义/漏项场景。
        路由：pass，或无需校验（execute 失败/追问载荷）→ reply_node 直接发送确定性结果；
        fail 且重试次数 attempt < MAX_VERIFY_ATTEMPTS → 写反馈，回 parse_query_node 重生成 IR；
        fail 且已达上限，或校验调用异常 → reply_node 兜底（数据确定性不编造、不阻断任务）。
    入参说明：
        state (StatState)：stat_agent 图状态，读取 final_struct、user_input、raw_ir、verify_attempt。
    返回值说明：
        Command：更新 verify_decision（reply/retry）、verify_feedback，必要时更新 verify_attempt；
            reply 时 goto="reply_node"，retry 时 goto="parse_query_node"。
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
    函数功能与逻辑描述：
        统计链路收尾节点（节点4）：将 final_struct 序列化为 JSON 字符串，经 a2a_bus.send_result
        以 task_id 为键回发调度主 Agent Orchestrator（send_result 为同步方法，禁止 await），
        随后流转 END 终止当前 stat 子图。
    入参说明：
        state (StatState)：stat_agent 图状态，读取 final_struct 与 task_id。
    返回值说明：
        Command：空 update（{}）并 goto=END，用于终止当前 stat 子图。
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