# ==============================================
# price_agent 节点实现 - 消费信息提取 + 城市物价对标
# 说明：本文件负责「从单条消费文本抽取城市/品类/金额 → 查城市基准均价 → 算溢价率」
#       口径与 orchestrator 预警链一致，统一复用 coreModules.price_compare。
# ==============================================
import json
import ast
import re
import time
import traceback
from typing import List, Dict, Tuple, Optional
from langgraph.types import Command
from langgraph.graph import END
from agents.price_agent.state import PriceState
from agentCore.parsers.json_parser import parse_json_output
from mcpGateway.client import mcp_client
from loguru import logger
from config.config import AGENT_LLM_CONFIG, DEFAULT_USER_CITY
from utils.common import parse_sql_result
from mcpGateway.a2a_queue import a2a_bus

# Agent常量定义（统一提取魔法值，方便运营调整）
PRICE_AGENT_ID = "price_agent"
# 溢价率计算（统一从 coreModules/price_compare 引入，与 orchestrator 预警链口径一致）
# M9（P1-7）：清除死导入 PREMIUM_THRESHOLD_*（仅 import 未使用，M9 走查 #1）
from coreModules.price_compare import calc_premium_evaluate
# 标准输出载荷固定key，和stat_agent完全统一
KEY_SUCCESS = "success"
KEY_AGENT_TYPE = "agent_type"
KEY_MSG = "msg"
KEY_DATA = "data"
KEY_ERROR = "error"

# 统一模板导入
from .prompts.prompt_loader import load_extract_system, render_extract_template


def build_price_extract_prompt(seg_text: str) -> Tuple[str, str]:
    """
    函数功能与逻辑描述：
        构建物价抽取链路的 LLM 提示词：系统提示词直接读 system.md（无变量需渲染），
        用户提示词用 user.j2 渲染并注入本条消费描述原文。本函数不做任何输入校验，
        seg_text 为空串时同样会渲染出提示词（由上层保证不空）。
    入参说明：
        seg_text (str)：用户单条消费描述文本。
    返回值说明：
        Tuple[str, str]：二元组 (system 提示词, 渲染后的 user 提示词)。
    """
    sys_prompt = load_extract_system()
    user_prompt = render_extract_template(seg_text=seg_text)
    return sys_prompt, user_prompt


def parse_llm_extract_output(raw_out: Dict, source_text: str) -> Dict:
    """
    函数功能与逻辑描述：
        统一解析 LLM 的抽取结果，对异常格式做分层兜底。
        前置约束：主调度 Agent 已前置校验文本存在消费品类 + 数字金额，本函数仅兜底 LLM 输出错乱场景。
        四步处理：① 城市直接取字符串并 strip；② 兼容 category 为列表或字符串两种形态——
        列表时取首个非空元素并告警（丢弃其余类目）；③ 金额非数字（ValueError/TypeError）时重置为 0 并告警；
        ④ 以原文是否含数字为准做强校验：原文无数字则**强制金额置 0**（覆盖 LLM 编造数值），
        原文有数字但 LLM 未提取出有效金额（1.8B 常漏提）时，用正则兜底取原文首个数字。
        注意 city 与 category 不做白名单校验与空值兜底，由 execute_node 分别处理。
    入参说明：
        raw_out (Dict)：LLM 返回的原始 IR 字典，期望含 city / category / amount 三键。
        source_text (str)：用户原始消费文本，用于判断是否存在数字并做金额兜底提取。
    返回值说明：
        Dict：标准化的三键字典 {city: str, category: str, amount: float}。
            city/category 可能为空串（未识别到），amount 为 0.0 表示无有效金额。
    """
    # 1. 提取城市
    city = raw_out.get("city", "").strip()

    # 2. 兼容category列表/字符串
    raw_category = raw_out.get("category", "")
    category = ""
    if isinstance(raw_category, list):
        valid_list = [c.strip() for c in raw_category if isinstance(c, str) and c.strip()]
        if valid_list:
            category = valid_list[0]
            if len(valid_list) > 1:
                logger.warning(f"[PRICE_PARSE] LLM返回多类目，仅取第一条：{valid_list}, 保留:{category}")
    elif isinstance(raw_category, str):
        category = raw_category.strip()

    # 3. 安全提取金额，捕获数值转换异常
    raw_amt = raw_out.get("amount", 0.0)
    try:
        llm_amount = float(raw_amt)
    except (ValueError, TypeError):
        logger.warning(f"[PRICE_PARSE] LLM金额输出非数字，重置为0，原始值：{raw_amt}")
        llm_amount = 0.0

    # 4. 原文无数字强制金额置0，覆盖LLM编造数值
    has_number = bool(re.search(r"\d+(\.\d+)?", source_text))
    amount = llm_amount
    if not has_number:
        amount = 0.0
        logger.warning(f"[PRICE_PARSE] 原文无数字，强制金额=0，LLM输出:{llm_amount}")
    elif amount <= 0:
        # 原文有数字但LLM未提取出有效金额（1.8B小模型常漏提）：正则兜底取首个数字
        m = re.search(r"\d+(\.\d+)?", source_text)
        if m:
            amount = float(m.group(0))
            logger.warning(f"[PRICE_PARSE] LLM金额无效({llm_amount})，正则兜底提取金额: {amount}")

    return {
        "city": city,
        "category": category,
        "amount": amount
    }


async def run_price_sql_query(sql: str, params: List) -> List[Dict]:
    """
    函数功能与逻辑描述：
        查询城市基准物价表 city_price，统一兼容项目 MCP 返回格式。
        分层异常处理：MCP 通信异常**向上抛出**（由 execute_node 捕获并生成错误载荷），
        解析异常/无数据由 parse_sql_result 吞掉并返回空列表。
        本函数自身不做参数校验、不做重试，仅做「调用 + 解析」两步。
    入参说明：
        sql (str)：参数化查询 SQL。
        params (List)：SQL 占位符参数列表（本链路固定为 [city, category]）。
    返回值说明：
        List[Dict]：基准价行列表，每行为 {列名: 值}（如 {"avg_price": 45.0}）；
            无数据或解析失败时返回空列表 []。
    异常说明：
        Exception：MCP 服务调用失败（连接异常、工具拒绝等）时直接向上抛出，不在此捕获。
    """
    raw_resp = await mcp_client.call_bill_sql(agent_id=PRICE_AGENT_ID, sql=sql, params=params)

    # 统一解析：兼容字符串与 [TextContent] 两种返回形态
    return parse_sql_result(raw_resp)


# ========== 主执行节点：提取消费信息 + 物价对比 + 生成结构化载荷 ==========
async def execute_node(state: PriceState):
    """
    函数功能与逻辑描述：
        物价对标链路的主执行节点（图中第 1 步），负责抽取消费信息、查基准价、
        算溢价率并生成结构化载荷；所有出口均 goto="reply_node"，无条件边。
        业务前置约束：
        1. 主调度 Agent 已完成准入校验：文本必须存在有效消费品类 + 明确数字金额，否则不会调度本节点；
        2. 本节点仅兜底 LLM 输出错乱、城市无基准物价、MCP 查询异常三类场景。
        城市规则：LLM 未提取城市时统一读取全局配置 DEFAULT_USER_CITY 兜底再查表；
        数据表无对应城市 + 品类基准价时返回失败提示，但城市兜底逻辑本身不变。
        流程分五个分支：① LLM 调用/解析彻底失败（含 1 次重试，保留完整异常堆栈随载荷返回）；
        ② 无有效品类；③ 无有效金额（<=0）；④ 城市为空则用 DEFAULT_USER_CITY 兜底；
        ⑤ 查 city_price 取基准均价——取不到（<=0）返回失败载荷，取到则调
        calc_premium_evaluate 计算溢价率与档次，组装成功载荷。
        city_price 查询在外层 try 中，异常时返回失败载荷并附带堆栈（写入 state.error_stack）。
    入参说明：
        state (PriceState)：物价对标状态字典，主要读取 raw_segments（多条消费描述，用换行合并）。
    返回值说明：
        Command：统一为 Command(update={...}, goto="reply_node")，update 中始终写入 final_struct
            （标准五键载荷：success / agent_type / msg / data / error），
            并在可用时附带 extract_info（标准化抽取结果）与 error_stack（异常堆栈）。
            成功时 data 含 city、category、user_consume_amount、city_base_avg_price、
            premium_rate、consume_level、conclusion；失败时 data 或为 None，
            或含 city/category/金额与置 None 的评价字段。
    """
    seg_text = "\n".join(state["raw_segments"])
    print(f"[PRICE] {time.strftime('%H:%M:%S')} execute_node 开始 raw={seg_text[:40]!r}", flush=True)
    logger.info(f"[PRICE_NODE] 原始输入文本: {seg_text}")
    full_exception_stack = ""
    max_retry = 1
    parse_out = None

    # LLM调用 + 异常重试
    for retry_times in range(max_retry + 1):
        try:
            sys_p, usr_p = build_price_extract_prompt(seg_text)
            parse_out = await parse_json_output(
                mcp_client.call_llm_base,
                sys_p, usr_p,
                schema_example='{"city":"深圳","category":"餐饮","amount":30.0}',
                agent_tag=PRICE_AGENT_ID
            )
            logger.info(f"[PRICE_NODE] LLM原始解析结果 parse_out = {parse_out}")
            break
        except Exception as e:
            full_exception_stack = traceback.format_exc()
            logger.error(f"[PRICE_NODE] LLM JSON解析异常堆栈:\n{full_exception_stack}")
            if retry_times < max_retry:
                continue

    # 分支1：LLM调用/解析彻底失败，返回标准错误载荷
    if not parse_out:
        payload = {
            KEY_SUCCESS: False,
            KEY_AGENT_TYPE: PRICE_AGENT_ID,
            KEY_MSG: "物价对比失败",
            KEY_DATA: None,
            KEY_ERROR: f"物价参数提取失败，完整异常堆栈：\n{full_exception_stack}"
        }
        return Command(update={"final_struct": payload, "error_stack": full_exception_stack}, goto="reply_node")

    # 标准化解析LLM输出
    extract_res = parse_llm_extract_output(parse_out, seg_text)
    city = extract_res["city"]
    category = extract_res["category"]
    amount = extract_res["amount"]

    # 品类归一化：账单品类只有五类（餐饮/交通/住宿/购物/娱乐），"火锅 / 奶茶 / 外卖 / 打车"等
    #   **子类表述必须归到父类**再查基准价，否则会拿 LLM 抽出的原词"火锅"去查 `city_price`
    #   （表中只有五大类）→ 回"没有当地火锅的平均价格信息"；而同一句话在**落库**侧其实已被
    #   `_fix_category` 归一成"餐饮"——两侧口径分叉。
    #   复用 bill_agent 的**同一张词典**（CATEGORY_KEYWORDS），保证"落库品类"与"比价品类"
    #   永远同源；无线索时 `_fix_category` 兜底"餐饮"，与本节点"必须存在有效消费品类"的
    #   前置约束相容。
    if category:
        from agents.bill_agent.nodes import _fix_category
        category = _fix_category(seg_text, category)

    # 分支2：LLM输出无有效消费品类
    if not category:
        payload = {
            KEY_SUCCESS: False,
            KEY_AGENT_TYPE: PRICE_AGENT_ID,
            KEY_MSG: "物价对比失败",
            KEY_DATA: None,
            KEY_ERROR: "未识别到任何有效消费场景，缺少必备消费项目，无法执行物价对标分析"
        }
        return Command(update={"final_struct": payload, "extract_info": extract_res}, goto="reply_node")

    # 分支3（★M18 改造）：无有效消费金额 → 进入「价位查询模式」
    #   起因：M17 契约化后，主 Agent 在 fit_level=inferable 场景会规划"无金额的比价"
    #   （如"我想去唱歌，预算多少合适"）；若此处仍直接失败，"花费前决策辅助"链路断裂。
    #   改造后：有品类、无金额 → 仍查 city_price，返回该品类**本地参考均价**（不做溢价率判断）。
    price_query_mode = False
    if amount <= 0:
        price_query_mode = True
        logger.info("[PRICE_NODE] 无有效金额，进入价位查询模式（仅返回品类基准均价，不做溢价率判断）")

    # 分支4：城市为空，读取全局配置默认城市兜底
    if not city.strip():
        city = DEFAULT_USER_CITY
        logger.warning(f"[PRICE_NODE] LLM未输出城市，读取全局配置默认城市：{city}")

    # 查询城市基准物价表 city_price
    sql = "SELECT avg_price FROM city_price WHERE city = ? AND category = ? LIMIT 1"
    params = [city, category]
    logger.info(f"[PRICE_NODE] 执行基准物价查询SQL: {sql}, 参数={params}")

    try:
        sql_result_list = await run_price_sql_query(sql, params)
        base_avg = 0.0
        if sql_result_list and len(sql_result_list) > 0:
            row = sql_result_list[0]
            val = row.get("avg_price")
            if val is not None:
                base_avg = float(val)
        logger.info(f"[PRICE_NODE] 匹配到的城市基准均价 base_avg={base_avg}")

        # 兜底城市查表后无对应品类基准数据
        if base_avg <= 0:
            payload = {
                KEY_SUCCESS: False,
                KEY_AGENT_TYPE: PRICE_AGENT_ID,
                KEY_MSG: "物价对比失败",
                KEY_DATA: {
                    "city": city,
                    "category": category,
                    "user_amount": amount,
                    "base_avg_price": 0.0,
                    "premium_rate": None,
                    "level": None,
                    "conclusion": None
                },
                KEY_ERROR: f"【{city}{category}】暂无城市基准物价数据，无法完成对标评价"
            }
            return Command(update={
                "final_struct": payload,
                "extract_info": extract_res
            }, goto="reply_node")

        # ★M18：价位查询模式——只返回参考均价，不做溢价率判断（无用户金额可比）
        if price_query_mode:
            full_data = {
                "city": city,
                "category": category,
                "user_consume_amount": None,
                "city_base_avg_price": base_avg,
                "premium_rate": None,
                "consume_level": None,
                "conclusion": None
            }
            payload = {
                KEY_SUCCESS: True,
                KEY_AGENT_TYPE: PRICE_AGENT_ID,
                KEY_MSG: "价位查询",
                KEY_DATA: full_data,
                KEY_ERROR: None
            }
            logger.info("[PRICE_NODE] 价位查询模式：返回品类本地参考均价")
        else:
            # 溢价率计算、档次评价（原逻辑）
            eval_data = calc_premium_evaluate(amount, base_avg)
            full_data = {
                "city": city,
                "category": category,
                "user_consume_amount": amount,
                "city_base_avg_price": base_avg,
                "premium_rate": eval_data["premium_rate"],
                "consume_level": eval_data["level"],
                "conclusion": eval_data["conclusion"]
            }
            payload = {
                KEY_SUCCESS: True,
                KEY_AGENT_TYPE: PRICE_AGENT_ID,
                KEY_MSG: "物价对比",
                KEY_DATA: full_data,
                KEY_ERROR: None
            }
            logger.info("[PRICE_NODE] 成功生成PRICE_SUCCESS结构化结果")

    except Exception as e:
        err_msg = str(e)
        stack = traceback.format_exc()
        logger.error(f"[PRICE_NODE] 基准物价SQL查询异常: {err_msg}\n堆栈:{stack}")
        payload = {
            KEY_SUCCESS: False,
            KEY_AGENT_TYPE: PRICE_AGENT_ID,
            KEY_MSG: "物价对比失败",
            KEY_DATA: {
                "city": city,
                "category": category,
                "user_amount": amount,
                "base_avg_price": None,
                "premium_rate": None,
                "level": None,
                "conclusion": None
            },
            KEY_ERROR: f"物价基准查询失败，异常信息：{err_msg}"
        }
        return Command(update={
            "final_struct": payload,
            "extract_info": extract_res,
            "error_stack": stack
        }, goto="reply_node")

    return Command(update={
        "final_struct": payload,
        "extract_info": extract_res
    }, goto="reply_node")


# ========== 收尾节点：结构化载荷序列化，A2A回传给调度主Agent ==========
async def reply_node(state: PriceState):
    """
    函数功能与逻辑描述：
        物价对标链路的收尾节点（图中第 2 步）：把 execute_node 写入 state["final_struct"]
        的字典序列化为 JSON 字符串（ensure_ascii=False，保证中文可读），
        再经 A2A 总线按 task_id 回传给编排器，格式与 stat_agent 完全对齐。
        同步投递、不 await；不做结果加工，只做序列化与搬运；执行后无条件 goto=END。
    入参说明：
        state (PriceState)：物价对标状态字典，需含 final_struct（标准五键载荷）与 task_id。
    返回值说明：
        Command：Command(update={}, goto=END)，不更新 state（结果已走 A2A 旁路回传）。
    """
    struct = state["final_struct"]
    result_str = json.dumps(struct, ensure_ascii=False)
    a2a_bus.send_result(
        task_id=state["task_id"],
        result=result_str
    )
    return Command(update={}, goto=END)


# 导出，移除recv_task_node
__all__ = ["execute_node", "reply_node", "PRICE_AGENT_ID"]
