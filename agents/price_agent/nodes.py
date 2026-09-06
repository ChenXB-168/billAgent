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
    构建消费提取LLM提示词
    Args:
        seg_text: 用户单条消费描述文本
    Returns:
        (system提示词, user渲染模板文本)
    """
    sys_prompt = load_extract_system()
    user_prompt = render_extract_template(seg_text=seg_text)
    return sys_prompt, user_prompt


def parse_llm_extract_output(raw_out: Dict, source_text: str) -> Dict:
    """
    统一解析LLM提取结果，分层兜底兼容异常格式
    前置约束：主调度Agent已前置校验文本存在消费品类+数字金额，仅兜底LLM输出错乱场景
    Args:
        raw_out: LLM返回原始IR字典
        source_text: 用户原始消费文本，用于判断是否存在数字
    Returns:
        标准化 {city:str, category:str, amount:float}
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
    查询城市基准物价，统一兼容项目MCP返回格式 [TextContent(text="单引号列表字面量")]
    分层异常处理：MCP通信异常向上抛出；解析异常返回空列表
    Args:
        sql: 参数化查询SQL
        params: 占位符参数
    Returns:
        基准价列表，无数据/解析失败返回空列表
    Raises:
        Exception: MCP服务调用失败直接抛出
    """
    raw_resp = await mcp_client.call_bill_sql(agent_id=PRICE_AGENT_ID, sql=sql, params=params)

    # 统一解析：兼容字符串与 [TextContent] 两种返回形态
    return parse_sql_result(raw_resp)


# ========== 主执行节点：提取消费信息 + 物价对比 + 生成结构化载荷 ==========
async def execute_node(state: PriceState):
    """
    业务前置约束：
    1. 主调度Agent已完成准入校验：文本必须存在有效消费品类+明确数字金额，否则不会调度本节点；
    2. 本节点仅兜底LLM输出错乱、城市无基准物价、MCP查询异常三类场景；
    城市规则：LLM未提取城市，统一读取全局配置DEFAULT_USER_CITY兜底再查表；
    数据表无对应城市+品类基准价，返回提示，但兜底城市逻辑不变。
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

    # 分支3：LLM输出无有效消费金额
    if amount <= 0:
        payload = {
            KEY_SUCCESS: False,
            KEY_AGENT_TYPE: PRICE_AGENT_ID,
            KEY_MSG: "物价对比失败",
            KEY_DATA: None,
            KEY_ERROR: "未识别到有效消费金额，缺少必备消费价格，无法计算物价溢价"
        }
        return Command(update={"final_struct": payload, "extract_info": extract_res}, goto="reply_node")

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

        # 溢价率计算、档次评价
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
    将标准化final_struct字典转为JSON字符串，和stat_agent完全对齐
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