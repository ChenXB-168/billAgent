# ==============================================
# 物价 Agent 节点单元测试：execute_node 提取/查表/异常分支与 reply_node 回传
# ==============================================
import json
import pytest
from unittest.mock import patch, AsyncMock
from langgraph.graph import END
from agents.price_agent.state import PriceState
from agents.price_agent.nodes import (
    execute_node,
    reply_node,
    PRICE_AGENT_ID
)
from config.config import DEFAULT_USER_CITY

# ===================== Fixture 基础price state =====================
@pytest.fixture
def base_price_state() -> PriceState:
    """
    函数功能与逻辑描述：
        提供一份基础的 PriceState：raw_segments/extract_info/final_struct/error_stack 均为空，
        各用例按需覆写字段，避免重复构造状态。function 作用域，无清理逻辑（纯内存对象）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        PriceState：初始化后的物价 Agent 状态对象（对齐 PriceState dataclass 定义）。
    """
    return PriceState(
        raw_segments=[],
        extract_info=None,
        final_struct=None,
        error_stack=None
    )

# ===================== 模拟MCP返回TextContent对象（和stat测试完全复用） =====================
class TextContent:
    """
    函数/类功能与逻辑描述：
        模拟 MCP 工具返回的 TextContent 对象（仅含 text 属性），用于伪造 call_bill_sql
        的返回形态，使被测代码走「列表内 TextContent.text 携带单引号列表字面量」的解析路径。
    构造入参说明：
        text：字符串，承载待解析的文本内容（如 "[{'avg_price': 28.0}]"）。
    返回值说明：
        构造返回 TextContent 实例，其 text 属性为传入文本。
    """
    def __init__(self, text):
        """
        函数功能与逻辑描述：
            构造 TextContent 替身：把传入文本保存到实例属性 `text`，使被测代码可沿
            「列表内 TextContent.text 携带单引号列表字面量」的路径解析基准均价；
            纯数据承载、无校验、无副作用。
        入参说明：
            text：字符串，承载待解析的文本内容（如 "[{'avg_price': 28.0}]"）。
        返回值说明：
            无（仅初始化实例，无副作用）。
        """
        self.text = text

# ===================== M18：价位查询模式（有品类、无金额） =====================
@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_price_query_mode_without_amount(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        ★M18 核心用例：验证「有品类、无金额」时走**价位查询**分支——
        不再直接失败，而是查 city_price 返回该品类**本地参考均价**；
        `premium_rate` 为 None（无用户金额可比，不做溢价率判断）。
        起因：M17 契约化后主 Agent 会在 fit_level=inferable 场景规划"无金额的比价"
        （如"我想去唱歌，预算多少合适"），若此处仍失败则"花费前决策辅助"链路断裂。
    入参说明：
        mock_call_llm：call_llm_base 替身，返回无金额的抽取 JSON。
        base_price_state：pytest fixture 提供的基础 PriceState。
    返回值说明：
        无（断言通过即成功）。
    """
    llm_out = json.dumps({"city": "深圳", "category": "娱乐", "amount": 0.0})
    mock_call_llm.return_value = llm_out

    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql") as mock_sql:
        mock_sql.return_value = [TextContent(text="[{'avg_price': 120.0}]")]
        state = base_price_state
        state["raw_segments"] = ["我想去唱歌，预算多少合适"]
        cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is True
    assert payload["msg"] == "价位查询"
    data = payload["data"]
    assert data["category"] == "娱乐"
    assert data["user_consume_amount"] is None       # ★ 无用户金额（未编造）
    assert data["city_base_avg_price"] == 120.0
    assert data["premium_rate"] is None              # ★ 不做溢价率判断
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_price_query_mode_no_base_data(mock_call_llm, base_price_state):
    """M18：价位查询模式下查不到基准价 → 仍如实失败（**不编价格**）。"""
    llm_out = json.dumps({"city": "深圳", "category": "娱乐", "amount": 0.0})
    mock_call_llm.return_value = llm_out

    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql") as mock_sql:
        mock_sql.return_value = []
        state = base_price_state
        state["raw_segments"] = ["我想去唱歌"]
        cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert "暂无城市基准物价数据" in payload["error"]


# ===================== execute_node 核心场景测试 =====================
@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_normal_success(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        验证正常场景：LLM 提取出城市「上海」、品类「餐饮」、金额 32.5，查表命中基准均价 28.0，
        execute_node 返回 success=True 的标准载荷（含 user_consume_amount=32.5、
        city_base_avg_price=28.0、premium_rate 为 float），并跳转 reply_node。
        覆盖场景：城市/品类/金额齐备且基准存在的成功路径。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，返回提取 JSON 字符串。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    llm_out = json.dumps({"city": "上海", "category": "餐饮", "amount": 32.5})
    mock_call_llm.return_value = llm_out

    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql") as mock_sql:
        mock_sql.return_value = [TextContent(text="[{'avg_price': 28.0}]")]
        state = base_price_state
        state["raw_segments"] = ["上海喝咖啡花了32.5"]
        cmd = await execute_node(state)

    # Command.update是dict，用字典访问
    payload = cmd.update["final_struct"]
    extract_info = cmd.update["extract_info"]
    assert extract_info["city"] == "上海"
    assert extract_info["category"] == "餐饮"
    assert extract_info["amount"] == 32.5
    assert payload["success"] is True
    assert payload["agent_type"] == PRICE_AGENT_ID
    assert payload["error"] is None
    data = payload["data"]
    assert data["city"] == "上海"
    assert data["category"] == "餐饮"
    assert data["user_consume_amount"] == 32.5
    assert data["city_base_avg_price"] == 28.0
    assert isinstance(data["premium_rate"], float)
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_no_city_use_config_default(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        验证城市兜底：LLM 未输出城市（city 为空串）时，execute_node 读取全局配置
        DEFAULT_USER_CITY（深圳）兜底并查表（本例命中基准 100.0），返回 success=True，
        data 中 city 等于 DEFAULT_USER_CITY。
        覆盖场景：LLM 城市缺省 → 配置默认城市兜底。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，返回 city 为空的提取 JSON。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    llm_out = json.dumps({"city": "", "category": "餐饮", "amount": 120.0})
    mock_call_llm.return_value = llm_out

    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql") as mock_sql:
        mock_sql.return_value = [TextContent(text="[{'avg_price': 100.0}]")]
        state = base_price_state
        state["raw_segments"] = ["晚上吃火锅花了120"]
        cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    data = payload["data"]
    assert data["city"] == DEFAULT_USER_CITY
    assert payload["success"] is True


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_no_category_error(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        验证品类缺失拦截：LLM 输出 category 为空时，execute_node 返回 success=False、
        error 含「缺少必备消费项目」、data 为 None，并跳转 reply_node。
        覆盖场景：无有效消费品类的错误分支。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，返回 category 为空的提取 JSON。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    llm_out = json.dumps({"city": "深圳", "category": "", "amount": 50.0})
    mock_call_llm.return_value = llm_out
    state = base_price_state
    state["raw_segments"] = ["随便一段话50块"]
    cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert "缺少必备消费项目" in payload["error"]
    assert payload["data"] is None
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_amount_zero_goes_to_price_query(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        ★M18 行为变更（原 test_execute_amount_zero_error 作废）：
        金额缺失（amount=0，如"买衣服花了几百"）**不再直接失败**，
        而是走「价位查询」分支，返回该品类本地参考均价（`premium_rate=None`）+ 引导补金额。
        理由：M17 契约化后主 Agent 会在 fit_level=inferable 场景规划无金额的 price；
        若此处失败，"花费前决策辅助"链路断裂。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，返回 amount=0.0 的提取 JSON。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    llm_out = json.dumps({"city": "深圳", "category": "购物", "amount": 0.0})
    mock_call_llm.return_value = llm_out

    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql") as mock_sql:
        mock_sql.return_value = [TextContent(text="[{'avg_price': 140.0}]")]
        state = base_price_state
        state["raw_segments"] = ["买衣服花了几百"]
        cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is True
    assert payload["msg"] == "价位查询"
    assert payload["data"]["city_base_avg_price"] == 140.0
    assert payload["data"]["premium_rate"] is None
    assert payload["data"]["user_consume_amount"] is None
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_city_category_no_base_data(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        验证基准数据缺失：城市（兜底）与品类均合法但查表无对应基准均价时，execute_node 返回
        success=False、error 含「暂无城市基准物价数据」、data 中 city 为 DEFAULT_USER_CITY 且
        premium_rate 为 None，并跳转 reply_node。
        覆盖场景：查表命中空结果 → base_avg<=0 的分支。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，返回 city 为空、品类「娱乐」的提取 JSON。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    llm_out = json.dumps({"city": "", "category": "娱乐", "amount": 88.0})
    mock_call_llm.return_value = llm_out
    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql") as mock_sql:
        mock_sql.return_value = []
        state = base_price_state
        state["raw_segments"] = ["周末看演出花了88"]
        cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert "暂无城市基准物价数据" in payload["error"]
    assert payload["data"]["city"] == DEFAULT_USER_CITY
    assert payload["data"]["premium_rate"] is None
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base", side_effect=ValueError("LLM JSON解析失败"))
async def test_execute_llm_call_exception(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        验证 LLM 调用异常：call_llm_base 抛出 ValueError（LLM JSON 解析失败）时，execute_node
        捕获后返回 success=False、error 含「物价参数提取失败」与原始异常文本，
        并把异常堆栈写入 state.error_stack，跳转 reply_node。
        覆盖场景：LLM 调用/解析彻底失败的分支（重试 1 次后仍失败）。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，side_effect=ValueError 模拟调用异常。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_price_state
    state["raw_segments"] = ["深圳奶茶花了30"]
    cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert "物价参数提取失败" in payload["error"]
    assert "LLM JSON解析失败" in payload["error"]
    assert cmd.update["error_stack"] is not None
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_sql_query_exception(mock_call_llm, base_price_state):
    """
    函数功能与逻辑描述：
        验证基准查询异常：LLM 提取正常但 call_bill_sql 抛异常（数据库连接失败）时，
        execute_node 捕获后返回 success=False、error 含「物价基准查询失败」与异常信息，
        data 中 base_avg_price 为 None，跳转 reply_node。
        覆盖场景：SQL 查询抛异常的兜底分支。
    入参说明：
        mock_call_llm：@patch 注入的 call_llm_base 替身，返回合法提取 JSON。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    llm_out = json.dumps({"city": "深圳", "category": "餐饮", "amount": 30.0})
    mock_call_llm.return_value = llm_out
    with patch("agents.price_agent.nodes.mcp_client.call_bill_sql", side_effect=Exception("数据库连接失败")) as mock_sql:
        state = base_price_state
        state["raw_segments"] = ["深圳奶茶花了30"]
        cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert "物价基准查询失败" in payload["error"]
    assert "数据库连接失败" in payload["error"]
    assert payload["data"]["base_avg_price"] is None
    assert cmd.goto == "reply_node"

# ===================== reply_node 出口测试 =====================
@pytest.mark.asyncio
# patch 目标为 nodes 模块内已导入的 a2a_bus 对象（其来源是 mcpGateway.a2a_queue）
@patch("agents.price_agent.nodes.a2a_bus.send_result")
async def test_reply_node_a2a_send(mock_send, base_price_state):
    """
    函数功能与逻辑描述：
        验证 reply_node 出口：把 final_struct 序列化为 JSON 字符串，经 a2a_bus.send_result
        回传给调度编排器；断言 send_result 被调用一次、task_id 透传为 None、result 为 str，
        节点以 END 结束。
        覆盖场景：携带完整物价评价字段的正常回传路径。
    入参说明：
        mock_send：@patch 注入的 a2a_bus.send_result 替身，用于断言调用参数。
        base_price_state：pytest fixture 注入，提供基础 PriceState（用例内覆写 task_id/final_struct）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_price_state
    state["session_id"] = "sess_test"
    state["task_id"] = None
    state["final_struct"] = {
        "success": True,
        "agent_type": PRICE_AGENT_ID,
        "data": {
            "city": "深圳",
            "category": "餐饮",
            "user_consume_amount": 32.5,
            "city_base_avg_price": 28.0,
            "premium_rate": 16.07,
            "consume_level": "偏高",
            "evaluate_conclusion": "高于城市基准均价，存在小幅消费溢价"
        },
        "error": None
    }
    cmd = await reply_node(state)
    mock_send.assert_called_once()
    call_args = mock_send.call_args.kwargs
    # ★M2 验收③：send_result 契约为 (task_id, result)，target_agent/session_id 死参数已删除
    assert call_args["task_id"] is None
    assert isinstance(call_args["result"], str)
    assert cmd.goto == END
