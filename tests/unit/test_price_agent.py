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
    """基础空物价Agent状态，对齐PriceState dataclass定义"""
    return PriceState(
        raw_segments=[],
        extract_info=None,
        final_struct=None,
        error_stack=None
    )

# ===================== 模拟MCP返回TextContent对象（和stat测试完全复用） =====================
class TextContent:
    def __init__(self, text):
        self.text = text

# ===================== execute_node 核心场景测试 =====================
@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_normal_success(mock_call_llm, base_price_state):
    """正常场景：有城市、有效品类、有效金额，查表存在基准均价，返回成功载荷"""
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
    """LLM未输出城市，读取全局配置DEFAULT_USER_CITY（深圳）兜底，查表存在基准"""
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
    """LLM输出category空，返回缺少消费项目错误"""
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
async def test_execute_amount_zero_error(mock_call_llm, base_price_state):
    """LLM提取amount=0，返回缺少有效消费价格错误"""
    llm_out = json.dumps({"city": "深圳", "category": "购物", "amount": 0.0})
    mock_call_llm.return_value = llm_out
    state = base_price_state
    state["raw_segments"] = ["买衣服花了几百"]
    cmd = await execute_node(state)

    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert "缺少必备消费价格" in payload["error"]
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
@patch("agents.price_agent.nodes.mcp_client.call_llm_base")
async def test_execute_city_category_no_base_data(mock_call_llm, base_price_state):
    """兜底城市+合法品类，但数据表无对应基准均价，返回提示错误"""
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
    """LLM调用抛出异常，堆栈存入state.error_stack，返回提取失败载荷"""
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
    """LLM提取正常，SQL数据库查询抛异常，捕获返回错误载荷"""
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
# 修复mock路径：a2a_bus 顶层变量，不是a2a_queue子模块
@patch("agents.price_agent.nodes.a2a_bus.send_result")
async def test_reply_node_a2a_send(mock_send, base_price_state):
    """测试载荷序列化并通过A2A发送给调度Orchestrator"""
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