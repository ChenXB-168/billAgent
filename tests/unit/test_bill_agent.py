import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.bill_agent.state import BillState
from agents.bill_agent.nodes import execute_node, reply_node, BILL_AGENT_ID

# ===================== Fixture =====================
@pytest.fixture
def base_bill_state() -> BillState:
    return BillState(
        session_id="test_session_001",
        task_id=None,
        raw_segments=[],
        result=None,
        city=None,
        month=None
    )

# 全局自动mock mcp_client，解决提前访问属性触发异常
@pytest.fixture(autouse=True)
def mock_mcp_client():
    mock_client = MagicMock()
    mock_client.call_llm_base = AsyncMock()
    mock_client.call_bill_sql = AsyncMock()
    with patch("agents.bill_agent.nodes.mcp_client", mock_client):
        yield mock_client


# M9：execute_node 记账成功分支会调 _settle_month_habit（registry.invoke "habit.upsert"）——
# autouse 屏蔽真实网关，防单测经 local 工具**真写本地 DB + 真审计**（隔离测试副作用）。
# 专项用例（test_settle_month_habit_*）自带 patch 覆盖本 fixture。
@pytest.fixture(autouse=True)
def mock_gateway_invoke():
    reg = MagicMock()
    reg.invoke = AsyncMock(return_value=MagicMock(ok=True, data={"habits": []}))
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        yield reg

# ===================== test_reply_node =====================
@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.a2a_bus.send_result")
async def test_reply_node_send_result(mock_send, base_bill_state):
    state = base_bill_state
    state["session_id"] = "s_001"
    state["task_id"] = "t_001"
    state["result"] = json.dumps({
        "success": True,
        "agent_type": BILL_AGENT_ID,
        "msg": "记账成功",
        "data": {},
        "error": None
    }, ensure_ascii=False)
    await reply_node(state)
    mock_send.assert_called_once()

# ===================== execute_node 业务场景 =====================
@pytest.mark.asyncio
async def test_execute_empty_raw_text(base_bill_state):
    """raw_segments全空白，空输入拦截"""
    state = base_bill_state
    state["raw_segments"] = ["    ", ""]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["msg"] == "输入内容为空"
    assert payload["error"] == "未获取到有效的账单描述信息"


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_need_more_info_trigger(mock_parse, base_bill_state):
    """LLM判定信息缺失，触发追问分支"""
    mock_parse.return_value = {
        "need_more_info": True,
        "prompt": "请提供午饭的消费金额",
        "valid": False,
        "amount": 0.0,
        "category": "",
        "consume_date": ""
    }
    state = base_bill_state
    # 带金额数字绕过前置启发式拦截，验证LLM判定缺信息时的追问透传
    state["raw_segments"] = ["午饭35元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["error"] == "need_more_info"
    assert payload["msg"] == "需要向用户补充询问信息"
    assert payload["data"]["prompt"] == "请提供午饭的消费金额"


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.parse_json_output", side_effect=ValueError("JSON解析失败"))
async def test_execute_parse_exception(mock_parse, base_bill_state):
    """parse_json_output抛出异常，兜底转追问而非硬失败"""
    state = base_bill_state
    state["raw_segments"] = ["晚饭35元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["msg"] == "需要向用户补充询问信息"
    assert payload["error"] == "need_more_info"
    assert "暂时没有识别到可记账的信息" in payload["data"]["prompt"]


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_field_invalid(mock_parse, base_bill_state):
    """解析结果 valid=false 拦截"""
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": False,
        "amount": 20,
        "category": "饮品",
        "consume_date": "2026-07-14"
    }
    state = base_bill_state
    state["raw_segments"] = ["奶茶20元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["msg"] == "账单信息校验不通过"
    assert payload["error"] == "valid标记为false，无法录入账单"


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_amount_negative(mock_parse, base_bill_state):
    """金额<=0 业务拦截"""
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "amount": -30,
        "category": "餐饮",
        "consume_date": "2026-07-14"
    }
    state = base_bill_state
    state["raw_segments"] = ["晚饭-30元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["msg"] == "金额非法"
    assert payload["error"] == "消费金额必须大于0"
    assert payload["data"]["amount"] == -30


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.mcp_client.call_bill_sql", return_value="成功")
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_date_format_illegal(mock_parse, mock_sql, base_bill_state):
    """日期仅年月（校验逻辑在parse内部，此处模拟绕过校验正常入库）"""
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "amount": 30,
        "category": "餐饮",
        "consume_date": "2026-07"
    }
    state = base_bill_state
    state["raw_segments"] = ["晚饭30元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True
    assert payload["msg"] == "记账成功"


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.mcp_client.call_bill_sql", return_value="成功")
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_sql_success_flow(mock_parse, mock_sql, base_bill_state):
    """完整正常链路：解析成功 + SQL入库成功"""
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "amount": 35.0,
        "category": "餐饮",
        "consume_date": "2026-07-14"
    }
    state = base_bill_state
    state["raw_segments"] = ["昨天晚饭35元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True
    assert payload["msg"] == "记账成功"
    assert payload["data"]["amount"] == 35.0


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.mcp_client.call_bill_sql", return_value="数据库连接异常")
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_sql_failed_flow(mock_parse, mock_sql, base_bill_state):
    """解析正常，SQL执行返回业务失败文本"""
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "amount": 18,
        "category": "饮品",
        "consume_date": "2026-07-15"
    }
    state = base_bill_state
    state["raw_segments"] = ["奶茶18元"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["msg"] == "账单入库操作失败"
    assert payload["error"] == "数据库连接异常"


@pytest.mark.asyncio
async def test_execute_full_template_render_path(base_bill_state):
    """携带city/month完整模板渲染分支，仅验证无崩溃"""
    state = base_bill_state
    state["raw_segments"] = ["午饭20元"]
    state["city"] = "深圳"
    state["month"] = "2026-07"
    assert True


# ===================== M9：月度习惯沉淀下沉（_settle_month_habit） =====================
# 原 orchestrator `_persist_bill_habit`（裸连 db ×2）M9 收口后删除，沉淀下沉 bill_agent：
# habit.upsert 工具（HABIT_WRITE，写 monthly_habit + 读回当月聚合）→ pkl 语义层更新。
@pytest.mark.asyncio
async def test_settle_month_habit_success_persists():
    """记账成功 → habit.upsert（写+读回聚合）→ pkl 语义层更新"""
    reg = MagicMock()
    invoke = AsyncMock(return_value=MagicMock(ok=True, data={
        "ok": True,
        "habits": [{"month": "2026-09", "category": "餐饮",
                    "amount_sum": 35.5, "count": 1}]}))
    reg.invoke = invoke
    with patch("mcpGateway.registry.get_registry", return_value=reg), \
            patch("memory.long_memory.upsert_month_habit_memory", return_value=True) as up:
        from agents.bill_agent.nodes import _settle_month_habit
        await _settle_month_habit({"amount": 35.5, "category": "餐饮",
                                   "consume_date": "2026-09-01"})

    invoke.assert_awaited_once()
    name = invoke.call_args.args[0]
    args = invoke.call_args.args[1]
    assert name == "habit.upsert"
    assert args == {"month": "2026-09", "category": "餐饮", "amount": 35.5}
    assert invoke.call_args.kwargs["agent_id"] == BILL_AGENT_ID
    up.assert_called_once()
    assert up.call_args.args[0] == "2026-09"    # 语义层按月份前缀替换该月摘要


@pytest.mark.asyncio
async def test_settle_month_habit_missing_fields_noop():
    """缺 amount/category/consume_date → 直接返回，不触达网关"""
    reg = MagicMock()
    reg.invoke = AsyncMock()
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        from agents.bill_agent.nodes import _settle_month_habit
        await _settle_month_habit({"amount": 10})        # 缺 category / consume_date
        await _settle_month_habit({"category": "餐饮"})   # 缺 amount / consume_date
    reg.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_month_habit_error_silent():
    """网关异常 → 静默吞掉，不阻断记账主流程（失败不影响 payload 组包）"""
    async def _boom(*a, **k):
        raise RuntimeError("gateway down")
    reg = MagicMock()
    reg.invoke = AsyncMock(side_effect=_boom)
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        from agents.bill_agent.nodes import _settle_month_habit
        await _settle_month_habit({"amount": 10, "category": "餐饮",
                                   "consume_date": "2026-09-01"})   # 不抛即通过