import json
import pytest
from unittest.mock import patch, AsyncMock
from langgraph.graph import END
from agents.stat_agent.state import StatState
from agents.stat_agent.nodes import (
    parse_query_node,
    execute_query_node,
    verify_query_node,
    reply_node,
    STAT_AGENT_ID,
    MAX_VERIFY_ATTEMPTS
)
from agents.stat_agent.graph import stat_agent_graph

# ===================== Fixture 基础state =====================
@pytest.fixture
def base_stat_state() -> StatState:
    return StatState(
        session_id="test_session_001",
        user_input="",
        session_memory={"history": []},
        target_control=None,
        raw_ir=None,
        final_struct=None
    )

# ===================== 模拟数据库返回对象 =====================
class TextContent:
    def __init__(self, text):
        self.text = text

# ===================== parse_query_node 测试 =====================
@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_parse_normal_ir(mock_call_llm, base_stat_state):
    mock_ir_json = json.dumps({
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "filter": {
            "time_start": "2026-07-01",
            "time_end": "2026-07-31",
            "category_in": ["餐饮"],
            "amount_min": None,
            "amount_max": None
        },
        "agg_ops": [
            {"op": "sum", "target_col": "amount", "group_by": []},
            {"op": "avg", "target_col": "amount", "group_by": []}
        ],
        "require_detail_rows": False
    })
    mock_call_llm.return_value = mock_ir_json
    state = base_stat_state
    state["user_input"] = "统计7月餐饮总花费、平均值"
    cmd = await parse_query_node(state)
    ir = cmd.update["raw_ir"]
    assert ir["need_more_info"] is False
    assert ir["valid"] is True
    assert len(ir["agg_ops"]) == 2
    assert cmd.goto == "execute_query_node"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_parse_need_more_info(mock_call_llm, base_stat_state):
    """need_more_info=true：parse阶段只透传IR，追问逻辑由execute_query_node前置拦截处理"""
    mock_ir_json = json.dumps({
        "need_more_info": True,
        "prompt": "请提供统计时间段",
        "valid": False,
        "filter": {
            "time_start": None,
            "time_end": None,
            "category_in": ["餐饮"],
            "amount_min": None,
            "amount_max": None
        },
        "agg_ops": [],
        "require_detail_rows": False
    })
    mock_call_llm.return_value = mock_ir_json
    state = base_stat_state
    state["user_input"] = "统计餐饮开销"
    cmd = await parse_query_node(state)
    ir = cmd.update["raw_ir"]
    assert ir["need_more_info"] is True
    assert ir["prompt"] == "请提供统计时间段"
    assert cmd.goto == "execute_query_node"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base", side_effect=ValueError("LLM JSON解析失败"))
async def test_parse_llm_exception(mock_call_llm, base_stat_state):
    state = base_stat_state
    state["user_input"] = "统计7月餐饮开销"
    cmd = await parse_query_node(state)
    # 新行为：解析失败降级为默认IR（本月总支出），走 execute_query_node 而非追问
    default_ir = cmd.update["raw_ir"]
    assert isinstance(default_ir, dict)
    assert default_ir["valid"] is True
    assert default_ir["need_more_info"] is False
    assert default_ir["filter"]["category_in"] == []
    assert default_ir["agg_ops"][0]["op"] == "sum"
    assert default_ir["agg_ops"][0]["target_col"] == "amount"
    assert cmd.goto == "execute_query_node"


@pytest.mark.asyncio
async def test_execute_valid_false_branch(base_stat_state):
    """valid=false：execute_query_node前置拦截，不执行SQL，直接返回不支持该操作"""
    state = base_stat_state
    state["raw_ir"] = {
        "need_more_info": False,
        "prompt": "无法识别统计维度",
        "valid": False,
        "filter": {},
        "agg_ops": [],
        "require_detail_rows": False
    }
    cmd = await execute_query_node(state)
    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert payload["msg"] == "不支持该操作"
    assert payload["error"] == "无法识别统计维度"
    assert cmd.goto == "reply_node"

# ===================== execute_query_node 测试 =====================
@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_bill_sql")
async def test_execute_normal_query(mock_call_sql, base_stat_state):
    mock_sum_data = [{"agg_val": 1260}]
    mock_avg_data = [{"agg_val": 63.0}]
    # 适配ast.literal_eval：手动写单引号字符串，不用json.dumps
    mock_call_sql.side_effect = [
        [TextContent(text="[{'agg_val': 1260}]")],
        [TextContent(text="[{'agg_val': 63.0}]")]
    ]
    state = base_stat_state
    state["raw_ir"] = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "filter": {
            "time_start": "2026-07-01",
            "time_end": "2026-07-31",
            "category_in": ["餐饮"],
            "amount_min": None,
            "amount_max": None
        },
        "agg_ops": [
            {"op": "sum", "target_col": "amount", "group_by": []},
            {"op": "avg", "target_col": "amount", "group_by": []}
        ],
        "require_detail_rows": False
    }
    cmd = await execute_query_node(state)
    payload = cmd.update["final_struct"]
    assert payload["success"] is True
    assert payload["data"]["total_amount"] == 1260
    assert payload["data"]["avg_amount"] == 63.0
    # D9 Reflection（M10）：成功结果先进 verify 自检
    assert cmd.goto == "verify_query_node"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_bill_sql", side_effect=Exception("数据库连接失败"))
async def test_execute_sql_exception(mock_call_sql, base_stat_state):
    state = base_stat_state
    state["raw_ir"] = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "filter": {
            "time_start": "2026-07-01",
            "time_end": "2026-07-31",
            "category_in": ["餐饮"],
            "amount_min": None,
            "amount_max": None
        },
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False
    }
    cmd = await execute_query_node(state)
    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert payload["msg"] == "统计数据库查询异常"
    assert "数据库连接失败" in payload["error"]
    assert cmd.goto == "reply_node"


@pytest.mark.asyncio
async def test_execute_need_more_branch(base_stat_state):
    state = base_stat_state
    state["raw_ir"] = {
        "need_more_info": True,
        "prompt": "请提供统计时间段",
        "valid": False,
        "filter": {
            "time_start": None,
            "time_end": None,
            "category_in": ["餐饮"],
            "amount_min": None,
            "amount_max": None
        },
        "agg_ops": [],
        "require_detail_rows": False
    }
    cmd = await execute_query_node(state)
    payload = cmd.update["final_struct"]
    assert payload["success"] is False
    assert payload["error"] == "need_more_info"
    assert payload["data"]["prompt"] == "请提供统计时间段"
    assert cmd.goto == "reply_node"

@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_bill_sql")
async def test_execute_with_detail_rows(mock_call_sql, base_stat_state):
    # 两次聚合 + 一次明细查询，共3次调用
    mock_call_sql.side_effect = [
        [TextContent(text="[{'agg_val': 1260}]")],
        [TextContent(text="[{'agg_val': 63.0}]")],
        [TextContent(text="[{'amount':100,'category':'餐饮','consume_time':'2026-07-01','remark':'午饭'}]")]
    ]
    state = base_stat_state
    state["raw_ir"] = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "filter": {
            "time_start": "2026-07-01",
            "time_end": "2026-07-31",
            "category_in": ["餐饮"],
            "amount_min": None,
            "amount_max": None
        },
        "agg_ops": [
            {"op": "sum", "target_col": "amount", "group_by": []},
            {"op": "avg", "target_col": "amount", "group_by": []}
        ],
        "require_detail_rows": True
    }
    cmd = await execute_query_node(state)
    payload = cmd.update["final_struct"]
    assert payload["success"] is True
    assert payload["data"]["total_amount"] == 1260
    assert payload["data"]["detail_list"][0]["remark"] == "午饭"
    # D9 Reflection（M10）：成功结果先进 verify 自检
    assert cmd.goto == "verify_query_node"

# ===================== reply_node 测试 =====================
@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.a2a_bus.send_result")
async def test_reply_node_send_result(mock_send, base_stat_state):
    state = base_stat_state
    state["session_id"] = "s_001"
    state["task_id"] = "task_001"
    state["final_struct"] = {
        "success": True,
        "agent_type": STAT_AGENT_ID,
        "msg": "月度消费统计",
        "data": {"total_amount": 1260},
        "error": None
    }
    cmd = await reply_node(state)
    mock_send.assert_called_once()
    call_args = mock_send.call_args.kwargs
    # ★M2 验收③：send_result 契约为 (task_id, result)，target_agent/session_id 死参数已删除
    assert call_args["task_id"] == "task_001"
    assert isinstance(call_args["result"], str)
    # 使用导入的END常量匹配，不再硬编码"END"
    assert cmd.goto == "__end__"


# ===================== verify_query_node（D9 Reflection · M10）测试 =====================

def _ok_state(base_stat_state, data=None, **extra) -> StatState:
    """构造 execute 成功后的状态（final_struct.success=True）"""
    st = base_stat_state
    st["user_input"] = "统计7月最大的一笔餐饮支出"
    st["raw_ir"] = {
        "need_more_info": False,
        "valid": True,
        "filter": {"time_start": "2026-07-01", "time_end": "2026-07-31",
                   "category_in": ["餐饮"], "amount_min": None, "amount_max": None},
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False,
    }
    st["final_struct"] = {
        "success": True,
        "agent_type": STAT_AGENT_ID,
        "msg": "月度消费统计",
        "data": data if data is not None else {"total_amount": 1260},
        "error": None,
    }
    for k, v in extra.items():
        st[k] = v
    return st


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_verify_pass_goes_reply(mock_call_llm, base_stat_state):
    """校验通过 → 直接发结果（goto reply_node）"""
    mock_call_llm.return_value = json.dumps({"pass": True, "issues": [], "feedback": ""})
    state = _ok_state(base_stat_state)
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_verify_fail_retries_parse(mock_call_llm, base_stat_state):
    """校验不过且未达上限 → 写反馈，回 parse_query_node 重生成 IR"""
    mock_call_llm.return_value = json.dumps({
        "pass": False,
        "issues": ["结果缺最大单笔：用户问最大支出但 agg_ops 无 max"],
        "feedback": "agg_ops 增加 {'op': 'max', 'target_col': 'amount', 'group_by': []}",
    })
    state = _ok_state(base_stat_state)
    cmd = await verify_query_node(state)
    assert cmd.goto == "parse_query_node"
    assert cmd.update["verify_decision"] == "retry"
    assert cmd.update["verify_attempt"] == 1
    assert "max" in cmd.update["verify_feedback"]
    # 当前确定性结果保留在 state 中（LangGraph 合并 update），兜底可发


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_verify_fail_reaches_limit_falls_back(mock_call_llm, base_stat_state):
    """校验不过且已达重试上限 → 转兜底直发（不无限循环）"""
    mock_call_llm.return_value = json.dumps({
        "pass": False,
        "issues": ["始终不一致"],
        "feedback": "仍不匹配",
    })
    state = _ok_state(base_stat_state, verify_attempt=MAX_VERIFY_ATTEMPTS)
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"
    # 兜底发送的是当前确定性结果
    assert cmd.update["verify_feedback"] is None


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base", side_effect=Exception("verify LLM调用失败"))
async def test_verify_llm_exception_falls_back(mock_call_llm, base_stat_state):
    """校验 LLM 调用异常 → 跳过校验直发（verify 是增强项，不阻断任务）"""
    state = _ok_state(base_stat_state)
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_verify_skips_for_failed_payload(mock_call_llm, base_stat_state):
    """execute 失败载荷（success=False）不参与语义自检 → 不调 LLM 直达 reply"""
    state = base_stat_state
    state["final_struct"] = {
        "success": False, "agent_type": STAT_AGENT_ID, "msg": "统计数据库查询异常",
        "data": None, "error": "boom",
    }
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    mock_call_llm.assert_not_called()


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_parse_injects_verify_feedback(mock_call_llm, base_stat_state):
    """parse 收到 verify_feedback → 注入到 user prompt（LLM 据此修正 IR）"""
    mock_ir_json = json.dumps({
        "need_more_info": False, "prompt": "", "valid": True,
        "filter": {"time_start": "2026-07-01", "time_end": "2026-07-31",
                   "category_in": ["餐饮"], "amount_min": None, "amount_max": None},
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False,
    })
    mock_call_llm.return_value = mock_ir_json
    state = base_stat_state
    state["user_input"] = "统计7月最大的一笔餐饮支出"
    state["verify_feedback"] = "agg_ops 增加 max 算子"
    cmd = await parse_query_node(state)
    # 反馈已消费（清空），避免泄漏到下一轮
    assert cmd.update["verify_feedback"] is None
    # 校验反馈确实注入到了调用 LLM 的 user prompt
    user_prompt_arg = mock_call_llm.call_args.args[1]
    assert "agg_ops 增加 max 算子" in user_prompt_arg


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_parse_normalizes_synonym_category(mock_call_llm, base_stat_state):
    """类目近义表达（吃饭→餐饮）→ _fix 归一为标准类目（M10 verify 收敛前提）"""
    ir_no_cat = json.dumps({
        "need_more_info": False, "prompt": "", "valid": True,
        "filter": {"time_start": "2026-07-01", "time_end": "2026-07-31",
                   "category_in": [], "amount_min": None, "amount_max": None},
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False,
    })
    mock_call_llm.return_value = ir_no_cat
    state = base_stat_state
    state["user_input"] = "统计一下7月吃饭一共花了多少钱"
    cmd = await parse_query_node(state)
    assert cmd.update["raw_ir"]["filter"]["category_in"] == ["餐饮"]


# ===================== 整图自纠闭环（D9 验收核心） =====================

@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.a2a_bus.send_result")
@patch("agents.stat_agent.nodes.mcp_client.call_bill_sql")
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_graph_verify_self_correction_loop(mock_call_llm, mock_call_sql, mock_send, base_stat_state):
    """结果缺陷自纠（近义类目错位——`_fix_stat_ir` 规则兜不住的场景）：
    用户问"吃饭"（=餐饮）→ LLM 漏输出 category_in（全类目统计，口径与问题不符）
    → verify 校验不过反馈 → 重生成限定餐饮 → 校验通过 → 结果口径正确"""
    ir_all_cat = {
        "need_more_info": False, "prompt": "", "valid": True,
        "filter": {"time_start": "2026-07-01", "time_end": "2026-07-31",
                   "category_in": [], "amount_min": None, "amount_max": None},
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False,
    }
    ir_dining_cat = {
        "need_more_info": False, "prompt": "", "valid": True,
        "filter": {"time_start": "2026-07-01", "time_end": "2026-07-31",
                   "category_in": ["餐饮"], "amount_min": None, "amount_max": None},
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False,
    }
    mock_call_llm.side_effect = [
        json.dumps(ir_all_cat),  # parse 第1轮（缺陷 IR：漏餐饮类目）
        json.dumps({"pass": False,
                    "issues": ["口径不一致：用户问'吃饭'（餐饮）花费，但 IR 未限定 category_in，统计为全类目"],
                    "feedback": "filter.category_in 应为 ['餐饮']，重查餐饮口径"}),  # verify 第1轮
        json.dumps(ir_dining_cat),                                 # parse 第2轮（修正 IR）
        json.dumps({"pass": True, "issues": [], "feedback": ""}),  # verify 第2轮
    ]
    mock_call_sql.side_effect = [
        [TextContent(text="[{'agg_val': 1260}]")],  # execute 第1轮：全类目 sum（口径错误）
        [TextContent(text="[{'agg_val': 1000}]")],  # execute 第2轮：餐饮 sum（口径正确）
    ]
    state = base_stat_state
    state["user_input"] = "统计一下7月吃饭一共花了多少钱"
    state["task_id"] = "task_verify_1"
    state["verify_attempt"] = 0
    state["verify_feedback"] = None
    final_state = await stat_agent_graph.ainvoke(state)

    data = final_state["final_struct"]["data"]
    # 自纠成功：结果为餐饮口径（total=1000），非全类目（1260）
    assert data["total_amount"] == 1000
    # 曾经历 1 次失败重生成（verify 反馈 → parse 修正生效）
    assert final_state["verify_attempt"] == 1
    # 反馈已被消费
    assert final_state["verify_feedback"] is None
    # 只发送一次结果
    mock_send.assert_called_once()


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.a2a_bus.send_result")
@patch("agents.stat_agent.nodes.mcp_client.call_bill_sql")
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_graph_verify_attempt_limit_falls_back(mock_call_llm, mock_call_sql, mock_send, base_stat_state):
    """verify 连续不过达到上限 → 转兜底发送当前确定性结果（链路不崩、有结果）"""
    ir = {
        "need_more_info": False, "prompt": "", "valid": True,
        "filter": {"time_start": "2026-07-01", "time_end": "2026-07-31",
                   "category_in": [], "amount_min": None, "amount_max": None},
        "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
        "require_detail_rows": False,
    }
    verdict_fail = json.dumps({"pass": False, "issues": ["仍不一致"], "feedback": "请再修正"})
    mock_call_llm.side_effect = [
        json.dumps(ir),       # parse 1
        verdict_fail,          # verify 1（attempt 0→1）
        json.dumps(ir),       # parse 2
        verdict_fail,          # verify 2（attempt 1→2）
        json.dumps(ir),       # parse 3
        verdict_fail,          # verify 3（attempt 2 达上限 → reply）
    ]
    mock_call_sql.side_effect = [
        [TextContent(text="[{'agg_val': 1260}]")],
        [TextContent(text="[{'agg_val': 1260}]")],
        [TextContent(text="[{'agg_val': 1260}]")],
    ]
    state = base_stat_state
    state["user_input"] = "统计本月总支出"
    state["task_id"] = "task_verify_2"
    state["verify_attempt"] = 0
    state["verify_feedback"] = None
    final_state = await stat_agent_graph.ainvoke(state)

    # 兜底：确定性结果仍发送（不编造、不阻断）
    assert final_state["final_struct"]["success"] is True
    assert final_state["final_struct"]["data"]["total_amount"] == 1260
    assert final_state["verify_attempt"] == MAX_VERIFY_ATTEMPTS
    mock_send.assert_called_once()