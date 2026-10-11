# ==============================================
# stat_agent（统计 Agent）单元测试
# 覆盖 parse→execute→verify→reply 四节点与 D9 Reflection 结果自纠闭环；
# mock 掉 LLM / SQL 通道，不触达真实模型与数据库。
# ==============================================
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
    """
    函数功能与逻辑描述：
        提供一份结构完整的空 StatState 基础状态（session_id 固定为 test_session_001），
        供各用例在其上覆写字段；function 作用域，每个用例独立，无清理逻辑。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        StatState：含默认空字段的统计 Agent 状态。
    """
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
    """
    函数/类功能与逻辑描述：
        模拟 MCP 工具返回的文本内容容器，仅持有 text 字段；
        供 call_llm_base / call_bill_sql 等替身返回值对齐真实 MCP 返回形态（与 price/finance 测试一致）。
    构造入参说明：
        text：工具返回的文本内容（IR JSON 字符串或单引号包裹的 SQL 结果串）。
    返回值说明：
        构造返回 TextContent 实例。
    """

    def __init__(self, text):
        """
        函数功能与逻辑描述：
            将入参文本记录到实例属性 text，作为 MCP 返回内容的模拟载体。
        入参说明：
            text：工具返回的文本内容。
        返回值说明：
            无（仅初始化实例，无副作用）。
        """
        self.text = text

# ===================== parse_query_node 测试 =====================
def test_merge_multi_groups_keeps_both_ranges():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（多组改造，方案 `设计/13` S6，决策 **D-3 = 原始 groups[]**）：
        多组时 `merge_stat_data` 必须输出 `{"groups": [...]}`，**每组各自保留区间与统计**，
        不得合并成一个区间（否则"本月 vs 上月"对比退化为"同一区间算两遍"）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    from agents.stat_agent.nodes import merge_stat_data

    group_results = [
        {"filter": {"time_start": "2026-09-01", "time_end": "2026-09-19", "category_in": []},
         "agg_ops": [({"op": "sum", "target_col": "amount", "group_by": []},
                      [{"agg_val": 1260.0}])],
         "detail_rows": []},
        {"filter": {"time_start": "2026-08-01", "time_end": "2026-08-31", "category_in": []},
         "agg_ops": [({"op": "sum", "target_col": "amount", "group_by": []},
                      [{"agg_val": 900.0}])],
         "detail_rows": []},
    ]
    data = merge_stat_data({}, [], [], group_results=group_results)
    assert "groups" in data, f"多组应输出 groups[]（D-3）: {data}"
    assert len(data["groups"]) == 2, data
    assert data["groups"][0]["time_range"] == ["2026-09-01", "2026-09-19"], data["groups"][0]
    assert data["groups"][1]["time_range"] == ["2026-08-01", "2026-08-31"], data["groups"][1]
    assert data["groups"][0]["total_amount"] == 1260.0, data["groups"][0]
    assert data["groups"][1]["total_amount"] == 900.0, data["groups"][1]


def test_merge_single_group_unchanged_shape():
    """
    函数功能与逻辑描述：
        ★兼容性用例：只有一组时 `merge_stat_data` 必须返回**与改造前一致的单组结构**
        （不含 groups 外壳），保证既有渲染路径与单测行为不变。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    from agents.stat_agent.nodes import merge_stat_data

    group_results = [
        {"filter": {"time_start": "2026-09-01", "time_end": "2026-09-19", "category_in": ["餐饮"]},
         "agg_ops": [({"op": "sum", "target_col": "amount", "group_by": []},
                      [{"agg_val": 100.0}])],
         "detail_rows": []},
    ]
    data = merge_stat_data({}, [], [], group_results=group_results)
    assert "groups" not in data, f"单组不应包 groups 外壳: {data}"
    assert data["total_amount"] == 100.0, data
    assert data["time_range"] == ["2026-09-01", "2026-09-19"], data
    assert data["query_categories"] == ["餐饮"], data


def test_fix_stat_ir_normalizes_legacy_flat_to_groups():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（S3）：`_fix_stat_ir` 必须把**旧扁平格式**（顶层 filter/agg_ops）
        归一为 `groups` 列表；且**仅一组时回写顶层**，保证尚未改造的消费方仍可读。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    from agents.stat_agent.nodes import _fix_stat_ir

    legacy = {"filter": {"time_start": "2026-09-01", "time_end": "2026-09-19",
                         "category_in": ["餐饮"], "amount_min": None, "amount_max": None},
              "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
              "require_detail_rows": False}
    out = _fix_stat_ir("本月餐饮花了多少", legacy)
    assert isinstance(out.get("groups"), list) and len(out["groups"]) == 1, out
    assert out["groups"][0]["filter"]["category_in"] == ["餐饮"], out["groups"][0]
    # 单组 → 回写顶层（向后兼容）
    assert out["filter"] is out["groups"][0]["filter"], "单组应回写顶层 filter"
    assert out["agg_ops"] == out["groups"][0]["agg_ops"], out


def test_fix_stat_ir_multi_groups_not_overwritten_by_whole_sentence():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（S3 **防串组**，方案 `13` §11.3 C-3）：多组时**不得按整句意图改写时间**。
        输入"这个月跟上个月比"同时含"这个月"与"上个月"；若按整句关键词统一改写，
        两组会被改成同一区间 → 对比失效。本用例锁定：两组区间**各自保留**。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    from agents.stat_agent.nodes import _fix_stat_ir

    multi = {"groups": [
        {"filter": {"time_start": "2026-09-01", "time_end": "2026-09-19",
                    "category_in": [], "amount_min": None, "amount_max": None},
         "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
         "require_detail_rows": False},
        {"filter": {"time_start": "2026-08-01", "time_end": "2026-08-31",
                    "category_in": [], "amount_min": None, "amount_max": None},
         "agg_ops": [{"op": "sum", "target_col": "amount", "group_by": []}],
         "require_detail_rows": False},
    ]}
    out = _fix_stat_ir("这个月跟上个月比，花得多了还是少了", multi)
    starts = [g["filter"]["time_start"] for g in out["groups"]]
    assert len(set(starts)) == 2, f"多组区间不得被整句意图改成同一个: {starts}"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_parse_normal_ir(mock_call_llm, base_stat_state):
    """
    函数功能与逻辑描述：
        验证 parse_query_node 正常解析 LLM 输出的 IR：输入含时间范围、品类与 sum/avg 两个 agg_ops 的
        IR JSON，断言 raw_ir 正常回填（need_more_info=False、valid=True、2 个 agg_ops），goto=execute_query_node。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回模拟 IR JSON 字符串。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        need_more_info=true 场景：parse 阶段只透传 IR，追问逻辑由 execute_query_node 前置拦截处理；
        断言 raw_ir.need_more_info=True、prompt 透传，goto=execute_query_node。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回 need_more_info 的 IR JSON。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        LLM 解析异常降级分支：call_llm_base 抛异常时降级为默认 IR（本月总支出），
        断言 valid=True、need_more_info=False、category_in 为空、首个 agg_op 为 amount sum，goto=execute_query_node。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，以 side_effect 抛 ValueError。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        valid=false 前置拦截：execute_query_node 不执行 SQL，直接返回"不支持该操作"，
        断言 success=False、msg 与 error 透传 prompt，goto=reply_node。
    入参说明：
        base_stat_state：pytest fixture 注入的基础 StatState（用例内覆写 raw_ir）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        正常执行 sum/avg 两个聚合：call_bill_sql 依次返回两段单引号结果串，
        断言 total_amount=1260、avg_amount=63.0，goto=verify_query_node（D9 Reflection 先进自检）。
    入参说明：
        mock_call_sql：patch 掉 `call_bill_sql`，以 side_effect 返回两次聚合结果。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        SQL 执行异常兜底：call_bill_sql 抛异常时返回失败载荷，
        断言 success=False、msg="统计数据库查询异常"、error 含异常信息，goto=reply_node。
    入参说明：
        mock_call_sql：patch 掉 `call_bill_sql`，以 side_effect 抛异常。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        need_more_info=true 前置拦截：execute_query_node 不查库，直接返回追问载荷，
        断言 success=False、error="need_more_info"、data.prompt 透传，goto=reply_node。
    入参说明：
        base_stat_state：pytest fixture 注入的基础 StatState（用例内覆写 raw_ir）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        require_detail_rows=True 时额外查明细（两次聚合 + 一次明细细共 3 次调用），
        断言 total_amount=1260 且 detail_list 首条 remark 透传，goto=verify_query_node。
    入参说明：
        mock_call_sql：patch 掉 `call_bill_sql`，以 side_effect 返回 sum/avg/明细三段结果。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证 reply_node 经 a2a_bus.send_result 回传结果：契约为 (task_id, result)，
        result 为字符串；断言只发送一次且 goto 为 LangGraph 终止节点 "__end__"。
    入参说明：
        mock_send：patch 掉 `a2a_bus.send_result`。
        base_stat_state：pytest fixture 注入的基础 StatState（用例内写入最终 struct）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    # 断言跳转至 LangGraph 终止节点 "__end__"（与导入的 END 常量取值一致）
    assert cmd.goto == "__end__"


# ===================== verify_query_node（D9 Reflection · M10）测试 =====================

def _ok_state(base_stat_state, data=None, **extra) -> StatState:
    """
    函数功能与逻辑描述：
        在 base_stat_state 基础上构造 execute 成功后的状态：固定 user_input 与一条 sum IR，
        写入 final_struct.success=True；data 可覆写结果数据，**extra 可注入 verify_attempt 等额外字段。
    入参说明：
        base_stat_state：基础状态 fixture（被就地修改后返回）。
        data：final_struct.data 的结果数据，默认 {"total_amount": 1260}。
        **extra：需额外写入状态的键值对（如 verify_attempt）。
    返回值说明：
        StatState：构造好的 execute 成功状态。
    """
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
    """
    函数功能与逻辑描述：
        校验通过分支：verify LLM 返回 pass=True，断言 goto=reply_node、verify_decision="reply"。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回通过的校验 JSON。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_call_llm.return_value = json.dumps({"pass": True, "issues": [], "feedback": ""})
    state = _ok_state(base_stat_state)
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_verify_fail_retries_parse(mock_call_llm, base_stat_state):
    """
    函数功能与逻辑描述：
        校验不过且未达上限：verify 返回 pass=False，断言写入反馈并回到 parse_query_node 重生成 IR
        （verify_decision="retry"、verify_attempt=1、verify_feedback 含修正提示）。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回校验不通过的 JSON。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        校验不过且已达重试上限（verify_attempt=MAX_VERIFY_ATTEMPTS）：转兜底直发当前确定性结果，
        断言 goto=reply_node、verify_decision="reply"、verify_feedback 置空（不无限循环）。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回始终不通过的校验 JSON。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        校验 LLM 调用异常兜底：抛异常时跳过校验直接发结果（verify 是增强项，不阻断任务），
        断言 goto=reply_node、verify_decision="reply"。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，以 side_effect 抛异常。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = _ok_state(base_stat_state)
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"


@pytest.mark.asyncio
@patch("agents.stat_agent.nodes.mcp_client.call_llm_base")
async def test_verify_skips_for_failed_payload(mock_call_llm, base_stat_state):
    """
    函数功能与逻辑描述：
        execute 失败载荷（success=False）不参与语义自检：断言直达 reply_node 且校验 LLM 未被调用。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`（断言未被调用）。
        base_stat_state：pytest fixture 注入的基础 StatState（用例内写入失败 struct）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        parse 收到 verify_feedback 时注入 user prompt（LLM 据此修正 IR），并清空 feedback 避免泄漏到下一轮；
        断言反馈被消费（置 None）且确实出现在调用 LLM 的 user prompt 中。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回合法 IR JSON。
        base_stat_state：pytest fixture 注入的基础 StatState（用例内写入 verify_feedback）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        类目近义表达归一：用户说"吃饭"但 LLM 输出空 category_in，经 _fix 归一后应补为 ["餐饮"]
        （M10 verify 收敛前提），断言 raw_ir.filter.category_in == ["餐饮"]。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，返回 category_in 为空的 IR JSON。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        结果缺陷自纠闭环（近义类目错位，`_fix_stat_ir` 规则兜不住的场景）：用户问"吃饭"（=餐饮），
        LLM 首版漏输出 category_in（全类目统计，口径不符）→ verify 反馈 → 重生成限定餐饮 → 通过发送；
        断言最终 total_amount=1000（餐饮口径而非全类目 1260）、verify_attempt=1、反馈已消费、只发送一次。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，以 side_effect 模拟 parse/verify 两轮交替返回。
        mock_call_sql：patch 掉 `call_bill_sql`，以 side_effect 返回两轮聚合结果。
        mock_send：patch 掉 `a2a_bus.send_result`（断言只发送一次）。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        整图兜底闭环：verify 连续不过达到上限（MAX_VERIFY_ATTEMPTS）时转兜底发送当前确定性结果，
        断言最终 success=True、total_amount=1260、verify_attempt=MAX_VERIFY_ATTEMPTS、只发送一次（链路不崩）。
    入参说明：
        mock_call_llm：patch 掉 `call_llm_base`，以 side_effect 模拟 3 轮 parse/verify 全部不通过的交替返回。
        mock_call_sql：patch 掉 `call_bill_sql`，以 side_effect 返回 3 轮聚合结果。
        mock_send：patch 掉 `a2a_bus.send_result`（断言只发送一次）。
        base_stat_state：pytest fixture 注入的基础 StatState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
