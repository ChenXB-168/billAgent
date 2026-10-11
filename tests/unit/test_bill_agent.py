# ==============================================
# 记账 Agent 节点单元测试：execute_node 校验/兜底分支、reply_node 回传、月度习惯沉淀
# ==============================================
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.bill_agent.state import BillState
from agents.bill_agent.nodes import execute_node, reply_node, BILL_AGENT_ID

# ===================== Fixture =====================
@pytest.fixture
def base_bill_state() -> BillState:
    """
    函数功能与逻辑描述：
        提供一份基础的 BillState：session_id 固定为 "test_session_001"，task_id/result/city/month
        均为 None，raw_segments 为空列表；各用例按需覆写字段，避免重复构造状态。
        function 作用域，无清理逻辑（纯内存对象）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        BillState：初始化后的记账 Agent 状态对象。
    """
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
    """
    函数功能与逻辑描述：
        autouse 的 function 作用域 fixture：把 `agents.bill_agent.nodes.mcp_client` 替换为 MagicMock，
        并将 call_llm_base / call_bill_sql 设为 AsyncMock，防止用例提前访问真实属性触发异常或触达远端。
        清理方式为 yield 后由 patch 上下文自动还原原始对象。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        MagicMock：注入的 mcp_client 替身，供用例设置返回值或断言调用。
    """
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
    """
    函数功能与逻辑描述：
        autouse 的 function 作用域 fixture：把 `mcpGateway.registry.get_registry` 打桩为返回
        一个 invoke 为 AsyncMock（默认 ok=True、data={"habits": []}）的假 registry，
        避免记账成功分支调用 habit.upsert 时真写本地 DB 与审计日志（隔离测试副作用）。
        清理方式为 yield 后由 patch 上下文还原。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        MagicMock：假 registry 对象；专项用例可再次 patch 覆盖其 invoke 行为。
    """
    reg = MagicMock()
    reg.invoke = AsyncMock(return_value=MagicMock(ok=True, data={"habits": []}))
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        yield reg

# ===================== test_reply_node =====================
@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.a2a_bus.send_result")
async def test_reply_node_send_result(mock_send, base_bill_state):
    """
    函数功能与逻辑描述：
        验证 reply_node 出口：state 中已有序列化 result 时，reply_node 调用一次
        a2a_bus.send_result 将结果回传给调度方。
        覆盖场景：正常回传路径（mock 掉 send_result，仅断言调用次数）。
    入参说明：
        mock_send：@patch 注入的 a2a_bus.send_result 替身，用于断言被调用一次。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 session_id/task_id/result）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证空输入拦截：raw_segments 全为空白时，execute_node 直接返回失败载荷，
        msg 为「输入内容为空」、error 为「未获取到有效的账单描述信息」，不进入 LLM 调用。
        覆盖场景：raw_segments = ["    ", ""] 的空白输入。
    入参说明：
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内把 raw_segments 置为全空白）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证 LLM 判定信息缺失时走追问分支：mock parse_json_output 返回 need_more_info=True、
        prompt 为「请提供午饭的消费金额」，execute_node 返回 need_more_info 追问载荷并透传该 prompt。
        覆盖场景：原文含金额数字（「午饭35元」）以绕过前置金额启发式拦截，验证追问透传。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，设定返回 need_more_info=True 的解析结果。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证解析异常兜底：parse_json_output 抛出 ValueError 时，execute_node 捕获异常并转为追问
        （而非硬失败），msg 为「需要向用户补充询问信息」、error 为 need_more_info，
        且 prompt 含兜底话术「暂时没有识别到可记账的信息」。
        覆盖场景：LLM 解析抛异常的兜底分支。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，side_effect=ValueError 模拟解析失败。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证字段合法性拦截：解析结果 valid=false（且 need_more_info=false）时，
        execute_node 返回 msg「账单信息校验不通过」、error「valid标记为false，无法录入账单」。
        覆盖场景：valid 标记为 false 的拦截分支。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，返回 valid=False 的解析结果。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
@patch("agents.bill_agent.nodes._settle_month_habit")
@patch("agents.bill_agent.nodes.mcp_client.call_bill_sql")
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_multi_items_both_persisted(mock_parse, mock_sql, mock_habit, base_bill_state):
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（方案 `设计/13` §3.2 多笔改造，防复发）：一次输入**两笔账**时，
        两笔都必须进入落库语句，且为**单条多值 INSERT**（原子，避免半笔账）。
        背景：改造前 `execute_node` 只读单个 amount/category、只执行一次单值 INSERT，
        且 `json_parser._try_loads_merged` 会把多个同构 JSON 浅合并成一笔 → **多笔静默丢账**。
    入参说明：
        mock_parse：`parse_json_output` 替身，返回含两条 items 的解析结果。
        mock_sql：`call_bill_sql` 替身，返回成功文案。
        mock_habit：`_settle_month_habit` 替身（避免用例真实写 monthly_habit）。
        base_bill_state：pytest fixture 提供的基础 BillState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "items": [
            {"amount": 30, "category": "餐饮", "consume_date": "2026-09-19", "remark": "午饭30"},
            {"amount": 20, "category": "交通", "consume_date": "2026-09-19", "remark": "打车20"},
        ],
    }
    mock_sql.return_value = "SQL执行成功"
    state = base_bill_state
    state["raw_segments"] = ["记午饭30，记打车20"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])

    # ① 两笔都成功、结构完整
    assert payload["success"] is True, payload
    assert payload["data"]["count"] == 2, payload["data"]
    assert [p["amount"] for p in payload["data"]["items"]] == [30, 20], payload["data"]
    assert [p["category"] for p in payload["data"]["items"]] == ["餐饮", "交通"], payload["data"]

    # ② 单条多值 INSERT（一次调用、两个占位组、参数含两笔金额）
    assert mock_sql.await_count == 1, f"应为单条多值 INSERT，实际调用 {mock_sql.await_count} 次"
    sql = mock_sql.await_args.args[1]
    params = mock_sql.await_args.args[2]
    assert sql.count("(?, ?, ?, ?, ?)") == 2, sql
    assert 30 in params and 20 in params, params

    # ③ 多笔共用同一个 task_id（同一次请求 = 同一个 task，对账依据）
    assert params.count(state.get("task_id")) == 2, params

    # ④ 习惯沉淀逐笔执行（各笔可能跨月）
    assert mock_habit.await_count == 2, f"应逐笔沉淀习惯，实际 {mock_habit.await_count}"


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.mcp_client.call_bill_sql")
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_multi_items_any_invalid_rejects_all(mock_parse, mock_sql, base_bill_state):
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（方案 `13` D-2 全回滚语义）：多笔中**任一笔金额非法**时，
        **整批拒绝**——一处都不落库，绝不产生"半笔账"。
    入参说明：
        mock_parse：返回两笔，其中第二笔金额为 -20（非法）。
        mock_sql：`call_bill_sql` 替身（预期**不被调用**）。
        base_bill_state：pytest fixture 提供的基础 BillState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_parse.return_value = {
        "need_more_info": False,
        "prompt": "",
        "valid": True,
        "items": [
            {"amount": 30, "category": "餐饮", "consume_date": "2026-09-19", "remark": "午饭30"},
            {"amount": -20, "category": "交通", "consume_date": "2026-09-19", "remark": "打车-20"},
        ],
    }
    state = base_bill_state
    state["raw_segments"] = ["记午饭30，记打车-20"]
    cmd = await execute_node(state)
    payload = json.loads(cmd.update["result"])

    assert payload["success"] is False, payload
    assert payload["msg"] == "金额非法", payload
    # 多笔场景给出"第几笔 + 共几笔 + 均未入库"
    assert "第2笔" in payload["error"] and "均未入库" in payload["error"], payload["error"]
    # ★关键：一次 SQL 都不能发出去（全回滚）
    assert mock_sql.await_count == 0, "任一笔记账非法时不得落库任何一笔"


@pytest.mark.asyncio
@patch("agents.bill_agent.nodes.parse_json_output")
async def test_execute_amount_negative(mock_parse, base_bill_state):
    """
    函数功能与逻辑描述：
        验证金额业务拦截：解析结果为 valid=true 但 amount=-30（原文「晚饭-30元」含负号），
        execute_node 返回 msg「金额非法」、error「消费金额必须大于0」，并在 data 中回带该金额。
        覆盖场景：负号需保留不被正则修正为正数，amount<=0 的拦截分支。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，返回 amount=-30、valid=True。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证仅年月日期在 mock 绕过 parse 层校验后仍可正常走完入库流程：
        parse_json_output 被 mock 为直接返回 consume_date="2026-07"，SQL 返回「成功」，
        execute_node 返回 success=True、msg「记账成功」（日期由代码层兜底修正后落库）。
        覆盖场景：parse 校验被绕过时的正常入库路径。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，返回仅年月的 consume_date。
        mock_sql：@patch 注入的 call_bill_sql 替身，固定返回「成功」。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证完整正常链路：解析成功（金额 35.0、日期 2026-07-14）+ SQL 入库返回「成功」，
        execute_node 返回 success=True、msg「记账成功」，data 中金额保持 35.0。
        覆盖场景：解析与落库均成功的端到端正常路径。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，返回合法完整字段。
        mock_sql：@patch 注入的 call_bill_sql 替身，固定返回「成功」。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证入库业务失败分支：解析正常但 call_bill_sql 返回业务失败文本「数据库连接异常」
        （不含「成功」），execute_node 返回 success=False、msg「账单入库操作失败」，
        并把该文本透传到 error。
        覆盖场景：SQL 返回非成功文本分支。
    入参说明：
        mock_parse：@patch 注入的 parse_json_output 替身，返回合法完整字段。
        mock_sql：@patch 注入的 call_bill_sql 替身，固定返回「数据库连接异常」。
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写 raw_segments）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        占位用例：仅在 state 上设置 raw_segments/city/month（完整模板渲染所需的字段），
        未调用任何节点函数，末尾 assert True 恒真，当前不产生实际断言效果。
        覆盖场景：无（不执行被测逻辑）。
    入参说明：
        base_bill_state：pytest fixture 注入，提供基础 BillState（用例内覆写三个字段）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证记账成功后的月度习惯沉淀：_settle_month_habit 经 registry.invoke 调用
        habit.upsert（入参 month/category/amount，agent_id=BILL_AGENT_ID），
        再把读回的当月聚合同步到 pkl 语义层（upsert_month_habit_memory，按月份前缀替换该月摘要）。
        覆盖场景：字段完整且网关返回 ok=True 的成功路径。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；用例内 patch get_registry 与 upsert_month_habit_memory）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证字段缺失时提前短路：data 缺 amount/category/consume_date 任一字段，
        _settle_month_habit 直接返回，registry.invoke 一次都不被调用（不触达网关）。
        覆盖场景：两次调用分别缺失不同字段的 no-op 分支。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；用例内 patch get_registry 并断言其 invoke 未被 await）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    reg = MagicMock()
    reg.invoke = AsyncMock()
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        from agents.bill_agent.nodes import _settle_month_habit
        await _settle_month_habit({"amount": 10})        # 缺 category / consume_date
        await _settle_month_habit({"category": "餐饮"})   # 缺 amount / consume_date
    reg.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_month_habit_error_silent():
    """
    函数功能与逻辑描述：
        验证静默失败：网关 invoke 抛 RuntimeError 时，_settle_month_habit 吞掉异常正常返回，
        不阻断记账主流程（失败仅影响习惯沉淀，不影响 payload 组包）。
        覆盖场景：网关异常的兜底分支（不抛即通过）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；用例内 patch get_registry 使其 invoke 抛异常）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    async def _boom(*a, **k):
        """
        函数功能与逻辑描述：
            模拟网关调用失败的 async 替身：接收任意位置/关键字参数后抛出 RuntimeError，
            用于验证 _settle_month_habit 对网关异常静默兜底、不阻断记账主流程。
        入参说明：
            *a：任意位置参数（本桩忽略，用于匹配 registry.invoke 的调用签名）。
            **k：任意关键字参数（本桩忽略）。
        返回值说明：
            无（恒定抛出 RuntimeError("gateway down")，不返回）。
        """
        raise RuntimeError("gateway down")
    reg = MagicMock()
    reg.invoke = AsyncMock(side_effect=_boom)
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        from agents.bill_agent.nodes import _settle_month_habit
        await _settle_month_habit({"amount": 10, "category": "餐饮",
                                   "consume_date": "2026-09-01"})   # 不抛即通过
