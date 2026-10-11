import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langgraph.types import Command
from agents.orchestrator.state import OrchState
from agents.orchestrator import nodes
from agents.orchestrator.nodes import (
    plan_node,
    dispatch_node,
    wait_result_node,
    collect_node,
    prune_tasks_by_heuristics,
    ORCH_AGENT_ID,
    TASK_AGENT_MAP,
    TASK_CONTEXT_DEPS,
    validate_task_list,
    _collect_llm_deps
)

# ===================== Fixture 基础状态 =====================
@pytest.fixture
def base_orch_state() -> OrchState:
    """
    函数功能与逻辑描述：
        构造 orchestrator 图节点单测所需的「干净初始状态」OrchState 实例，
        所有业务字段均为空值/零值，确保用例之间无状态串扰；
        用例可在拿到实例后按需覆写 user_input / task_plan / current_task 等字段。
    入参说明：
        无（pytest fixture 注入，无外部入参）。
    返回值说明：
        OrchState：session_id 固定为 test_sess_001，user_input 与 session_history 为空串、
            task_plan 为空字典、all_task_results 为空列表、dispatch_round=0，
            current_agent 为 ORCH_AGENT_ID。
    """
    return OrchState(
        session_id="test_sess_001",
        user_input="",
        session_history="",
        task_plan={},
        current_task=None,
        all_task_results=[],
        error_msg=None,
        final_reply="",
        current_agent=ORCH_AGENT_ID,
        current_sub_agent_struct=None,
        agent_result_cache={},
        dispatch_round=0
    )

# ===================== 全局通用mock =====================
@pytest.fixture(autouse=True)
def mock_deps():
    """
    函数功能与逻辑描述：
        全局 autouse fixture，为全部用例自动打桩外部依赖，避免触碰真实会话记忆与数据库：
        以 MagicMock 替换 nodes.get_session_memory（get_history 返回空串、export_all 返回空字典、
        incr_ask_round 返回 1 以保持「首轮追问」语义、reset_ask_round/clear_draft 为空操作），
        以 AsyncMock 替换 nodes.get_user_target_config（返回月预算 3000、目标储蓄率 0.3）。
        作用域：function，每个用例执行前自动进入、用例结束由 patch 上下文还原 mock。
    入参说明：
        无（pytest fixture 注入，无外部入参）。
    返回值说明：
        无（yield 上下文 fixture，仅提供打桩环境，不向用例返回数据）。
    """
    mock_mem = MagicMock()
    mock_mem.get_history.return_value = ""
    mock_mem.export_all.return_value = {}
    mock_mem.add_msg = MagicMock()
    # ★M5（D5）：collect_node 新增会话级追问计数，测试替身需对齐新契约——
    #   默认返回 1（计数 1 < ASK_ROUND_LIMIT，config 默认 3、可用 BILLAGENT_ASK_ROUND_LIMIT 覆盖；
    #   nodes.py 判定为 `incr_ask_round(key) >= ASK_ROUND_LIMIT` 才放弃），
    #   保持既有"正常追问"用例语义不被放弃分支改变
    mock_mem.incr_ask_round = MagicMock(return_value=1)
    mock_mem.reset_ask_round = MagicMock()
    mock_mem.clear_draft = MagicMock()

    with patch("agents.orchestrator.nodes.get_session_memory", return_value=mock_mem), \
         patch("agents.orchestrator.nodes.get_user_target_config",
               AsyncMock(return_value={"month_budget": 3000, "target_save_rate": 0.3})):
        yield

# ===================== plan_node 场景测试 =====================
@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_pre_check_block(mock_parse, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 plan_node 在规划 LLM 返回 pre_check_pass=false 时执行顶层拦截：
        打桩 parse_json_output 返回空 tasks + pre_check_pass=False + block_tip 文案；
        断言不生成任何任务（task_plan 仍为空字典）、final_reply 直接透传 block_tip，
        且控制流跳转到 collect_node（绕过 dispatch 直接汇总）。
    入参说明：
        mock_parse：patch 到 agents.orchestrator.nodes.parse_json_output 的替身，构造顶层拦截的假返回。
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例将 user_input 覆写为"随便输入"。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["user_input"] = "随便输入"
    mock_parse.return_value = {
        "tasks": [],
        "pre_check_pass": False,
        "block_tip": "当前不允许记账，请先完成身份验证"
    }
    cmd: Command = await plan_node(base_orch_state)
    assert cmd.update["task_plan"] == {}
    assert cmd.update["final_reply"] == "当前不允许记账，请先完成身份验证"
    assert cmd.goto == "collect_node"


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output", side_effect=ValueError("JSON解析出错"))
async def test_plan_node_parse_exception(mock_parse, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 plan_node 在规划 LLM 输出解析抛异常时的容错分支：
        打桩 parse_json_output 以 side_effect=ValueError("JSON解析出错") 模拟非法输出；
        断言 error_msg 中包含"任务规划解析失败："前缀与原始异常文案，
        且控制流跳转 collect_node，不进入 dispatch 派发。
    入参说明：
        mock_parse：patch 到 agents.orchestrator.nodes.parse_json_output 的替身，本用例令其抛 ValueError。
        base_orch_state：pytest fixture 注入的干净 OrchState，user_input 覆写为"晚饭30元"。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["user_input"] = "晚饭30元"
    cmd: Command = await plan_node(base_orch_state)
    assert "任务规划解析失败：JSON解析出错" in cmd.update["error_msg"]
    assert cmd.goto == "collect_node"


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_single_bill_task(mock_parse, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 plan_node 正常路径下（pre_check_pass=true）能生成单一 bill 任务：
        打桩 parse_json_output 返回仅含 bill 的 tasks；断言 task_plan 中落库 bill 任务、
        其 deps 为空列表（无上游依赖）、status 初始为 pending，
        且控制流跳转 dispatch_node 准备派发。
    入参说明：
        mock_parse：patch 到 agents.orchestrator.nodes.parse_json_output 的替身，返回单任务规划结果。
        base_orch_state：pytest fixture 注入的干净 OrchState，user_input 覆写为带记账指令的"记一下晚饭30元"。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["user_input"] = "记一下晚饭30元"
    mock_parse.return_value = {
        "tasks": [
            {"name": "bill", "raw_segments": ["晚饭30元"], "operate_sub_type": "add"}
        ],
        "pre_check_pass": True,
        "block_tip": ""
    }
    cmd: Command = await plan_node(base_orch_state)
    tp = cmd.update["task_plan"]
    assert "bill" in tp
    assert tp["bill"]["deps"] == []
    assert tp["bill"]["status"] == "pending"
    assert cmd.goto == "dispatch_node"


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_cycle_detect(mock_parse, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 plan_node 对任务依赖环的检测与终止：临时篡改全局 TASK_CONTEXT_DEPS，
        令 bill→stat→bill 互指形成死循环，并打桩 parse_json_output 返回 bill 与 stat 双任务；
        断言 error_msg 中包含"任务依赖存在死循环"、控制流跳转 collect_node 终止派发，
        用例末尾用备份 original_deps 还原 TASK_CONTEXT_DEPS，避免污染其它用例。
    入参说明：
        mock_parse：patch 到 agents.orchestrator.nodes.parse_json_output 的替身，返回双任务规划结果。
        base_orch_state：pytest fixture 注入的干净 OrchState，user_input 覆写为同时触发记账与统计的文本。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 文本须同时触发 bill 与 stat 启发式保留（记账指令+金额、统计词），否则无法构造双任务环
    base_orch_state["user_input"] = "记一笔100元，查询本月开销"
    # 临时修改模拟循环依赖
    original_deps = TASK_CONTEXT_DEPS.copy()
    TASK_CONTEXT_DEPS["bill"] = ["stat"]
    TASK_CONTEXT_DEPS["stat"] = ["bill"]

    mock_parse.return_value = {
        "tasks": [
            {"name": "bill", "raw_segments": ["x"], "operate_sub_type": "add"},
            {"name": "stat", "raw_segments": ["x"], "operate_sub_type": "analyse"}
        ],
        "pre_check_pass": True,
        "block_tip": ""
    }
    cmd: Command = await plan_node(base_orch_state)
    assert "任务依赖存在死循环" in cmd.update["error_msg"]
    assert cmd.goto == "collect_node"
    # 恢复原始
    TASK_CONTEXT_DEPS.clear()
    TASK_CONTEXT_DEPS.update(original_deps)

# ===================== dispatch_node =====================
@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.a2a_bus.send_task")
async def test_dispatch_ready_task(mock_send, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 dispatch_node 对「依赖已满足的就绪任务」的正常派发路径：
        构造 bill 任务（deps 为空、status=pending），打桩 a2a_bus.send_task 以隔离真实总线；
        断言派发后该任务 status 由 pending 置为 running、控制流跳转 wait_result_node 等待结果，
        且 send_task 恰好被调用一次。
    入参说明：
        mock_send：patch 到 agents.orchestrator.nodes.a2a_bus.send_task 的替身，用于拦截并校验派发调用。
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为单个就绪 bill 任务。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "pending",
            "deps": [],
            "raw_segments": ["午饭20"],
            "operate_sub_type": "add",
            "result": None,
            "agent_id": "bill_agent"
        }
    }
    cmd: Command = await dispatch_node(base_orch_state)
    tp = cmd.update["task_plan"]
    assert tp["bill"]["status"] == "running"
    assert cmd.goto == "wait_result_node"
    mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_all_task_done(base_orch_state):
    """
    函数功能与逻辑描述：
        验证 dispatch_node 在「无就绪任务、全部任务已 done」时的收口分支：
        构造唯一 bill 任务且 status=done；断言控制流跳转 collect_node 进入汇总，
        不再派发任何子任务（无新的 running 任务）。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为单个已 done 的 bill 任务。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "bill": {"status": "done", "deps": [], "raw_segments": [], "operate_sub_type": "", "result": {}, "agent_id": "bill_agent"}
    }
    cmd: Command = await dispatch_node(base_orch_state)
    assert cmd.goto == "collect_node"

# ===================== wait_result_node =====================
@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.a2a_bus.wait_result")
async def test_wait_result_normal_struct(mock_wait, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 wait_result_node 在子 agent 正常返回结构化 JSON 时的处理：
        构造 running 状态的 bill 任务并指定 current_task，打桩 a2a_bus.wait_result
        返回含 success/agent_type/msg/data 的标准响应包字符串；
        断言任务状态被置为 done、结果按 agent_type（bill_agent）写入 agent_result_cache 且 success 为 True，
        控制流回到 dispatch_node 继续调度后续任务。
    入参说明：
        mock_wait：patch 到 agents.orchestrator.nodes.a2a_bus.wait_result 的替身，返回结构化 JSON 字符串。
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 current_task 与 task_plan。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["current_task"] = "bill"
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "running",
            "deps": [],
            "raw_segments": ["晚饭30"],
            "operate_sub_type": "add",
            "result": None,
            "agent_id": "bill_agent",
            "task_id": "t123"
        }
    }
    struct_payload = {
        "success": True,
        "agent_type": "bill_agent",
        "msg": "记账成功",
        "data": {"category": "餐饮", "amount": 30, "consume_date": "2026-07-16", "remark": "晚饭30"},
        "error": None
    }
    mock_wait.return_value = json.dumps(struct_payload, ensure_ascii=False)
    cmd: Command = await wait_result_node(base_orch_state)
    tp = cmd.update["task_plan"]
    cache = cmd.update["agent_result_cache"]
    assert tp["bill"]["status"] == "done"
    assert cache["bill_agent"]["success"] is True
    assert cmd.goto == "dispatch_node"


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.a2a_bus.wait_result", side_effect=TimeoutError())
async def test_wait_result_timeout_fallback(mock_wait, base_orch_state):
    """
    函数功能与逻辑描述：
        验证 wait_result_node 的等待超时兜底分支：打桩 a2a_bus.wait_result
        以 side_effect=TimeoutError() 模拟子 agent 超时；
        断言节点自动构造失败结果 struct（success=False、msg 含"任务执行异常"）写入
        current_sub_agent_struct，且**按 M19（D20）终止语义收口到 collect_node**。
        ★行为变更（M19）：改造前本分支回到 dispatch_node 继续推进 —— 那会让失败结果溜进
        collect 的汇总素材，产出"看似正常但无数据"的回复（静默降级）。M19 起：
        任务级失败（超时/异常）→ **终止本轮**，与 M16 用户取消共用收口路径，
        按 `abort_reason` 区分回执（见 `设计/02` D20 与契约 `09` §2.17）。
    入参说明：
        mock_wait：patch 到 agents.orchestrator.nodes.a2a_bus.wait_result 的替身，本用例令其抛 TimeoutError。
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 current_task 与 running 任务计划。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["current_task"] = "bill"
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "running",
            "deps": [],
            "raw_segments": ["晚饭30"],
            "operate_sub_type": "add",
            "result": None,
            "agent_id": "bill_agent",
            "task_id": "t123"
        }
    }
    cmd: Command = await wait_result_node(base_orch_state)
    struct = cmd.update["current_sub_agent_struct"]
    assert struct["success"] is False
    assert "任务执行异常" in struct["msg"]
    # ★M19：任务级失败 → 终止本轮（不再派发后续波次），交 collect 出对应回执
    assert cmd.goto == "collect_node"
    assert cmd.update["cancelled"] is True
    # 超时兜底载荷的 error 不含"超时"字样 → 归为 task_error（"超时"才是 task_timeout）
    assert cmd.update["abort_reason"] == "task_error"

# ===================== collect_node 汇总节点 =====================
@pytest.mark.asyncio
async def test_collect_need_more_info(base_orch_state):
    """
    函数功能与逻辑描述：
        验证 collect_node 对「子 agent 需要追问」结果的透传：构造 bill 任务结果
        为 success=False、error="need_more_info" 且 data 内含 prompt 文案；
        断言汇总节点不生成新文案，而是把 data.prompt 原样写入 final_reply 交由用户补充信息。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为带追问结果的已完成任务。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "done",
            "deps": [],
            "raw_segments": [],
            "operate_sub_type": "",
            "agent_id": "bill_agent",
            "result": {
                "success": False,
                "agent_type": "bill_agent",
                "msg": "需要向用户补充询问信息",
                "data": {"prompt": "请告诉我晚饭花了多少钱？"},
                "error": "need_more_info"
            }
        }
    }
    cmd: Command = await collect_node(base_orch_state)
    assert cmd.update["final_reply"] == "请告诉我晚饭花了多少钱？"


@pytest.mark.asyncio
async def test_collect_single_bill_success(base_orch_state, monkeypatch):
    """
    函数功能与逻辑描述：
        验证 collect_node 单一记账任务成功时的汇总路径：不再硬编码成功文案，
        而是把任务素材交给汇总 LLM 按汇总提示词生成 final_reply。
        以 monkeypatch 隔离真实依赖：_calc_month_budget_status 返回 None、_calc_alert_facts 返回空预警事实、
        mcp_client.call_llm_base 替换为假实现；假 LLM 内部反向断言 sys_prompt 含"最终回复生成器"
        且 user_text 携带"餐饮"/"30"，用于确认走的是汇总提示词体系并收到任务素材；
        断言节点 final_reply 包含假 LLM 返回的确定性文案。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为记账成功结果。
        monkeypatch：pytest 内置 fixture，用于临时替换 nodes 内的预算状态/预警事实计算与 LLM 调用。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "done",
            "deps": [],
            "raw_segments": [],
            "operate_sub_type": "",
            "agent_id": "bill_agent",
            "result": {
                "success": True,
                "agent_type": "bill_agent",
                "msg": "记账成功",
                "data": {"category": "餐饮", "amount": 30, "consume_date": "2026-07-16", "remark": "晚饭30"},
                "error": None
            }
        }
    }
    # 隔离真实DB：预算状态/预警事实计算走 mock，汇总LLM返回确定性文案并校验收到汇总提示词与任务素材
    async def _no_status():
        """
        函数功能与逻辑描述：
            测试替身：将 _calc_month_budget_status 替换为返回 None 的协程，
            模拟"无可用预算状态"，从而隔离真实数据库查询。
        入参说明：
            无。
        返回值说明：
            None：表示没有可用的月度预算状态。
        """
        return None
    monkeypatch.setattr(nodes, "_calc_month_budget_status", _no_status)

    async def _no_facts(_status, _amt, _cat, _city):
        """
        函数功能与逻辑描述：
            测试替身：将 _calc_alert_facts 替换为返回空预警事实的协程，
            使汇总渲染不依赖真实预算与消费数据。
        入参说明：
            _status：月度预算状态（本替身忽略）。
            _amt：本次消费金额（忽略）。
            _cat：消费品类（忽略）。
            _city：城市（忽略）。
        返回值说明：
            dict：三层预警事实全为空的结构 {"l1": None, "l2": None, "l3": None}。
        """
        return {"l1": None, "l2": None, "l3": None}
    monkeypatch.setattr(nodes, "_calc_alert_facts", _no_facts)

    async def _fake_summarize_llm(sys_prompt, user_text, agent_tag="unknown"):
        """
        函数功能与逻辑描述：
            假汇总 LLM：先反向断言收到的 sys_prompt 含"最终回复生成器"、
            user_text 含"餐饮"与"30"（确认提示词体系与任务素材均已注入），再返回确定性文案。
        入参说明：
            sys_prompt：系统提示词，应为汇总提示词体系。
            user_text：用户侧素材文本，应携带任务结果。
            agent_tag：agent 标签，默认 "unknown"（本替身未使用）。
        返回值说明：
            str：固定文案"记账成功：消费类目 餐饮，金额 30 元，日期 2026-07-16"。
        """
        assert "最终回复生成器" in sys_prompt  # 确认走的是汇总提示词体系
        assert "餐饮" in user_text
        assert "30" in user_text
        return "记账成功：消费类目 餐饮，金额 30 元，日期 2026-07-16"
    monkeypatch.setattr(nodes.mcp_client, "call_llm_base", _fake_summarize_llm)

    cmd: Command = await collect_node(base_orch_state)
    assert "记账成功：消费类目 餐饮，金额 30 元，日期 2026-07-16" in cmd.update["final_reply"]


@pytest.mark.asyncio
async def test_collect_multi_task_merge(base_orch_state, monkeypatch):
    """
    函数功能与逻辑描述：
        验证 collect_node 在多任务全部成功时的汇总：构造 bill（记账成功）与 stat
        （月度消费统计，deps 依赖 bill）两个已完成任务，确定性地把各任务结果渲染为文本素材，
        再交汇总 LLM 生成最终回复。打桩 mcp_client.call_llm_base，假 LLM 内部反向断言
        sys_prompt 含"最终回复生成器"、user_text 同时注入记账素材（餐饮/30）与统计素材（月度消费统计）；
        断言 final_reply 包含假 LLM 返回文案中的"已完成以下处理""记账成功""月度消费统计"。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为 bill+stat 双成功任务。
        monkeypatch：pytest 内置 fixture，用于替换 nodes.mcp_client.call_llm_base 为假 LLM。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "done",
            "deps": [],
            "raw_segments": [],
            "operate_sub_type": "",
            "agent_id": "bill_agent",
            "result": {
                "success": True,
                "agent_type": "bill_agent",
                "msg": "记账成功",
                "data": {"category": "餐饮", "amount": 30, "consume_date": "2026-07-16", "remark": "晚饭30"},
                "error": None
            }
        },
        "stat": {
            "status": "done",
            "deps": ["bill"],
            "raw_segments": [],
            "operate_sub_type": "",
            "agent_id": "stat_agent",
            "result": {
                "success": True,
                "agent_type": "stat_agent",
                "msg": "月度消费统计",
                "data": {"month_total": 30},
                "error": None
            }
        }
    }
    async def _fake_summarize_llm(sys_prompt, user_text, agent_tag="unknown"):
        """
        函数功能与逻辑描述：
            假汇总 LLM：反向断言 sys_prompt 含"最终回复生成器"、
            user_text 同时注入记账素材（餐饮/30）与统计素材（月度消费统计），
            用于确认真实汇总路径与素材拼接均生效，再返回确定性多任务文案。
        入参说明：
            sys_prompt：系统提示词，应为汇总提示词体系。
            user_text：用户侧素材文本，应同时含记账与统计结果。
            agent_tag：agent 标签，默认 "unknown"（本替身未使用）。
        返回值说明：
            str：固定文案"已完成以下处理：1.记账成功（餐饮 30 元）；2.月度消费统计（本月共 30 元）。"。
        """
        assert "最终回复生成器" in sys_prompt
        assert "餐饮" in user_text and "30" in user_text        # 记账素材已注入
        assert "月度消费统计" in user_text                       # 统计素材已注入
        return "已完成以下处理：1.记账成功（餐饮 30 元）；2.月度消费统计（本月共 30 元）。"
    monkeypatch.setattr(nodes.mcp_client, "call_llm_base", _fake_summarize_llm)

    cmd: Command = await collect_node(base_orch_state)
    final = cmd.update["final_reply"]
    assert "已完成以下处理" in final
    assert "记账成功" in final
    assert "月度消费统计" in final


@pytest.mark.asyncio
async def test_collect_success_plus_followup_appended(base_orch_state, monkeypatch):
    """
    函数功能与逻辑描述：
        验证 collect_node 的混合场景（同轮既有成功结果又有追问结果）：
        构造 bill 成功、stat 返回 need_more_info 追问的已完成任务计划，
        并以 monkeypatch 隔离真实 DB 与外部 LLM（_calc_month_budget_status 返回 None、
        _calc_alert_facts 返回空事实、EXTERNAL_LLM_ENABLED=False 走本地确定性渲染）；
        断言 final_reply 含成功文案（记账成功/餐饮/56），且以「\\n\\n + prompt」形式
        把追问内容附加到末尾（final 以追问文案结尾、文本中存在空行分隔符）。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为成功 task + 追问 task。
        monkeypatch：pytest 内置 fixture，用于替换预算状态/预警事实计算并关闭外部 LLM。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "bill": {
            "status": "done", "deps": [], "raw_segments": [], "operate_sub_type": "",
            "agent_id": "bill_agent",
            "result": {"success": True, "agent_type": "bill_agent", "msg": "记账成功",
                       "data": {"category": "餐饮", "amount": 56, "consume_date": "2026-09-13",
                                "remark": "午饭56"},
                       "error": None},
        },
        "stat": {
            "status": "done", "deps": ["bill"], "raw_segments": [], "operate_sub_type": "",
            "agent_id": "stat_agent",
            "result": {"success": False, "agent_type": "stat_agent", "msg": "需要补充信息",
                       "data": {"prompt": "请提供统计的时间范围"},
                       "error": "need_more_info"},
        },
    }

    async def _no_status():
        """
        函数功能与逻辑描述：
            测试替身：将 _calc_month_budget_status 替换为返回 None 的协程，模拟无可用预算状态。
        入参说明：
            无。
        返回值说明：
            None：表示没有可用的月度预算状态。
        """
        return None
    monkeypatch.setattr(nodes, "_calc_month_budget_status", _no_status)

    async def _no_facts(_status, _amt, _cat, _city):
        """
        函数功能与逻辑描述：
            测试替身：将 _calc_alert_facts 替换为返回空预警事实的协程，隔离真实数据依赖。
        入参说明：
            _status：月度预算状态（本替身忽略）。
            _amt：本次消费金额（忽略）。
            _cat：消费品类（忽略）。
            _city：城市（忽略）。
        返回值说明：
            dict：三层预警事实全为空的结构 {"l1": None, "l2": None, "l3": None}。
        """
        return {"l1": None, "l2": None, "l3": None}
    monkeypatch.setattr(nodes, "_calc_alert_facts", _no_facts)
    # 本地确定性渲染，不依赖外部 LLM
    monkeypatch.setattr(nodes, "EXTERNAL_LLM_ENABLED", False)

    cmd: Command = await collect_node(base_orch_state)
    final = cmd.update["final_reply"]
    assert "记账成功" in final and "餐饮" in final and "56" in final   # 成功文案
    assert final.endswith("请提供统计的时间范围")                      # 追问作为补充附加在末尾
    assert "\n\n" in final                                            # 以空行分隔


@pytest.mark.asyncio
async def test_collect_multiple_followups_only_first_appended(base_orch_state, monkeypatch):
    """
    函数功能与逻辑描述：
        锁定现状（回归保护）：同轮出现「1 个成功任务 + 多个追问任务」时，
        collect_node 只把第一个追问（按 task_plan 迭代顺序取到 stat）附加到回复末尾，
        其余追问静默丢弃。构造 bill 成功、stat 与 price 均 need_more_info 的任务计划，
        以 monkeypatch 隔离真实 DB 与外部 LLM（预算状态返回 None、空预警事实、EXTERNAL_LLM_ENABLED=False）；
        断言 final_reply 含第一个追问"请提供统计的时间范围"，但不含第二个追问"请提供对比的城市"。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为 1 成功 + 2 追问任务。
        monkeypatch：pytest 内置 fixture，用于替换预算状态/预警事实计算并关闭外部 LLM。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    def _followup_task(agent_id: str, prompt: str) -> dict:
        """
        函数功能与逻辑描述：
            用例内局部辅助工厂，避免重复书写"追问类子任务"的完整 task_plan 结构；
            生成 status=done、error=need_more_info 且 data.prompt 为指定文案的任务字典。
        入参说明：
            agent_id (str)：子 agent 标识，同时用作任务的 agent_id 与结果 agent_type。
            prompt (str)：追问文案，写入 result.data.prompt。
        返回值说明：
            dict：单个追问任务的 task_plan 条目（含 status/deps/raw_segments/operate_sub_type/agent_id/result）。
        """
        return {
            "status": "done", "deps": [], "raw_segments": [], "operate_sub_type": "",
            "agent_id": agent_id,
            "result": {"success": False, "agent_type": agent_id, "msg": "需要补充信息",
                       "data": {"prompt": prompt}, "error": "need_more_info"},
        }

    base_orch_state["task_plan"] = {
        "bill": {
            "status": "done", "deps": [], "raw_segments": [], "operate_sub_type": "",
            "agent_id": "bill_agent",
            "result": {"success": True, "agent_type": "bill_agent", "msg": "记账成功",
                       "data": {"category": "交通", "amount": 35, "consume_date": "2026-09-13",
                                "remark": "打车35"},
                       "error": None},
        },
        "stat": _followup_task("stat_agent", "请提供统计的时间范围"),
        "price": _followup_task("price_agent", "请提供对比的城市"),
    }

    async def _no_status():
        """
        函数功能与逻辑描述：
            测试替身：将 _calc_month_budget_status 替换为返回 None 的协程，模拟无可用预算状态。
        入参说明：
            无。
        返回值说明：
            None：表示没有可用的月度预算状态。
        """
        return None
    monkeypatch.setattr(nodes, "_calc_month_budget_status", _no_status)

    async def _no_facts(_status, _amt, _cat, _city):
        """
        函数功能与逻辑描述：
            测试替身：将 _calc_alert_facts 替换为返回空预警事实的协程，隔离真实数据依赖。
        入参说明：
            _status：月度预算状态（本替身忽略）。
            _amt：本次消费金额（忽略）。
            _cat：消费品类（忽略）。
            _city：城市（忽略）。
        返回值说明：
            dict：三层预警事实全为空的结构 {"l1": None, "l2": None, "l3": None}。
        """
        return {"l1": None, "l2": None, "l3": None}
    monkeypatch.setattr(nodes, "_calc_alert_facts", _no_facts)
    monkeypatch.setattr(nodes, "EXTERNAL_LLM_ENABLED", False)

    cmd: Command = await collect_node(base_orch_state)
    final = cmd.update["final_reply"]
    assert "请提供统计的时间范围" in final          # 第一个追问被附加
    assert "请提供对比的城市" not in final          # 第二个追问被丢弃（现状）


@pytest.mark.asyncio
async def test_collect_error_global(base_orch_state):
    """
    函数功能与逻辑描述：
        验证 collect_node 对全局错误（task_plan 为空、error_msg 非空）的兜底展示：
        直接置 error_msg 为终止原因，不构造任何任务结果；
        断言 final_reply 为"执行终止："前缀拼接原始 error_msg 文案，保证用户能看到失败原因。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例仅覆写 error_msg。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["error_msg"] = "任务依赖死锁，执行终止"
    cmd: Command = await collect_node(base_orch_state)
    assert cmd.update["final_reply"] == "执行终止：任务依赖死锁，执行终止"


# ===================== collect_node 防幻觉：数字白名单闸门（2026-08-30 修复回归） =====================
# 注：_extract_numbers / _numbers_inside_whitelist 的实现已抽到 utils/number_whitelist.py
#     （D16/M11 finance 防幻觉单一来源，含 2026-08-30 序号误判修复说明），
#     nodes.py 仅 re-export（nodes.py 顶部 `from utils.number_whitelist import ...`），
#     故此处经 nodes 命名空间调用仍然有效。
def test_extract_numbers_strips_inline_list_ordinals():
    """
    函数功能与逻辑描述：
        回归用例（2026-08-30 修复）：验证 collect_node 防幻觉数字白名单闸门的取数逻辑
        nodes._extract_numbers 能剔除单行列举序号（"1.""2."），避免序号被误判为编造数字
        （修复前会误杀正常回复并回退本地渲染）。
        断言对含两个列举序号且金额均为 30 的文本，取数结果恰为 {30.0}。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._extract_numbers(
        "已完成以下处理：1.记账成功（餐饮 30 元）；2.月度消费统计（本月共 30 元）。"
    ) == {30.0}


def test_extract_numbers_keeps_decimals_and_amounts():
    """
    函数功能与逻辑描述：
        边界用例：验证序号剔除逻辑不会误伤真实小数与金额。
        断言对"金额 1.5 元，本月共 280 元"，_extract_numbers 同时保留小数 1.5 与整数金额 280.0。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._extract_numbers("金额 1.5 元，本月共 280 元") == {1.5, 280.0}


def test_extract_numbers_strips_newline_ordinals():
    """
    函数功能与逻辑描述：
        锁定原有行为：换行分隔的列举序号（"\\n1.""\\n2."）同样必须剔除。
        断言对"完成：\\n1.记账 30\\n2.统计 30"，取数结果恰为 {30.0}。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._extract_numbers("完成：\n1.记账 30\n2.统计 30") == {30.0}


def test_numbers_inside_whitelist_still_blocks_true_hallucination():
    """
    函数功能与逻辑描述：
        反向锁死用例：确保序号修复未放松防幻觉闸门——白名单外数字仍被判定为编造。
        断言含白名单外数字 300 的文本返回 False（拦截），仅含白名单内 30 的文本返回 True（放行）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._numbers_inside_whitelist("记账成功 30 元，另外共 300 元", {30.0}) is False
    assert nodes._numbers_inside_whitelist("记账成功 30 元", {30.0}) is True


# ===================== M4 需求5：prune_tasks_by_heuristics 意图裁剪/兜底（纯函数单测） =====================
def _names(task_list: list) -> list:
    """
    函数功能与逻辑描述：
        用例级公共辅助函数：从任务列表提取 name 字段序列，便于与期望的任务名列表直接比对。
    入参说明：
        task_list (list)：任务字典列表，每项预期含 "name" 键。
    返回值说明：
        list：按输入顺序排列的 name 列表；缺失 name 的任务以 None 占位。
    """
    return [t.get("name") for t in task_list]


def test_prune_single_bill_no_extra():
    """
    函数功能与逻辑描述：
        验证 prune_tasks_by_heuristics 对「含显式记账指令 + 金额」的输入保留 bill 任务，
        且不额外脑补其它任务。输入"记奶茶15块"并令模型只规划 bill，
        断言输出任务名序列恰为 ["bill"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内固定 tasks 为单条 bill（add）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "bill", "raw_segments": ["记奶茶15块"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("记奶茶15块", tasks)
    assert _names(out) == ["bill"]


def test_prune_plain_consume_no_record():
    """
    函数功能与逻辑描述：
        验证 2026-09-12 方案C：无记账指令的纯消费陈述（"奶茶15块""那打车花了80块呢"）
        必须裁掉 bill 绝不落库，并默认按咨询口径兜底补出 price 对标任务
        （配套的"不回生硬引导文案"由 plan_node 顶层准入放行分支负责——pre_check_pass=false
        时纯消费陈述被放行而非回 block_tip，本用例只覆盖裁剪函数自身的补任务行为）。
        对两条文本分别执行裁剪，断言输出任务名序列均为 ["price"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内以 for 循环覆盖两条消费陈述，tasks 固定为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("奶茶15块", "那打车花了80块呢"):
        tasks = [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}]
        out = prune_tasks_by_heuristics(text, tasks)
        assert _names(out) == ["price"]


def test_prune_single_bill_force_when_model_omits():
    """
    函数功能与逻辑描述：
        验证记账兜底：输入含记账指令"记奶茶15块"但模型漏规划（误规划为 stat），
        prune_tasks_by_heuristics 应强制补出 bill 任务以保障记账不丢失；
        断言输出任务名序列为 ["bill"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条误规划的 stat。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "stat", "raw_segments": ["记奶茶15块"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("记奶茶15块", tasks)
    assert _names(out) == ["bill"]


def test_prune_consult_keeps_finance_prunes_bill():
    """
    函数功能与逻辑描述：
        验证咨询场景裁剪："花了300买耳机，值吗"无记账指令，
        模型同时规划 bill 与 finance 时，应裁掉 bill 仅保留 finance 咨询；
        断言输出任务名序列为 ["finance"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 为 bill+finance 两条。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [
        {"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"},
        {"name": "finance", "raw_segments": ["值吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("花了300买耳机，值吗", tasks)
    # ★2026-09-17 期望更新（用户指令"price 与 finance 不互斥"）：合理性疑问同样要比价。
    #   finance 已在输入里被保留，price 由补齐追加在其后。
    assert _names(out) == ["finance", "price"]


def test_prune_consult_finance_backfilled():
    """
    函数功能与逻辑描述：
        验证咨询兜底补齐：输入"花了300买耳机，值吗"但模型漏规划 finance（只给 bill），
        prune_tasks_by_heuristics 应裁掉 bill 并补齐 finance 咨询（不补 bill）；
        断言输出任务名序列为 ["finance"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("花了300买耳机，值吗", tasks)
    # ★2026-09-17 期望更新：bill 被裁（无记账指令）；price 与 finance 由补齐产生（price 先补 → 在前）。
    assert _names(out) == ["price", "finance"]


def test_prune_consult_price_rewritten_to_finance():
    """
    函数功能与逻辑描述：
        验证误规划纠正：输入"花了300买耳机，值吗"，模型误给 bill+price
        （"值吗"不等于"值不值"，不构成比价意图），裁剪器应同时裁掉 bill 与 price，
        由 finance 兜底；断言输出任务名序列为 ["finance"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 为 bill+price 两条。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [
        {"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"},
        {"name": "price", "raw_segments": ["值吗"], "operate_sub_type": "query"},
    ]
    out = prune_tasks_by_heuristics("花了300买耳机，值吗", tasks)
    # ★2026-09-17 期望更新：price 不再被"合理性疑问"排除，与 finance 并存（"值吗"本就命中比价词）。
    assert _names(out) == ["price", "finance"]


def test_prune_consume_price_word_keeps_price_and_finance():
    """
    函数功能与逻辑描述：
        ★2026-09-17 修正（用户反馈，实测 bug）：**price 与 finance 不互斥**。
        文本同时含金额、消费实体与比价词（"贵不贵 / 值不值 / 划算吗"）时，
        既需要**比价**（拿当地同品类均价算溢价率）也需要**合理性分析**，两个任务都应保留。
        原实现只留 finance、把 price 裁掉，导致"吃火锅花了80，贵不贵"**从未发生比价**，
        用户收到"没有当地火锅的平均价格信息"（本该按餐饮均价给出溢价率）。
        bill 仍被裁（该输入无记账指令）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内以 for 循环覆盖三条文本，tasks 均为 bill+price。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("打车80贵不贵", "打车80值不值", "打车80划算吗"):
        tasks = [
            {"name": "bill", "raw_segments": [text], "operate_sub_type": "add"},
            {"name": "price", "raw_segments": [text], "operate_sub_type": "query"},
        ]
        out = prune_tasks_by_heuristics(text, tasks)
        assert _names(out) == ["price", "finance"], f"{text} 应同时比价与分析: {_names(out)}"


def test_prune_price_word_no_amount_goes_price():
    """
    函数功能与逻辑描述：
        验证无金额场景不误判：输入"打车贵不贵"（无比价对象金额），
        应仍归 price（市场比价意图），词表重叠不影响判定；
        断言输出任务名序列为 ["price"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 price。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "price", "raw_segments": ["打车贵不贵"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("打车贵不贵", tasks)
    assert _names(out) == ["price"]


def test_prune_review_bill_finance():
    """
    函数功能与逻辑描述：
        M4 用例：输入"记一笔打车35元，合理吗"同时含显式记账指令与合理性疑问，
        模型给出 bill+price+finance 三条，裁剪器应裁掉 price 而保留 bill 与 finance；
        断言输出任务名序列为 ["bill", "finance"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 为 bill+price+finance 三条。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [
        {"name": "bill", "raw_segments": ["记一笔打车35元"], "operate_sub_type": "add"},
        {"name": "price", "raw_segments": ["合理吗"], "operate_sub_type": "query"},
        {"name": "finance", "raw_segments": ["合理吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("记一笔打车35元，合理吗", tasks)
    # ★2026-09-17 期望更新（用户指令"price 与 finance 不互斥"）：显式记账 + 合理性疑问 →
    #   bill 保留（有记账指令）、finance 保留，**price 同样要保留**（"合理吗"属合理性疑问，
    #   按新规则同时比价；它在输入里被裁后由补齐追加在末位）。
    assert _names(out) == ["bill", "finance", "price"]


def test_prune_consume_consult_normal_question():
    """
    函数功能与逻辑描述：
        2026-09-05 回归用例（源于用户投诉句）："吃一顿饭花了600块正常吗"无记账指令，
        属消费咨询；模型同时给出 bill 与 finance 时，应裁掉 bill 仅保留 finance 咨询；
        断言输出任务名序列为 ["finance"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 为 bill+finance 两条。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [
        {"name": "bill", "raw_segments": ["吃一顿饭花了600块"], "operate_sub_type": "add"},
        {"name": "finance", "raw_segments": ["吃一顿饭花了600块正常吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("吃一顿饭花了600块正常吗", tasks)
    assert _names(out) == ["finance"]


def test_prune_record_cmd_overrides_question():
    """
    函数功能与逻辑描述：
        与上一条用例对照：同样的消费陈述但带记账指令（"帮我记一下打车80块，正常吗"），
        记账意图优先，bill 与 finance 均应保留；
        断言输出任务名序列为 ["bill", "finance"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 为 bill+finance 两条。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [
        {"name": "bill", "raw_segments": ["打车80块"], "operate_sub_type": "add"},
        {"name": "finance", "raw_segments": ["正常吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("帮我记一下打车80块，正常吗", tasks)
    # ★2026-09-17 期望更新：记账指令优先（bill 保留）+ 合理性疑问（finance）+ 比价（price 补齐）。
    #   三者并存体现"price 与 finance 不互斥"。
    assert _names(out) == ["bill", "finance", "price"]


def test_validate_task_list_duplicate_skipped_not_failed():
    """
    函数功能与逻辑描述：
        ★2026-09-17 新增（实测驱动，防复发）：`validate_task_list` 对**同轮同名重复任务**
        必须**跳过重复条目并继续**，而不是把整轮规划判为失败。来源：e2e `empty_habits`
        实测外部 LLM 输出 `tasks=['finance','finance','finance']`；原实现虽已 `continue`
        跳过重复项，却又记入 `errors` → 上游判失败 → `task_plan` 置空 → **用户输入被完全丢弃**。
        本项与同函数内"name 非法即剔除、不判失败"的既有口径保持一致。
        同时锁定"name 非法仍剔除且不判失败"的既有行为不被回退。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # ① 三个同名 finance：只保留首条，且**整体校验通过**（不再整轮失败）
    #    ★2026-09-19 用户更正约束：**只有 finance 至多一条**，其余类型允许多条。
    ok, msg, fixed = validate_task_list([
        {"name": "finance", "raw_segments": ["给点建议"]},
        {"name": "finance", "raw_segments": ["给点建议"]},
        {"name": "finance", "raw_segments": ["给点建议"]},
    ])
    assert ok is True, f"重复 finance 不应判整轮失败: {msg}"
    assert msg == "", f"重复 finance 不应留错误信息: {msg}"
    assert [t["name"] for t in fixed] == ["finance"], fixed
    # ★finance 硬约束落地：至多一条
    assert fixed[0]["raw_segments"] == ["给点建议"], fixed[0]

    # ①' 重复条目必须**丢弃**，**严禁合并**。
    #   理由：finance 多条**本身即规划错误**（如把"查开销"塞进 finance，那是 stat 的职责），
    #   合并会把越权诉求固化进执行。此处锁定"只留首条、多余片段不得出现"。
    _ok, _msg, dropped = validate_task_list([
        {"name": "finance", "raw_segments": ["给点建议"]},
        {"name": "finance", "raw_segments": ["查查这个月花了多少"]},   # 越权诉求（应属 stat）→ 整条丢弃
        {"name": "finance", "raw_segments": ["给点建议"]},
    ])
    assert [t["name"] for t in dropped] == ["finance"], dropped
    assert dropped[0]["raw_segments"] == ["给点建议"], dropped[0]
    # 越权片段**绝不能被并入**首条（防"合并实现"回潮）
    assert "查查这个月花了多少" not in (dropped[0]["raw_segments"] or []), dropped[0]
    # 首条的 operate_sub_type 应补默认（finance → analyse）
    assert dropped[0]["operate_sub_type"] == "analyse", dropped[0]

    # ②' ★2026-09-19 用户更正约束：**bill / stat / price 允许多条**，不得去重。
    #    他们是互不替代的诉求：多笔记账、多个时间范围统计、多个品类比价。
    ok4, _msg4, fixed4 = validate_task_list([
        {"name": "bill", "raw_segments": ["记午饭30"]},
        {"name": "bill", "raw_segments": ["记打车20"]},
        {"name": "stat", "raw_segments": ["查本月"]},
        {"name": "stat", "raw_segments": ["查上月"]},
        {"name": "price", "raw_segments": ["对比火锅"]},
        {"name": "price", "raw_segments": ["对比奶茶"]},
    ])
    assert ok4 is True, f"bill/stat/price 多条不应判失败: {_msg4}"
    assert [t["name"] for t in fixed4] == ["bill", "bill", "stat", "stat", "price", "price"], fixed4
    # 各条的片段必须**各自保留**（不得被去重吃掉）
    assert [t["raw_segments"][0] for t in fixed4] == [
        "记午饭30", "记打车20", "查本月", "查上月", "对比火锅", "对比奶茶",
    ], fixed4

    # ② 混合场景：**finance 之外的重复必须全部保留**（约束已于 2026-09-19 收窄）。
    #    两个 bill 各自对应"午饭"与"打车"两笔真实记账诉求，去重会吃掉一笔账。
    ok2, msg2, fixed2 = validate_task_list([
        {"name": "bill", "raw_segments": ["记午饭30"]},
        {"name": "stat", "raw_segments": ["查开销"]},
        {"name": "bill", "raw_segments": ["记打车20"]},
        {"name": "price", "raw_segments": ["对比物价"]},
    ])
    assert ok2 is True, f"含重复任务时不应整体失败: {msg2}"
    assert [t["name"] for t in fixed2] == ["bill", "stat", "bill", "price"], fixed2
    assert fixed2[0]["raw_segments"] == ["记午饭30"], fixed2
    assert fixed2[2]["raw_segments"] == ["记打车20"], fixed2

    # ③ 既有行为不回退：name 非法 → 剔除该条但**不判失败**
    ok3, _msg3, fixed3 = validate_task_list([
        {"name": "not_a_task", "raw_segments": ["x"]},
        {"name": "bill", "raw_segments": ["午饭32元记一下"]},
    ])
    assert ok3 is True, "name 非法应剔除而非判失败（既有口径）"
    assert [t["name"] for t in fixed3] == ["bill"], fixed3


def test_prune_finance_data_dependent_backfills_stat():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（实测驱动，防复发）：finance 的分析若**依赖当期账目**，必须自动补 stat。
        背景：用例输入"我想分析一下我的消费习惯，算算剩下预算够不够撑到月底，给我一些省钱建议"
        ——"算预算够不够/撑到月底"依赖**当月支出**，而只有 stat 会去查账；LLM 却只规划了 finance
        （实测甚至重复 3 条），导致 finance 拿不到 month_spend、只能回"暂缺当月支出"，
        用户什么也得不到。`TASK_CONTEXT_DEPS` 早已声明 finance 依赖 stat，本用例锁定该声明被兑现。
        三条分支：① 命中数据依赖词 → 补 stat；② 不依赖账目 → **不得**补（避免强加统计任务）；
        ③ 已存在 stat → 不重复补。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    def _fin(text):
        return [{"name": "finance", "raw_segments": [text], "operate_sub_type": "analyse"}]

    # ① 依赖账目（"够不够撑到月底"）→ 必须补出 stat，与 finance 并存
    t1 = "我想分析一下我的消费习惯，算算剩下预算够不够撑到月底，给我一些省钱建议"
    names1 = [t["name"] for t in prune_tasks_by_heuristics(t1, _fin(t1))]
    assert "finance" in names1, f"finance 应保留: {names1}"
    assert "stat" in names1, f"依赖账目的 finance 分析应自动补 stat: {names1}"

    # ② 不依赖账目（纯理财原则）→ **不得**补 stat（避免强加统计任务）
    t2 = "怎么理财比较好，给点建议"
    names2 = [t["name"] for t in prune_tasks_by_heuristics(t2, _fin(t2))]
    assert "finance" in names2, f"finance 应保留: {names2}"
    assert "stat" not in names2, f"无数据依赖时不应强加 stat: {names2}"

    # ③ 已有 stat → 不重复补
    t3 = "算算剩下预算够不够撑到月底"
    tasks3 = _fin(t3) + [{"name": "stat", "raw_segments": [t3], "operate_sub_type": "query"}]
    names3 = [t["name"] for t in prune_tasks_by_heuristics(t3, tasks3)]
    assert names3.count("stat") == 1, f"stat 不应重复补齐: {names3}"


def test_prune_finance_data_dependent_keeps_stat():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（用户指正 + 根因定位）：**裁剪器不得把 finance 依赖的 stat 删掉**。
        背景：`prune_tasks_by_heuristics` 原用 `has_stat_hint`（词表为 统计/查询/汇总/这个月/
        花了多少/支出/开销/余额…）决定 stat 去留；而输入"我想分析一下我的消费习惯，算算剩下预算
        够不够撑到月底，给我一些省钱建议"**一个词都不命中** → **即使 LLM 正确规划了 stat，
        也会被裁剪掉** → finance 拿不到 month_spend → 只能回"暂缺当月支出数据"，用户什么也得不到。
        这解释了本用例所属链路"规划看起来没错、结果却没有数据"的现象。
        本用例锁定：**finance 需要数据时，LLM 给出的 stat 必须保留**。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    text = "我想分析一下我的消费习惯，算算剩下预算够不够撑到月底，给我一些省钱建议"
    tasks = [
        {"name": "stat", "raw_segments": [text], "operate_sub_type": "query"},
        {"name": "finance", "raw_segments": [text], "operate_sub_type": "analyse"},
    ]
    names = [t["name"] for t in prune_tasks_by_heuristics(text, tasks)]
    assert "finance" in names, f"finance 应保留: {names}"
    assert "stat" in names, f"finance 依赖数据的场景下，LLM 给出的 stat 不得被裁剪掉: {names}"


def test_collect_llm_deps_drops_reverse_edge():
    """
    函数功能与逻辑描述：
        ★2026-09-19 新增（实测驱动，防复发）：`_collect_llm_deps` 必须**剔除与静态表反向的依赖**。
        背景：真链路日志显示 LLM 输出 `动态依赖生效 {'stat': ['finance']}` —— 而静态表
        `TASK_CONTEXT_DEPS` 声明的是 **finance 依赖 stat**（stat→finance 是真实数据流），
        LLM 把方向写反了；反向边与静态表互指成环 → `_has_cycle` 判真 → 回退**整张**依赖表，
        把 LLM 其它**正确**的附加依赖一并丢掉（惩罚过重）。
        本用例锁定：① 反向边被剔除；② 同向的合理补充仍保留；③ 自依赖/批外名照旧剔除。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    names = {"bill", "stat", "price", "finance"}
    # ① 反向边（stat 依赖 finance）→ 必须被剔除
    out = _collect_llm_deps(
        [{"name": "stat", "deps": ["finance"]}, {"name": "finance", "deps": ["stat"]}],
        names,
    )
    assert "stat" not in out, f"反向依赖 stat→finance 应被剔除: {out}"
    # 同向边 finance→stat（与静态表一致）应保留
    assert out.get("finance") == {"stat"}, out

    # ② 自依赖与批外任务名仍按既有口径剔除。
    #    注意必须用**同向**边（price→bill，与静态表 `TASK_CONTEXT_DEPS["price"] = ["bill"]` 一致）：
    #    若用 bill→stat 之类反向边，会先被本轮新增的"反向边剔除"过滤掉，就测不出原有口径了。
    out2 = _collect_llm_deps(
        [{"name": "price", "deps": ["price", "not_a_task", "bill"]}],
        names,
    )
    assert out2.get("price") == {"bill"}, f"自依赖与批外名应剔除: {out2}"


def test_prune_refuse_bill_single():
    """
    函数功能与逻辑描述：
        M4 拒绝记账用例：输入"别记这笔"（模型因含"记"字误判为 bill），
        裁剪器应识别否定意图并返回空任务列表，坚决不记账；
        断言输出为空列表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "bill", "raw_segments": ["别记这笔"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("别记这笔", tasks)
    assert out == []


def test_prune_refuse_bill_with_amount():
    """
    函数功能与逻辑描述：
        M4 拒绝记账用例：输入"先不记这300块"虽带金额，但含明确否定记账意图，
        裁剪器应返回空任务列表（金额的存在不改变拒绝结论）；
        断言输出为空列表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "bill", "raw_segments": ["先不记这300块"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("先不记这300块", tasks)
    assert out == []


def test_prune_refuse_bill_keep_stat():
    """
    函数功能与逻辑描述：
        M4 拒绝记账 + 保留统计用例：输入"别记这笔，看看我最近吃火锅多不多"，
        模型规划 bill+stat，裁剪器应裁掉 bill 但保留 stat（不记账但仍做统计查询）；
        断言输出任务名序列为 ["stat"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 为 bill+stat 两条。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [
        {"name": "bill", "raw_segments": ["别记这笔"], "operate_sub_type": "add"},
        {"name": "stat", "raw_segments": ["看看我最近吃火锅多不多"], "operate_sub_type": "query"},
    ]
    out = prune_tasks_by_heuristics("别记这笔，看看我最近吃火锅多不多", tasks)
    assert _names(out) == ["stat"]


def test_prune_refuse_bill_just_asking():
    """
    函数功能与逻辑描述：
        M4 拒绝记账用例：输入"奶茶15块，我只是问问"表明仅咨询而非记账，
        裁剪器应拒绝记账并返回空任务列表；断言输出为空列表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "bill", "raw_segments": ["奶茶15块，我只是问问"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("奶茶15块，我只是问问", tasks)
    assert out == []


def test_prune_price_review_no_amount_trusts_model():
    """
    函数功能与逻辑描述：
        M4 回归用例：输入"吃饭多少钱值吗"既无金额也无记账上下文，
        裁剪器应保持信任模型的原始规划 price（不做改写或兜底）；
        断言输出任务名序列为 ["price"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 price。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "price", "raw_segments": ["吃饭多少钱值吗"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("吃饭多少钱值吗", tasks)
    assert _names(out) == ["price"]


def test_prune_income_intercept_regression():
    """
    函数功能与逻辑描述：
        M4 回归用例：输入"工资8000到账"属收入场景，即便模型规划了 bill，
        裁剪器仍应拦截记账（收入不应计入消费账单）；断言输出为空列表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "bill", "raw_segments": ["工资8000到账"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("工资8000到账", tasks)
    assert out == []


def test_prune_stat_hint_more_less():
    """
    函数功能与逻辑描述：
        M4 用例：验证"多不多"这类频次询问计入统计意图，
        模型规划的 stat 任务在裁剪后应被保留；断言输出任务名序列为 ["stat"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内 tasks 固定为单条 stat。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    tasks = [{"name": "stat", "raw_segments": ["火锅多不多"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("看看我最近吃火锅多不多", tasks)
    assert _names(out) == ["stat"]


# ===================== M4 需求5：plan_node 多意图金额上下文补全（finance 也补全） =====================
@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_finance_param_backfill(mock_parse, base_orch_state):
    """
    函数功能与逻辑描述：
        M4 用例：验证 plan_node 的多意图金额上下文补全——输入"记一下花了300买耳机，值吗"，
        模型给出的 finance 片段只有"值吗"而缺金额，节点应把含金额/品类的原文补进 finance.raw_segments；
        断言 task_plan 同时含 bill 与 finance、finance 的 raw_segments 拼接后包含"300"与"耳机"，
        且 finance 的 deps 依赖 bill（先记账后分析）。
    入参说明：
        mock_parse：patch 到 agents.orchestrator.nodes.parse_json_output 的替身，返回 bill+finance 规划。
        base_orch_state：pytest fixture 注入的干净 OrchState，user_input 覆写为多意图语句。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["user_input"] = "记一下花了300买耳机，值吗"
    mock_parse.return_value = {
        "tasks": [
            {"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"},
            {"name": "finance", "raw_segments": ["值吗"], "operate_sub_type": "analyse"},
        ],
        "pre_check_pass": True,
        "block_tip": ""
    }
    cmd: Command = await plan_node(base_orch_state)
    tp = cmd.update["task_plan"]
    assert "bill" in tp and "finance" in tp
    # finance 片段被补全：除"值吗"外还追加含金额/品类的完整原文
    finance_segs = " ".join(tp["finance"]["raw_segments"])
    assert "300" in finance_segs and "耳机" in finance_segs
    # 依赖：finance 等待 bill
    assert "bill" in tp["finance"]["deps"]


# ===================== 胡乱记账修复：伪记账词剥离（2026-09-05） =====================
def test_has_record_cmd_strict_verbs():
    """
    函数功能与逻辑描述：
        验证伪记账词剥离修复未误伤真实指令：nodes._has_record_cmd 对
        记/记录/记账/记一下/记一笔/写入/入账 等真实记账表达均应命中；
        用例遍历 11 条正样本逐条断言返回真，任一未命中即失败。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；正样本文本在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in (
        "记打车35元", "记午饭56元", "记录奶茶18", "记账打车80",
        "帮我记一下打车80", "记一笔奶茶15", "写入账本300", "入账打车80",
        "帮我记个账，打车35", "把奶茶记下来", "记上打车80",
    ):
        assert nodes._has_record_cmd(text), f"应命中真实记账指令: {text}"


def test_has_record_cmd_pseudo_words_not_trigger():
    """
    函数功能与逻辑描述：
        验证「胡乱记账」修复核心：仅含"记/记录"字串但语义非记账的词
        （记得/记不清/记住/日记/标记/记录仪/笔记/记性 等）一律不得命中 _has_record_cmd；
        用例遍历 9 条负样本逐条断言返回假，任一误命中即失败。
        边界：负样本中仍夹带金额等消费要素，用于确认命中与否只由指令词决定。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；负样本文本在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in (
        "我记得打车花了80",            # 记得 = 回忆，非记账
        "我记不清打车花了多少",        # 记不清
        "我记住了这个月别乱花钱",      # 记住
        "我日记里写了买书80",          # 日记
        "标记一下这个支出",            # 标记
        "行车记录仪花了300块",          # 记录仪（设备）
        "买了个记录仪300块",
        "写笔记记单词",                # 笔记/记单词
        "你记性真差",                  # 记性
    ):
        assert not nodes._has_record_cmd(text), f"伪记账词不应命中: {text}"


def test_prune_pseudo_record_word_prunes_bill():
    """
    函数功能与逻辑描述：
        验证伪记账词场景下模型误规划 bill 的处理：输入"我记得打车花了80"等
        伪记账词 + 金额文本，裁剪器应裁掉 bill 绝不落库，并按咨询口径补出 price 任务；
        用例遍历 3 条文本逐条断言输出任务名序列为 ["price"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；三组输入文本在用例内硬编码，tasks 均为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("我记得打车花了80", "买行车记录仪花了300", "我日记里写了买书80"):
        tasks = [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}]
        out = prune_tasks_by_heuristics(text, tasks)
        assert _names(out) == ["price"], f"伪记账词场景 bill 应裁剪并按咨询补 price: {text}"


def test_prune_pseudo_record_word_no_bill():
    """
    函数功能与逻辑描述：
        验证伪记账词场景下模型漏规划（只给 stat）时不得强制补 bill：
        伪记账词剥离后已无真实记账指令，故裁剪结果中不应出现 bill；
        用例遍历 2 条文本断言输出任务名集合不含 "bill"。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；输入文本在用例内硬编码，tasks 均为单条 stat。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("我记得打车花了80", "行车记录仪花了300"):
        tasks = [{"name": "stat", "raw_segments": [text], "operate_sub_type": "query"}]
        out = prune_tasks_by_heuristics(text, tasks)
        assert "bill" not in [t.get("name") for t in out], f"不应强制记账: {text}"


def test_prune_mixed_pseudo_plus_real_record_cmd():
    """
    函数功能与逻辑描述：
        验证伪记账词与真实指令混用时的判定优先级：输入"记得把打车80记上"
        虽含"记得"这类伪记账词，但同时含真实指令"记上"，应判定为记账并保留 bill；
        断言输出任务名序列为 ["bill"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；文本在用例内硬编码，tasks 为单条 bill。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    text = "记得把打车80记上"
    tasks = [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics(text, tasks)
    assert [t.get("name") for t in out] == ["bill"]


# ===================== 规划提示词日期注入（2026-09-10） =====================
from datetime import date as _date
from agents.orchestrator.prompts.prompt_loader import render_user as _render_user


def test_plan_prompt_injects_today_date():
    """
    函数功能与逻辑描述：
        验证规划提示词注入「今天日期」（2026-09-10 新增）：这是 LLM 换算
        「昨天/这个月/上个月」等相对时间的基准，须与 bill_agent / stat_agent 两侧
        已注入的 today_date 保持同一范式。调用 build_task_plan_prompt（不 mock LLM）
        取渲染后的 user_prompt，断言其中包含系统当天 ISO 日期字符串。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内以 "上个月花了多少" + ["bill"] 调用提示词渲染。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    today = _date.today().isoformat()
    _, user_prompt = nodes.build_task_plan_prompt("上个月花了多少", "", ["bill"])
    assert today in user_prompt, f"规划提示词缺少今天日期 {today}: {user_prompt[:300]}"


def test_render_user_default_today_date_when_not_passed():
    """
    函数功能与逻辑描述：
        验证 prompt_loader.render_user 的向后兼容：未显式传 today_date 时自动填充当天日期，
        使既有调用方与单测无需改动即可获得日期注入；
        断言渲染结果的 user 文本中包含系统当天 ISO 日期。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内固定传入 sys="x"、user="y"、agent_tag="bill"。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    out = _render_user("x", "y", "bill")
    assert _date.today().isoformat() in out


def test_render_user_respects_explicit_today_date():
    """
    函数功能与逻辑描述：
        验证显式传入 today_date 时以传入值为准，供测试做确定性渲染、
        避免断言依赖真实当天；断言渲染结果包含"今天日期：2026-01-15"。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内显式传入 today_date="2026-01-15"。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    out = _render_user("x", "y", "bill", today_date="2026-01-15")
    assert "今天日期：2026-01-15" in out


# ===================== 方案C：无记账指令消费陈述默认咨询（2026-09-12） =====================
def test_is_consume_statement_true_for_plain_consume():
    """
    函数功能与逻辑描述：
        验证方案C（2026-09-12）消费陈述识别正样本：含金额或消费实体词、
        无记账指令、非收入、非拒绝表达的文本应判为消费陈述（后续按咨询处理）；
        用例遍历 4 条文本逐条断言 nodes._is_consume_statement 返回真。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；正样本文本在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("奶茶15块", "那打车花了80块呢", "住酒店一晚388元", "今天吃饭花了50"):
        assert nodes._is_consume_statement(text), f"应判为消费陈述: {text}"


def test_is_consume_statement_false_for_record_cmd():
    """
    函数功能与逻辑描述：
        验证含显式记账指令的文本不属于消费陈述（应走 bill 记账而非默认咨询兜底）；
        用例遍历 4 条带指令文本逐条断言 _is_consume_statement 返回假。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；负样本文本在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("记打车80元", "帮我记一下奶茶18", "记账午饭56", "记一笔打车35元"):
        assert not nodes._is_consume_statement(text), f"有记账指令不应判消费陈述: {text}"


def test_is_consume_statement_false_for_income_and_refuse():
    """
    函数功能与逻辑描述：
        验证收入与拒绝记账两类文本均不属于消费陈述（走确定性拦截，不补 price 对标）；
        分别断言"工资8000到账"（收入）、"奶茶15块，我只是问问"（拒绝）、"别记这300块"（拒绝）返回假。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；三条文本在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert not nodes._is_consume_statement("工资8000到账")
    assert not nodes._is_consume_statement("奶茶15块，我只是问问")
    assert not nodes._is_consume_statement("别记这300块")


def test_is_consume_statement_false_for_review_question():
    """
    函数功能与逻辑描述：
        验证带合理性疑问（正常吗/值吗/合理吗）的文本不属于「纯」消费陈述，
        应走 finance 分析而非默认补 price；用例遍历 2 条疑问文本断言返回假。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；疑问文本在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for text in ("吃一顿饭花了600块正常吗", "花了300买耳机，值吗"):
        assert not nodes._is_consume_statement(text), f"合理性疑问不应判纯消费陈述: {text}"


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_consume_statement_not_blocked(mock_parse, base_orch_state):
    """
    函数功能与逻辑描述：
        验证方案C：当规划 LLM 返回 pre_check_pass=false 但用户输入实为无记账指令的消费陈述时，
        plan_node 不回生硬引导文案拦截，而是放行到裁剪层补出 price 对标任务；
        打桩 parse_json_output 返回空 tasks + pre_check_pass=False + block_tip；
        断言 task_plan 中存在 price 任务、其 operate_sub_type 为 query，且跳转 dispatch_node。
    入参说明：
        mock_parse：patch 到 agents.orchestrator.nodes.parse_json_output 的替身，构造拦截态返回。
        base_orch_state：pytest fixture 注入的干净 OrchState，user_input 覆写为消费陈述"那打车花了80块呢"。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["user_input"] = "那打车花了80块呢"
    mock_parse.return_value = {
        "tasks": [],
        "pre_check_pass": False,
        "block_tip": "当前不允许记账"
    }
    cmd: Command = await plan_node(base_orch_state)
    tp = cmd.update["task_plan"]
    # 放行则补 price 并继续派发；不放行则 task_plan={} 且 goto=collect_node
    assert "price" in tp, f"消费陈述应放行补 price 而非拦截: {tp}"
    assert tp["price"]["operate_sub_type"] == "query"
    assert cmd.goto == "dispatch_node"


# ===================== 长期记忆沉淀：_build_monthly_summary 摘要构建（2026-09-13 补充） =====================
def _stat_struct(**data_overrides) -> dict:
    """
    函数功能与逻辑描述：
        用例级辅助工厂：构造 stat_agent 汇总成功的标准 struct（success=True、
        agent_type=stat_agent、data 含 time_range/total_amount/category_summary/total_count/max_record），
        便于各摘要构建用例复用；调用方可通过关键字参数覆盖任意 data 字段（含置 None 以测缺失分支）。
    入参说明：
        **data_overrides：可选关键字参数，键名须为 data 内字段名，用于覆盖默认值。
    返回值说明：
        dict：stat_agent 成功结果结构；先落默认 data，再以 data_overrides 更新后返回。
    """
    data = {
        "time_range": ["2026-01-01", "2026-01-31"],
        "total_amount": 5000,
        "category_summary": {"餐饮": 800, "交通": 200, "房租": 3000, "购物": 500, "娱乐": 500},
        "total_count": 45,
        "max_record": {"amount": 600, "category": "娱乐"},
    }
    data.update(data_overrides)
    return {"success": True, "agent_type": "stat_agent", "msg": "统计完成", "data": data}


def test_build_monthly_summary_full_fields_matches_seed_format():
    """
    函数功能与逻辑描述：
        验证 _build_monthly_summary 在字段齐全时输出固定 5 段格式摘要
        （时间段/总支出/分类/笔数/单笔最大），且与 evaluation/seed_memory_data.py
        的种子摘要逐字一致（供召回评测复用同一形态）。
        断言完整字段输入下返回的摘要字符串与期望字面量完全相等。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内以 _stat_struct() 默认字段构造输入。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._build_monthly_summary(_stat_struct()) == (
        "【2026-01-01~2026-01-31 消费摘要】; 总支出: 5000元; "
        "分类: 餐饮: 800元, 交通: 200元, 房租: 3000元, 购物: 500元, 娱乐: 500元; "
        "共45笔; 单笔最大: 600元(娱乐)"
    )


def test_build_monthly_summary_without_time_range_returns_none():
    """
    函数功能与逻辑描述：
        验证无 time_range 时不沉淀长期记忆（无时间范围则摘要无意义）：
        覆盖空 dict、data 为空 dict、data 为 None、time_range 为空数组四种边界，
        逐条断言 _build_monthly_summary 返回 None。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；四种缺失形态在用例内硬编码。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._build_monthly_summary({}) is None
    assert nodes._build_monthly_summary({"data": {}}) is None
    assert nodes._build_monthly_summary({"data": None}) is None
    assert nodes._build_monthly_summary({"data": {"time_range": []}}) is None


def test_build_monthly_summary_time_range_only_returns_none():
    """
    函数功能与逻辑描述：
        验证仅有 time_range、无任何统计字段时不沉淀：实现以「段数」为判据
        （len(parts) < 2，即除时间范围外不足 1 段有效信息）返回 None；
        断言该输入下 _build_monthly_summary 返回 None。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；输入为仅含 time_range 的 struct。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._build_monthly_summary({"data": {"time_range": ["2026-01-01", "2026-01-31"]}}) is None


def test_build_monthly_summary_single_stat_field_is_enough():
    """
    函数功能与逻辑描述：
        验证「只要有 1 个统计字段即视为有效聚合」的下限门槛：仅保留 total_amount=100、
        其余统计字段置 None 时不得返回 None；断言输出为仅含时间段与总支出的摘要字符串。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内通过 _stat_struct 覆盖
        category_summary/total_count/max_record 为 None，仅留 total_amount=100。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._build_monthly_summary(_stat_struct(
        total_amount=100, category_summary=None, total_count=None, max_record=None,
    )) == "【2026-01-01~2026-01-31 消费摘要】; 总支出: 100元"


def test_build_monthly_summary_missing_max_record_skips_segment():
    """
    函数功能与逻辑描述：
        验证 max_record 缺失时跳过「单笔最大」段而不抛错，分段渲染是条件拼接的；
        断言输出为时间段 + 总支出 + 笔数三段，不含单笔最大部分。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内覆盖 category_summary 与 max_record 为 None。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._build_monthly_summary(_stat_struct(
        total_amount=100, total_count=3, category_summary=None, max_record=None,
    )) == "【2026-01-01~2026-01-31 消费摘要】; 总支出: 100元; 共3笔"


def test_build_monthly_summary_zero_values_kept():
    """
    函数功能与逻辑描述：
        验证 0 属有效值：摘要渲染以 is not None 判空而非真值判断，
        故 total_count=0 不得被丢弃；断言输出包含"共0笔"。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）；用例内仅保留 total_count=0，其余统计字段置 None。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._build_monthly_summary(_stat_struct(
        total_amount=None, category_summary=None, total_count=0, max_record=None,
    )) == "【2026-01-01~2026-01-31 消费摘要】; 共0笔"


@pytest.mark.asyncio
async def test_collect_node_stat_success_persists_monthly_summary(base_orch_state):
    """
    函数功能与逻辑描述：
        接线验证：stat_agent 汇总成功时，collect_node 应触发 _build_monthly_summary
        并把生成摘要写入长期记忆（save_consume_memory）。
        构造唯一 stat 任务（status=done、result 由 _stat_struct 提供），
        并以 patch.object 打桩 nodes.save_consume_memory 与 EXTERNAL_LLM_ENABLED=False 隔离副作用；
        断言 save_consume_memory 恰被调用一次，且其首个位置参数包含"消费摘要"与"总支出: 5000元"。
    入参说明：
        base_orch_state：pytest fixture 注入的干净 OrchState，本用例覆写 task_plan 为已完成的 stat 任务。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    base_orch_state["task_plan"] = {
        "stat": {
            "status": "done",
            "deps": [],
            "raw_segments": ["查本月开销"],
            "operate_sub_type": "query",
            "result": _stat_struct(),
            "agent_id": "stat_agent",
        }
    }
    with patch.object(nodes, "save_consume_memory") as mock_save, \
         patch.object(nodes, "EXTERNAL_LLM_ENABLED", False):
        await collect_node(base_orch_state)

    mock_save.assert_called_once()
    saved = mock_save.call_args[0][0]
    assert "消费摘要" in saved and "总支出: 5000元" in saved