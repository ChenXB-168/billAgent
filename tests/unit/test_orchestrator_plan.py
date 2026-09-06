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
    TASK_CONTEXT_DEPS
)

# ===================== Fixture 基础状态 =====================
@pytest.fixture
def base_orch_state() -> OrchState:
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
    mock_mem = MagicMock()
    mock_mem.get_history.return_value = ""
    mock_mem.export_all.return_value = {}
    mock_mem.add_msg = MagicMock()
    # ★M5（D5）：collect_node 新增会话级追问计数，测试替身需对齐新契约——
    #   默认返回 1（第 1 轮 < 上限 3），保持既有"正常追问"用例语义不被放弃分支改变
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
    """pre_check_pass=false 顶层拦截，不生成任务"""
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
    """规划阶段LLM解析异常，直接进入汇总"""
    base_orch_state["user_input"] = "晚饭30元"
    cmd: Command = await plan_node(base_orch_state)
    assert "任务规划解析失败：JSON解析出错" in cmd.update["error_msg"]
    assert cmd.goto == "collect_node"


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_single_bill_task(mock_parse, base_orch_state):
    """仅生成bill单一任务，无依赖（用户带记账指令"记一下"）"""
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
    """构造循环依赖，检测后终止"""
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
    """存在就绪任务，正常派发"""
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
    """全部任务done，跳转到collect"""
    base_orch_state["task_plan"] = {
        "bill": {"status": "done", "deps": [], "raw_segments": [], "operate_sub_type": "", "result": {}, "agent_id": "bill_agent"}
    }
    cmd: Command = await dispatch_node(base_orch_state)
    assert cmd.goto == "collect_node"

# ===================== wait_result_node =====================
@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.a2a_bus.wait_result")
async def test_wait_result_normal_struct(mock_wait, base_orch_state):
    """正常返回结构化JSON"""
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
    """任务超时，自动构造错误struct"""
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
    assert cmd.goto == "dispatch_node"

# ===================== collect_node 汇总节点 =====================
@pytest.mark.asyncio
async def test_collect_need_more_info(base_orch_state):
    """子agent需要追问，直接透传prompt"""
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
    """单一账单任务：汇总节点将成功结果交由外部LLM按汇总提示词生成最终回复（不再硬编码文案）"""
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
        return None
    monkeypatch.setattr(nodes, "_calc_month_budget_status", _no_status)

    async def _no_facts(_status, _amt, _cat, _city):
        return {"l1": None, "l2": None, "l3": None}
    monkeypatch.setattr(nodes, "_calc_alert_facts", _no_facts)

    async def _fake_summarize_llm(sys_prompt, user_text, agent_tag="unknown"):
        assert "最终回复生成器" in sys_prompt  # 确认走的是汇总提示词体系
        assert "餐饮" in user_text
        assert "30" in user_text
        return "记账成功：消费类目 餐饮，金额 30 元，日期 2026-07-16"
    monkeypatch.setattr(nodes.mcp_client, "call_llm_base", _fake_summarize_llm)

    cmd: Command = await collect_node(base_orch_state)
    assert "记账成功：消费类目 餐饮，金额 30 元，日期 2026-07-16" in cmd.update["final_reply"]


@pytest.mark.asyncio
async def test_collect_multi_task_merge(base_orch_state, monkeypatch):
    """多任务结果：确定性渲染各任务文本素材，汇总LLM按提示词生成最终回复"""
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
async def test_collect_error_global(base_orch_state):
    """全局error_msg直接展示"""
    base_orch_state["error_msg"] = "任务依赖死锁，执行终止"
    cmd: Command = await collect_node(base_orch_state)
    assert cmd.update["final_reply"] == "执行终止：任务依赖死锁，执行终止"


# ===================== collect_node 防幻觉：数字白名单闸门（2026-08-30 修复回归） =====================
def test_extract_numbers_strips_inline_list_ordinals():
    """回归：单行列举序号（"1.…2.…"）必须剔除，不得被误判为编造数字（修复前误杀 → 回退本地渲染）"""
    assert nodes._extract_numbers(
        "已完成以下处理：1.记账成功（餐饮 30 元）；2.月度消费统计（本月共 30 元）。"
    ) == {30.0}


def test_extract_numbers_keeps_decimals_and_amounts():
    """边界：小数与金额不得被序号剔除误伤"""
    assert nodes._extract_numbers("金额 1.5 元，本月共 280 元") == {1.5, 280.0}


def test_extract_numbers_strips_newline_ordinals():
    """原行为保持：换行序号仍剔除"""
    assert nodes._extract_numbers("完成：\n1.记账 30\n2.统计 30") == {30.0}


def test_numbers_inside_whitelist_still_blocks_true_hallucination():
    """反向锁死：闸门不得因序号修复而放松——真编造数字仍必须被拦截"""
    assert nodes._numbers_inside_whitelist("记账成功 30 元，另外共 300 元", {30.0}) is False
    assert nodes._numbers_inside_whitelist("记账成功 30 元", {30.0}) is True


# ===================== M4 需求5：prune_tasks_by_heuristics 意图裁剪/兜底（纯函数单测） =====================
def _names(task_list: list) -> list:
    return [t.get("name") for t in task_list]


def test_prune_single_bill_no_extra():
    """2026-09-05: '记奶茶15块' 含显式记账指令+金额 → bill 保留（且不脑补其它任务）"""
    tasks = [{"name": "bill", "raw_segments": ["记奶茶15块"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("记奶茶15块", tasks)
    assert _names(out) == ["bill"]


def test_prune_plain_consume_no_record():
    """2026-09-05 产品规则回归：无记账指令的消费陈述（'奶茶15块''那打车花了80块呢'）→ bill 一律裁掉，绝不落库"""
    for text in ("奶茶15块", "那打车花了80块呢"):
        tasks = [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}]
        out = prune_tasks_by_heuristics(text, tasks)
        assert out == []


def test_prune_single_bill_force_when_model_omits():
    """2026-09-05: '记奶茶15块' 含记账指令但模型漏规划（误规划 stat）→ force_bill 兜底补 bill"""
    tasks = [{"name": "stat", "raw_segments": ["记奶茶15块"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("记奶茶15块", tasks)
    assert _names(out) == ["bill"]


def test_prune_consult_keeps_finance_prunes_bill():
    """2026-09-05: '花了300买耳机，值吗'（无记账指令）→ bill 裁掉，仅保留 finance 咨询"""
    tasks = [
        {"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"},
        {"name": "finance", "raw_segments": ["值吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("花了300买耳机，值吗", tasks)
    assert _names(out) == ["finance"]


def test_prune_consult_finance_backfilled():
    """2026-09-05: '花了300买耳机，值吗' 模型漏规划 finance → 咨询兜底补齐 finance（不补 bill）"""
    tasks = [{"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("花了300买耳机，值吗", tasks)
    assert _names(out) == ["finance"]


def test_prune_consult_price_rewritten_to_finance():
    """2026-09-05: '花了300买耳机，值吗' 模型误规划 bill+price（'值吗'≠'值不值'）→ bill/price裁剪，finance兜底"""
    tasks = [
        {"name": "bill", "raw_segments": ["花了300买耳机"], "operate_sub_type": "add"},
        {"name": "price", "raw_segments": ["值吗"], "operate_sub_type": "query"},
    ]
    out = prune_tasks_by_heuristics("花了300买耳机，值吗", tasks)
    assert _names(out) == ["finance"]


def test_prune_review_bill_finance():
    """M4: '记一笔打车35元，合理吗' → bill+finance（显式记账指令 + 合理性疑问）"""
    tasks = [
        {"name": "bill", "raw_segments": ["记一笔打车35元"], "operate_sub_type": "add"},
        {"name": "price", "raw_segments": ["合理吗"], "operate_sub_type": "query"},
        {"name": "finance", "raw_segments": ["合理吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("记一笔打车35元，合理吗", tasks)
    assert _names(out) == ["bill", "finance"]


def test_prune_consume_consult_normal_question():
    """2026-09-05 回归（用户投诉句）：'吃一顿饭花了600块正常吗' 无记账指令 → 仅 finance 咨询，bill 裁掉"""
    tasks = [
        {"name": "bill", "raw_segments": ["吃一顿饭花了600块"], "operate_sub_type": "add"},
        {"name": "finance", "raw_segments": ["吃一顿饭花了600块正常吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("吃一顿饭花了600块正常吗", tasks)
    assert _names(out) == ["finance"]


def test_prune_record_cmd_overrides_question():
    """2026-09-05 对照：同一消费带记账指令（'帮我记一下打车80块，正常吗'）→ bill+finance 都保留"""
    tasks = [
        {"name": "bill", "raw_segments": ["打车80块"], "operate_sub_type": "add"},
        {"name": "finance", "raw_segments": ["正常吗"], "operate_sub_type": "analyse"},
    ]
    out = prune_tasks_by_heuristics("帮我记一下打车80块，正常吗", tasks)
    assert _names(out) == ["bill", "finance"]


def test_prune_refuse_bill_single():
    """M4: '别记这笔' 模型规划 bill（含'记'字误判）→ 返回空任务，坚决不记账"""
    tasks = [{"name": "bill", "raw_segments": ["别记这笔"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("别记这笔", tasks)
    assert out == []


def test_prune_refuse_bill_with_amount():
    """M4: '先不记这300块' 即使带金额，也拒绝记账"""
    tasks = [{"name": "bill", "raw_segments": ["先不记这300块"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("先不记这300块", tasks)
    assert out == []


def test_prune_refuse_bill_keep_stat():
    """M4: '别记这笔，看看我最近吃火锅多不多' 模型规划 bill+stat → 只保留 stat（不记账但统计）"""
    tasks = [
        {"name": "bill", "raw_segments": ["别记这笔"], "operate_sub_type": "add"},
        {"name": "stat", "raw_segments": ["看看我最近吃火锅多不多"], "operate_sub_type": "query"},
    ]
    out = prune_tasks_by_heuristics("别记这笔，看看我最近吃火锅多不多", tasks)
    assert _names(out) == ["stat"]


def test_prune_refuse_bill_just_asking():
    """M4: '奶茶15块，我只是问问' → 拒绝记账"""
    tasks = [{"name": "bill", "raw_segments": ["奶茶15块，我只是问问"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("奶茶15块，我只是问问", tasks)
    assert out == []


def test_prune_pure_price_review_unchanged():
    """M4 回归: '吃饭多少钱值吗'（纯比价，无金额无记账上下文）→ 保持信任模型原始规划 price"""
    tasks = [{"name": "price", "raw_segments": ["吃饭多少钱值吗"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("吃饭多少钱值吗", tasks)
    assert _names(out) == ["price"]


def test_prune_income_intercept_regression():
    """M4 回归: 收入场景依旧拦截记账"""
    tasks = [{"name": "bill", "raw_segments": ["工资8000到账"], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics("工资8000到账", tasks)
    assert out == []


def test_prune_stat_hint_more_less():
    """M4: '多不多' 计入统计意图，模型规划 stat 保留"""
    tasks = [{"name": "stat", "raw_segments": ["火锅多不多"], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics("看看我最近吃火锅多不多", tasks)
    assert _names(out) == ["stat"]


# ===================== M4 需求5：plan_node 多意图金额上下文补全（finance 也补全） =====================
@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_plan_node_finance_param_backfill(mock_parse, base_orch_state):
    """M4: '记一下花了300买耳机，值吗' 模型 finance 片段缺金额 → 金额上下文补全到 finance"""
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
    """真实记账指令词仍命中：记/记录/记账/记一下/记一笔/写入/入账等"""
    for text in (
        "记打车35元", "记午饭56元", "记录奶茶18", "记账打车80",
        "帮我记一下打车80", "记一笔奶茶15", "写入账本300", "入账打车80",
        "帮我记个账，打车35", "把奶茶记下来", "记上打车80",
    ):
        assert nodes._has_record_cmd(text), f"应命中真实记账指令: {text}"


def test_has_record_cmd_pseudo_words_not_trigger():
    """含'记/记录'字串但非记账指令的词一律不命中（修复胡乱记账核心）"""
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
    """'我记得打车花了80' 模型误规划 bill → 裁剪掉，绝不落库"""
    for text in ("我记得打车花了80", "买行车记录仪花了300", "我日记里写了买书80"):
        tasks = [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}]
        out = prune_tasks_by_heuristics(text, tasks)
        assert out == [], f"伪记账词场景 bill 应被裁剪: {text}"


def test_prune_pseudo_record_word_no_force_bill():
    """'我记得打车花了80' 模型漏规划（空/stat）→ 不触发 force_bill 补 bill"""
    for text in ("我记得打车花了80", "行车记录仪花了300"):
        tasks = [{"name": "stat", "raw_segments": [text], "operate_sub_type": "query"}]
        out = prune_tasks_by_heuristics(text, tasks)
        assert "bill" not in [t.get("name") for t in out], f"不应强制记账: {text}"


def test_prune_mixed_pseudo_plus_real_record_cmd():
    """'记得把打车80记上' 虽含'记得'但另有真实指令'记上' → 仍记账"""
    text = "记得把打车80记上"
    tasks = [{"name": "bill", "raw_segments": [text], "operate_sub_type": "add"}]
    out = prune_tasks_by_heuristics(text, tasks)
    assert [t.get("name") for t in out] == ["bill"]