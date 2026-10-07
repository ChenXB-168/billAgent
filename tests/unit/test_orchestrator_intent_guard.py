# -*- coding: utf-8 -*-
"""
M16（D18）意图层能力补齐专项单测 —— 契约 `09` §2.14 / 方案 `设计/14` §5.3。

覆盖六组：
  ① 三分支（OOD / 缺参 / 低置信）各自「不派发 + 走 collect_node 旁路」
  ② 硬判断优先：明确记账指令（_is_force_bill）不被软判断误伤
  ③ 契约校验 _check_contract（代码判定 ∪ LLM 自报；bill 的 edit/delete 豁免）
  ④ 参数落地校验 _verify_grounding（编造数字被剔除 / 原文数字保留）
  ⑤ 预规划追问 key（_preplan_ask_key）
  ⑥ 向后兼容：规划输出缺新字段时走旧逻辑（零回归保护）
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langgraph.types import Command
from agents.orchestrator.state import OrchState
from agents.orchestrator.nodes import (
    plan_node, ORCH_AGENT_ID,
    _check_contract, _verify_grounding, _preplan_ask_key, _contract_required,
    _check_inferable_grounding, _has_text_basis,
)


def _make_state(user_input: str) -> OrchState:
    """构造干净的 OrchState（与 test_orchestrator_plan.py 的 base_orch_state 同口径）。"""
    return OrchState(
        session_id="test_sess_m16",
        user_input=user_input,
        session_history="",
        task_plan={},
        current_task=None,
        all_task_results=[],
        error_msg=None,
        final_reply="",
        current_agent=ORCH_AGENT_ID,
        current_sub_agent_struct=None,
        agent_result_cache={},
        dispatch_round=0,
    )


@pytest.fixture(autouse=True)
def mock_deps():
    """打桩会话记忆与用户配置（与 test_orchestrator_plan.py 同口径，避免触碰真实记忆/DB）。"""
    mock_mem = MagicMock()
    mock_mem.get_history.return_value = ""
    mock_mem.export_all.return_value = {}
    mock_mem.add_msg = MagicMock()
    mock_mem.incr_ask_round = MagicMock(return_value=1)
    mock_mem.reset_ask_round = MagicMock()
    mock_mem.clear_draft = MagicMock()
    with patch("agents.orchestrator.nodes.get_session_memory", return_value=mock_mem), \
         patch("agents.orchestrator.nodes.get_user_target_config",
               AsyncMock(return_value={"month_budget": 3000, "target_save_rate": 0.3})):
        yield


# ======================= ① 三分支 =======================

@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_ood_out_of_scope_no_dispatch(mock_parse):
    """超范围（scope_check=out_of_scope）→ 不派发、回引导式兜底、走 collect_node。"""
    mock_parse.return_value = {
        "tasks": [],
        "pre_check_pass": True,
        "block_tip": "",
        "scope_check": "out_of_scope",
        "confidence": 0.95,
        "missing": {},
        "clarify_options": [],
        "boundary_reason": "订票非本系统能力范畴",
    }
    cmd: Command = await plan_node(_make_state("帮我订张机票"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}
    assert "暂时处理不了" in cmd.update["final_reply"]


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_missing_asks_before_dispatch(mock_parse):
    """比价缺金额（LLM 自报 missing）→ 不派发、追问（优先用候选选项）。"""
    mock_parse.return_value = {
        "tasks": [{"name": "price", "raw_segments": ["KTV 消费 是否偏高"],
                   "operate_sub_type": "query"}],
        "pre_check_pass": True,
        "block_tip": "",
        "scope_check": "in_scope",
        "confidence": 0.8,
        "missing": {"price": ["amount"]},
        "clarify_options": ["查KTV一般价位", "评估一笔具体消费是否合理"],
        "boundary_reason": "",
    }
    cmd: Command = await plan_node(_make_state("我打算去KTV，选择什么价位合适"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}
    assert "KTV" in cmd.update["final_reply"]


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_missing_detected_by_code_not_llm(mock_parse):
    """缺参由【代码判定】发现（LLM 未自报 missing）→ 仍不派发（写任务缺参场景）。

    注：M18 起 `price` 的 amount 已可选，故此用 `bill.add`（写任务）验证"代码判定兜底"路径。
    """
    mock_parse.return_value = {
        "tasks": [{"name": "bill", "raw_segments": ["记一下"],
                   "operate_sub_type": "add"}],
        "pre_check_pass": True,
        "block_tip": "",
        "scope_check": "in_scope",
        "confidence": 0.9,
        "missing": {},                      # ★ LLM 没报，靠代码判定兜住
        "clarify_options": [],
        "boundary_reason": "",
    }
    cmd: Command = await plan_node(_make_state("记一下"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}
    assert "金额" in cmd.update["final_reply"]      # 话术用中文标签而非 amount


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_low_confidence_clarify(mock_parse):
    """低置信（confidence < 阈值）→ 不派发、回候选澄清。"""
    mock_parse.return_value = {
        "tasks": [{"name": "finance", "raw_segments": ["帮我看看"],
                   "operate_sub_type": "analyse"}],
        "pre_check_pass": True,
        "block_tip": "",
        "scope_check": "in_scope",
        "confidence": 0.2,
        "missing": {},
        "clarify_options": ["查统计", "要建议"],
        "boundary_reason": "",
    }
    cmd: Command = await plan_node(_make_state("帮我弄一下"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}
    assert "您是想要" in cmd.update["final_reply"]


# ======================= ② 硬判断优先 =======================

@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_force_bill_bypasses_three_branches(mock_parse):
    """明确记账指令 → 硬判断放行，即使模型误报 out_of_scope + 低置信（红线 3）。"""
    mock_parse.return_value = {
        "tasks": [{"name": "bill", "raw_segments": ["打车 交通 35元"],
                   "operate_sub_type": "add"}],
        "pre_check_pass": True,
        "block_tip": "",
        "scope_check": "out_of_scope",      # ★ 模型误报
        "confidence": 0.1,                  # ★ 模型低置信
        "missing": {},
        "clarify_options": [],
        "boundary_reason": "",
    }
    cmd: Command = await plan_node(_make_state("记打车35元"))
    assert cmd.goto == "dispatch_node"
    assert "bill" in cmd.update["task_plan"]


# ======================= ③ 契约校验 =======================

def test_contract_required_bill_add_needs_amount():
    assert "amount" in _contract_required("bill", "add")


def test_contract_required_bill_edit_is_exempt():
    """bill 的 edit/delete 豁免金额（bill_agent/nodes.py:276 已放宽前置拦截）。"""
    assert _contract_required("bill", "edit") == []
    assert _contract_required("bill", "delete") == []


def test_contract_required_price_amount_now_optional():
    """★M18：price 的 amount 已改为可选（有品类无金额 → 走「价位查询」分支）。"""
    assert _contract_required("price", "query") == []


def test_contract_required_stat_finance_empty():
    assert _contract_required("stat", "query") == []
    assert _contract_required("finance", "analyse") == []


def test_check_contract_code_detection():
    """代码判定：**写任务**（bill.add）片段无金额 → missing。

    注：M18 起 price 无金额合法（价位查询），故改用 bill.add 验证代码判定。
    """
    tasks = [{"name": "bill", "raw_segments": ["记奶茶"],
              "operate_sub_type": "add"}]
    assert _check_contract(tasks, "记奶茶", {}) == {"bill": ["amount"]}


def test_check_contract_llm_reported_read_missing():
    """LLM 自报的**只读**缺参仍被采纳（代码判定之外的第二个来源）。"""
    tasks = [{"name": "price", "raw_segments": ["唱歌 是否偏高"],
              "operate_sub_type": "query"}]
    assert _check_contract(tasks, "唱歌", {"price": ["amount"]}) == {"price": ["amount"]}


def test_check_contract_amount_from_user_text():
    """金额在用户原文里（OR 范围）→ 不算缺参。"""
    tasks = [{"name": "price", "raw_segments": ["火锅 是否偏高"],
              "operate_sub_type": "query"}]
    assert _check_contract(tasks, "火锅120贵不贵", {}) == {}


def test_check_contract_union_with_llm():
    """代码判定 ∪ LLM 自报（保守取并集）。"""
    tasks = [{"name": "bill", "raw_segments": ["奶茶"], "operate_sub_type": "add"}]
    got = _check_contract(tasks, "记奶茶", {"bill": ["category"]})
    assert set(got["bill"]) == {"amount", "category"}


def test_check_contract_bill_edit_no_missing():
    """bill.edit 全链路无缺失（豁免生效）。"""
    tasks = [{"name": "bill", "raw_segments": ["把刚才那笔改成60"],
              "operate_sub_type": "edit"}]
    assert _check_contract(tasks, "把刚才那笔改成60", {}) == {}


# ======================= ④ 参数落地校验 =======================

def test_verify_grounding_keeps_origin_number():
    """原文有的数字 → 保留。"""
    kept, dropped = _verify_grounding(["打车 交通 80元 是否偏高"], "那打车花了80块呢")
    assert dropped == []
    assert "80" in kept[0]


def test_verify_grounding_drops_fabricated_number():
    """归一化编造的数字（原文没有 300）→ 被剔除并留痕（红线 1）。"""
    kept, dropped = _verify_grounding(["KTV 300元 是否偏高"], "我打算去KTV")
    assert "300" in dropped
    assert "300" not in kept[0]


# ======================= ⑤ 预规划追问 key =======================

def test_preplan_ask_key():
    assert _preplan_ask_key("missing") == "preplan_missing"
    assert _preplan_ask_key("") == "preplan_unknown"


def test_missing_ask_does_not_leak_internal_task_name():
    """
    兜底话术**不得**向用户暴露内部任务名（如 price）——2026-10-08 用户实测反馈：
    原话术输出 "price 缺金额"，用户看不懂。改后应使用中文标签「比价」。
    """
    from agents.orchestrator.nodes import _build_missing_ask
    msg = _build_missing_ask({"price": ["amount"]}, [])
    assert "price" not in msg
    assert "比价" in msg
    assert "金额" in msg


# ======================= ⑥ 向后兼容（回归保护） =======================

@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_legacy_output_without_new_fields(mock_parse):
    """规划输出缺 5 个新字段 → 走旧逻辑，不触发任何分支（零回归）。"""
    mock_parse.return_value = {
        "tasks": [{"name": "bill", "raw_segments": ["午饭 餐饮 56元"],
                   "operate_sub_type": "add"}],
        "pre_check_pass": True,
        "block_tip": "",
        # ★ 无 scope_check / confidence / missing / clarify_options / boundary_reason
    }
    cmd: Command = await plan_node(_make_state("记午饭56元"))
    assert cmd.goto == "dispatch_node"
    assert "bill" in cmd.update["task_plan"]


# ======================= ⑦ M17：分档决策（fit_level 四档） =======================

def _fit_return(fit_level, tasks, **kw):
    """构造带 fit_level 的规划返回。"""
    base = {
        "tasks": tasks,
        "pre_check_pass": True,
        "block_tip": "",
        "scope_check": "out_of_scope" if fit_level == "out_of_scope" else "in_scope",
        "confidence": 0.8,
        "missing": {},
        "clarify_options": [],
        "boundary_reason": "",
        "fit_level": fit_level,
        "fit_reason": "test",
    }
    base.update(kw)
    return base


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_inferable_conservative_plan_dispatches(mock_parse):
    """
    ★核心用例：fit_level=inferable + 参数齐备 + 有话语依据
    → **放行**（保守规划生效），且**严禁规划 bill**。
    """
    mock_parse.return_value = _fit_return("inferable", [
        {"name": "stat", "raw_segments": ["查询本月预算余额与娱乐支出"],
         "operate_sub_type": "query"},
        {"name": "finance", "raw_segments": ["结合剩余预算建议唱歌花费"],
         "operate_sub_type": "analyse"},
    ])
    cmd: Command = await plan_node(_make_state("我想去唱歌，预算多少合适"))
    assert cmd.goto == "dispatch_node"
    assert "stat" in cmd.update["task_plan"]
    assert "finance" in cmd.update["task_plan"]
    assert "bill" not in cmd.update["task_plan"]      # ★ 写操作永不推断


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_inferable_without_basis_downgrades(mock_parse):
    """fit_level=inferable 但片段与原文无公共子串（瞎猜）→ 降级追问。"""
    mock_parse.return_value = _fit_return("inferable", [
        {"name": "stat", "raw_segments": ["查询本月北京房租均价"],
         "operate_sub_type": "query"},
    ])
    cmd: Command = await plan_node(_make_state("帮我弄一下"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_inferable_price_without_amount_dispatches(mock_parse):
    """★M18 核心用例：inferable + price 无金额（价位查询合法）→ **放行**。

    这是"花费前决策辅助"链路的关键一环：
    「我想去唱歌，预算多少合适」→ price（价位）+ stat（余额）+ finance（建议）全部放行。
    """
    mock_parse.return_value = _fit_return("inferable", [
        {"name": "price", "raw_segments": ["唱歌 娱乐 价位"], "operate_sub_type": "query"},
        {"name": "stat", "raw_segments": ["查询本月预算余额"], "operate_sub_type": "query"},
        {"name": "finance", "raw_segments": ["结合剩余预算建议"], "operate_sub_type": "analyse"},
    ])
    cmd: Command = await plan_node(_make_state("我想去唱歌，预算多少合适"))
    assert cmd.goto == "dispatch_node"
    assert "price" in cmd.update["task_plan"]
    assert "stat" in cmd.update["task_plan"]
    assert "finance" in cmd.update["task_plan"]
    assert "bill" not in cmd.update["task_plan"]


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_write_missing_asks_even_in_inferable(mock_parse):
    """★写任务缺参 → 即使 fit_level=inferable 也一律追问（写永不推断）。"""
    mock_parse.return_value = _fit_return("inferable", [
        {"name": "bill", "raw_segments": ["记一下"], "operate_sub_type": "add"},
    ])
    cmd: Command = await plan_node(_make_state("记一下"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}
    assert "金额" in cmd.update["final_reply"]


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_need_user_asks(mock_parse):
    """fit_level=need_user（已授权但缺参）→ 追问。"""
    mock_parse.return_value = _fit_return("need_user", [
        {"name": "bill", "raw_segments": ["记一下"], "operate_sub_type": "add"},
    ], missing={"bill": ["amount"]},
        clarify_options=["这笔花了多少钱？"])
    cmd: Command = await plan_node(_make_state("记一下"))
    assert cmd.goto == "collect_node"
    assert cmd.update["task_plan"] == {}


@pytest.mark.asyncio
@patch("agents.orchestrator.nodes.parse_json_output")
async def test_fit_level_defaults_from_scope_check(mock_parse):
    """未输出 fit_level 时由 scope_check 推导：out_of_scope → 同档；否则 direct。"""
    mock_parse.return_value = {
        "tasks": [],
        "pre_check_pass": True, "block_tip": "",
        "scope_check": "out_of_scope", "confidence": 0.9,
        "missing": {}, "clarify_options": [], "boundary_reason": "",
        # ★ 无 fit_level
    }
    cmd: Command = await plan_node(_make_state("帮我订张机票"))
    assert cmd.goto == "collect_node"
    assert "暂时处理不了" in cmd.update["final_reply"]


def test_has_text_basis():
    assert _has_text_basis("唱歌 是否偏高", "我想去唱歌") is True
    assert _has_text_basis("查询北京房租均价", "帮我弄一下") is False


# ======================= ⑧ M18：价位查询结果的渲染 =======================

def test_render_price_display_text_price_query_mode():
    """★M18：价位查询结果（premium_rate=None 但有均价）→ 渲染为可读参考价位，不 dump dict。"""
    from agents.orchestrator.nodes import _render_price_display_text
    data = {
        "city": "深圳", "category": "娱乐",
        "user_consume_amount": None, "city_base_avg_price": 120.0,
        "premium_rate": None, "consume_level": None, "conclusion": None,
    }
    text = _render_price_display_text("价位查询", data)
    assert "120" in text
    assert "参考均价" in text
    assert "{" not in text          # ★ 不得 dump dict


def test_render_price_display_text_premium_mode_unchanged():
    """M18：有溢价率的原路径渲染保持不变（零回归）。"""
    from agents.orchestrator.nodes import _render_price_display_text
    data = {"premium_rate": 15.0, "consume_level": "偏高", "conclusion": "略高"}
    text = _render_price_display_text("物价对比", data)
    assert "溢价率" in text and "偏高" in text


def test_check_inferable_grounding_ok():
    tasks = [{"name": "stat", "raw_segments": ["查询本月预算余额"],
              "operate_sub_type": "query"}]
    ok, reason = _check_inferable_grounding(tasks, "我想去唱歌，预算多少合适")
    assert ok is True


def test_check_inferable_grounding_write_missing_fails():
    """防御性：**写任务**缺参 → 依据校验不成立（参数齐备判据）。

    注：实际路径中写任务缺参已被决策 ② 先行拦下，此处仅锁守门人自身行为。
    """
    tasks = [{"name": "bill", "raw_segments": ["记一下"], "operate_sub_type": "add"}]
    ok, reason = _check_inferable_grounding(tasks, "记一下")
    assert ok is False
    assert "缺参" in reason


def test_check_inferable_grounding_price_no_amount_is_ok():
    """★M18：price 无金额不再导致依据校验失败（价位查询合法）。"""
    tasks = [{"name": "price", "raw_segments": ["唱歌 是否偏高"],
              "operate_sub_type": "query"}]
    ok, reason = _check_inferable_grounding(tasks, "唱歌一般多少钱")
    assert ok is True


def test_check_inferable_grounding_no_basis_fails():
    tasks = [{"name": "stat", "raw_segments": ["查询北京房租均价"],
              "operate_sub_type": "query"}]
    ok, reason = _check_inferable_grounding(tasks, "帮我弄一下")
    assert ok is False
