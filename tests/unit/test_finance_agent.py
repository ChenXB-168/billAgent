# ==============================================
# finance_agent（理财分析 Agent）单元测试
# 覆盖 build_prompt→execute→verify→reply 四节点、D16 事实底稿与数字白名单防幻觉闸门、
# D9 Reflection 语义自检与整图自纠闭环；mock 掉 LLM 与 A2A 回传，不触达真实服务。
# ==============================================
import json
import pytest
from unittest.mock import patch, AsyncMock
from langgraph.graph import END
from agents.finance_agent.state import FinanceState
from agents.finance_agent.graph import finance_agent_graph
from agents.finance_agent.nodes import (
    build_prompt_node,
    execute_finance_node,
    verify_query_node,
    reply_node,
    FINANCE_AGENT_ID,
    MAX_VERIFY_ATTEMPTS,
    KEY_SUCCESS,
    KEY_AGENT_TYPE,
    KEY_DATA,
    KEY_ERROR,
    _build_fact_sheet,
    _render_fact_fallback,
    _build_finance_whitelist,
    _suggestion_exempt_numbers,
    _violating_numbers,
)

# 统一模拟MCP返回文本容器，和price/stat测试保持一致
class TextContent:
    """
    函数/类功能与逻辑描述：
        模拟 MCP 工具返回的文本内容容器，仅持有 text 字段；
        与 price/stat 测试保持同一替身形态。
    构造入参说明：
        text：工具返回的文本内容。
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


@pytest.fixture
def base_finance_state() -> FinanceState:
    """
    函数功能与逻辑描述：
        提供一份结构完整的空 FinanceState 基础夹具（operate_sub_type 固定 "analyse"），
        供各用例在其上覆写字段；function 作用域，每个用例独立，无清理逻辑。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        FinanceState：含默认空字段的理财 Agent 状态。
    """
    return FinanceState(
        raw_segments=[],
        operate_sub_type="analyse",
        session_memory={},
        target_control=None,
        bill_result=None,
        stat_result=None,
        price_result=None,
        user_query=None,
        llm_input_prompt=None,
        llm_raw_answer=None,
        final_struct=None,
        error_stack=None
    )


# 用例1：仅询问单笔消费是否合适（LLM自动输出极简短句）
@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_finance_single_simple_ask(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        单笔消费是否合适的咨询：build_prompt_node 拼接提示词须含用户原文，execute_finance_node 调用理财模型；
        断言成功载荷含 raw_analysis_text、has_price/stat/target_data 均为 True、fallback_used=False，
        且底稿 monthly_budget=3000（D16：成功载荷携带底稿与兜底标记）。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回固定分析文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = "该消费高于本地同品类均价，本月餐饮预算已超支，不建议购买。"
    state = base_finance_state
    state["raw_segments"] = ["这家咖啡35一杯，现在买合适吗？"]
    # 填充前置数据
    state["target_control"] = {"month_budget": 3000, "target_save_rate": 0.3}
    state["stat_result"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3300, "category_ratio": {"餐饮": 0.45}}
    }
    state["price_result"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: "price_agent",
        KEY_DATA: {"avg_price": 22, "user_amount": 35, "premium_rate": 59.0}
    }

    # 执行拼接prompt节点
    cmd_build = await build_prompt_node(state)
    state_after_build = cmd_build.update
    assert state_after_build["llm_input_prompt"] is not None
    assert "这家咖啡35一杯，现在买合适吗？" in state_after_build["llm_input_prompt"]

    # 执行理财模型调用节点
    cmd_exec = await execute_finance_node(state_after_build)
    payload = cmd_exec.update["final_struct"]
    assert payload[KEY_SUCCESS] is True
    assert payload[KEY_AGENT_TYPE] == FINANCE_AGENT_ID
    assert "raw_analysis_text" in payload[KEY_DATA]
    assert "高于本地同品类均价" in payload[KEY_DATA]["raw_analysis_text"]
    assert payload[KEY_DATA]["has_price_data"] is True
    assert payload[KEY_DATA]["has_stat_data"] is True
    assert payload[KEY_DATA]["has_target_data"] is True
    # D16（M11）：成功载荷携带底稿与兜底标记
    assert payload[KEY_DATA]["fallback_used"] is False
    assert payload[KEY_DATA]["fact_sheet"]["monthly_budget"] == 3000


# 用例2：用户询问月度整体开销，需要标准四段式分析
@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_finance_month_stat_analyse(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        月度整体开销咨询（标准四段式分析）：构造预算 3000、支出 3200 的统计结果且无物价数据；
        断言分析文本含"月度消费概况""可落地省钱建议"、has_price_data=False、fallback_used=False，
        且底稿 usage_percent=107、over_by=200。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回四段式分析文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = """一、月度消费概况
本月总支出3200元，月度预算3000元，已超出预算200元，当前支出已达预算的107%，消费节奏偏宽松。
二、消费问题诊断
餐饮类目（占42%）溢价消费较多，多次购买高于城市均价的饮品，拉高整体开销。
三、可落地省钱建议
1. 减少溢价咖啡奶茶消费，优先把超支的200元缺口补回来；
2. 控制线下餐饮频次，压低42%的餐饮占比。
四、预算优化方案
下月总预算维持3000元，其余类目按实际消费节奏分配。"""
    state = base_finance_state
    state["raw_segments"] = ["帮我看看这个月整体开销，有没有超预算，哪里可以省钱"]
    state["target_control"] = {"month_budget": 3000, "target_save_rate": 0.3}
    state["stat_result"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200, "category_ratio": {"餐饮": 0.42, "交通": 0.15}}
    }
    # 无单笔物价数据
    state["price_result"] = None

    cmd_build = await build_prompt_node(state)
    state_after_build = cmd_build.update
    cmd_exec = await execute_finance_node(state_after_build)
    payload = cmd_exec.update["final_struct"]
    text = payload[KEY_DATA]["raw_analysis_text"]
    assert "月度消费概况" in text
    assert "可落地省钱建议" in text
    assert payload[KEY_DATA]["has_price_data"] is False
    # D16（M11）：白名单放行无兜底，底稿数字随载荷下发
    assert payload[KEY_DATA]["fallback_used"] is False
    assert payload[KEY_DATA]["fact_sheet"]["usage_percent"] == 107
    assert payload[KEY_DATA]["fact_sheet"]["over_by"] == 200


# 用例3：用户索要长期存钱规划，深度完整方案
@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_finance_long_term_save_plan(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        长期存钱规划咨询：构造预算 3000、支出 2900、储蓄率 0.2 的统计结果且无物价数据；
        断言传给理财模型的 user_input 含"长期存钱方案"、文本含 30%/20%、
        fallback_used=False，且底稿 save_rate_percent=20、usage_percent=97。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回存钱方案文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = """一、月度消费概况
本月总支出2900元，未超3000元预算，当前支出为预算的97%，预算内剩余100元；储蓄率20%，未达30%储蓄目标。
二、消费问题诊断
休闲餐饮溢价消费占比较高，非必要冲动消费较多，挤压储蓄空间。
三、可落地省钱建议
1. 奶茶咖啡消费频次减半，把储蓄率从20%逐步提升到30%目标；
2. 外卖替换家常做饭，进一步压缩月度非必要餐饮支出；
3. 非刚需服饰延后购买，把结余的100元转入储蓄。
四、预算优化方案
下月总预算维持3000元，支出结构按30%储蓄目标反向压缩非必要类目。"""
    state = base_finance_state
    state["raw_segments"] = ["给我一套长期存钱方案，调整每月消费结构，提高储蓄比例"]
    state["target_control"] = {"month_budget": 3000, "target_save_rate": 0.3}
    state["stat_result"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 2900, "save_rate": 0.2}
    }
    state["price_result"] = None

    cmd_build = await build_prompt_node(state)
    state_after_build = cmd_build.update
    cmd_exec = await execute_finance_node(state_after_build)
    payload = cmd_exec.update["final_struct"]
    text = payload[KEY_DATA]["raw_analysis_text"]
    # 拆分三元入参 agent_id, sys, user
    _, sys_content, user_input = mock_finance_llm.call_args[0]
    assert "长期存钱方案" in user_input
    # D16（M11）：建议只引用白名单内数字（储蓄率 20%→30% 目标来自注入事实，非编造额度）
    assert "30%" in text and "20%" in text
    assert payload[KEY_DATA]["fallback_used"] is False
    assert payload[KEY_DATA]["fact_sheet"]["save_rate_percent"] == 20
    assert payload[KEY_DATA]["fact_sheet"]["usage_percent"] == 97


# 用例4：无预算、无统计数据，触发need_more_info多轮追问
@pytest.mark.asyncio
async def test_finance_missing_core_data_need_prompt(base_finance_state):
    """
    函数功能与逻辑描述：
        关键数据缺失场景：无预算、无统计数据、无物价数据时 build_prompt_node 直接返回追问载荷；
        断言 success=False、error="need_more_info"、prompt 含"请先记录账单或设置月度预算目标"。
    入参说明：
        base_finance_state：pytest fixture 注入的基础 FinanceState（用例内清空关键数据）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["raw_segments"] = ["分析一下我的消费情况，怎么省钱"]
    # 关键数据全部为空
    state["target_control"] = None
    state["stat_result"] = None
    state["price_result"] = None

    cmd = await build_prompt_node(state)
    payload = cmd.update["final_struct"]
    assert payload[KEY_SUCCESS] is False
    assert payload[KEY_ERROR] == "need_more_info"
    assert "请先记录账单或设置月度预算目标" in payload[KEY_DATA]["prompt"]


# 用例5：理财专用LLM调用异常分支捕获
@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance", side_effect=Exception("模型服务超时断开"))
async def test_finance_llm_call_exception(mock_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        理财专用 LLM 调用异常分支：call_llm_finance 抛异常时返回失败载荷；
        断言 success=False、error 含"理财分析服务异常"、data 为 None。
    入参说明：
        mock_llm：patch 掉 `mcp_client.call_llm_finance`，以 side_effect 抛异常。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["raw_segments"] = ["分析本月开销"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 2600}
    }

    cmd_build = await build_prompt_node(state)
    state_after_build = cmd_build.update
    cmd_exec = await execute_finance_node(state_after_build)
    payload = cmd_exec.update["final_struct"]
    assert payload[KEY_SUCCESS] is False
    assert "理财分析服务异常" in payload[KEY_ERROR]
    assert payload[KEY_DATA] is None


# 用例6：prompt组装阶段代码异常捕获
# 正确patch路径：agents.finance_agent.nodes.render_finance_user
@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.render_finance_user", side_effect=Exception("模板渲染失败"))
async def test_build_prompt_render_error(mock_render, base_finance_state):
    """
    函数功能与逻辑描述：
        prompt 组装阶段异常捕获：render_finance_user 抛异常时返回失败载荷；
        断言 success=False 且 error 含"模板渲染失败"。
    入参说明：
        mock_render：patch 掉 `render_finance_user`，以 side_effect 抛异常。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["raw_segments"] = ["看看这笔消费划算吗"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {"success": True, "agent_type": "stat_agent", "data": {}}

    cmd = await build_prompt_node(state)
    payload = cmd.update["final_struct"]
    assert payload[KEY_SUCCESS] is False
    assert "模板渲染失败" in payload[KEY_ERROR]


# 用例7：reply_node A2A消息推送校验，验证正常发给orchestrator
@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.a2a_bus.send_result")
async def test_finance_reply_a2a_send(mock_send, base_finance_state):
    """
    函数功能与逻辑描述：
        reply_node 经 a2a_bus.send_result 回传结果：契约为 (task_id, result)，
        result 为字符串；断言只发送一次且 goto 为 LangGraph 终止节点 END。
    入参说明：
        mock_send：patch 掉 `a2a_bus.send_result`。
        base_finance_state：pytest fixture 注入的基础 FinanceState（用例内写入最终 struct）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["session_id"] = "sess_test"
    state["task_id"] = "task_test"
    state["final_struct"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: FINANCE_AGENT_ID,
        "msg": "理财分析完成",
        KEY_DATA: {
            "raw_analysis_text": "本月预算充足，该笔小额消费无压力，可以正常消费",
            "has_price_data": True,
            "has_stat_data": True,
            "has_target_data": True
        },
        KEY_ERROR: None
    }

    cmd = await reply_node(state)
    # 校验A2A发送一次
    mock_send.assert_called_once()
    call_args = mock_send.call_args.kwargs
    # ★M2 验收③：send_result 契约为 (task_id, result)，target_agent/session_id 死参数已删除
    assert call_args["task_id"] == "task_test"
    assert isinstance(call_args["result"], str)
    assert cmd.goto == END


# ===================== M11 D16 防幻觉：底稿构建 / 白名单闸门 =====================

def test_fact_sheet_percent_integerized_and_over_fields():
    """
    函数功能与逻辑描述：
        事实底稿（确定性计算层）：由预算/统计/物价三份数据计算预算、支出、结余、超支字段，
        并将小数比例转为整数百分比；断言 monthly_budget=3000、month_spend=3200、balance=-200、
        is_over_budget=True、over_by=200、usage_percent=107、over_percent=7，
        比例字段 target_save_rate_percent=30 / save_rate_percent=20 / category_ratio_percent={42,15}，
        且 price_data 的 premium_rate 原样收进底稿。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sheet = _build_fact_sheet(
        {"month_budget": 3000, "target_save_rate": 0.3},
        {"month_total_spend": 3200, "category_ratio": {"餐饮": 0.42, "交通": 0.15}, "save_rate": 0.2},
        {"avg_price": 22, "user_amount": 35, "premium_rate": 59.0},
    )
    assert sheet["monthly_budget"] == 3000
    assert sheet["month_spend"] == 3200
    assert sheet["balance"] == -200
    assert sheet["is_over_budget"] is True
    assert sheet["over_by"] == 200
    assert sheet["usage_percent"] == 107
    assert sheet["over_percent"] == 7
    # 小数比例 → 整数百分比（0.3→30 / 0.2→20 / 0.42→42）
    assert sheet["target_save_rate_percent"] == 30
    assert sheet["save_rate_percent"] == 20
    assert sheet["category_ratio_percent"] == {"餐饮": 42, "交通": 15}
    # price_data 确定性基准原样收进底稿
    assert sheet["premium_rate"] == 59.0


def test_fact_sheet_omits_missing_fields():
    """
    函数功能与逻辑描述：
        底稿只收录存在项：未超支时不写超支字段，无储蓄率/物价时不写对应键；
        断言 balance=100、usage_percent=97、is_over_budget=False，且 over_by/over_percent/
        save_rate_percent/avg_price 均不在底稿中。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sheet = _build_fact_sheet(
        {"month_budget": 3000},
        {"month_total_spend": 2900},
        {},
    )
    assert sheet["monthly_budget"] == 3000
    assert sheet["month_spend"] == 2900
    assert sheet["balance"] == 100
    assert sheet["usage_percent"] == 97
    assert sheet["is_over_budget"] is False
    assert "over_by" not in sheet and "over_percent" not in sheet
    assert "save_rate_percent" not in sheet
    assert "avg_price" not in sheet


def test_finance_whitelist_sources():
    """
    函数功能与逻辑描述：
        白名单组成校验：用户原文 ∪ 确定性注入数据（目标/统计/物价/习惯/底稿）中的数字应被收录，
        用户资料（user_docs）数字不入白名单；断言 3000/3200/35/0.3/0.42/22/400.0 均被收录，
        而 25000/5000 不被收录。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = FinanceState(
        raw_segments=[],
        operate_sub_type="analyse",
        session_memory={},
        target_control={"month_budget": 3000, "target_save_rate": 0.3},
        bill_result=None,
        stat_result={
            KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
            KEY_DATA: {"month_total_spend": 3200, "category_ratio": {"餐饮": 0.42}},
        },
        price_result={
            KEY_SUCCESS: True, KEY_AGENT_TYPE: "price_agent",
            KEY_DATA: {"avg_price": 22, "user_amount": 35, "premium_rate": 59.0},
        },
        user_query="这家咖啡35一杯值得吗？",
        llm_input_prompt=None,
        llm_raw_answer=None,
        final_struct=None,
        error_stack=None,
    )
    state["fact_sheet"] = {"monthly_budget": 3000, "month_spend": 3200}
    state["habit_data"] = [{"monthly_habit": "日均1杯奶茶", "last_month_avg": 400.0}]
    state["user_docs"] = [{"doc_id": "d1", "text": "预计下月收入2万5，可投资5000元", "score": 0.8}]
    wl = _build_finance_whitelist(state)
    # 确定性来源数字应被收录（含 400.0 习惯均额）
    for n in (3000, 3200, 35, 0.3, 0.42, 22, 400.0):
        assert any(abs(n - w) < 0.5 for w in wl), f"{n} 应被白名单收录"
    # 用户资料（user_docs）数字不入白名单（25000 / 5000）
    for n in (25000, 5000):
        assert not any(abs(n - w) < 0.5 for w in wl), f"{n} 不应入白名单（用户资料非确定性来源）"


def test_fallback_render_integerized():
    """
    函数功能与逻辑描述：
        规则兜底渲染：只渲染底稿确定性数字，不出现小数比例/编造建议；
        断言文本含"数据底稿""当月支出合计 3200 元""已超出预算 200 元（达预算的 107%）"，
        且不含"0.42""106."。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    text = _render_fact_fallback(
        {
            "monthly_budget": 3000, "month_spend": 3200, "balance": -200,
            "usage_percent": 107, "is_over_budget": True, "over_by": 200, "over_percent": 7,
        },
        "这个月超支了吗",
    )
    assert "数据底稿" in text
    assert "当月支出合计 3200 元" in text
    assert "已超出预算 200 元（达预算的 107%）" in text
    assert "0.42" not in text and "106." not in text


def test_fallback_render_states_missing_data_and_next_step():
    """
    函数功能与逻辑描述：
        ★2026-09-17 新增（实测驱动）：兜底渲染在**底稿数据不全**时，必须说明"缺什么 + 下一步"，
        而不是只堆已有数字。来源：e2e `empty_habits`（空习惯且本轮无 stat 任务）时底稿只有
        `monthly_budget`，原实现整段输出仅「【消费诊断 · 数据底稿】+ 月度预算 X 元」——用户问的是
        规划建议，却只拿到一个数字，既不知为何没有分析、也不知该做什么。
        本用例锁定三条分支：① 只有预算；② 只有支出；③ 两项齐全（不得出现缺失提示）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # ① 只有预算（无当月支出）：须点明缺"当月支出"并给出可行下一步
    only_budget = _render_fact_fallback({"monthly_budget": 2000.0}, "下个月怎么规划")
    assert "数据底稿" in only_budget
    assert "月度预算 2000 元" in only_budget
    assert "暂缺当月支出" in only_budget
    assert "记账或查询当月开销后" in only_budget

    # ② 只有支出（无预算）：须点明缺"月度预算"并给出可行下一步
    only_spend = _render_fact_fallback({"month_spend": 320.0}, "怎么省点钱")
    assert "当月支出合计 320 元" in only_spend
    assert "暂缺月度预算" in only_spend
    assert "设置月度预算后" in only_spend

    # ③ 两项齐全时**不得**出现缺失提示（原有分支不受本次改动影响）
    full = _render_fact_fallback(
        {
            "monthly_budget": 3000, "month_spend": 3200, "balance": -200,
            "usage_percent": 107, "is_over_budget": True, "over_by": 200, "over_percent": 7,
        },
        "这个月超支了吗",
    )
    assert "暂缺" not in full


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_whitelist_gate_passes_for_traceable_numbers(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        白名单闸门放行：模型输出数字全部可溯源时 fallback_used=False、载荷携带底稿；
        断言 goto=verify_query_node、底稿 monthly_budget=3000/over_by=200、只调用理财模型一次。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回可溯源数字文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = "本月总支出3200元，预算3000元，已超支200元（达预算107%）。"
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    cmd_build = await build_prompt_node(state)
    state_after_build = cmd_build.update
    cmd_exec = await execute_finance_node(state_after_build)
    data = cmd_exec.update["final_struct"][KEY_DATA]
    assert cmd_exec.goto == "verify_query_node"
    assert data["fallback_used"] is False
    assert data["fact_sheet"]["monthly_budget"] == 3000
    assert data["fact_sheet"]["over_by"] == 200
    assert "已超支200元" in data["raw_analysis_text"]
    assert mock_finance_llm.call_count == 1


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_whitelist_intercept_then_corrected_on_retry(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        拦截→修正（M11.1 口径=只拦事实断言）：首版把白名单外数字当事实（支出3500/超支500）→
        注入清单修正 1 次 → 次版事实数字全部改用白名单数值 → 放行；
        断言 fallback_used=False、模型调用 2 次、修正版文本不含"3500"且含"转入储蓄"。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，以 side_effect 返回首版违规与修正版文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.side_effect = [
        "本月支出3500元，已超支500元。",        # 首版：3500/500 为事实断言、底稿外 → 违规
        "本月支出3200元，已超支200元。建议把可压缩开销转入储蓄。",  # 修正版：事实数字全合规
    ]
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    cmd_build = await build_prompt_node(state)
    cmd_exec = await execute_finance_node(cmd_build.update)
    data = cmd_exec.update["final_struct"][KEY_DATA]
    assert data["fallback_used"] is False
    assert mock_finance_llm.call_count == 2
    assert "3500" not in data["raw_analysis_text"]
    assert "转入储蓄" in data["raw_analysis_text"]


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_whitelist_intercept_then_fallback_render(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        拦截→修正仍把白名单外数字当事实 → 回退确定性底稿渲染（fallback_used=True，数字 100% 来自底稿）；
        断言 success=True、"3500"不在文本中，且文本含底稿渲染的"已超出预算 200 元""达预算的 107%"。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，每次均返回编造事实的文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = "本月支出3500元，已超支500元，赶紧减少开销。"  # 每次都编造事实
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    cmd_build = await build_prompt_node(state)
    cmd_exec = await execute_finance_node(cmd_build.update)
    payload = cmd_exec.update["final_struct"]
    data = payload[KEY_DATA]
    assert payload[KEY_SUCCESS] is True
    assert data["fallback_used"] is True
    assert "3500" not in data["raw_analysis_text"]
    assert "已超出预算 200 元" in data["raw_analysis_text"]
    assert "达预算的 107%" in data["raw_analysis_text"]


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_suggestion_numbers_exempt_from_whitelist(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        M11.1 建议性数字豁免放行：模式B/C 的节约幅度/预算下调（1100/100 不在白名单），
        只要用建议词框定且位于限定词之后 → 硬闸门放行（fallback_used=False）不逼死表达力；
        断言只调用模型一次且文本保留建议数字与超支事实。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回含建议数字的文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = (
        "本月总支出3200元，预算3000元，已超支200元（达预算107%）。"
        "建议将餐饮预算下调至1100元，每月约可省下100元。"
    )
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    cmd_build = await build_prompt_node(state)
    cmd_exec = await execute_finance_node(cmd_build.update)
    data = cmd_exec.update["final_struct"][KEY_DATA]
    assert data["fallback_used"] is False
    assert mock_finance_llm.call_count == 1
    text = data["raw_analysis_text"]
    assert "下调至1100元" in text and "省下100元" in text
    assert "已超支200元" in text


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_hedging_words_not_exempt_factual_misclaim(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        规避守卫：事实断言加"约/仅/共"等模糊词**不豁免**（这些不是建议限定词）；
        "本月支出约3300元"仍被拦截并修正，断言 fallback_used=False、模型调用 2 次、
        最终文本不含"3300"且含"超支200元"。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，以 side_effect 返回模糊词违规版与修正版。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.side_effect = [
        "本月支出约3300元，共超支约500元。",        # 首版：约不豁免 → 违规
        "本月支出3200元，超支200元。",             # 修正版：合规
    ]
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    cmd_build = await build_prompt_node(state)
    cmd_exec = await execute_finance_node(cmd_build.update)
    data = cmd_exec.update["final_struct"][KEY_DATA]
    assert data["fallback_used"] is False
    assert mock_finance_llm.call_count == 2
    assert "3300" not in data["raw_analysis_text"]
    assert "超支200元" in data["raw_analysis_text"]


def test_suggestion_exempt_scope_and_factual_gate_boundary():
    """
    函数功能与逻辑描述：
        豁免粒度边界：建议限定词**之前**的事实数字仍被拦截（防"本月支出3300元，建议…"规避）；
        限定词之后子句的数字才豁免（1100）；纯事实句（无限定词）不豁免（999 仍违规）；
        并锁定"建议词框定但数字位于事实位"的整句规避属 verify 语义层职责（本层不拦）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    text = "本月支出3300元，建议压缩餐饮。餐饮预算下调至1100元。"
    exempt = _suggestion_exempt_numbers(text)
    assert exempt == {1100.0}
    # 白名单含 3200（底稿支出），违规数字只剩 3300（限定词前的 3300 不受豁免）
    viol = _violating_numbers(text, {3200.0})
    assert viol == {3300.0}
    # 无建议词框定的纯事实断言（如凭空"超支999元"）——无豁免，白名单外即违规
    assert _violating_numbers("本月超支999元，请尽快调整。", {200.0}) == {999.0}
    # 建议词框定但数字是"事实位"的整句规避属 verify 语义层职责（本层不拦，见 checklist ①③）
    assert _violating_numbers("建议：本月实际支出3500元。", {3200.0}) == set()


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_no_whitelist_gate_when_fact_sheet_empty(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        无确定性底稿（无预算/无金额支出，仅笔数）→ 不下白名单闸门：纯建议文本不误杀直发；
        断言 fact_sheet 为空 dict、fallback_used=False 且文本原样输出。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回纯建议文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = "建议每月固定储蓄200元，长期坚持。"
    state = base_finance_state
    state["raw_segments"] = ["怎么存钱"]
    state["target_control"] = {}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"total_count": 3},   # 仅笔数、无金额 → 底稿无预算/支出
    }
    state["price_result"] = None
    state["habit_data"] = []
    cmd_build = await build_prompt_node(state)
    state_after_build = cmd_build.update
    assert state_after_build["fact_sheet"] == {}
    cmd_exec = await execute_finance_node(state_after_build)
    data = cmd_exec.update["final_struct"][KEY_DATA]
    assert data["fallback_used"] is False
    assert data["raw_analysis_text"] == "建议每月固定储蓄200元，长期坚持。"


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_execute_injects_verify_feedback(mock_finance_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        execute 收到 verify_feedback 时追加到 user prompt 引导重生成修正；
        断言 fallback_used=False 且传给模型的 user 段含反馈文本"已超支200元，请勿表述为未超支"。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，返回合规文本。
        base_finance_state：pytest fixture 注入的基础 FinanceState（用例内写入 verify_feedback）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_finance_llm.return_value = "本月支出3200元，已超出预算200元。"
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    state["verify_feedback"] = "底稿显示已超支200元，请勿表述为未超支"
    cmd_build = await build_prompt_node(state)
    cmd_exec = await execute_finance_node(cmd_build.update)
    assert cmd_exec.update["final_struct"][KEY_DATA]["fallback_used"] is False
    _, _, user_part = mock_finance_llm.call_args[0]
    assert "已超支200元，请勿表述为未超支" in user_part


# ===================== M11 D9 Reflection：语义自检节点 =====================

def _ok_finance_state(base_finance_state, text=None, attempt=0):
    """
    函数功能与逻辑描述：
        构造 execute 成功后的 finance 状态：写入固定 task_id/user_query 与一份超支底稿，
        并组装 final_struct.success=True（含 raw_analysis_text/fact_sheet/fallback_used 等）；
        attempt 用于注入 verify_attempt，text 可覆写分析文本。
    入参说明：
        base_finance_state：基础状态夹具（被就地修改后返回）。
        text：raw_analysis_text 文本，默认"本月支出3200元，已超出预算200元。"。
        attempt：verify_attempt 值，默认 0。
    返回值说明：
        FinanceState：构造好的 execute 成功状态。
    """
    st = base_finance_state
    st["task_id"] = "task_fin_verify"
    st["user_query"] = "这个月是不是超支了？"
    st["fact_sheet"] = {
        "monthly_budget": 3000, "month_spend": 3200, "balance": -200,
        "usage_percent": 107, "is_over_budget": True, "over_by": 200, "over_percent": 7,
        "category_ratio_percent": {"餐饮": 42}, "target_save_rate_percent": 30,
    }
    st["verify_attempt"] = attempt
    st["verify_feedback"] = None
    st["final_struct"] = {
        KEY_SUCCESS: True,
        KEY_AGENT_TYPE: FINANCE_AGENT_ID,
        "msg": "理财分析完成",
        KEY_DATA: {
            "raw_analysis_text": text or "本月支出3200元，已超出预算200元。",
            "fact_sheet": st["fact_sheet"], "fallback_used": False,
            "has_price_data": False, "has_stat_data": True, "has_target_data": True,
        },
        KEY_ERROR: None,
    }
    return st


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
async def test_finance_verify_pass_goes_reply(mock_parse, base_finance_state):
    """
    函数功能与逻辑描述：
        语义自检通过分支：校验返回 pass=True，断言 goto=reply_node、verify_decision="reply"。
    入参说明：
        mock_parse：patch 掉 `parse_json_output`（AsyncMock），返回通过的校验结果。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_parse.return_value = {"pass": True, "issues": [], "feedback": ""}
    cmd = await verify_query_node(_ok_finance_state(base_finance_state))
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
async def test_finance_verify_fail_feedback_retry_execute(mock_parse, base_finance_state):
    """
    函数功能与逻辑描述：
        语义校验不过且未达上限：断言反馈回 execute_finance_node 重生成（verify_decision="retry"、
        verify_attempt=1、verify_feedback 携带反馈）。
    入参说明：
        mock_parse：patch 掉 `parse_json_output`（AsyncMock），返回不通过的校验结果。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_parse.return_value = {
        "pass": False,
        "issues": ["底稿显示已超出预算200元，文本却称'预算内尚有结余'"],
        "feedback": "改为超支口径：已超出预算200元（达预算107%）",
    }
    cmd = await verify_query_node(_ok_finance_state(base_finance_state))
    assert cmd.goto == "execute_finance_node"
    assert cmd.update["verify_decision"] == "retry"
    assert cmd.update["verify_attempt"] == 1
    assert "超支口径" in cmd.update["verify_feedback"]


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
async def test_finance_verify_attempt_limit_goes_reply(mock_parse, base_finance_state):
    """
    函数功能与逻辑描述：
        校验连续不过达上限：兜底直发当前结果（不无限循环），断言 goto=reply_node、
        verify_decision="reply"、verify_feedback 置空。
    入参说明：
        mock_parse：patch 掉 `parse_json_output`（AsyncMock），返回始终不通过的结果。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_parse.return_value = {"pass": False, "issues": ["仍不一致"], "feedback": "请再修正"}
    cmd = await verify_query_node(
        _ok_finance_state(base_finance_state, attempt=MAX_VERIFY_ATTEMPTS)
    )
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"
    assert cmd.update["verify_feedback"] is None


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
async def test_finance_verify_llm_exception_skips(mock_parse, base_finance_state):
    """
    函数功能与逻辑描述：
        校验调用异常时跳过自检直发（verify 是增强项，不阻断业务）；
        断言 goto=reply_node、verify_decision="reply"。
    入参说明：
        mock_parse：patch 掉 `parse_json_output`（AsyncMock），以 side_effect 抛异常。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_parse.side_effect = Exception("verify LLM调用失败")
    cmd = await verify_query_node(_ok_finance_state(base_finance_state))
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
async def test_finance_verify_skips_failed_payload(mock_parse, base_finance_state):
    """
    函数功能与逻辑描述：
        execute 失败/追问载荷（success=False）不参与语义自检：断言直达 reply_node 且校验 LLM 未被调用。
    入参说明：
        mock_parse：patch 掉 `parse_json_output`（AsyncMock，断言未被调用）。
        base_finance_state：pytest fixture 注入的基础 FinanceState（用例内写入失败 struct）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["final_struct"] = {
        KEY_SUCCESS: False, KEY_AGENT_TYPE: FINANCE_AGENT_ID, "msg": "理财模型调用失败",
        KEY_DATA: None, KEY_ERROR: "理财分析服务异常",
    }
    cmd = await verify_query_node(state)
    assert cmd.goto == "reply_node"
    mock_parse.assert_not_called()


# ===================== M11 D9 Reflection：整图自纠闭环（验收核心） =====================

@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.a2a_bus.send_result")
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_finance_graph_verify_self_correction_loop(mock_finance_llm, mock_parse, mock_send, base_finance_state):
    """
    函数功能与逻辑描述：
        语义缺陷自纠闭环：文本称"预算内尚有结余"（与超支底稿矛盾）→ verify 反馈重生成 → 通过发送；
        断言最终文本含"已超出预算200元"、不含"预算内尚有结余"、verify_attempt=1、反馈已清空、只发送一次。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，以 side_effect 返回矛盾版与修正版文本。
        mock_parse：patch 掉 `parse_json_output`（AsyncMock），以 side_effect 返回不通过/通过两轮校验。
        mock_send：patch 掉 `a2a_bus.send_result`（断言只发送一次）。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    state["habit_data"] = []
    state["task_id"] = "task_fin_verify_1"
    state["verify_attempt"] = 0
    state["verify_feedback"] = None
    # 两轮文本只含白名单数字（通过硬闸门）；第1轮语义与底稿矛盾、第2轮修正
    mock_finance_llm.side_effect = [
        "本月支出3200元，预算3000元，支出达预算的107%，预算内尚有结余，无需调整。",
        "本月支出3200元，预算3000元，已超出预算200元，支出达预算的107%，建议压缩非必要消费。",
    ]
    mock_parse.side_effect = [
        {"pass": False, "issues": ["底稿显示已超支200元，文本却说'预算内尚有结余'"],
         "feedback": "请按超支口径改写：已超出预算200元"},
        {"pass": True, "issues": [], "feedback": ""},
    ]
    final_state = await finance_agent_graph.ainvoke(state)

    payload = final_state["final_struct"]
    assert payload[KEY_SUCCESS] is True
    text = payload[KEY_DATA]["raw_analysis_text"]
    assert "已超出预算200元" in text
    assert "预算内尚有结余" not in text
    assert final_state["verify_attempt"] == 1
    assert final_state["verify_feedback"] is None
    mock_send.assert_called_once()


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.a2a_bus.send_result")
@patch("agents.finance_agent.nodes.parse_json_output", new_callable=AsyncMock)
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_finance_graph_verify_attempt_limit_falls_back(mock_finance_llm, mock_parse, mock_send, base_finance_state):
    """
    函数功能与逻辑描述：
        verify 连续不过达上限时兜底发送当前确定性结果（链路不崩、必有结果）：
        断言 success=True、文本等于确定性文本、verify_attempt=MAX_VERIFY_ATTEMPTS、只发送一次。
    入参说明：
        mock_finance_llm：patch 掉 `mcp_client.call_llm_finance`，以 side_effect 返回 3 轮确定性文本。
        mock_parse：patch 掉 `parse_json_output`（AsyncMock），以 side_effect 返回 3 轮不通过校验。
        mock_send：patch 掉 `a2a_bus.send_result`（断言只发送一次）。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    state = base_finance_state
    state["raw_segments"] = ["这个月是不是超支了？"]
    state["target_control"] = {"month_budget": 3000}
    state["stat_result"] = {
        KEY_SUCCESS: True, KEY_AGENT_TYPE: "stat_agent",
        KEY_DATA: {"month_total_spend": 3200},
    }
    state["price_result"] = None
    state["habit_data"] = []
    state["task_id"] = "task_fin_verify_2"
    state["verify_attempt"] = 0
    state["verify_feedback"] = None
    text_ok = "本月支出3200元，预算3000元，已超出预算200元。"
    verdict_fail = {"pass": False, "issues": ["仍不一致"], "feedback": "请再修正"}
    # 3 轮 execute（初版+2次重生成） + 3 轮 verify（第3轮 attempt=2 达上限 → reply）
    mock_finance_llm.side_effect = [text_ok, text_ok, text_ok]
    mock_parse.side_effect = [verdict_fail, verdict_fail, verdict_fail]
    final_state = await finance_agent_graph.ainvoke(state)

    payload = final_state["final_struct"]
    assert payload[KEY_SUCCESS] is True
    assert payload[KEY_DATA]["raw_analysis_text"] == text_ok
    assert final_state["verify_attempt"] == MAX_VERIFY_ATTEMPTS
    mock_send.assert_called_once()


@pytest.mark.asyncio
@patch("agents.finance_agent.nodes.mcp_client.call_llm_finance")
async def test_finance_verify_uses_finance_llm_channel_not_base(mock_fin_llm, base_finance_state):
    """
    函数功能与逻辑描述：
        生产路径校验：finance verify 走 `LLM_FINANCE` 通道（RBAC 无 LLM_BASE，`06` 权威表）；
        不 mock parse_json_output，让真实解析器跑通 _verify_llm_call → call_llm_finance，
        证明 verify 不会因 finance 缺 LLM_BASE 权限在生产被引擎拒绝而恒跳过；
        断言 goto=reply_node、模型被 await、首参为 FINANCE_AGENT_ID 且提示词含 checklist。
    入参说明：
        mock_fin_llm：patch 掉 `mcp_client.call_llm_finance`，返回通过的校验 JSON。
        base_finance_state：pytest fixture 注入的基础 FinanceState。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    mock_fin_llm.return_value = json.dumps(
        {"pass": True, "issues": [], "feedback": ""}, ensure_ascii=False
    )
    cmd = await verify_query_node(_ok_finance_state(base_finance_state))
    assert cmd.goto == "reply_node"
    assert cmd.update["verify_decision"] == "reply"
    mock_fin_llm.assert_awaited()
    # 首参应为 finance_agent 自身标识（call_llm_finance 第一参）
    args = mock_fin_llm.await_args[0]
    assert args[0] == FINANCE_AGENT_ID
    assert "checklist" in args[2]
