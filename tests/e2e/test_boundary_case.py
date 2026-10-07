# ==============================================
# 边界场景 e2e：无记账指令 / 收入内容 / 无效输入 / 多轮历史引用
# 均走 orchestrator 全链路（真实外部 LLM）。
# ==============================================
import pytest
import uuid
from agents.orchestrator.graph import orchestrator_graph


def _run_graph(user_input: str):
    """
    函数功能与逻辑描述：
        辅助函数：构造 orchestrator 初始 state（随机 session_id、空历史/空任务计划），
        返回 orchestrator_graph.ainvoke(...) 的协程，供各用例 await 触发全链路执行。
    入参说明：
        user_input (str)：待编排的用户自然语言输入，作为 state["user_input"]。
    返回值说明：
        返回 orchestrator_graph.ainvoke(init_state) 的协程（Coroutine），调用方需 await。
    """
    init_state = {
        "user_input": user_input,
        "session_id": f"e2e_bound_{uuid.uuid4().hex[:8]}",
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent"
    }
    return orchestrator_graph.ainvoke(init_state)


@pytest.mark.asyncio
async def test_e2e_no_bill_without_record_cmd():
    """
    函数功能与逻辑描述：
        场景「仅咨询、无记账指令」：输入“今天喝奶茶花了18，是不是花多了”（含消费金额但无记账动词），
        断言最终回复不含"记账成功"（未落库记账），且回复长度 >10（返回了分析类实质内容）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    user_input = "今天喝奶茶花了18，是不是花多了"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    assert "记账成功" not in final
    assert len(final) > 10


@pytest.mark.asyncio
async def test_e2e_income_content():
    """
    函数功能与逻辑描述：
        场景「收入内容不入库」：输入“发奖金5000元，帮我记上”（含记账动词但为收入），
        断言最终回复不含"记账成功"，即收入被拦截、不写入账单表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    user_input = "发奖金5000元，帮我记上"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    assert "记账成功" not in final


@pytest.mark.asyncio
async def test_e2e_empty_invalid_input():
    """
    函数功能与逻辑描述：
        场景「无效/闲聊输入」：输入“哈哈哈随便聊聊”，断言 final_reply 非 None 且为 str，
        即系统兜底返回提示而不崩溃。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    user_input = "哈哈哈随便聊聊"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    assert final is not None
    assert isinstance(final, str)


@pytest.mark.asyncio
async def test_e2e_multi_round_history():
    """
    函数功能与逻辑描述：
        场景「多轮历史引用」：先以随机 session_id 构造 state 执行第一轮
        “晚饭30元帮我记一下 [测试标记]”完成记账；随后 init_state2 = init_state1.copy()
        （session_history 仍为空字符串，仅替换 user_input 为“再帮我看看这个月总共花了多少”）
        执行第二轮。两轮共享同一 session_id，故统计可读到第一轮落库的账单，
        断言第二轮回复含统计业务事实"总支出"。
        注：本用例直连 orchestrator_graph 而非走 _run_graph，因两轮必须复用同一 session_id。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = f"e2e_multi_{uuid.uuid4().hex[:8]}"
    init_state1 = {
        "user_input": "晚饭30元帮我记一下 [测试标记]",
        "session_id": sid,
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent"
    }
    # 第一轮
    res1 = await orchestrator_graph.ainvoke(init_state1)

    # 第二轮
    init_state2 = init_state1.copy()
    init_state2["user_input"] = "再帮我看看这个月总共花了多少"
    res2 = await orchestrator_graph.ainvoke(init_state2)

    # ★2026-08-31 断言调整：统计回复为外部LLM自然语言（summarize.md 术语规范
    #   强制"总支出"表述），"月度消费统计"标题仅存在于本地确定性渲染。
    assert "总支出" in res2["final_reply"]