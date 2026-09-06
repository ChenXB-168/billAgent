import pytest
import uuid
from agents.orchestrator.graph import orchestrator_graph


def _run_graph(user_input: str):
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
    """无记账指令：不生成账单，仅理财分析"""
    user_input = "今天喝奶茶花了18，是不是花多了"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    assert "记账成功" not in final
    assert len(final) > 10


@pytest.mark.asyncio
async def test_e2e_income_content():
    """收入内容：不入库，返回提示"""
    user_input = "发奖金5000元，帮我记上"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    assert "记账成功" not in final


@pytest.mark.asyncio
async def test_e2e_empty_invalid_input():
    """无效输入：返回提示，不崩溃"""
    user_input = "哈哈哈随便聊聊"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    assert final is not None
    assert isinstance(final, str)


@pytest.mark.asyncio
async def test_e2e_multi_round_history():
    """多轮对话：第二轮能引用第一轮上下文"""
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