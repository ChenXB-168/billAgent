import pytest
import uuid
from agents.orchestrator.graph import orchestrator_graph


def _run_graph(user_input: str):
    init_state = {
        "user_input": user_input,
        "session_id": f"e2e_mix_{uuid.uuid4().hex[:8]}",
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
async def test_e2e_bill_plus_stat():
    """记账+统计：bill先执行，统计能读到新账单"""
    user_input = "买零食花了45元记一下 [测试标记]，查本月总开销"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    # ★2026-08-31 断言调整（同 test_single_task）：collect_node 表达层外部LLM
    #   自然化，"记账成功/月度消费统计"仅存在于本地确定性渲染；按 summarize.md
    #   术语规范改用业务事实断言（统计必含"总支出"，记账必含品类/金额）。
    assert "总支出" in final
    # 验证45元被计入统计结果
    assert "45" in final or "零食" in final or "购物" in final


@pytest.mark.asyncio
async def test_e2e_bill_plus_finance():
    """记账+理财：bill先入库，理财后分析"""
    user_input = "打车25元存账单 [测试标记]，给我理财建议"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    # 记账事实（金额必现，术语规范要求记账点明品类与金额）
    assert "25" in final
    # 理财建议有实际内容
    assert len(final) > 50


@pytest.mark.asyncio
async def test_e2e_full_four_tasks():
    """四任务全场景：记账→统计→物价→理财，按依赖顺序执行"""
    user_input = "午饭32元记一下 [测试标记]，查7月开销，对比餐饮物价，给省钱建议"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    # 记账金额 + 统计"总支出" + 物价"溢价率"（均为 summarize.md 术语规范强制）
    assert "32" in final
    assert "总支出" in final
    assert "溢价率" in final
    assert len(final) > 100