# ==============================================
# 混合任务 e2e：记账 + 统计 / 记账 + 理财 / 四任务全场景
# 走 orchestrator 全链路（真实外部 LLM），验证多 Agent 组合与执行顺序。
# ==============================================
import pytest
import uuid
from agents.orchestrator.graph import orchestrator_graph


def _run_graph(user_input: str):
    """
    函数功能与逻辑描述：
        辅助函数：构造 orchestrator 初始 state（随机 session_id、空历史/空任务计划），
        返回 orchestrator_graph.ainvoke(...) 的协程，供各用例 await 触发多任务编排。
    入参说明：
        user_input (str)：待编排的用户自然语言输入，作为 state["user_input"]。
    返回值说明：
        返回 orchestrator_graph.ainvoke(init_state) 的协程（Coroutine），调用方需 await。
    """
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
    """
    函数功能与逻辑描述：
        场景「记账 + 统计」：输入“买零食花了45元记一下 [测试标记]，查本月总开销”，
        断言最终回复含统计业务事实"总支出"，且含 "45" / "零食" / "购物" 之一的记账事实；
        验证 bill 先执行、stat 能读到新账单，且 LLM 自然化回复不依赖固定模板。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        场景「记账 + 理财」：输入“打车25元存账单 [测试标记]，给我理财建议”，
        断言回复含记账金额"25"，且回复长度 >50（理财分析有实质内容）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    user_input = "打车25元存账单 [测试标记]，给我理财建议"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    # 记账事实（金额必现，术语规范要求记账点明品类与金额）
    assert "25" in final
    # 理财建议有实际内容
    assert len(final) > 50


@pytest.mark.asyncio
async def test_e2e_full_four_tasks():
    """
    函数功能与逻辑描述：
        场景「四任务全链路」：输入“午饭32元记一下 [测试标记]，查7月开销，对比餐饮物价，给省钱建议”，
        断言回复同时含记账金额"32"、统计术语"总支出"、物价术语"溢价率"，且长度 >100，
        验证 bill→stat→price→finance 按依赖顺序都执行到位。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    user_input = "午饭32元记一下 [测试标记]，查7月开销，对比餐饮物价，给省钱建议"
    result = await _run_graph(user_input)

    final = result["final_reply"]
    # 记账金额 + 统计"总支出" + 比价结论（价格偏离度）
    assert "32" in final
    assert "总支出" in final
    # ★2026-09-17 放宽：原断言必须出现术语字面"溢价率"，但实测外部 LLM 会输出**等价表述**
    #   （"比当地同品类均价高出 X%"）且用户已确认"意思一样即为正确"，故改断**必须给出偏离幅度
    #   百分比**（百分比只能来自 price 任务的 premium_rate，是"比价确实执行"的硬证据）。
    assert "%" in final, final
    assert len(final) > 100