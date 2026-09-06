import pytest
import uuid
from agents.orchestrator.graph import orchestrator_graph


def _run_graph(user_input: str):
    """辅助函数：执行编排器全流程"""
    init_state = {
        "user_input": user_input,
        "session_id": f"e2e_{uuid.uuid4().hex[:8]}",
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
async def test_e2e_single_bill():
    """端到端：单记账任务"""
    result = await _run_graph("晚饭38元帮我记一下 [测试标记]")
    # ★2026-08-31 断言调整（M1 实测）：collect_node 表达层已自然化——外部LLM按
    #   summarize.md 生成自然语言回复（"已帮你记录：…消费38.0元（餐饮）…"），
    #   "记账成功"固定短语仅存在于本地确定性渲染分支。断言改为验证记账事实
    #   （分类 + 金额）正确落入回复，不依赖固定模板措辞。
    assert "餐饮" in result["final_reply"]
    assert "38" in result["final_reply"]


@pytest.mark.asyncio
async def test_e2e_single_stat():
    """端到端：单统计任务"""
    result = await _run_graph("看看这个月花了多少钱")
    # ★2026-08-31 断言调整：外部LLM统计回复为自然语言（"这个月您的总支出是X元"），
    #   "月度消费统计"标题仅存在于本地确定性渲染分支。断言改为验证统计事实"总支出"。
    assert "总支出" in result["final_reply"]


@pytest.mark.asyncio
async def test_e2e_single_price():
    """端到端：单物价对比任务"""
    result = await _run_graph("吃火锅花了80，贵不贵")
    # ★2026-08-31 断言调整：外部LLM自然回复不含任务类型标题"物价对比"（该词仅
    #   存在于本地确定性渲染），断言改为验证业务事实：用户金额80 + 标准术语"溢价率"
    #   （summarize.md 术语规范已强制该表述）。
    assert "80" in result["final_reply"]
    assert "溢价率" in result["final_reply"]


@pytest.mark.asyncio
async def test_e2e_single_finance():
    """端到端：单理财建议任务"""
    result = await _run_graph("给我点本月省钱建议")
    assert len(result["final_reply"]) > 20
    assert "失败" not in result["final_reply"]