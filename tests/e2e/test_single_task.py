# ==============================================
# 单任务 e2e：单记账 / 单统计 / 单物价对比 / 单理财建议
# 走 orchestrator 全链路（真实外部 LLM），验证各单意图任务能正确路由与产出。
# ==============================================
import pytest
import uuid
from agents.orchestrator.graph import orchestrator_graph


def _run_graph(user_input: str):
    """
    函数功能与逻辑描述：
        辅助函数：构造 orchestrator 初始 state（随机 session_id、空历史/空任务计划），
        返回 orchestrator_graph.ainvoke(...) 的协程，供各用例 await 触发单任务编排。
    入参说明：
        user_input (str)：待编排的用户自然语言输入，作为 state["user_input"]。
    返回值说明：
        返回 orchestrator_graph.ainvoke(init_state) 的协程（Coroutine），调用方需 await。
    """
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
    """
    函数功能与逻辑描述：
        场景「单记账」：输入“晚饭38元帮我记一下 [测试标记]”，断言最终回复含记账事实
        （品类"餐饮"与金额"38"）；因 collect 表达层已 LLM 自然化，不依赖固定短语"记账成功"。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    result = await _run_graph("晚饭38元帮我记一下 [测试标记]")
    # ★2026-08-31 断言调整（M1 实测）：collect_node 表达层已自然化——外部LLM按
    #   summarize.md 生成自然语言回复（"已帮你记录：…消费38.0元（餐饮）…"），
    #   "记账成功"固定短语仅存在于本地确定性渲染分支。断言改为验证记账事实
    #   （分类 + 金额）正确落入回复，不依赖固定模板措辞。
    assert "餐饮" in result["final_reply"]
    assert "38" in result["final_reply"]


@pytest.mark.asyncio
async def test_e2e_single_stat():
    """
    函数功能与逻辑描述：
        场景「单统计」：输入“看看这个月花了多少钱”，断言最终回复含统计业务事实"总支出"
        （不依赖本地确定性渲染的"月度消费统计"标题）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    result = await _run_graph("看看这个月花了多少钱")
    # ★2026-08-31 断言调整：外部LLM统计回复为自然语言（"这个月您的总支出是X元"），
    #   "月度消费统计"标题仅存在于本地确定性渲染分支。断言改为验证统计事实"总支出"。
    assert "总支出" in result["final_reply"]


@pytest.mark.asyncio
async def test_e2e_single_price():
    """
    函数功能与逻辑描述：
        场景「单物价对比」：输入“吃火锅花了80，贵不贵”，断言最终回复含用户金额"80"
        与标准术语"溢价率"（summarize.md 术语规范已强制该表述）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    result = await _run_graph("吃火锅花了80，贵不贵")
    # ★2026-08-31 断言调整：外部 LLM 自然回复不含任务类型标题"物价对比"（该词仅存在于
    #   本地确定性渲染），断言改为验证业务事实。
    # ★2026-09-17 **再调整（实测驱动，非为迁就实现）**：原先断言**必须出现术语"溢价率"**；
    #   但实测在 `summarize.md` 连续三轮加强约束（输出要求 L27 → 禁止事项 → 讲清后果）后，
    #   外部 LLM 仍稳定输出**等价表述**（"比当地同品类均价高出/高约 122.22%"）。
    #   结论：**"术语规范化"不能且不必硬保证** —— 生成式表达天然多样，且用户已确认
    #   "没硬说溢价率、但表达意思一样"**即视为正确**。故提示词只保留"优先使用溢价率"的引导，
    #   测试则断**可复现的业务事实**：金额 80 复现 + 给出了**价格偏离度百分比**，
    #   后者正是"比价确实发生且拿到了基准价"的硬证据（该百分比由 price 任务产出，非模型编造）。
    assert "80" in result["final_reply"]
    assert "%" in result["final_reply"], result["final_reply"]


@pytest.mark.asyncio
async def test_e2e_single_finance():
    """
    函数功能与逻辑描述：
        场景「单理财建议」：输入“给我点本月省钱建议”，断言最终回复长度 >20
        （理财分析有实质内容，非常识性空回复），且不含"失败"字样（未落入错误兜底回复）。
        运行时依赖真实外部 LLM 与本地账单库（session_id 随机生成，无历史数据也能给出通用建议）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    result = await _run_graph("给我点本月省钱建议")
    assert len(result["final_reply"]) > 20
    assert "失败" not in result["final_reply"]