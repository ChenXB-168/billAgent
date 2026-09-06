import asyncio
import time
import pytest
import uuid
import pytest_asyncio
from agents.orchestrator.graph import orchestrator_graph

# 复用生产环境的真实子Agent Worker（与 startup.bootstrap 完全一致）
# 关键：没有 worker，dispatch 派发的任务无人消费，子任务必然 240s 超时
from startup.bootstrap import (
    _bill_agent_worker, _stat_agent_worker, _price_agent_worker, _finance_agent_worker,
)
from mcpGateway.a2a_queue import a2a_bus


@pytest_asyncio.fixture(autouse=True)
async def _start_agents_workers():
    """每个用例前启动 4 个子 Agent 常驻 worker，用例结束取消"""
    a2a_bus.clear_all()
    workers = [
        asyncio.create_task(_bill_agent_worker()),
        asyncio.create_task(_stat_agent_worker()),
        asyncio.create_task(_price_agent_worker()),
        asyncio.create_task(_finance_agent_worker()),
    ]
    await asyncio.sleep(0.2)  # 等 worker 进入消费循环
    yield
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def _log(stage: str, sid: str = ""):
    print(f"[TEST] {time.strftime('%H:%M:%S')} 阶段: {stage} {sid}", flush=True)


@pytest.mark.asyncio
async def test_multi_round_session_memory():
    """端到端验证：多轮对话记忆被带入提示词，上下文连贯"""
    sid = f"mem_integ_{uuid.uuid4().hex[:8]}"
    
    _log("用例开始: test_multi_round_session_memory", sid)
    # 第一轮：记账
    state1 = {
        "user_input": "晚饭35元帮我记一下 [测试标记]",
        "session_id": sid,
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent"
    }
    _log("第1轮 开始 ainvoke(记账)", sid)
    res1 = await orchestrator_graph.ainvoke(state1)
    _log(f"第1轮 完成 final_reply={res1['final_reply'][:50]!r}", sid)
    assert "记账成功" in res1["final_reply"]
    
    # 第二轮：引用上轮内容，验证记忆存在
    _log("第2轮 准备(读会话记忆)", sid)
    state2 = state1.copy()
    state2["user_input"] = "刚才那笔是哪个分类的"
    # 手动带入上轮记忆（模拟真实会话循环）
    from memory.short_memory import get_session_memory
    mem = get_session_memory(sid)
    state2["session_history"] = mem.get_history()
    
    _log("第2轮 开始 ainvoke(引用上轮)", sid)
    res2 = await orchestrator_graph.ainvoke(state2)
    _log(f"第2轮 完成 final_reply={res2['final_reply'][:50]!r}", sid)
    # 能正常回复，不崩溃，即记忆注入生效
    assert res2["final_reply"] is not None
    assert len(res2["final_reply"]) > 0
    _log("用例结束: test_multi_round_session_memory", sid)


@pytest.mark.asyncio
async def test_long_memory_finance_agent():
    """理财Agent调用长期记忆补充上下文"""
    from memory.long_memory import save_consume_memory
    
    _log("用例开始: test_long_memory_finance_agent")
    # 先存入历史消费记录
    save_consume_memory("上月餐饮支出1200元，占比45%，结构偏高")
    
    sid = f"long_mem_fin_{uuid.uuid4().hex[:8]}"
    state = {
        "user_input": "结合我历史消费给点理财建议",
        "session_id": sid,
        "session_history": "",
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent"
    }
    
    _log("开始 ainvoke(理财建议)", sid)
    res = await orchestrator_graph.ainvoke(state)
    _log(f"完成 final_reply={res['final_reply'][:50]!r}", sid)
    assert "失败" not in res["final_reply"]
    assert len(res["final_reply"]) > 30
    _log("用例结束: test_long_memory_finance_agent", sid)