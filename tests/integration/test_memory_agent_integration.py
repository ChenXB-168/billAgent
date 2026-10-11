"""会话记忆 / 长期记忆 与 orchestrator 全链路的集成测试（真实外部 LLM，非 mock）。

运行时前置依赖：
- MCP 常驻服务（**全部 3 个**：sql_bill / llm_base / llm_finance）：由 tests/conftest.py 的
  session 级 autouse fixture `spawn_mcp_servers` 以**同步 subprocess** 按
  `mcpGateway.client.SERVER_CMD_MAP` 一概拉起（BILLAGENT_FORCE_EXTERNAL=1 走外部强模型）。
- 子 Agent 消费方：由本文件 `_start_agents_workers` fixture 拉起 4 个常驻 worker；
- 本地 SQLite（bill.db）与 memory 长期记忆（pkl + FAISS 索引）。
"""
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
    """
    函数功能与逻辑描述：
        pytest 异步 fixture（function 作用域，autouse=True），保证每个用例都有子 Agent 消费方：
        用例前清空 a2a_bus，再创建 4 个常驻 worker 任务（bill/stat/price/finance，复用
        startup.bootstrap 的生产 worker），并 sleep 0.2s 等待 worker 进入消费循环；
        用例结束后 cancel 并 gather 回收，避免进程残留与跨用例串消息。
        不向用例注入数据；若没有 worker，dispatch 派发的子任务无人消费必然 240s 超时。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无显式返回值（yield 仅用于分离 setup/teardown，不向用例提供数据）。
    """
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
    """
    函数功能与逻辑描述：
        用例内阶段打点辅助：以 "[TEST] 时:分:秒 阶段: <stage> <sid>" 形式立即输出到 stdout
        （flush=True，避免长耗时用例日志被缓冲），便于人工观察多轮/长链路执行进度。
    入参说明：
        stage (str)：阶段描述文本（如"第1轮 开始 ainvoke(记账)"）。
        sid (str)：可选会话标识，缺省为空串。
    返回值说明：
        无（副作用：向 stdout 打印一行日志）。
    """
    print(f"[TEST] {time.strftime('%H:%M:%S')} 阶段: {stage} {sid}", flush=True)


@pytest.mark.asyncio
async def test_multi_round_session_memory():
    """
    函数功能与逻辑描述：
        端到端验证多轮对话记忆注入：以同一随机 session_id 连续执行两轮 orchestrator——
        第一轮"晚饭35元帮我记一下 [测试标记]"（断言 final_reply 含"记账成功"），
        第二轮"刚才那笔是哪个分类的"并由用例手工把第一轮的会话历史
        （get_session_memory(sid).get_history()，默认 3 轮窗口）写回 state["session_history"]，
        断言第二轮 final_reply 非 None 且长度大于 0（即带历史上下文仍能正常回复、不崩溃）。
        ✅ 脆弱点已修（2026-09-17）：第一轮断言原为 `assert "记账成功" in final_reply`，
        而 collect 表达层在外部 LLM 下已**自然化**（该短语仅存在于本地确定性渲染分支），
        实测同环境同日**一次通过、两次失败**。现改为断言**可复现事实**（回复非空且回显
        品类 / 金额 / 记账语义之一），口径与 e2e 的"2026-08-31 断言调整"一致。
        运行时前置依赖：真实外部 LLM（tests/conftest.py 拉起 llm_base MCP）、bill.db 与 sql_bill MCP、
        本文件 fixture 拉起的 4 个常驻 worker、会话短期记忆（进程内全局字典）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    # ✅ 已修（2026-09-17）：本断言原为 `assert "记账成功" in res1["final_reply"]`，即文件
    #   docstring 早已登记的**脆弱点**——`"记账成功"` 只存在于**本地确定性渲染分支**，而
    #   `tests/conftest.py` 强制外部 LLM（`BILLAGENT_FORCE_EXTERNAL=1`）时 collect 表达层已把
    #   回复**自然化**，措辞随机（实测出现过"好的，已经帮您记下了晚饭的消费，餐饮类别，
    #   金额是35元。"）→ 同环境同日**一次通过、两次失败**（典型 flaky）。
    #   口径与 e2e 的对齐（见 `test_long_memory_e2e.py` 的"2026-08-31 断言调整：final_reply 为
    #   外部 LLM 自然表达……此处仅验证记账链路已返回品类"）：只断言**可复现事实**——
    #   回复非空且回显了品类/金额/记账语义之一，不锁具体措辞。
    reply1 = res1["final_reply"] or ""
    assert reply1, "第一轮 final_reply 为空"
    assert ("餐饮" in reply1) or ("35" in reply1) or ("记账" in reply1), reply1
    
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
    """
    函数功能与逻辑描述：
        验证长期记忆对理财链路的上下文补充：先用 save_consume_memory 写入一条历史消费摘要
        （"上月餐饮支出1200元，占比45%，结构偏高"，进 pkl + FAISS 向量索引），再以随机
        session_id 执行"结合我历史消费给点理财建议"的 orchestrator 全链路，断言最终回复
        不含"失败"且长度大于 30（理财建议有实质内容、未走错误兜底）。
        本用例不清理已写入的长期记忆（仅追加，不影响断言）；注入是否真正命中向量检索未做强断言。
        运行时前置依赖：真实外部 LLM（tests/conftest.py 拉起 llm_base MCP）、memory 长期记忆
        （embedding 模型 + pkl/FAISS）、bill.db 与 sql_bill MCP、本文件 fixture 拉起的 4 个常驻 worker。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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