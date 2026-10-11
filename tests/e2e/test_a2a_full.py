# ==============================================
# A2A 消息总线 e2e 测试：bill_agent 全链路集成（依赖真实外部 LLM）。
# 纯总线 API 契约测试见 tests/unit/test_a2a_bus.py，本层不重复。
# ==============================================
import pytest,sys
import asyncio
import pytest_asyncio
from pathlib import Path

# 本文件上两级目录（tests/），追加到 sys.path 便于导入项目内模块
BASE_PROJECT = Path(__file__).parent.parent
sys.path.append(str(BASE_PROJECT))

# 只在函数内部导入A2A，避免顶层加载server_llm_base触发mcp缺失报错
TEST_SESSION = "test_session_001"


# ── 覆盖 e2e conftest 的 autouse worker fixture（本文件不需要常驻 worker）──
# 原因：conftest 的 _start_agents_workers 会拉起 4 个常驻 worker，而 worker 调
#      recv_task_blocking 时不传 session_id（不过滤会话），会把本用例 send_task 投递的
#      bill 任务一并取走并**再执行一遍 bill 图**——与本用例的直接图执行重复写账。
#      本用例自行执行图、不走 worker 消费，故覆盖为空实现（用例内部自行 clear_all 隔离）。
@pytest_asyncio.fixture(autouse=True)
async def _start_agents_workers():
    """
    函数功能与逻辑描述：
        覆盖 tests/e2e/conftest.py 中的同名 autouse fixture：本文件不拉起常驻 worker，
        避免 worker 取走本用例投递的任务并重复执行同一份 bill 图。用例内部自行 clear_all 隔离。
    入参说明：
        无。
    返回值说明：
        无（仅占位以覆盖父级 fixture，不向用例提供数据）。
    """
    yield


# ===================== bill_agent 全链路集成（依赖真实外部 LLM） =====================
@pytest.mark.asyncio
async def test_bill_agent_a2a_integrate():
    """
    函数功能与逻辑描述：
        bill_agent 全链路集成：先向 A2A 投递 raw_segments=["晚餐50"] 的 bill 任务，
        再以 asyncio.wait_for（超时 180s）执行 bill_agent_graph；按执行结果构造
        {success, agent, data/error} 应答，经 send_result 回传，最后以 wait_result（timeout=120）
        校验应答为 dict 且含 success 字段。
        注：bill_agent_graph 已移除内部 recv_task_node，改为直接以传入的 state 执行，
        任务接收由外层 worker（a2a_bus.recv_task_blocking）承担；超时/取消/异常均被捕获，
        统一构造 success=False 的应答后仍继续断言流程。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    import json
    import asyncio
    from mcpGateway.a2a_queue import a2a_bus
    from agents.bill_agent.graph import bill_agent_graph

    a2a_bus.clear_all()
    target_agent_id = "bill_agent"
    tid = "t99"
    input_segments = ["晚餐50"]

    print(f"\n【集成测试】1. Orchestrator 发送任务 task_id={tid}")
    a2a_bus.send_task(
        target_agent_id,
        TEST_SESSION,
        tid,
        {"raw_segments": input_segments}
    )

    # 注：bill_agent_graph 已移除内部 recv_task_node，改为直接以传入 state 执行，
    #     任务接收由外层 worker（a2a_bus.recv_task_blocking）承担。
    print(f"【集成测试】2. 启动 bill_agent_graph（直接以传入 state 执行）")
    graph_input = {
        "session_id": TEST_SESSION,
        "task_id": tid,
        "raw_segments": []
    }
    graph_coro = asyncio.create_task(bill_agent_graph.ainvoke(graph_input))

    graph_result = None
    error_msg = ""
    try:
        # 提升超时到180秒，给LLM足够响应窗口
        graph_result = await asyncio.wait_for(graph_coro, timeout=180)
        print(f"【集成测试】3. bill_agent graph执行成功，输出：{graph_result}")
    except asyncio.TimeoutError:
        error_msg = "任务执行超时，LLM响应过慢"
        print(f"【集成测试】!! {error_msg}")
        graph_coro.cancel()
    except asyncio.CancelledError:
        error_msg = "任务被外部取消"
        print(f"【集成测试】!! {error_msg}")
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"【集成测试】!! bill_agent graph异常：{error_msg}")

    # 根据有无错误构造应答
    if error_msg == "":
        resp_payload = json.dumps({
            "success": True,
            "agent": "bill_agent",
            "data": graph_result
        })
    else:
        resp_payload = json.dumps({
            "success": False,
            "agent": "bill_agent",
            "error": error_msg
        })
    a2a_bus.send_result(
        task_id=tid,
        result=resp_payload
    )
    print(f"【集成测试】4. bill_agent 调用send_result回传结果")

    # Orchestrator 阻塞等待应答
    print(f"【集成测试】5. Orchestrator 阻塞等待结果 task_id={tid}")
    raw_res = await a2a_bus.wait_result(tid, timeout=120)
    final_res = json.loads(raw_res)
    print(f"【集成测试】6. Orchestrator 收到应答：{final_res}")

    assert isinstance(final_res, dict)
    assert "success" in final_res

    if final_res["success"]:
        print("✅ bill Agent A2A集成测试【成功】")
    else:
        raise AssertionError(f"Agent运行失败：{final_res.get('error')}")


# 本地直接运行入口
async def main():
    """
    函数功能与逻辑描述：
        模块本地调试入口：直接 await 执行 test_bill_agent_a2a_integrate 用例
        （不经 pytest，故不会启用 conftest 中的 autouse fixture）。
    入参说明：
        无。
    返回值说明：
        无。
    """
    await test_bill_agent_a2a_integrate()


if __name__ == "__main__":
    asyncio.run(main())
