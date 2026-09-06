import pytest,sys
import asyncio
import json
from pathlib import Path

# 项目根路径
BASE_PROJECT = Path(__file__).parent.parent
sys.path.append(str(BASE_PROJECT))

# 只在函数内部导入A2A，避免顶层加载server_llm_base触发mcp缺失报错
TEST_SESSION = "test_session_001"
TIMEOUT = 3

# ===================== 第一部分：纯A2A总线单元测试（无任何Agent，只测收发逻辑） =====================
@pytest.mark.asyncio
async def test_a2a_basic_send_recv():
    from mcpGateway.a2a_queue import a2a_bus
    a2a_bus.clear_all()
    task_id = "t001"
    target = "bill_agent"
    send_data = {"raw": "奶茶20元"}

    print(f"\n【调试】开始发送任务 task_id={task_id}")
    a2a_bus.send_task(target, TEST_SESSION, task_id, send_data)

    print("【调试】调用recv_task等待消息")
    rec_msg = await a2a_bus.recv_task(target, TEST_SESSION, max_wait=TIMEOUT)
    print(f"【调试】成功收到消息：{rec_msg}")

    assert rec_msg["task_id"] == task_id
    assert rec_msg["task_data"]["raw"] == "奶茶20元"

    mock_res = json.dumps({"success": True, "agent_type": target})
    print(f"【调试】准备调用send_result，payload={mock_res}")
    a2a_bus.send_result(task_id, mock_res)
    print("【调试】send_result执行完毕")

    # 打印bus内部状态，重点观察事件与存储
    print(f"【调试】_result_store keys = {list(a2a_bus._result_store.keys())}")
    print(f"【调试】_result_events keys = {list(a2a_bus._result_events.keys())}")
    if task_id in a2a_bus._result_events:
        print(f"【调试】event is_set() = {a2a_bus._result_events[task_id].is_set()}")

    print("【调试】进入wait_result阻塞等待")
    res = await a2a_bus.wait_result(task_id)
    print(f"【调试】wait_result拿到返回：{res}")

    assert json.loads(res)["success"] is True
    print("✅ A2A基础收发测试通过")


@pytest.mark.asyncio
async def test_a2a_session_isolation():
    from mcpGateway.a2a_queue import a2a_bus
    a2a_bus.clear_all()
    s1 = "s001"
    s2 = "s002"
    a2a_bus.send_task("bill_agent", s1, "t1", {"val": 111})
    a2a_bus.send_task("bill_agent", s2, "t2", {"val": 222})
    msg1 = await a2a_bus.recv_task("bill_agent", s1, max_wait=TIMEOUT)
    msg2 = await a2a_bus.recv_task("bill_agent", s2, max_wait=TIMEOUT)
    assert msg1["task_data"]["val"] == 111
    assert msg2["task_data"]["val"] == 222
    print("✅ A2A会话隔离测试通过")


@pytest.mark.asyncio
async def test_a2a_recv_timeout():
    from mcpGateway.a2a_queue import a2a_bus
    print("✅ A2A读取超时测试开始")
    a2a_bus.clear_all()
    with pytest.raises(TimeoutError):
        await a2a_bus.recv_task("bill_agent", "empty_session", max_wait=1)
    print("✅ A2A读取超时测试通过")


@pytest.mark.asyncio
async def test_a2a_concurrent_msg():
    from mcpGateway.a2a_queue import a2a_bus
    a2a_bus.clear_all()
    target = "stat_agent"
    # 批量发消息
    for i in range(3):
        a2a_bus.send_task(target, TEST_SESSION, f"t{i}", {"num": i})
    # 循环读取
    nums = []
    for _ in range(3):
        m = await a2a_bus.recv_task(target, TEST_SESSION, max_wait=TIMEOUT)
        nums.append(m["task_data"]["num"])
    nums.sort()
    assert nums == [0, 1, 2]
    print("✅ A2A并发消息测试通过")

# ===================== 第二部分（可选，Agent集成测试，按需打开） =====================
@pytest.mark.asyncio
async def test_bill_agent_a2a_integrate():
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

    print(f"【集成测试】2. 启动 bill_agent_graph（内部自行执行recv_task）")
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
    await test_bill_agent_a2a_integrate()
    

if __name__ == "__main__":
    asyncio.run(main())
