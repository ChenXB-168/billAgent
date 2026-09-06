import pytest
import asyncio
from mcpGateway.a2a_queue import a2a_bus


@pytest.mark.asyncio
async def test_task_send_and_receive(test_session_id):
    """测试基础任务投递与接收"""
    task_id = "task_001"
    test_data = {"raw_segments": ["测试记账文本"]}

    a2a_bus.send_task(
        target_agent="bill_agent",
        session_id=test_session_id,
        task_id=task_id,
        task_data=test_data
    )

    msg = await a2a_bus.recv_task("bill_agent", test_session_id)

    assert msg["session_id"] == test_session_id
    assert msg["task_id"] == task_id
    assert msg["task_data"] == test_data


@pytest.mark.asyncio
async def test_result_send_and_wait(test_session_id):
    """测试结果回传与等待机制"""
    task_id = "task_002"

    async def mock_worker():
        await asyncio.sleep(0.1)
        a2a_bus.send_result(
            task_id=task_id,
            result="记账成功"
        )

    asyncio.create_task(mock_worker())
    result = await a2a_bus.wait_result(task_id, timeout=2)
    assert result == "记账成功"


@pytest.mark.asyncio
async def test_wait_result_timeout():
    """测试超时熔断机制"""
    with pytest.raises(TimeoutError, match="执行超时"):
        await a2a_bus.wait_result("not_exist_task", timeout=0.5)


@pytest.mark.asyncio
async def test_session_isolation():
    """测试不同会话消息隔离，不会串数据"""
    sess1 = "session_A"
    sess2 = "session_B"

    a2a_bus.send_task("stat_agent", sess1, "t1", {"data": "A的任务"})
    a2a_bus.send_task("stat_agent", sess2, "t2", {"data": "B的任务"})

    msg_b = await a2a_bus.recv_task("stat_agent", sess2)
    assert msg_b["task_data"]["data"] == "B的任务"

    msg_a = await a2a_bus.recv_task("stat_agent", sess1)
    assert msg_a["task_data"]["data"] == "A的任务"


@pytest.mark.asyncio
async def test_other_session_message_put_back(test_session_id):
    """测试非当前会话的消息会放回队列，不会丢失"""
    agent = "price_agent"
    other_sess = "other_session"
    payload = {"data": "其他会话"}

    # 发送一条其他会话消息
    a2a_bus.send_task(agent, other_sess, "t_other", payload)

    # 尝试读取当前会话，预期超时，消息放回队列
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            a2a_bus.recv_task(agent, test_session_id),
            timeout=0.2
        )

    # 直接读取原消息，不需要重新发送
    msg = await a2a_bus.recv_task(agent, other_sess)
    # 修复：只对比data字段
    assert msg["task_data"]["data"] == payload["data"]