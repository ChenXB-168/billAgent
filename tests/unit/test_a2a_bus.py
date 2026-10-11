# ==============================================
# A2A 消息总线单元测试：任务投递/接收、结果回传与等待、会话隔离与超时熔断
# ==============================================
import pytest
import asyncio
from mcpGateway.a2a_queue import a2a_bus
from config.config import A2A_QUEUE_MAXSIZE


# ===================== M20（D21）：队列背压与深度可观测 =====================
def test_send_task_returns_true_when_not_full(test_session_id):
    """M20：正常投递返回 True（既有行为不变，仅新增返回值）。"""
    a2a_bus.clear_all()
    assert a2a_bus.send_task("ok_agent", test_session_id, "t_ok", {"x": 1}) is True


def test_queue_backpressure_rejects_when_full(test_session_id):
    """★M20 核心用例：队列满时 send_task 返回 False（拒绝），**不阻塞、不抛异常**。"""
    a2a_bus.clear_all()
    agent = "backpressure_agent"
    for i in range(A2A_QUEUE_MAXSIZE):
        assert a2a_bus.send_task(agent, test_session_id, f"t{i}", {"i": i}) is True
    # 队列已满 → 拒绝
    assert a2a_bus.send_task(agent, test_session_id, "overflow", {}) is False
    assert a2a_bus.queue_depth(agent) == A2A_QUEUE_MAXSIZE


def test_queue_depth_and_snapshot(test_session_id):
    """M20：queue_depth / snapshot_depths 可观测接口；未创建的 Agent 不产生副作用。"""
    a2a_bus.clear_all()
    assert a2a_bus.queue_depth("nobody") == 0        # 未创建 → 0
    assert a2a_bus.snapshot_depths() == {}
    a2a_bus.send_task("depth_agent", test_session_id, "d1", {})
    assert a2a_bus.queue_depth("depth_agent") == 1
    assert a2a_bus.snapshot_depths() == {"depth_agent": 1}


def test_queue_still_bounded_after_clear_all(test_session_id):
    """★M20 易漏点：clear_all() 重建后队列**仍有界**（两处 factory 必须一致）。"""
    a2a_bus.clear_all()
    agent = "bounded_after_clear"
    for i in range(A2A_QUEUE_MAXSIZE):
        a2a_bus.send_task(agent, test_session_id, f"c{i}", {})
    assert a2a_bus.send_task(agent, test_session_id, "overflow", {}) is False


@pytest.mark.asyncio
async def test_task_send_and_receive(test_session_id):
    """
    函数功能与逻辑描述：
        验证基础任务投递与接收：向 bill_agent 投递一条任务后，按同一会话读取，
        收到的消息中 session_id / task_id / task_data 与投递内容一致。
        覆盖场景：单会话、单消息的正常收发（不触发超时与暂存分支）。
    入参说明：
        test_session_id：pytest fixture（function 作用域，定义于 tests/conftest.py），
            提供形如 "test_<uuid4>" 的唯一会话 ID，用于隔离用例间的会话消息。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证结果回传与等待机制：后台协程延迟 0.1s 调用 send_result，
        主协程通过 wait_result(task_id, timeout=2) 阻塞等待并拿到该结果字符串。
        覆盖场景：结果先于/后于等待到达均可被正确领取（此处为等待方先挂起、结果方后写入）。
    入参说明：
        test_session_id：pytest fixture（function 作用域，定义于 tests/conftest.py），
            提供唯一会话 ID；本用例仅借用其隔离语义，结果等待以 task_id 为键，不使用该值。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    task_id = "task_002"

    async def mock_worker():
        """
        函数功能与逻辑描述：
            用例内的模拟 worker 协程：延迟 0.1s 后调用 a2a_bus.send_result 回传结果 "记账成功"，
            用于制造「等待方先挂起、结果方后写入」的时序，验证 wait_result 能正确领取结果。
        入参说明：
            无。
        返回值说明：
            无（副作用：向 a2a_bus 写入 task_id 对应的结果后结束）。
        """
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
    """
    函数功能与逻辑描述：
        验证超时熔断机制：等待一个从未回传结果的 task_id，超时后必须抛出
        TimeoutError 且异常信息含「执行超时」。
        覆盖场景：结果永不到达的边界路径。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with pytest.raises(TimeoutError, match="执行超时"):
        await a2a_bus.wait_result("not_exist_task", timeout=0.5)


@pytest.mark.asyncio
async def test_session_isolation():
    """
    函数功能与逻辑描述：
        验证不同会话的消息互不串扰：同一 agent 向两个会话各投递一条任务后，
        分别读取各自会话，只能取到属于本会话的那条消息。
        覆盖场景：同 agent 多会话并发排队时的会话隔离。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证非当前会话的消息不会丢失：向 other_session 投递一条消息后，用当前会话读取会超时；
        该不匹配消息由总线旁路暂存（`_parked`），随后按原会话再次读取仍能取回同一条，
        无需重新投递（D4 去轮询改造后不再「放回主队尾」，而是旁路暂存以免退化为主队忙循环）。
        覆盖场景：会话不匹配 → 暂存 → 按原会话取回。
    入参说明：
        test_session_id：pytest fixture（function 作用域，定义于 tests/conftest.py），
            提供形如 "test_<uuid4>" 的唯一会话 ID，用作「当前会话」以触发不匹配暂存分支。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    agent = "price_agent"
    other_sess = "other_session"
    payload = {"data": "其他会话"}

    # 发送一条其他会话消息
    a2a_bus.send_task(agent, other_sess, "t_other", payload)

    # 尝试读取当前会话，预期超时；不匹配消息被旁路暂存（非放回主队尾）
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            a2a_bus.recv_task(agent, test_session_id),
            timeout=0.2
        )

    # 直接读取原会话消息，不需要重新发送
    msg = await a2a_bus.recv_task(agent, other_sess)
    # 只对比 data 字段
    assert msg["task_data"]["data"] == payload["data"]
