import asyncio
from typing import Any, Dict, List, Optional
from collections import defaultdict

class A2AMessageBus:
    def __init__(self):
        # 私有内部变量，外部任何文件禁止直接访问
        self._task_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._result_events: Dict[str, asyncio.Event] = {}
        self._result_store: Dict[str, Any] = {}
        # 会话不匹配消息的暂存区（按 agent_id 分组）
        # 【D4 去轮询引入】不再"放回主队尾部"——放回会让 await queue.get() 立刻再取到同一条，
        # 退化为 100% CPU 忙循环。改为旁路暂存，取消息时优先消费，保证同会话 FIFO 且不丢消息。
        self._parked: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._shutdown_flag: bool = False

    # ===================== 对外公开方法：供测试/业务调用 =====================
    def clear_all(self):
        """【公开接口】清空所有消息队列、结果缓存，仅测试场景使用"""
        # 【D4 去轮询配套】直接重建队列字典，而非逐个 get_nowait 清空：
        # asyncio.Queue 在首次 await get() 时会绑定所属 event loop，跨 loop 复用会抛
        # "is bound to a different event loop"。原实现只用 get_nowait/put_nowait（不触发
        # 绑定校验）故从未暴露此问题。重建后队列全新、未绑定任何 loop，测试与重启场景均安全。
        self._task_queues = defaultdict(asyncio.Queue)
        # 清空结果事件与缓存
        self._result_events.clear()
        self._result_store.clear()
        # 清空会话不匹配暂存区（否则测试用例之间会串消息）
        self._parked.clear()

    def send_task(self, target_agent: str, session_id: str, task_id: str, task_data: Dict[str, Any]):
        """向目标Agent投递任务"""
        msg = {
            "session_id": session_id,
            "task_id": task_id,
            "task_data": task_data
        }
        self._task_queues[target_agent].put_nowait(msg)
        
        print(f"[A2A send_task] target_agent={target_agent} session={session_id} task_id={task_id} 入队完成")

    async def recv_task(self, agent_id: str, session_id: Optional[str] = None, max_wait=3) -> Dict[str, Any]:
        """
        业务/测试用：带超时的阻塞接收。

        【D4 去轮询改造】原实现 = get_nowait + sleep(0.01) 轮询：
        空队列时每 10ms 唤醒一次空转，消息到达后平均还需多等 5ms 才被取走。
        现改为 wait_for(queue.get())：空队列时挂起（零空转），消息入队即刻被唤醒。
        """
        queue = self._task_queues[agent_id]
        parked = self._parked[agent_id]
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max_wait

        while True:
            # 优先消费暂存区中"上次会话不匹配"的消息（保持同会话 FIFO、不丢消息）
            for idx, msg in enumerate(parked):
                if session_id is None or msg["session_id"] == session_id:
                    return parked.pop(idx)

            # 快路径：队列非空 → 同步取走，不让出控制权（保持原 get_nowait 语义）。
            # 若此处直接 await queue.get()，会让出控制权，同 loop 内的常驻 worker 将抢先
            # 消费掉本该由调用方接收的消息（e2e 总线用例正是这个场景）。
            try:
                msg = queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                if session_id is None or msg["session_id"] == session_id:
                    return msg
                parked.append(msg)
                continue

            # 慢路径：队列空 → 挂起等待（零空转），到期仍未等到则超时
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"等待Agent[{agent_id}]消息超时{max_wait}s，无匹配会话数据")

            try:
                msg = await asyncio.wait_for(queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f"等待Agent[{agent_id}]消息超时{max_wait}s，无匹配会话数据")

            if session_id is None or msg["session_id"] == session_id:
                return msg
            # 会话不匹配 → 暂存旁路（不放回主队，见 __init__ 中 _parked 注释）
            parked.append(msg)

    async def recv_task_blocking(self, agent_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Worker常驻消费专用：无限阻塞，无超时
        持续等待匹配消息，不会抛出TimeoutError

        【D4 去轮询改造】原实现 = get_nowait + sleep(0.1)：
        空闲时每 100ms 空转唤醒一次（4 个常驻 worker ≈ 每秒 40 次无谓唤醒），
        且任务投递后平均多等 50ms 才被取走。
        现改为 await queue.get()：无消息时挂起、消息入队即刻唤醒——零空转、零额外延迟。
        """
        queue = self._task_queues[agent_id]
        parked = self._parked[agent_id]

        print(f"[A2A recv_task_blocking START] agent={agent_id} expect_session={session_id}")

        while True:
            # 优先消费暂存区中"上次会话不匹配"的消息
            for idx, msg in enumerate(parked):
                if session_id is None or msg["session_id"] == session_id:
                    return parked.pop(idx)

            # 队列空 → 挂起等待（零空转）；有消息 → 立即唤醒
            msg = await queue.get()

            msg_sid = msg["session_id"]
            msg_tid = msg["task_id"]

            print(f"[A2A recv_task_blocking pop] agent={agent_id} msg_session={msg_sid} task_id={msg_tid}")

            if session_id is None or msg["session_id"] == session_id:
                print(f"[A2A recv_task_blocking MATCH] agent={agent_id} session={msg_sid} task_id={msg_tid} 返回消息")
                return msg
            # 会话不匹配 → 暂存旁路，不回主队
            parked.append(msg)

            print(f"[A2A recv_task_blocking MISMATCH] expect={session_id}, msg={msg_sid}, task_id={msg_tid} 暂存旁路")

    def send_result(self, task_id: str, result: Any):
        """
        Worker回传执行结果（按 task_id 路由，orchestrator 用 wait_result(task_id) 领取）

        【D4 验收③】删除死参数 `target_agent` / `session_id`：
        结果存储与唤醒均以 `task_id` 为唯一键，这两个形参从未被使用；保留会误导调用方，
        以为存在"按 Agent 路由 / 按会话隔离"的语义。
        """
        # 无论是否存在等待者，先存入结果
        self._result_store[task_id] = result
        # Event不存在则主动创建
        if task_id not in self._result_events:
            self._result_events[task_id] = asyncio.Event()
        self._result_events[task_id].set()


    async def wait_result(self, task_id: str, timeout: int = 240) -> Any:
        """
        编排器阻塞等待任务结果，带超时熔断
        默认 240s：本地 CPU 推理单次 LLM 调用可能超过 120s，必须与
        config.MODEL_TIMEOUT(240s) 对齐，避免编排器过早熔断
        """
        if task_id not in self._result_events:
            self._result_events[task_id] = asyncio.Event()
        event = self._result_events[task_id]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            res = self._result_store.pop(task_id, None)
            self._result_events.pop(task_id, None)
            return res
        except asyncio.TimeoutError:
            self._result_events.pop(task_id, None)
            self._result_store.pop(task_id, None)
            raise TimeoutError(f"任务[{task_id}]执行超时")

    def cleanup_session(self, session_id: str):
        """会话结束清理（预留扩展）"""
        pass

# 全局单例消息总线
a2a_bus = A2AMessageBus()

__all__ = ["a2a_bus"]