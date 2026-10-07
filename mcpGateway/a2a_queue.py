# ==============================================
# A2A 消息总线 - Agent 间异步任务派发与结果回传
# 说明：编排器向业务 Agent 投递任务、业务 Agent 回传结果、编排器领取结果
#       全部经本模块完成，是「编排器 ↔ 常驻 Worker」之间唯一的数据通路。
#
# ★接口分工（生产 vs 测试）—— 两个接收口不可互换：
#   生产：recv_task_blocking（永久阻塞、无超时）——4 个常驻 worker 专用，
#         调用方 startup/bootstrap.py:191/234/274/311（bill / stat / price / finance）。
#   测试：recv_task（带超时，默认 3s）——纯测试接口，零生产调用；
#         调用方 tests/unit/test_a2a_bus.py、tests/e2e/test_a2a_full.py。
#   为何不合并：测试用 blocking 会永久挂死；worker 用 recv_task 会每 3s 抛 TimeoutError。
# ==============================================
import asyncio
from typing import Any, Dict, List, Optional
from collections import defaultdict

# M20（D21）：队列容量治理常量（拷打 F3）——有界 + 堆积预警阈值
# M22：等待上限改为**引用** MODEL_TIMEOUT（单一来源），不再硬编码 240
from config.config import A2A_QUEUE_MAXSIZE, A2A_QUEUE_WARN_DEPTH, MODEL_TIMEOUT
from utils.common import logger

# M21（T-2）：`_abandoned` 标记集合的容量上限。
#   超时（wait_result 240s 未等到结果）属**低频**事件，正常远达不到该上限；
#   设上限只是为集合自身兜底（避免极端情况下反向泄漏）。
_ABANDONED_MAX = 1000


class A2AMessageBus:
    """
    函数/类功能与逻辑描述：
        Agent 间消息总线，基于内存队列实现同一进程内的任务派发与结果回传。
        按 target_agent 分队列投递任务，按 task_id 暂存与唤醒结果；同会话（session_id）严格 FIFO，
        会话不匹配的消息进旁路暂存区（_parked）而不放回主队——放回会导致
        `await queue.get()` 立刻再取到同一条，退化为 100% CPU 忙循环。
        每条消息的会话隔离靠 session_id 过滤实现，不匹配的消息不会丢失，只会被推迟到匹配时消费。
        全部内部变量以单下划线前缀标记为私有，外部模块不得直接访问。
    构造入参说明：
        无（队列、事件与暂存区均在构造时初始化为空）。
    返回值说明：
        构造返回 A2AMessageBus 实例；业务方法见 send_task / recv_task / wait_result 等。
    """

    def __init__(self):
        """
        函数功能与逻辑描述：
            初始化总线内部状态：按 Agent 分组的任务队列、按 task_id 索引的结果事件与结果缓存、
            会话不匹配消息的旁路暂存区。
            _task_queues 与 _parked 均使用 defaultdict，因此对任意 agent_id 首次访问即自动建空容器，
            调用方无需预注册 Agent。
            关停由 `startup/bootstrap.py` 的全局 SHUTDOWN_FLAG + `task.cancel()` 驱动。
        入参说明：
            无。
        返回值说明：
            无（仅初始化实例属性，无外部副作用）。
        """
        # 私有内部变量，外部任何文件禁止直接访问
        # ★M20（D21）：**有界**队列——满时由 send_task **拒绝**（返回 False + WARNING），
        #   绝不阻塞（阻塞会连派发方 Orchestrator 一起挂死，详见 config 注释）。
        #   ★factory 必须与 clear_all() 中的 factory 保持一致（否则清空后退化为无界）。
        self._task_queues: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=A2A_QUEUE_MAXSIZE))
        self._result_events: Dict[str, asyncio.Event] = {}
        self._result_store: Dict[str, Any] = {}
        # 会话不匹配消息的暂存区（按 agent_id 分组）
        # 【D4 去轮询引入】不再"放回主队尾部"——放回会让 await queue.get() 立刻再取到同一条，
        # 退化为 100% CPU 忙循环。改为旁路暂存，取消息时优先消费，保证同会话 FIFO 且不丢消息。
        self._parked: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # ★M21（T-2）：编排器**已放弃**的 task_id（保序集合，dict 当有序 set 用）——
        #   防止"worker 事后回传"造成内存泄漏。
        #   场景：`wait_result` 超时后已 pop 掉结果缓存与事件，但 worker 侧的 runner 仍在跑
        #   （编排器只是"不等了"）；待其结束调 `send_result` 时会**再次写入** _result_store /
        #   _result_events ——而**再无人消费** → 两条记录常驻内存（按超时次数累积）。
        #   ★为什么不能改成"无等待者就丢弃"：存在**竞态**——`dispatch_node` 投递后、
        #   `wait_result` 建 Event 之前，worker 可能已快速返回（`track()` 里 `mark_running`
        #   发现任务已被取消即立即 `_notify_result`）。故**只在"超时放弃"这一明确路径上登记**，
        #   正常路径语义零变更。
        self._abandoned: Dict[str, None] = {}

    # ===================== M20（D21）：队列深度可观测（供监控轮询）=====================
    def queue_depth(self, agent_id: str) -> int:
        """
        函数功能与逻辑描述：
            【公开接口】返回某 Agent 队列的当前深度（待消费任务数），供监控 / 健康检查轮询；
            只读、不改变任何状态。★用 `.get()` 而非下标访问，避免 defaultdict 副作用建空队列。
        入参说明：
            agent_id (str)：目标 Agent 标识（如 "bill_agent"）。
        返回值说明：
            int：当前队列深度；该 Agent 尚无队列时返回 0。
        """
        q = self._task_queues.get(agent_id)
        return q.qsize() if q is not None else 0

    def snapshot_depths(self) -> Dict[str, int]:
        """
        函数功能与逻辑描述：
            【公开接口】返回全部队列的深度快照 `{agent_id: depth}`，供监控 / 排障一次性拉取。
            仅含**已创建过**队列的 Agent（defaultdict 按需创建，从未出现的 Agent 不在结果中）。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Dict[str, int]：Agent 标识 → 当前队列深度。
        """
        return {aid: q.qsize() for aid, q in self._task_queues.items()}

    # ===================== 对外公开方法：供测试/业务调用 =====================
    def clear_all(self):
        """
        函数功能与逻辑描述：
            【公开接口】清空全部任务队列、结果事件、结果缓存与会话暂存区。两个使用场景：
            ① 启动路径在崩溃恢复前调用（startup/bootstrap.bootstrap_all），保证
              resubmit_submitted 从【空队列】起点重投，不与本进程既有任务混在一起；
            ② 测试用例之间做隔离。
            注：进程重启后本对象是新建的（模块级单例随模块重新导入），这些容器本就是空的，
            故场景① 在首次启动时属**幂等保证**；它真正不可省略的情形是 bootstrap_all 被重复调用。
            实现上直接重建队列字典而非逐个 get_nowait 清空，原因：asyncio.Queue 在首次
            `await get()` 时会绑定所属 event loop，跨 loop 复用会抛
            "is bound to a different event loop"；原实现只用 get_nowait/put_nowait
            （不触发绑定校验）故未暴露此问题。重建后的队列全新且未绑定任何 loop，
            测试与重启场景均安全。
        入参说明：
            无（隐式 self）。
        返回值说明：
            无（就地重置实例内部状态，无返回值）。
        """
        # 【去轮询配套】直接重建队列字典，而非逐个 get_nowait 清空：
        # asyncio.Queue 在首次 await get() 时会绑定所属 event loop，跨 loop 复用会抛
        # "is bound to a different event loop"。原实现只用 get_nowait/put_nowait（不触发
        # 绑定校验）故从未暴露此问题。重建后队列全新、未绑定任何 loop，测试与重启场景均安全。
        # ★M20：重建时**必须同样带 maxsize**——否则清空队列后退化为无界（易漏点）
        self._task_queues = defaultdict(lambda: asyncio.Queue(maxsize=A2A_QUEUE_MAXSIZE))
        # 清空结果事件与缓存
        self._result_events.clear()
        self._result_store.clear()
        # ★M21（T-2）：清空"已放弃"标记（与结果缓存同生命周期）
        self._abandoned.clear()
        # 清空会话不匹配暂存区（否则测试用例之间会串消息）
        self._parked.clear()

    def send_task(self, target_agent: str, session_id: str, task_id: str, task_data: Dict[str, Any]):
        """
        函数功能与逻辑描述：
            向目标 Agent 的专属队列投递一条任务消息（非阻塞 put_nowait），消息体固定为
            {session_id, task_id, task_data} 三键。同步方法、不做超时控制，
            目标 Agent 若不是常驻 worker，消息会一直留在队列中等待后续消费。
        入参说明：
            target_agent (str)：目标 Agent 标识，作为队列分组键（如 stat_agent / bill_agent）。
            session_id (str)：会话标识，供接收侧做会话过滤。
            task_id (str)：任务唯一标识，业务 Agent 回传结果时以它作为路由键。
            task_data (Dict[str, Any])：任务载荷，如 query_ir、原始用户问题等。
        返回值说明：
            无（消息入队，另打印一条 "[A2A send_task] ... 入队完成" 到标准输出）。
        """
        msg = {
            "session_id": session_id,
            "task_id": task_id,
            "task_data": task_data
        }
        queue = self._task_queues[target_agent]
        try:
            queue.put_nowait(msg)
        except asyncio.QueueFull:
            # ★M20（D21）：满时**拒绝**而非阻塞——阻塞会把派发方（Orchestrator，用户请求
            #   持有者）一并挂死，属故障扩散。任务在 DB 侧仍为 submitted，交既有超时/重投
            #   机制兜底（队列只是下游投影，`04` §3.1）。
            logger.warning(
                f"[A2A send_task] 队列已满，拒绝投递: target={target_agent} "
                f"depth={queue.qsize()}/{A2A_QUEUE_MAXSIZE} task_id={task_id}")
            return False
        depth = queue.qsize()
        if depth >= A2A_QUEUE_WARN_DEPTH:
            # 堆积预警（不阻断）：队列深度是"饱和度"黄金信号
            logger.warning(
                f"[A2A send_task] 队列堆积预警: target={target_agent} "
                f"depth={depth}/{A2A_QUEUE_MAXSIZE} task_id={task_id}")
        print(f"[A2A send_task] target_agent={target_agent} session={session_id} task_id={task_id} 入队完成")
        return True

    async def recv_task(self, agent_id: str, session_id: Optional[str] = None, max_wait=3) -> Dict[str, Any]:
        """
        函数功能与逻辑描述：
            【测试专用接口】带超时的阻塞接收，在 max_wait 秒内拿到一条会话匹配的消息，否则抛 TimeoutError。
            ★使用方：生产路径**不调用**本方法——常驻 worker 一律走 recv_task_blocking（永久阻塞）。
            二者不可互换：测试用 blocking 会永久挂死；worker 用本方法会每 3s 抛 TimeoutError。
            调用方清单：tests/unit/test_a2a_bus.py（收发闭环 / 会话隔离 / 并发回归）、
            tests/e2e/test_a2a_full.py（含 max_wait=1 的超时用例）。
            取消息顺序为：① 先消费旁路暂存区（_parked）中会话匹配的历史消息，保证不丢；
            ② 未取到则挂起等待 `await asyncio.wait_for(queue.get())`，零空转
            （原实现为 get_nowait + sleep(0.01) 轮询，空队列时每 10ms 空转一次，
            消息到达后平均还需多等 5ms）。会话不匹配的消息进旁路暂存，不放回主队。
        入参说明：
            agent_id (str)：接收方 Agent 标识，决定消费哪个队列。
            session_id (Optional[str])：期望的会话标识；为 None 表示不过滤会话，任意消息都接收。
            max_wait：最长等待秒数，默认 3；从进入函数起算的总预算，而非每次循环重置。
        返回值说明：
            Dict[str, Any]：匹配到的消息，结构为 {session_id, task_id, task_data}。
        异常说明：
            TimeoutError：在预算内未等到匹配会话的消息（文案含 agent_id 与 max_wait）。
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

            # 未取到 → 挂起等待（零空转），到期仍未等到则超时
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
        函数功能与逻辑描述：
            【生产专用接口】常驻 Worker 接收口：无限阻塞、无超时，永不抛 TimeoutError，只能靠任务关停终止。
            ★使用方：唯一调用方为 4 个常驻 worker（startup/bootstrap.py 的
            _bill_agent_worker / _stat_agent_worker / _price_agent_worker / _finance_agent_worker）。
            测试**不使用**本方法（无超时会永久挂死）；测试场景请用 recv_task（带超时）。
            取消息顺序与 recv_task 一致——先消费旁路暂存区再 `await queue.get()` 挂起等待。
            （原实现为 get_nowait + sleep(0.1) 轮询：空闲时每 100ms 空转唤醒一次，
            4 个常驻 worker 约每秒 40 次无谓唤醒，且任务投递后平均多等 50ms 才被取走。）
            每次取出消息都会打印 pop / MATCH / MISMATCH 三类调试日志，便于排查会话错配。
        入参说明：
            agent_id (str)：Worker 所属 Agent 标识，决定消费哪个队列。
            session_id (Optional[str])：期望会话；为 None 表示不过滤，任意会话消息都接收。
        返回值说明：
            Dict[str, Any]：匹配到的消息，结构为 {session_id, task_id, task_data}。
        """
        queue = self._task_queues[agent_id]
        parked = self._parked[agent_id]

        print(f"[A2A recv_task_blocking START] agent={agent_id} expect_session={session_id}")

        while True:
            # 优先消费暂存区中"上次会话不匹配"的消息,当前项目场景用不到，单纯为了后面可能用到的多用户场景预留
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
        函数功能与逻辑描述：
            Worker 回传执行结果：先无条件把结果存入 _result_store（无论当前是否有等待者，
            避免等待者稍后才调用 wait_result 时结果已丢失），再确保该 task_id 有对应的
            asyncio.Event 并 set() 唤醒等待中的编排器。路由与唤醒均以 task_id 为唯一键。
            ★M21（T-2）：若该 task_id 已被编排器**放弃**（`wait_result` 超时清理，见
            `_remember_abandoned`），则本次回传**无人消费** → 直接丢弃并消费掉标记，
            避免 `_result_store` / `_result_events` 常驻泄漏。
            【D4 验收③】历史上曾有 `target_agent` / `session_id` 两个形参，实际从未被使用，
            保留会误导调用方以为存在「按 Agent 路由 / 按会话隔离」语义，故已删除。
        入参说明：
            task_id (str)：任务唯一标识，与 send_task 投递时使用的 task_id 一致。
            result (Any)：任意结果对象，由业务 Agent 决定结构，总线不做校验与序列化。
        返回值说明：
            无（写入结果缓存并唤醒等待者；若任务已被放弃则丢弃，两种情况均不返回值）。
        """
        # ★M21（T-2）：编排器已放弃该任务 → 回传无人消费，直接丢弃（并消费掉标记）
        if task_id in self._abandoned:
            self._abandoned.pop(task_id, None)
            logger.warning(f"[M21] 结果无人消费（编排器已放弃，直接丢弃）: task_id={task_id}")
            return
        # 无论是否存在等待者，先存入结果
        self._result_store[task_id] = result
        # Event不存在则主动创建
        if task_id not in self._result_events:
            self._result_events[task_id] = asyncio.Event()
        self._result_events[task_id].set()

    def _remember_abandoned(self, task_id: str) -> None:
        """
        函数功能与逻辑描述：
            M21（T-2）：登记"编排器已放弃"的 task_id，供 `send_result` 识别并丢弃无人消费的结果。
            容量有界：达到 `_ABANDONED_MAX` 时按**插入序**淘汰最旧一项（dict 在 3.7+ 保序，
            故用 dict 当有序 set）——本集合仅是防泄漏的**辅助标记**，万一误淘汰，后果只是
            "那一条结果恢复原泄漏行为"，不影响任何正确性。
        入参说明：
            task_id (str)：被放弃的任务标识。
        返回值说明：
            无（就地修改 self._abandoned）。
        """
        if len(self._abandoned) >= _ABANDONED_MAX:
            self._abandoned.pop(next(iter(self._abandoned)), None)
        self._abandoned[task_id] = None

    async def wait_result(self, task_id: str, timeout: int = MODEL_TIMEOUT) -> Any:
        """
        函数功能与逻辑描述：
            编排器阻塞等待某个任务的执行结果，带超时熔断。若该 task_id 尚无对应 Event 则先补建，
            避免结果先到、等待后至的时序问题。成功时取出结果并同时清理结果缓存与事件，
            防止同一 task_id 在长生命周期进程中无限累积。
            ★M22：默认值改为**引用 `config.MODEL_TIMEOUT`**（现 150s），不再硬编码 240
            ——该值在 M19/M22 期间几经调整，硬编码会静默漂移。
            原注释"本地 CPU 推理单次可能超 120s"系**本地 1.8B 基线**（单次实测 189s）；
            M22 起本地通道已弃用、生产走外部强模型（**秒级**），150s 留有充分余量（`设计/20`）。
        入参说明：
            task_id (str)：要等待的任务标识。
            timeout (int)：最长等待秒数，默认取 config.MODEL_TIMEOUT。
        返回值说明：
            Any：任务结果对象（由 send_result 写入的原始对象）。
        异常说明：
            TimeoutError：超时未等到结果，文案为 "任务[{task_id}]执行超时"；
                抛出前会清理该 task_id 的事件与结果缓存。
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
            # ★M21（T-2）：登记"已放弃"——worker 事后回传时据此丢弃，避免无人消费的常驻泄漏
            self._remember_abandoned(task_id)
            raise TimeoutError(f"任务[{task_id}]执行超时")

# 注：会话级资源清理当前无需求；若将来要做，需同时处理 _task_queues / _parked 的按会话剔除。


# 全局单例消息总线
a2a_bus = A2AMessageBus()

__all__ = ["a2a_bus"]
