# -*- coding: utf-8 -*-
"""M5（D2）Durable Execution：任务状态机 + 崩溃恢复 + 重投（TaskManager）。

设计依据：
- `04` §2.4（`task_state` 表与状态语义）/ **§3.1 修订版**（崩溃对账：`bill` 按 `task_id` 反查）
  / §3.2（`task_id` 生成与"只防重投、不防重说"边界）/ §3.3（`retriable` 标志）
- `08` §2.3（DDL + 状态机 + `recover_orphans`）/ §6 Q4（独立库）/ Q7（优雅关闭）
- `03` §8.1 **落点**：`startup/task_manager.py` + `task_state` 表
  （★`11` M5 原文写 `mcpGateway/task_state.py` 与之冲突，以 `03` §8.1 为准并已勘误——
   同 M7「Tracer 落点以 `03` §8.1 为准」的规矩；TaskManager 管的是 **A2A 任务投递**，
   与 MCP 工具平台无关，放 `mcpGateway` 语义也不对）

状态机：
```
                    ┌────────── retry_cnt<3 且 retriable=1 ───────────┐
                    ▼                                                  │
  submitted ──(worker 领取)──→ running ──(成功)──→ completed           │
                                  ├──(异常)──→ failed ─────────────────┘
                                  ├──(优雅关闭)──→ interrupted ──(重启扫描)──→ submitted / completed
                                  └──(用户取消)──→ cancelled  ← 【终态】不重投、不回复、不进恢复扫描
```

★四个关键语义（避免误用）：
1. **completed 判据是"执行完成"而非"业务成功"**：worker 未抛异常即 `completed`，
   业务成败（如 `need_more_info` 追问、收入不入账拦截）由 `result` 字段承载。
   否则"正常的业务拒绝"会被标 failed，恢复时触发误告警。
2. **DB 是队列事实的唯一来源**（`04` §3.1 补注）：`recover_orphans()` 只负责把状态改对，
   `submitted` 即"声明可重投"，再由 `resubmit_submitted()` 重发 `send_task`——
   队列是 DB 状态的下游投影，重启即失，故不直接改内存队列。
3. **对账只针对有副作用的任务**（`bill_agent`）：`stat`/`price`/`finance` 无落库副作用，
   崩溃后直接重投即可，无需对账。
4. ★**`interrupted` 与 `cancelled` 是两种语义相反的"中断"**（M16 用户取消）：
   - `interrupted`：进程优雅关闭遗留，**重启后参与恢复扫描、会被重投**（"任务没做完，继续做"）；
   - `cancelled`：**用户主动要求停止**，是**终态**——`recover_orphans` 的扫描条件
     （`status IN ('running','interrupted')`）天然不含它，`resubmit_submitted` 也只投
     `submitted`，故**既不会重投、也不会再触发一次自动回复**。
   ★二者**不可复用同一状态**：若把用户取消也标成 `interrupted`，重启后会自动重投并回复，
   与"用户已明确不要了"直接冲突。
"""
import asyncio
import json
import time
from typing import Any, Callable, Dict

from config.config import TASK_STATE_DB, REQUEUE_STALE_AFTER_SEC, TASK_TIMEOUT
from utils.common import DatabaseCRUD, logger

# 有副作用（会写库）的任务：崩溃后**禁止直接重投**，必须先按 task_id 对账（`04` §3.3）
SIDE_EFFECT_AGENTS = {"bill_agent"}
# 重投上限（与 `04` §2.4 / `08` §2.3 一致）：超过则标记 failed，需人工介入
MAX_RETRY = 3

# 终态：不再参与恢复扫描（cancelled 是用户取消终态，见模块 docstring 第 4 条）
TERMINAL_STATUS = ("completed", "failed", "cancelled")

# 在途（非终态）状态：可被用户取消的集合
INFLIGHT_STATUS = ("submitted", "running")


class TaskManager:
    """
    函数/类功能与逻辑描述：
        任务状态机（独立库 `database/task_state.db`）：派发前建单（submitted），
        worker 侧包装状态流转（running → completed / failed / interrupted），
        启动时扫描遗留 running/interrupted 任务做崩溃对账与重投决策。
        所有写操作为同步 SQLite（单次 ~1ms，与既有 `db.` 调用一致，不引入事件循环阻塞风险）。
    构造入参说明：
        db_path：任务状态库文件路径，默认 config.TASK_STATE_DB（database/task_state.db）。
    返回值说明：
        构造返回 TaskManager 实例；构造期即幂等建表（详见 __init__）。
    """

    def __init__(self, db_path=TASK_STATE_DB):
        """
        函数功能与逻辑描述：
            建立任务状态库连接并幂等建表：不依赖调用方"先跑过 init_all_tables"的调用顺序，
            保证新建库 / 独立测试环境不会 `no such table`。DDL 单一来源见
            `database/init_db.py`（延迟导入以避免启动期循环依赖）。
        入参说明：
            db_path：任务状态库文件路径，默认 config.TASK_STATE_DB
                （database/task_state.db）。
        返回值说明：
            无（仅初始化 self._db 并确保 task_state 表存在）。
        """
        # 独立库：Durable 每次状态流转一次写，与 bill.db 同库会争抢写锁（`08` §6 Q4）
        self._db = DatabaseCRUD(db_path)
        # ★在途 runner Task 登记表（task_id -> asyncio.Task），供用户取消时精准中断单个子任务。
        #   存在的理由：worker 消费循环是 `while True` 常驻协程，若直接 cancel worker 协程本身，
        #   CancelledError 会穿透循环的 `except Exception`（CancelledError 继承 BaseException），
        #   把整个 worker 打死——那是"进程关停"的语义，不是"取消一个任务"。
        #   故 track() 把 runner() 包成独立 Task 再 await，取消只针对这个内层 Task，
        #   worker 消费循环不受影响、可继续领取后续任务。
        #   内存态：进程重启即失，但取消请求本身已落 `cancelled` 状态（DB 是唯一事实源）。
        self._running_tasks: Dict[str, asyncio.Task] = {}
        # ★自包含建表（幂等）：不依赖"启动时先跑过 init_all_tables"的调用顺序，
        #   否则新建库 / 独立测试环境会 `no such table`。DDL 单一来源见 `database/init_db.py`。
        from database.init_db import ensure_task_state_tables  # noqa: PLC0415 —— 延迟导入避免启动期循环

        ensure_task_state_tables(self._db)

    # ─────────── 写入侧：状态流转 ───────────
    def create_task(self, task_id: str, session_id: str, agent_id: str,
                    payload: dict | None = None, retriable: int | None = None) -> None:
        """
        函数功能与逻辑描述：
            派发前建单（状态 `submitted`），使 DB 成为任务事实的唯一来源；用
            INSERT OR IGNORE，以 task_id 主键充当幂等键——重复投递同一 task_id 不会
            重复建单、也不会覆盖首次 payload（`04` §3.2：幂等键防"重投"，不防"重说"）。
        入参说明：
            task_id (str)：任务唯一 ID，主键兼幂等键。
            session_id (str)：所属会话 ID。
            agent_id (str)：目标子 Agent 名，取值 bill_agent / stat_agent /
                price_agent / finance_agent。
            payload (dict | None)：派发参数，原样 JSON 落库（崩溃后据此原样回填重投），
                默认 None（按 {} 存）。
            retriable (int | None)：是否允许自动重投，取值 0（否）/ 1（是）；
                None = 按 agent_id 推断（SIDE_EFFECT_AGENTS 内的写任务 → 0，其余 → 1），
                默认 None。
        返回值说明：
            无（写入失败由 execute_sql 内部记日志并返回 False，不向上抛）。
        """
        if retriable is None:
            retriable = 0 if agent_id in SIDE_EFFECT_AGENTS else 1
        now = time.time()
        self._db.execute_sql(
            "INSERT OR IGNORE INTO task_state "
            "(task_id, session_id, agent_id, status, payload, retry_cnt, retriable, created_at, updated_at) "
            "VALUES (?, ?, ?, 'submitted', ?, 0, ?, ?, ?)",
            (task_id, session_id, agent_id,
             json.dumps(payload or {}, ensure_ascii=False), retriable, now, now))
        # INSERT OR IGNORE：task_id 主键即幂等键——重复投递同一 task_id 不会重复建单
        # （`04` §3.2：幂等键防"重投"，不防"重说"）

    def mark_running(self, task_id: str) -> bool:
        """
        函数功能与逻辑描述：
            worker 领取任务后把状态流转为 `running`（`04` §2.4 状态机）。
            ★M16：返回值携带"能否执行"的判定——若该任务在**排队期间**已被用户取消
            （状态已是 `cancelled`），必须返回 False 让 worker 跳过执行，否则会把终态
            覆盖回 `running` 并真的跑一遍。这是"取消队列中待执行任务"能生效的关键：
            取消只改 DB 状态（队列里的消息无法安全剔除），领取时必须靠状态做二次拦截。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            bool：True = 状态已置 `running`，可继续执行；False = 已被取消，应跳过执行。
        """
        row = self.get_task(task_id)
        if row and row.get("status") == "cancelled":
            logger.info(f"[M16] task={task_id} 排队期间已被取消 → 跳过执行")
            return False
        self._update_status(task_id, "running")
        return True

    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """
        函数功能与逻辑描述：
            标记任务 `completed`。★判据是"worker 未抛异常"即执行完成，而非"业务成功"，
            业务成败由 result 字段承载——否则正常的业务拒绝（need_more_info 追问、
            收入不入账拦截）会被标 failed，恢复时触发误告警。
        入参说明：
            task_id (str)：任务 ID。
            result (Any)：执行结果，JSON 序列化后写入 result 列（不可序列化时降级为
                str），默认 None（不写 result 列）。
        返回值说明：
            无（仅更新状态 / result / updated_at）。
        """
        self._update_status(task_id, "completed", result=result)

    def mark_failed(self, task_id: str, result: Any = None) -> None:
        """
        函数功能与逻辑描述：
            标记任务 `failed`（worker 抛非取消异常，或恢复扫描判定不可再重投）。
            写入的是任务状态而非业务结论，具体错误由 result 承载。
        入参说明：
            task_id (str)：任务 ID。
            result (Any)：失败信息，序列化后写入 result 列；track 调用时传
                {"error": "<异常类型>: <异常信息>"}，默认 None（不写 result 列）。
        返回值说明：
            无（仅更新状态 / result / updated_at）。
        """
        self._update_status(task_id, "failed", result=result)

    def mark_interrupted(self, task_id: str) -> None:
        """
        函数功能与逻辑描述：
            优雅关闭：worker 被 cancel 时标记 `interrupted`（`08` §6 Q7）——区别于
            failed 的关键在于语义：任务并非执行失败，重启后可参与恢复扫描重投，
            避免"账已落库却因 retriable=0 被判 failed"。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            无（仅更新状态 / updated_at）。
        """
        self._update_status(task_id, "interrupted")

    def mark_cancelled(self, task_id: str) -> None:
        """
        函数功能与逻辑描述：
            把任务标记为**用户取消终态** `cancelled`（M16）。★与 `interrupted` 语义相反：
            本状态**不参与** `recover_orphans` 的恢复扫描（其条件是
            `status IN ('running','interrupted')`），`resubmit_submitted` 也只投 `submitted`，
            因此重启后**既不会重投、也不会再次触发自动回复**——这正是"用户已明确要求停止"
            应有的行为。`cancelled` 属于 TERMINAL_STATUS（终态）。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            无（仅更新状态 / updated_at）。
        """
        self._update_status(task_id, "cancelled")

    def is_cancelled(self, task_id: str) -> bool:
        """
        函数功能与逻辑描述：
            判定某任务是否已被用户取消（M16）。供 `track()` 捕获 `CancelledError` 时
            区分两种语义完全不同的中断来源：
              - DB 状态为 `cancelled` → **用户主动取消** → 标终态、吞掉异常、worker 继续消费；
              - 其它（含状态仍是 `running`）→ **进程优雅关闭** → 标 `interrupted` 并向上传播，
                由关停逻辑统一收敛。
            ★之所以靠查 DB 而非靠异常类型区分：Python 的 `task.cancel()` 一律抛
            `asyncio.CancelledError`，无法携带"谁取消的"信息；而取消请求在执行 cancel **之前**
            已落库，故查询必然命中（DB 是唯一事实源）。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            bool：True = 已被用户取消；False = 未取消或记录不存在。
        """
        row = self.get_task(task_id)
        return bool(row and row.get("status") == "cancelled")

    def cancel_running(self, task_id: str) -> bool:
        """
        函数功能与逻辑描述：
            对某任务**正在执行的 runner Task** 发 `asyncio.cancel()`（M16）。
            ★只取消内层 runner Task（`track()` 中 `create_task(runner())` 的产物），
            **不取消 worker 消费循环协程**——后者是常驻的，取消它等于关停该 Agent 的整条
            消费链路（详见 `__init__` 中对 `_running_tasks` 的说明）。
            返回 False 的两种情形：任务仍在队列里未领取（尚无 runner Task）、
            或已执行完毕（Task 已 done）——二者都无需中断。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            bool：True = 已发出取消；False = 当前无在途 runner Task。
        """
        task = self._running_tasks.get(task_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def cancel_session(self, session_id: str) -> Dict[str, Any]:
        """
        函数功能与逻辑描述：
            【用户级取消入口】取消某会话当前全部在途任务（`submitted` / `running`），M16。
            执行两步，**顺序不可颠倒**：
              ① **先落库**：把在途任务的 status 直接置 `cancelled`——DB 是唯一事实源，
                 且**必须在 cancel 协程之前完成**，否则 `track()` 捕获 `CancelledError` 时
                 无法区分"用户取消"与"进程关闭"（会把取消误标成 `interrupted`，
                 重启后自动重投并再次回复，与用户意图冲突）；
              ② **再中断**：对正在执行的 runner Task 发 `cancel()`（内层 Task，不杀 worker）。
            队列中尚未领取的任务（`submitted`）无法从 `asyncio.Queue` 安全剔除，
            靠 `mark_running()` 的二次拦截生效（worker 领取时发现已是 `cancelled` 即跳过）。
            已 `completed` / `failed` 的任务不受影响。
            ★已产生的副作用（如已落库的账单）**不回滚**——那属于补偿事务范畴，与
            "停止执行"是两个问题：用户取消的语义是"别再继续"，不是"当作没发生"。
        入参说明：
            session_id (str)：要取消的会话标识。
        返回值说明：
            dict：{"cancelled": [已取消的 task_id 列表],
                   "interrupted": [其中真正中断了执行的 task_id 列表]}。
        """
        rows = self._db.query_sql(
            "SELECT task_id FROM task_state "
            "WHERE session_id=? AND status IN (?,?)",
            (session_id, INFLIGHT_STATUS[0], INFLIGHT_STATUS[1]))
        cancelled: list = []
        interrupted: list = []
        for r in rows:
            tid = r["task_id"]
            # ① 先落库（必须早于 cancel，见 docstring 中的时序说明）
            self.mark_cancelled(tid)
            cancelled.append(tid)
            # ② 再中断正在执行的 runner（队列中的任务无 Task 可取消，靠 mark_running 拦截）
            if self.cancel_running(tid):
                interrupted.append(tid)
        if cancelled:
            logger.warning(
                f"[M16] 用户取消：session={session_id} 共 {len(cancelled)} 个在途任务 → cancelled"
                f"（其中 {len(interrupted)} 个正在执行、已发出中断）")
        return {"cancelled": cancelled, "interrupted": interrupted}

    def _update_status(self, task_id: str, status: str, result: Any = None) -> None:
        """
        函数功能与逻辑描述：
            状态流转的统一落库实现：result 为空时只更新 status / updated_at；
            result 非空时把结果 JSON 序列化后一并写入。序列化失败（如含不可序列化对象）
            降级为 str(result)，保证结果序列化问题不阻断状态流转。
        入参说明：
            task_id (str)：任务 ID。
            status (str)：目标状态，取值
                submitted / running / completed / failed / interrupted / cancelled。
            result (Any)：结果内容，默认 None（不写 result 列）。
        返回值说明：
            无（仅执行 UPDATE，失败由 execute_sql 内部吞掉）。
        """
        now = time.time()
        if result is None:
            self._db.execute_sql(
                "UPDATE task_state SET status=?, updated_at=? WHERE task_id=?",
                (status, now, task_id))
            return
        try:
            payload = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover —— 结果不可序列化时不阻断状态流转
            payload = str(result)
        self._db.execute_sql(
            "UPDATE task_state SET status=?, result=?, updated_at=? WHERE task_id=?",
            (status, payload, now, task_id))

    # ─────────── 读取侧 ───────────
    def get_task(self, task_id: str) -> dict | None:
        """
        函数功能与逻辑描述：
            按 task_id 读取任务行（含 status / result / retry_cnt / retriable / payload），
            供崩溃恢复与测试断言使用；纯查询、无副作用。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            dict | None：命中返回该行全部列（列名做 key）的字典；未命中或查询异常返回 None。
        """
        rows = self._db.query_sql("SELECT * FROM task_state WHERE task_id=?", (task_id,))
        return rows[0] if rows else None

    def count_by_status(self, status: str) -> int:
        """
        函数功能与逻辑描述：
            统计处于指定状态的任务条数（SELECT COUNT(1)），用于观测 / 测试断言。
        入参说明：
            status (str)：目标状态，取值
                submitted / running / completed / failed / interrupted / cancelled。
        返回值说明：
            int：该状态的任务条数；无记录或查询异常返回 0。
        """
        rows = self._db.query_sql("SELECT COUNT(1) AS c FROM task_state WHERE status=?", (status,))
        return int(rows[0]["c"]) if rows else 0

    # ─────────── worker 执行包装 ───────────
    def _agent_of(self, task_id: str) -> str:
        """
        函数功能与逻辑描述：
            取任务对应的 agent_id，用于构造与子 Agent 回传同构的取消结果
            （编排器 `_wait_one` 按 `{success, agent_type, msg, data, error}` 解析，
            取消结果必须同构，否则会被当成"数据损坏"）。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            str：agent_id；记录不存在时返回空字符串（编排器侧按未知类型容错）。
        """
        row = self.get_task(task_id)
        return str(row.get("agent_id") or "") if row else ""

    def _cancelled_result(self, task_id: str) -> dict:
        """
        函数功能与逻辑描述：
            构造"任务被用户取消"的标准结果结构（M16）——与子 Agent 正常回传的结构同构，
            使编排器无需特判即可识别为一次失败结果，进而走收口路径而不是无限等待。
        入参说明：
            task_id (str)：任务 ID。
        返回值说明：
            dict：{"success": False, "agent_type": ..., "msg": "任务已被用户取消",
                   "data": {}, "error": "用户取消"}。
        """
        return {"success": False, "agent_type": self._agent_of(task_id),
                "msg": "任务已被用户取消", "data": {}, "error": "用户取消"}

    def _failed_result(self, task_id: str, error_text: str) -> dict:
        """
        函数功能与逻辑描述：
            M19（D20）新增：构造"任务执行失败（超时 / 异常）"的标准结果结构——与子 Agent
            正常回传的结构**同构**（五键），使编排器 `_wait_one` 无需特判即可解析。
            存在的意义同 `_cancelled_result`：**失败路径也必须有人回传结果**，否则编排器
            只能盲等到 `wait_result` 的 240s 超时才发现出事。
        入参说明：
            task_id (str)：任务 ID，用于反查 agent_type。
            error_text (str)：失败原因文案（超时则为"任务执行超时（Ns）"，异常则为
                "类型名: 信息"），同时写入结果 payload 供编排器识别与回执。
        返回值说明：
            dict：{"success": False, "agent_type": ..., "msg": "任务执行失败",
                   "data": {}, "error": error_text}。
        """
        return {"success": False, "agent_type": self._agent_of(task_id),
                "msg": "任务执行失败", "data": {}, "error": error_text}

    @staticmethod
    def _notify_result(task_id: str, result: dict) -> None:
        """
        函数功能与逻辑描述：
            ★M16 取消路径的关键补偿：把取消结果回传到 A2A 总线，唤醒正在等待的编排器。
            为什么必须做：正常路径下结果由子 Agent 图内的 reply_node 调 `a2a_bus.send_result`
            回传；而任务被取消时 runner（整张子图）已被中断，reply_node **根本不会执行**，
            若不在此代传结果，编排器的 `wait_result(tid, timeout=240)` 会**一直阻塞到 240 秒
            超时**才继续——用户点了"停止"却要等 4 分钟才有反应。
            回传格式与子 Agent 一致（JSON 字符串），使 `_wait_one` 的 `json.loads` 路径无差别。
            延迟导入 a2a_bus 以避免 `startup` ↔ `mcpGateway` 的 import 期循环。
            本方法吞掉一切异常（可观测/补偿失败不阻断主链路）。
        入参说明：
            task_id (str)：任务 ID，与编排器 wait_result 的 key 一致。
            result (dict)：要回传的结构化结果。
        返回值说明：
            无（副作用：向消息总线写入结果并唤醒等待者；失败仅记 warning）。
        """
        try:
            from mcpGateway.a2a_queue import a2a_bus  # noqa: PLC0415 —— 延迟导入避免循环
            a2a_bus.send_result(task_id, json.dumps(result, ensure_ascii=False))
        except Exception as e:  # noqa: BLE001 —— 补偿失败不得阻断主链路
            logger.warning(f"[M16] 取消结果回传失败（编排器将等待至超时）: {e}")

    async def track(self, msg: dict, runner: Callable) -> Any:
        """
        函数功能与逻辑描述：
            worker 侧统一包装：mark_running → 执行 runner → 按结果流转终态。
            M19 起为**五分支**口径（M16 四分支 + 任务级超时）：
              ① runner 正常返回 → `mark_completed` 并回传结果；
              ② 抛 `CancelledError` 且 DB 状态为 `cancelled`（**用户取消**）→ 返回取消结果、
                 **不向上抛**，worker 消费循环继续领取后续任务；
              ③ 抛 `CancelledError` 且非用户取消（**进程优雅关闭**）→ `mark_interrupted`
                 后**继续抛**，由关停逻辑收敛——不能标 failed，否则重启后"账已记"却因
                 retriable=0 被判 failed（`04` §3.1）；
              ④ **任务级超时**（`TASK_TIMEOUT` 秒，M19 新增）→ `mark_failed` + **代传失败结果**
                 + 吞掉异常，worker 继续服务后续任务；防"挂死的 runner 永久占住单协程消费循环"；
              ⑤ 其它异常 → `mark_failed` + **代传失败结果**（M19 补）后继续抛，
                 依赖调用方的 except 保证消费循环不中断。
            ★M19 两处补偿的理由（`设计/17` D20）：改造前异常分支只落状态、**不回传**，
              子图异常穿透时编排器拿不到任何结果，只能**盲等到 240s 超时**才收口。
            ★为什么把 runner 包成独立 Task：用户取消要能只中断"这一个任务"，而 worker 是
            常驻 `while True` 协程。若直接 cancel worker 协程，`CancelledError` 会穿透循环的
            `except Exception`（`CancelledError` 继承 `BaseException`，不被捕获）从而打死整个
            worker——那是"进程关停"语义，不是"取消一个任务"。包成内层 Task 后，取消只作用
            于它，worker 循环照常继续。
        入参说明：
            msg (dict)：A2A 消息，至少含 task_id（session_id / task_data 供 runner 使用）。
            runner (Callable)：无参可等待对象，真正执行子 Agent 图。
        返回值说明：
            Any：runner() 的返回值；被用户取消时返回 `_cancelled_result()` 形态的失败结果；
                runner() 抛出的**非取消**异常仍原样向上抛出、不在此吞掉。
        """
        task_id = msg["task_id"]
        # ★排队期间已被用户取消 → 不进 running、不执行（取消"队列中任务"的拦截点）
        if not self.mark_running(task_id):
            result = self._cancelled_result(task_id)
            self._notify_result(task_id, result)     # ★必须回传，否则编排器等到 240s 超时
            return result

        task = asyncio.create_task(runner())
        self._running_tasks[task_id] = task          # 登记，供 cancel_running 精准中断
        try:
            # ★M19 引入 / ★M22 调整语义：任务级超时兜底——防"挂死的 runner 永久占住 worker
            #   单协程消费循环"（worker 是 `while True` 串行消费，一个卡死的任务会让该 Agent
            #   通道持续堆积、彻底停摆）。
            #   ★M22 前（TASK_TIMEOUT=300 > MODEL_TIMEOUT=240）：编排器先超时放弃，本超时兜底。
            #   ★M22 起（TASK_TIMEOUT=120 < MODEL_TIMEOUT=150）：**worker 侧先超时**——好处是
            #   `track()` 会走 ④ 分支 `mark_failed` + **代传失败结果**，编排器**立即**收到失败
            #   并触发 M19 终止语义，而不是自己盲等到 150s 才发现。<=0 = 关闭（回滚开关）。
            if TASK_TIMEOUT and TASK_TIMEOUT > 0:
                res = await asyncio.wait_for(task, timeout=TASK_TIMEOUT)
            else:
                res = await task
        except asyncio.TimeoutError:
            # ④ 任务级超时：wait_for 超时时已自动取消并等待内层 Task 结束
            #   （CPython `_cancel_and_wait`），下方 task.cancel() 属防御性冗余。
            task.cancel()
            err = f"任务执行超时（{TASK_TIMEOUT}s）"
            logger.warning(f"[M19] task={task_id} {err}")
            self.mark_failed(task_id, {"error": err})
            result = self._failed_result(task_id, err)
            self._notify_result(task_id, result)
            return result
        except asyncio.CancelledError:
            if self.is_cancelled(task_id):
                # ② 用户取消：吞掉异常，worker 消费循环继续（★不 raise）
                logger.info(f"[M16] task={task_id} 已按用户要求中断执行")
                result = self._cancelled_result(task_id)
                self._notify_result(task_id, result)  # ★必须回传（runner 已中断，reply_node 不会执行）
                return result
            # ③ 进程优雅关闭：标 interrupted 并向上传播，由关停逻辑收敛
            self.mark_interrupted(task_id)
            raise
        except Exception as e:
            # ⑤ 其它异常：★M19 起补代传失败结果（改造前只落状态 → 编排器盲等 240s 超时）
            err = f"{type(e).__name__}: {e}"
            self.mark_failed(task_id, {"error": err})
            self._notify_result(task_id, self._failed_result(task_id, err))
            raise
        finally:
            self._running_tasks.pop(task_id, None)
            if not task.done():
                # ★兜底：当前协程被取消时 await 会提前退出，内层 Task 不能变成孤儿
                task.cancel()
        self.mark_completed(task_id, res)
        return res

    # ─────────── 崩溃恢复 ───────────
    def recover_orphans(self, bill_db=None) -> dict:
        """
        函数功能与逻辑描述：
            启动时扫描上次进程遗留的 `running` / `interrupted` 任务并定状态——本方法只负责
            由 DB 把状态改对，真正的重投由 resubmit_submitted 完成。
            对账规则（仅对 SIDE_EFFECT_AGENTS 内有副作用的 `bill_agent`）：
              - `bill` 表存在该 `task_id` 记录 → 副作用已发生 → 置 `completed`（不是 failed！）
              - 无记录 → 确实没执行 → `retriable=1` 且 `retry_cnt < MAX_RETRY` 时
                置 `submitted` 并把 retry_cnt+1（等待重建队列重投），否则置 `failed`
            无副作用任务不做对账，直接走上述重投/failed 分支。有遗留任务时打一条告警日志。
            ★M16：扫描条件是 `status IN ('running','interrupted')`——`cancelled`（用户取消终态）
            **不在其中**，因此被用户取消的任务重启后**不会被重投、也不会再次触发自动回复**，
            这正是取消语义应有的行为（区别于 `interrupted` 的"继续做完"）。
        入参说明：
            bill_db：业务库访问对象（需提供 `query_sql`），默认 None 表示使用
                `utils.common.db` 单例（延迟导入避免循环依赖）；测试可注入替身。
        返回值说明：
            dict：处理计数 {"completed": int, "resubmitted": int, "failed": int}，
                便于启动日志与测试断言。
        """
        if bill_db is None:
            from utils.common import db as bill_db  #延迟导入避免循环依赖
        rows = self._db.query_sql(
            "SELECT task_id, agent_id, retriable, retry_cnt FROM task_state "
            "WHERE status IN ('running','interrupted')")
        stat = {"completed": 0, "resubmitted": 0, "failed": 0}
        for r in rows:
            task_id, agent_id = r["task_id"], r["agent_id"]
            cnt = int(r["retry_cnt"] or 0)
            retriable = int(r["retriable"] or 0)
            # ★有副作用任务先对账：确认副作用到底有没有发生
            if agent_id in SIDE_EFFECT_AGENTS and self._bill_exists(bill_db, task_id):
                self._db.execute_sql(
                    "UPDATE task_state SET status='completed', updated_at=? WHERE task_id=?",
                    (time.time(), task_id))
                stat["completed"] += 1
                logger.warning(f"[M5] 崩溃对账：task={task_id} 账单已落库 → 标记 completed（非 failed）")
                continue
            if retriable and cnt < MAX_RETRY:
                self._db.execute_sql(
                    "UPDATE task_state SET status='submitted', retry_cnt=?, updated_at=? WHERE task_id=?",
                    (cnt + 1, time.time(), task_id))
                stat["resubmitted"] += 1
            else:
                self._db.execute_sql(
                    "UPDATE task_state SET status='failed', updated_at=? WHERE task_id=?",
                    (time.time(), task_id))
                stat["failed"] += 1
        if rows:
            logger.warning(f"[M5] 崩溃恢复完成：{stat}")
        return stat

    @staticmethod
    def _bill_exists(bill_db, task_id: str) -> bool:
        """
        函数功能与逻辑描述：
            崩溃对账辅助查询：确认业务库 `bill` 表是否已有该 task_id 的记账记录，
            用于判定"副作用到底有没有发生"。查询异常按"未落库"保守处理——宁可不谎报成功，
            只记错误日志后返回 False。
        入参说明：
            bill_db：业务库访问对象（需提供 `query_sql`）。
            task_id (str)：任务 ID。
        返回值说明：
            bool：True = bill 表已有该 task_id 记录（副作用已发生）；False = 无记录或查询异常。
        """
        try:
            rows = bill_db.query_sql("SELECT 1 FROM bill WHERE task_id=?", (task_id,))
        except Exception as e:  # pragma: no cover —— 对账查询失败按"未落库"处理（保守：不谎报成功）
            logger.error(f"[M5] 对账查询失败（按未落库处理）: {e}")
            return False
        return bool(rows)

    def resubmit_submitted(self, send_fn: Callable) -> int:
        """
        函数功能与逻辑描述：
            把 DB 中 `status='submitted'` 的任务重投到队列（`04` §3.1 补注：DB 是唯一事实源，
            队列只是下游投影）。★只在启动路径调用一次：此时队列刚 clear_all()，submitted
            全部是上次进程遗留任务；运行期重复调用会把"正在排队等待领取"的新任务重复投递
            （worker 无感知 → 重复执行）。
            过期过滤（【2026-09-05 演示健壮性】）：created_at 距今超过 REQUEUE_STALE_AFTER_SEC
            （默认 1800s）的 submitted 视为测试/演示残留——崩溃前正在排队的任务在队列里只停留
            毫秒级，能躺 >30 分钟只可能是历史脏数据，盲目重投会让每次演示启动都真调 LLM /
            真写账；此类任务直接标记 failed，不再重投。
        入参说明：
            send_fn (Callable)：投递函数，签名 (agent_id, session_id, task_id, payload)，
                生产路径传 a2a_bus.send_task。
        返回值说明：
            int：实际重投的任务条数（不含被判过期标 failed 的条数）。
        """
        rows = self._db.query_sql(
            "SELECT task_id, session_id, agent_id, payload, created_at FROM task_state "
            "WHERE status='submitted'")
        now = time.time()
        fresh: list[dict] = []
        stale: list[dict] = []
        for r in rows:
            try:
                created = float(r["created_at"] or 0)
            except (TypeError, ValueError):  # pragma: no cover
                created = 0.0
            (fresh if now - created <= REQUEUE_STALE_AFTER_SEC else stale).append(r)
        for r in stale:
            self._db.execute_sql(
                "UPDATE task_state SET status='failed', updated_at=? WHERE task_id=?",
                (now, r["task_id"]))
        for r in fresh:
            try:
                payload = json.loads(r["payload"] or "{}")
            except Exception:  # pragma: no cover
                payload = {}
            send_fn(r["agent_id"], r["session_id"], r["task_id"], payload)
        if rows:
            if fresh:
                logger.info(f"[M5] 重建队列：重投 {len(fresh)} 条 submitted 任务")
            if stale:
                logger.warning(
                    f"[M5] 过滤 {len(stale)} 条过期 submitted 遗留任务"
                    f"（> {int(REQUEUE_STALE_AFTER_SEC)}s，视为测试/演示残留）→ 标记 failed")
        return len(fresh)

    def close(self) -> None:
        """
        函数功能与逻辑描述：
            关闭任务状态库连接，复用 DatabaseCRUD.close() 的幂等保护（重复调用不报错）。
        入参说明：
            无。
        返回值说明：
            无（仅关闭连接，失败只记日志）。
        """
        self._db.close()


# 进程级单例（懒加载）：orchestrator 建单与 bootstrap 恢复/重投必须共用同一实例。
# ★懒加载而非模块级实例化：`agents/*` 的导入链会经 `startup` 反向引入，
#   模块级建连接会在 import 期产生副作用；懒加载把连接推迟到真正使用时。
_MANAGER: "TaskManager | None" = None


def get_task_manager() -> "TaskManager":
    """
    函数功能与逻辑描述：
        获取进程级 TaskManager 单例（懒加载）：首次调用时构造实例并缓存到模块级 _MANAGER，
        之后每次调用直接返回同一实例。
        ★必须懒加载而非模块级实例化：`agents/*` 的导入链会经 `startup` 反向引入本模块，
        模块级建连接会在 import 期产生副作用（提前拉起数据库连接）。
        编排器的任务建单与启动期 bootstrap 的恢复/重投必须共用同一实例，
        否则会各自持有独立连接与内存状态，导致对账不一致。
        本函数非线程安全（无锁保护），依赖「事件循环单线程 + 启动期串行调用」的前提。
    入参说明：
        无。
    返回值说明：
        TaskManager：全局唯一实例（首次调用时创建，之后复用）。
    """
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TaskManager()
    return _MANAGER


__all__ = ["TaskManager", "get_task_manager", "SIDE_EFFECT_AGENTS", "MAX_RETRY",
           "TERMINAL_STATUS", "INFLIGHT_STATUS"]
