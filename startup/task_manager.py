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
                                  └──(异常)──→ failed ─────────────────┘
                                  └──(优雅关闭)──→ interrupted ──(重启扫描)──→ submitted / completed
```

★三个关键语义（避免误用）：
1. **completed 判据是"执行完成"而非"业务成功"**：worker 未抛异常即 `completed`，
   业务成败（如 `need_more_info` 追问、收入不入账拦截）由 `result` 字段承载。
   否则"正常的业务拒绝"会被标 failed，恢复时触发误告警。
2. **DB 是队列事实的唯一来源**（`04` §3.1 补注）：`recover_orphans()` 只负责把状态改对，
   `submitted` 即"声明可重投"，再由 `resubmit_submitted()` 重发 `send_task`——
   队列是 DB 状态的下游投影，重启即失，故不直接改内存队列。
3. **对账只针对有副作用的任务**（`bill_agent`）：`stat`/`price`/`finance` 无落库副作用，
   崩溃后直接重投即可，无需对账。
"""
import asyncio
import json
import time
from typing import Any, Callable

from config.config import TASK_STATE_DB, REQUEUE_STALE_AFTER_SEC
from utils.common import DatabaseCRUD, logger

# 有副作用（会写库）的任务：崩溃后**禁止直接重投**，必须先按 task_id 对账（`04` §3.3）
SIDE_EFFECT_AGENTS = {"bill_agent"}
# 重投上限（与 `04` §2.4 / `08` §2.3 一致）：超过则标记 failed，需人工介入
MAX_RETRY = 3

# 终态：不再参与恢复扫描
TERMINAL_STATUS = ("completed", "failed")


class TaskManager:
    """任务状态机（独立库 `database/task_state.db`）。

    所有写操作为同步 SQLite（单次 ~1ms，与既有 `db.` 调用一致，不引入事件循环阻塞风险）。
    """

    def __init__(self, db_path=TASK_STATE_DB):
        # 独立库：Durable 每次状态流转一次写，与 bill.db 同库会争抢写锁（`08` §6 Q4）
        self._db = DatabaseCRUD(db_path)
        # ★自包含建表（幂等）：不依赖"启动时先跑过 init_all_tables"的调用顺序，
        #   否则新建库 / 独立测试环境会 `no such table`。DDL 单一来源见 `database/init_db.py`。
        from database.init_db import ensure_task_state_tables  # noqa: PLC0415 —— 延迟导入避免启动期循环

        ensure_task_state_tables(self._db)

    # ─────────── 写入侧：状态流转 ───────────
    def create_task(self, task_id: str, session_id: str, agent_id: str,
                    payload: dict | None = None, retriable: int | None = None) -> None:
        """派发前建单（`submitted`）。

        :param retriable: 是否允许自动重投；None = 按 `agent_id` 推断（写操作 → 0）。
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

    def mark_running(self, task_id: str) -> None:
        self._update_status(task_id, "running")

    def mark_completed(self, task_id: str, result: Any = None) -> None:
        self._update_status(task_id, "completed", result=result)

    def mark_failed(self, task_id: str, result: Any = None) -> None:
        self._update_status(task_id, "failed", result=result)

    def mark_interrupted(self, task_id: str) -> None:
        """优雅关闭：worker 被 cancel（`08` §6 Q7）——区别于 failed，重启后可重投"""
        self._update_status(task_id, "interrupted")

    def _update_status(self, task_id: str, status: str, result: Any = None) -> None:
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
        rows = self._db.query_sql("SELECT * FROM task_state WHERE task_id=?", (task_id,))
        return rows[0] if rows else None

    def count_by_status(self, status: str) -> int:
        rows = self._db.query_sql("SELECT COUNT(1) AS c FROM task_state WHERE status=?", (status,))
        return int(rows[0]["c"]) if rows else 0

    # ─────────── worker 执行包装 ───────────
    async def track(self, msg: dict, runner: Callable) -> Any:
        """worker 侧统一包装：running → 执行 → completed / failed / interrupted。

        :param msg: A2A 消息（含 task_id / session_id / task_data）
        :param runner: 无参协程（真正执行子 Agent 图）
        """
        task_id = msg["task_id"]
        self.mark_running(task_id)
        try:
            res = await runner()
        except asyncio.CancelledError:
            # 优雅关闭信号（Python 3.8+ 为 BaseException，故必须显式捕获）→ interrupted，
            # 不能标 failed：否则重启后"账已记"却因 retriable=0 被判 failed（`04` §3.1）
            self.mark_interrupted(task_id)
            raise
        except Exception as e:
            self.mark_failed(task_id, {"error": f"{type(e).__name__}: {e}"})
            raise
        self.mark_completed(task_id, res)
        return res

    # ─────────── 崩溃恢复 ───────────
    def recover_orphans(self, bill_db=None) -> dict:
        """启动时扫描上次进程遗留的 `running` / `interrupted` 任务并定状态（`04` §3.1 修订版）。

        对账规则（仅对有副作用的 `bill_agent`）：
          - `bill` 表存在 `task_id` 记录 → 副作用**已发生** → `completed`（不是 failed！）
          - 无记录 → 确实没执行 → 按 `retriable` 决定重投或 failed

        :param bill_db: 业务库访问对象（默认 `utils.common.db` 单例；测试可注入）
        :return: 各类处理计数，便于启动日志与测试断言
        """
        if bill_db is None:
            from utils.common import db as bill_db  # noqa: PLC0415 —— 延迟导入避免循环依赖
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
        try:
            rows = bill_db.query_sql("SELECT 1 FROM bill WHERE task_id=?", (task_id,))
        except Exception as e:  # pragma: no cover —— 对账查询失败按"未落库"处理（保守：不谎报成功）
            logger.error(f"[M5] 对账查询失败（按未落库处理）: {e}")
            return False
        return bool(rows)

    def resubmit_submitted(self, send_fn: Callable) -> int:
        """把 DB 中 `status='submitted'` 的任务重投到队列（`04` §3.1 补注：DB 是唯一事实源）。

        ★**只在启动路径调用一次**：此时队列刚 `clear_all()`，`submitted` 全部是遗留任务。
        运行期重复调用会把"正在排队等待领取"的新任务重复投递（worker 无感知 → 重复执行）。

        【2026-09-05 演示健壮性】超过 `REQUEUE_STALE_AFTER_SEC`（默认 1800s）的 submitted
        视为测试/演示残留：崩溃前正在排队的任务在队列里只停留毫秒级，能躺 >30 分钟只可能是
        历史脏数据。若盲目重投，每次演示启动都会自动执行一批旧任务（真调 LLM / 真写账）。
        此类任务直接标记 failed，不再重投。
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
        self._db.close()


# 进程级单例（懒加载）：orchestrator 建单与 bootstrap 恢复/重投必须共用同一实例。
# ★懒加载而非模块级实例化：`agents/*` 的导入链会经 `startup` 反向引入，
#   模块级建连接会在 import 期产生副作用；懒加载把连接推迟到真正使用时。
_MANAGER: "TaskManager | None" = None


def get_task_manager() -> "TaskManager":
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TaskManager()
    return _MANAGER


__all__ = ["TaskManager", "get_task_manager", "SIDE_EFFECT_AGENTS", "MAX_RETRY"]
