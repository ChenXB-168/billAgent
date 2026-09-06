# -*- coding: utf-8 -*-
"""
D3 可观测：LLM 调用 token/耗时归因 + run_id 贯穿的 span 树

设计依据：
  - `设计/02_现状盘点与缺陷分析.md:302-308`  D3 缺陷现状：无 run_id / 无 span 树 /
                                            无耗时分段 / **无 token 成本归因**
  - `设计/02_现状盘点与缺陷分析.md:69`       D3 属 **P0 架构级**缺陷
  - `设计/03_逻辑架构.md:363`                D3 落点 = Tracer（utils/tracer.py），➕ 新增模块
  - `设计/03_逻辑架构.md:517`                职责：run_id 贯穿的 span 树：自动计时 +
                                            失败归因 + token/耗时统计
  - `设计/03_逻辑架构.md:534`                **铁律**：可观测属横切能力，失败**不阻断**主链路
  - `设计/03_逻辑架构.md:546`                接入点仅 3 处：编排入口 / worker 执行 / Registry.invoke
  - `设计/08_架构演进方案与决策记录.md`      M7（D3）完成记录见 08 §5.2 / 11 §3 M7

阶段划分：
  【阶段1 · M1 前置】 LLM 调用埋点（两处模型入口采集 token/耗时/成败 → 落 llm_call_stat 表）
                      + run_id / span 的进程内上下文（contextvars），为阶段2 预留
  【阶段2 · M7 完成】 ① 编排入口 `orchestrator.run` root span（chat_loop / webUI respond）
                      ② worker 执行 `{agent}.run` span（startup/bootstrap 4 worker）
                      ③ 工具层 `tool.{name}` span（mcpGateway/executor.py ⑨ 打点）
                      + `trace_span` 表持久化 span 树（database/init_db.ensure_trace_tables）
                      + run_id 按 `r{ms}{hex3}` 规则生成（08 §3.2，与 task_id 同构）
                      + 查看脚本 `python utils/trace_view.py <run_id>`（08 §3.4 Q10）

★跨进程说明：LLM 调用发生在 MCP 子进程（mcpGateway/server_llm_base.py / server_llm_finance.py）。
  M7 已打通 run_id 跨进程透传：主进程经 mcpGateway/adapters.py 按 `binding["inject_run_id"]`
  把 `ctx.trace_id`（= current_run_id()）作为隐藏参数注入 llm 工具请求；server 侧工具函数
  签名增加可选 `run_id`，函数内 `bind_run_id(run_id)` 写入**本进程** contextvars，
  随后 `record_llm_call` 即可带上 run_id 落库（子进程不回传新 id）。
  本地 spec 刻意不声明 run_id（同 agent_id 防注入先例），`_schema_diff` 将其计入预期差异豁免集。

★本模块所有写库/查询**永不抛异常**（`03`:534 铁律：可观测失败不阻断主链路）。
"""
from __future__ import annotations

import secrets
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

from config.config import TRACER_ENABLED

# ────────────────────────── 进程内上下文（contextvars） ──────────────────────────
# contextvars 对 asyncio / 线程池均安全，父子协程与 to_thread 任务都能正确继承。
_RUN_ID: ContextVar[Optional[str]] = ContextVar("billagent_run_id", default=None)
_SESSION_ID: ContextVar[Optional[str]] = ContextVar("billagent_session_id", default=None)
_SPAN_STACK: ContextVar[tuple] = ContextVar("billagent_span_stack", default=())

_INSERT_SQL = """
INSERT INTO llm_call_stat (
    run_id, span_id, parent_span_id, session_id, agent_tag, model, channel,
    prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
    latency_ms, success, error_type
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

_INSERT_SPAN_SQL = """
INSERT INTO trace_span (
    run_id, span_id, parent_span_id, name, kind, agent_id, status, duration_ms, error
) VALUES (?,?,?,?,?,?,?,?,?)
"""

# trace_span 表惰性建表标记（GIL 保护足够：可观测层不追求严格竞态语义）
_SPAN_TABLE_READY = False


def new_run_id() -> str:
    """生成一次端到端请求的 run_id。

    M7 起与 task_id 同构（`08` §3.2 / `04` §3.2）：`r{ms}{hex3}`，跨 run / 跨 task 时序可辨。
    """
    return f"r{int(time.time() * 1000)}{secrets.token_hex(3)}"


def new_span_id() -> str:
    return uuid.uuid4().hex[:8]


def current_run_id() -> Optional[str]:
    return _RUN_ID.get()


def current_span_id() -> Optional[str]:
    stack = _SPAN_STACK.get()
    return stack[-1] if stack else None


def current_parent_span_id() -> Optional[str]:
    stack = _SPAN_STACK.get()
    return stack[-2] if len(stack) >= 2 else None


@contextmanager
def run_scope(
    session_id: Optional[str] = None, run_id: Optional[str] = None
) -> Iterator[str]:
    """把一次请求的 run_id / session_id 绑定到当前上下文。

    - 编排入口（chat_loop / webUI respond）：不传 run_id → 生成新 run_id。
    - worker 执行（bootstrap）：传 `run_id`（来自 task_data）恢复主进程链——contextvars
      不跨独立协程传播，worker 必须显式重建，见 `11` §3 M7。
    """
    rid = run_id or new_run_id()
    token_run = _RUN_ID.set(rid)
    token_sess = _SESSION_ID.set(session_id)
    try:
        yield rid
    finally:
        _RUN_ID.reset(token_run)
        _SESSION_ID.reset(token_sess)


def bind_run_id(run_id: Optional[str], session_id: Optional[str] = None) -> None:
    """把外部传入的 run_id 写入**当前进程**上下文（MCP server 子进程透传用）。

    不 reset：server 工具函数处于本请求独享的执行上下文（FastMCP/线程池任务），
    请求结束即被丢弃，无需清理。
    """
    if run_id:
        _RUN_ID.set(run_id)
    if session_id is not None:
        _SESSION_ID.set(session_id)


@contextmanager
def span(
    name: str,
    *,
    kind: Optional[str] = None,
    agent_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Iterator[str]:
    """开启一个 span；结束时自动计时并落 `trace_span` 表（阶段2 持久化 span 树）。

    kind：orchestrator（编排入口 root）/ agent（worker 子图执行）/ tool（工具调用）。
    异常 → status=error + 归因文本，随后**照常向上抛**（打点不吞业务异常）；
    root span（无父）由入口层负责，因此这里不兜底生成 run_id。

    parent_span_id：跨协程恢复用（bootstrap worker）——worker 与 orchestrator 不在
    同一协程链，span 栈不共享，故由编排侧把根 span id 注入 task_data 显式指定。
    """
    sid = new_span_id()
    rid = _RUN_ID.get()
    parent = parent_span_id or current_span_id()
    token = _SPAN_STACK.set(_SPAN_STACK.get() + (sid,))
    t0 = time.perf_counter()
    status, err = "ok", None
    try:
        yield sid
    except BaseException as e:  # noqa: BLE001 —— span 打点不改变异常传播
        status, err = "error", f"{type(e).__name__}: {e}"[:500]
        raise
    finally:
        cost_ms = int((time.perf_counter() - t0) * 1000)
        _SPAN_STACK.reset(token)
        _insert_span_row(
            rid, sid, parent, name, kind, agent_id, status, cost_ms, err
        )


def record_tool_span(
    *,
    tool_name: str,
    agent_id: Optional[str] = None,
    ok: bool = True,
    duration_ms: int = 0,
    error: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """记录一次工具调用 span（mcpGateway/executor.py ⑨ 打点 → `tool.{name}`）。

    run_id 优先取调用方显式传入（engine 的 ctx.trace_id）；无 run 上下文则静默跳过，
    避免独立/测试调用污染统计。
    """
    rid = run_id or _RUN_ID.get()
    if not rid:
        return
    parent = current_span_id()
    _insert_span_row(
        rid, new_span_id(), parent, f"tool.{tool_name}", "tool", agent_id,
        "ok" if ok else "error", int(duration_ms or 0),
        error[:500] if error else None,
    )


def _ensure_span_table() -> None:
    """惰性建 trace_span 表（幂等）。正常启动已由 init_all_tables 建好；此处兜底
    独立测试环境 / 新建库——避免埋点依赖启动顺序（同 `04` 表结构单一来源原则）。"""
    global _SPAN_TABLE_READY
    if _SPAN_TABLE_READY:
        return
    try:
        from database.init_db import ensure_trace_tables
        from utils.common import db
        ensure_trace_tables(db)
        _SPAN_TABLE_READY = True
    except Exception:  # noqa: BLE001
        pass


def _insert_span_row(
    run_id, span_id, parent_span_id, name, kind, agent_id, status, duration_ms, error,
) -> None:
    if not TRACER_ENABLED:
        return
    _ensure_span_table()
    try:
        from utils.common import db
        db.execute_sql(
            _INSERT_SPAN_SQL,
            (run_id, span_id, parent_span_id, name, kind, agent_id, status, duration_ms, error),
        )
    except Exception as e:  # noqa: BLE001 —— 可观测层必须吞掉一切异常
        try:
            from utils.common import logger
            logger.warning(f"[TRACER] span 落库失败（不阻断主链路）：{e}")
        except Exception:
            pass


def record_llm_call(
    *,
    agent_tag: str,
    model: str,
    channel: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    total_tokens: Optional[int] = None,
    latency_ms: int = 0,
    success: bool = True,
    error_type: Optional[str] = None,
) -> None:
    """记录一次 LLM 调用的 token / 耗时 / 成败。

    ★铁律：本函数**永不抛异常**（可观测失败不阻断主链路，见 `03`:534）。
    最坏情况只是统计缺失，绝不改变业务返回值。
    """
    if not TRACER_ENABLED:
        return
    try:
        p = int(prompt_tokens or 0)
        c = int(completion_tokens or 0)
        r = int(reasoning_tokens or 0)
        total = int(total_tokens) if total_tokens is not None else (p + c + r)
        # 延迟 import：避免模块导入即建立 DB 连接 / 注册日志 sink（MCP 子进程场景）
        from utils.common import db
        db.execute_sql(
            _INSERT_SQL,
            (
                current_run_id(), current_span_id(), current_parent_span_id(),
                _SESSION_ID.get(), agent_tag, model, channel,
                p, c, r, total,
                int(latency_ms or 0), 1 if success else 0, error_type,
            ),
        )
    except Exception as e:  # noqa: BLE001 —— 可观测层必须吞掉一切异常
        try:
            from utils.common import logger
            logger.warning(f"[TRACER] LLM调用统计落库失败（不阻断主链路）：{e}")
        except Exception:
            pass


# ────────────────────────── 查询 / 导出（查看脚本 trace_view 用） ──────────────────────────

def load_spans(run_id: str) -> list:
    """按 run_id 拉 span 行（id 正序 = 创建序）"""
    try:
        from utils.common import db
        return db.query_sql(
            "SELECT run_id, span_id, parent_span_id, name, kind, agent_id,"
            " status, duration_ms, error, created_at"
            " FROM trace_span WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
    except Exception:  # noqa: BLE001
        return []


def load_recent_runs(limit: int = 10) -> list:
    """列出最近有 span 记录的 run（按创建时间倒序，trace_view 无参入口用）"""
    try:
        from utils.common import db
        return db.query_sql(
            "SELECT run_id, MIN(created_at) AS started_at,"
            " SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS error_cnt,"
            " COUNT(*) AS span_cnt"
            " FROM trace_span WHERE run_id IS NOT NULL"
            " GROUP BY run_id ORDER BY MAX(id) DESC LIMIT ?",
            (limit,),
        )
    except Exception:  # noqa: BLE001
        return []


def summary_by_agent() -> list:
    """按 agent 聚合 token/耗时/失败率（运维排查与成本归因入口）"""
    sql = """
    SELECT agent_tag,
           channel,
           COUNT(*)                        AS call_cnt,
           SUM(prompt_tokens)              AS prompt_tokens,
           SUM(completion_tokens)          AS completion_tokens,
           SUM(reasoning_tokens)           AS reasoning_tokens,
           SUM(total_tokens)               AS total_tokens,
           AVG(latency_ms)                 AS avg_latency_ms,
           SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) AS fail_cnt
    FROM llm_call_stat
    GROUP BY agent_tag, channel
    ORDER BY total_tokens DESC
    """
    try:
        from utils.common import db
        return db.query_sql(sql)
    except Exception as e:  # noqa: BLE001
        try:
            from utils.common import logger
            logger.warning(f"[TRACER] 统计聚合查询失败：{e}")
        except Exception:
            pass
        return []


def summary_by_run(run_id: str) -> list:
    """查询某次请求的全链路 LLM 调用明细（按时间正序）"""
    sql = """
    SELECT agent_tag, model, channel, prompt_tokens, completion_tokens,
           reasoning_tokens, total_tokens, latency_ms, success, error_type, created_at
    FROM llm_call_stat WHERE run_id = ? ORDER BY id ASC
    """
    try:
        from utils.common import db
        return db.query_sql(sql, (run_id,))
    except Exception as e:  # noqa: BLE001
        try:
            from utils.common import logger
            logger.warning(f"[TRACER] run 明细查询失败：{e}")
        except Exception:
            pass
        return []
