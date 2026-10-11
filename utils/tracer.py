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
    """
    函数功能与逻辑描述：
        生成一次端到端请求的 run_id。M7 起与 task_id 同构（`08` §3.2 / `04` §3.2）：
        格式为 `r{毫秒时间戳}{3字节随机hex}`，前缀 r 便于与 task_id 区分类型，
        毫秒时间戳保证跨 run / 跨 task 的时序可辨，6 位随机 hex 保证同毫秒内不撞号。
        run_id 随后经 mcpGateway 的隐藏参数跨进程透传给 MCP server，用于 LLM 调用归因。
    入参说明：
        无。
    返回值说明：
        str：新生成的 run_id，形如 "r1712345678901a3f2c1"。
    """
    return f"r{int(time.time() * 1000)}{secrets.token_hex(3)}"


def new_span_id() -> str:
    """
    函数功能与逻辑描述：
        生成一个 span 的唯一标识，用作 trace_span 主键与父子关联键。
        取 uuid4 的 hex 前 8 位——单次 run 内 span 数量在个位数到几十量级，
        8 位 hex（32 bit）碰撞概率可忽略，且比完整 uuid 更利于日志阅读与手工检索。
    入参说明：
        无。
    返回值说明：
        str：8 位十六进制字符的 span 标识。
    """
    return uuid.uuid4().hex[:8]


def current_run_id() -> Optional[str]:
    """
    函数功能与逻辑描述：
        读取当前上下文中的 run_id（contextvars）。用于埋点落库时自动带上链路归属，
        以及适配器把 run_id 作为隐藏参数注入 MCP 请求（M7 跨进程透传的取值来源）。
    入参说明：
        无。
    返回值说明：
        Optional[str]：当前 run_id；不在 run_scope 内时为 None。
    """
    return _RUN_ID.get()


def current_span_id() -> Optional[str]:
    """
    函数功能与逻辑描述：
        读取当前上下文 span 栈的**栈顶** span id，即最近进入、尚未退出的 span。
        新开的子 span 会以它作为默认父节点，从而自动构成树形层级。
    入参说明：
        无。
    返回值说明：
        Optional[str]：栈顶 span id；span 栈为空（不在任何 span 内）时返回 None。
    """
    stack = _SPAN_STACK.get()
    return stack[-1] if stack else None


def current_parent_span_id() -> Optional[str]:
    """
    函数功能与逻辑描述：
        读取当前上下文 span 栈的**次栈顶**，即当前 span 的父 span id。
        用于 LLM 调用埋点写入 parent_span_id 列——记录一次模型调用时，
        它自身不属于任何 span，其归属应挂在「发起调用的那个 span」的父级关系上。
    入参说明：
        无。
    返回值说明：
        Optional[str]：父 span id；栈中不足两层（根 span 或不在任何 span 内）时返回 None。
    """
    stack = _SPAN_STACK.get()
    return stack[-2] if len(stack) >= 2 else None


@contextmanager
def run_scope(
    session_id: Optional[str] = None, run_id: Optional[str] = None
) -> Iterator[str]:
    """
    函数功能与逻辑描述：
        把一次请求的 run_id / session_id 绑定到当前上下文（contextvars），
        退出时通过 ContextVar token 精确复位，避免污染外层上下文。
        两个典型用法：
        - 编排入口（chat_loop / webUI respond）：不传 run_id → 内部调用 new_run_id() 生成新链路；
        - worker 执行（bootstrap）：传 run_id（来自 task_data）恢复主进程链路——因为
          contextvars 不跨独立协程传播，worker 必须显式重建上下文，见 `11` §3 M7。
    入参说明：
        session_id (Optional[str])：会话标识，默认 None；为 None 时同样会被写入上下文
            （表示「显式无会话」，与「未绑定」在语义上区分度有限，读取方需自行容错）。
        run_id (Optional[str])：要恢复的 run_id，默认 None 表示新建。
    返回值说明：
        Iterator[str]：上下文管理器，with 语句的绑定值为本次实际使用的 run_id
            （传参则回显该值，否则为新生成值）。
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
    """
    函数功能与逻辑描述：
        把外部传入的 run_id 写入**当前进程**上下文，供 MCP server 子进程接收主进程透传值时使用。
        与 run_scope 的关键差异：本函数**不做 reset**——原因有两层，
        ① server 工具函数处于本请求独享的执行上下文（FastMCP 的线程池任务），请求结束即被丢弃；
        ② bind 与调用发生在函数体内而非 with 块内，无法对称保存 token。
        run_id 为空（None 或空串）时跳过 run_id 写入，但仍会写 session_id（若显式传入）。
    入参说明：
        run_id (Optional[str])：主进程透传的链路标识；空值时保持上下文原状。
        session_id (Optional[str])：会话标识，默认 None；仅当**非 None** 时才覆盖写入。
    返回值说明：
        无（副作用：改写本进程 _RUN_ID / _SESSION_ID 上下文变量，且不自动复位）。
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
    """
    函数功能与逻辑描述：
        开启一个 span：进入时压栈并记录起始时间，退出时自动计算耗时、弹栈并把一行
        写入 `trace_span` 表（阶段2 持久化 span 树）。kind 取值约定：
        orchestrator（编排入口 root span）/ agent（worker 子图执行）/ tool（工具调用）。
        异常处理：捕获 BaseException 后把 status 置为 error、错误文本截断至 500 字符写库，
        随后**照常向上抛出**——打点绝不吞业务异常。注意这里不兜底生成 run_id：
        root span（无父）由入口层的 run_scope 负责建立链路，本函数只读取现状。
        parent_span_id 用于**跨协程恢复**：worker 与 orchestrator 不在同一协程链，
        span 栈不共享，因此由编排侧把根 span id 注入 task_data 显式指定父节点。
    入参说明：
        name (str)：span 名称（如 "orchestrator.run"、"bill_agent.run"）。
        kind (Optional[str])：span 类型，取值 orchestrator / agent / tool，默认 None。
        agent_id (Optional[str])：归属 Agent 标识，默认 None。
        parent_span_id (Optional[str])：显式指定的父 span id；默认 None 表示取当前 span 栈栈顶。
    返回值说明：
        Iterator[str]：上下文管理器，with 语句的绑定值为本次新建的 span id。
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
    """
    函数功能与逻辑描述：
        记录一次**工具调用** span（由 mcpGateway/executor.py 在调用结束时打点），
        span 名固定为 `tool.{tool_name}`、kind 固定为 "tool"。
        与 span 上下文管理器的区别：工具调用的耗时与成败由引擎侧已知，
        因此这里直接以一次性写入代替进入/退出配对，避免在异步边界上维护嵌套栈。
        span id 每次新生成，父节点取当前 span 栈栈顶（即发起调用的 agent span）。
        重要保护：run_id 优先取调用方显式传入（引擎的 ctx.trace_id），
        **若无任何可用 run 上下文则静默返回**，避免独立调用 / 单测场景污染链路统计。
    入参说明：
        tool_name (str)：工具名（不含 tool. 前缀），写入 span 名时会自动拼接。
        agent_id (Optional[str])：调用方 Agent 标识，默认 None。
        ok (bool)：调用是否成功，默认 True；False 时 status 记为 "error"。
        duration_ms (int)：调用耗时毫秒，默认 0；None 按 0 处理。
        error (Optional[str])：错误文本，默认 None；写入前截断至 500 字符。
        run_id (Optional[str])：显式指定的链路标识；默认 None 表示取当前上下文，
            两者皆无则直接返回、不落库。
    返回值说明：
        无（副作用：向 trace_span 表插入一行；TRACER_ENABLED 关闭时不写）。
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
    """
    函数功能与逻辑描述：
        惰性建 trace_span 表（幂等），带进程级缓存标记 _SPAN_TABLE_READY 避免重复执行 DDL。
        正常启动路径已由 database.init_db.init_all_tables 建好，此处只为兜底
        独立测试环境或新建库，避免埋点依赖启动顺序（遵循 `04` 表结构单一来源原则）。
        建表函数与 db 都在函数内延迟导入（避免 import 期建立 DB 连接），
        任一步失败都被吞掉——可观测层不得因建表失败而阻断主链路。
    入参说明：
        无。
    返回值说明：
        无（副作用：可能创建 trace_span / llm_call_stat 相关表，并置位 _SPAN_TABLE_READY；
            失败时标记保持 False，下次调用会重试）。
    """
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
    """
    函数功能与逻辑描述：
        span 行的统一落库实现（span 与 record_tool_span 共用），是 span 树的最终写入点。
        流程：先判 TRACER_ENABLED 总开关（关闭则直接返回，零开销），再确保表存在，
        然后延迟导入 db 并执行插入。
        失败处理遵循可观测层铁律：捕获一切异常并只记 warning，**绝不向上抛出**；
        连 logger 本身都再包一层 try，以防日志依赖未就绪时二次抛错。
        参数全部原样拼装进 _INSERT_SPAN_SQL，列顺序为
        (run_id, span_id, parent_span_id, name, kind, agent_id, status, duration_ms, error)。
    入参说明：
        run_id：链路标识；为 None 时该行无链路归属（load_recent_runs 会以 run_id IS NOT NULL 过滤掉）。
        span_id：本 span 唯一标识。
        parent_span_id：父 span 标识；根 span 为 None。
        name (str)：span 名称。
        kind：span 类型（orchestrator / agent / tool），可为 None。
        agent_id：归属 Agent 标识，可为 None。
        status (str)：状态，取值 "ok" / "error"。
        duration_ms (int)：耗时毫秒。
        error：错误文本（已由调用方截断），成功时为 None。
    返回值说明：
        无（副作用：向 trace_span 插入一行；开关关闭或异常时静默跳过）。
    """
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
    """
    函数功能与逻辑描述：
        记录一次 LLM 调用的 token / 耗时 / 成败，是 D3「token 成本归因」的唯一落库入口，
        由 modelService.router._record 在**每次尝试**（含重试）时调用。
        run_id / span_id / parent_span_id / session_id 全部从 contextvars 自动读取，
        因此调用方无需显式传递链路信息（跨进程场景由 bind_run_id 预先绑定）。
        总 token 的取值规则：显式传入 total_tokens 时尊重传入值，否则按
        prompt + completion + reasoning 自算（与 openai_compat.parse_usage 口径一致）。
        延迟导入 db 是为了避免模块导入即建立 DB 连接 / 注册日志 sink（MCP 子进程场景下尤其重要）。
        ★铁律：本函数**永不抛异常**（可观测失败不阻断主链路，见 `03`:534）——
        最坏情况只是统计缺失，绝不改变业务返回值；连 logger 失败也被二次兜底。
    入参说明：
        agent_tag (str)：调用方 Agent 标识（关键字参数，必填）。
        model (str)：模型名（关键字参数，必填）。
        channel (str)：通道，取值 local / external（关键字参数，必填）。
        prompt_tokens (int)：输入 token，默认 0；None 按 0 处理。
        completion_tokens (int)：输出 token，默认 0。
        reasoning_tokens (int)：推理 token，默认 0（非推理模型恒为 0）。
        total_tokens (Optional[int])：总 token，默认 None 表示按三段之和自算。
        latency_ms (int)：本次调用耗时毫秒，默认 0。
        success (bool)：是否成功，默认 True；写库时转为 1/0。
        error_type (Optional[str])：失败时的错误类型（如 Timeout / HTTP_429），默认 None。
    返回值说明：
        无（副作用：向 llm_call_stat 插入一行；TRACER_ENABLED 关闭或异常时静默跳过）。
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
    """
    函数功能与逻辑描述：
        按 run_id 拉取该链路的全部 span 行，用于 trace_view 渲染 span 树。
        排序用 `ORDER BY id ASC`（自增主键）而非 created_at——同一毫秒内可能产生多行，
        以插入顺序（= 创建序）为准才能保证父子节点的呈现顺序稳定。
        查询失败返回空列表而不抛异常（可观测层铁律）。
    入参说明：
        run_id (str)：要查询的链路标识。
    返回值说明：
        list：字典列表，每行含 run_id、span_id、parent_span_id、name、kind、agent_id、
            status、duration_ms、error、created_at；无数据或查询异常时返回 []。
    """
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
    """
    函数功能与逻辑描述：
        列出最近产生过 span 记录的 run，供 trace_view 的无参入口展示「可查看的链路清单」。
        聚合口径：以 run_id 分组，started_at 取该组最早的 created_at，
        span_cnt 为该组 span 总数，error_cnt 只统计 status='error' 的行。
        排序用 `MAX(id) DESC` 而非时间列，理由同 load_spans（避免同毫秒排序不稳定）；
        并以 `run_id IS NOT NULL` 过滤掉无链路归属的孤立 span。
        查询失败返回空列表而不抛异常。
    入参说明：
        limit (int)：返回的 run 数量上限，默认 10。
    返回值说明：
        list：字典列表，每行含 run_id、started_at、error_cnt、span_cnt（按最近创建倒序）；
            无数据或查询异常时返回 []。
    """
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
    """
    函数功能与逻辑描述：
        按 (agent_tag, channel) 维度聚合 llm_call_stat，输出调用次数、四类 token 合计、
        平均耗时与失败次数，是运维排查与成本归因的主要入口。
        排序用 total_tokens DESC，让「最烧钱的 Agent/通道」排在前面。
        注意：本函数**不传 run_id 也不限定时间范围**，统计的是全表历史累计数据。
        失败处理比其它查询更完整：先尝试记 warning，再返回空列表，全程不抛异常。
    入参说明：
        无。
    返回值说明：
        list：字典列表，每行含 agent_tag、channel、call_cnt、prompt_tokens、
            completion_tokens、reasoning_tokens、total_tokens、avg_latency_ms、fail_cnt；
            无数据或查询异常时返回 []。
    """
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
    """
    函数功能与逻辑描述：
        查询某一次请求（单个 run_id）的全链路 LLM 调用明细，按 `ORDER BY id ASC`
        即实际发生顺序返回，供 trace_view 打印「LLM 调用明细（llm_call_stat）」表。
        与 summary_by_agent 的区别：本函数是明细（每次尝试一行，含重试），
        后者是跨 run 的聚合。失败处理同样只记 warning 并返回空列表，不抛异常。
    入参说明：
        run_id (str)：要查询的链路标识。
    返回值说明：
        list：字典列表，每行含 agent_tag、model、channel、prompt_tokens、completion_tokens、
            reasoning_tokens、total_tokens、latency_ms、success、error_type、created_at；
            无数据或查询异常时返回 []。
    """
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
