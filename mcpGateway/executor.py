# -*- coding: utf-8 -*-
"""ExecutionEngine：把散落各处的治理逻辑收拢成一条固定的 9 步流水线。

设计依据：`设计/08_架构演进方案与决策记录.md` §1.7。

现状对照（这些能力原本要么没有、要么散落在调用点手写）：

| 能力 | 原本状态 | 收拢后 |
|---|---|---|
| 鉴权 | `client.py` 按 SQL 前缀粗判 | 统一按 `ToolSpec.required_perm` 静态判定 |
| 参数校验 | skills.py 手写 if | JSON Schema 统一校验 |
| 超时 | 仅握手超时，调用无超时 | 统一 `asyncio.wait_for(spec.timeout)` |
| 重试 | 硬编码 2 次，仅连接异常 | 按 kind + **副作用** 决策 |
| 审计 | 各调用点手写，易漏 | 引擎内自动，不可能漏 |
| 可观测 | 无 | 引擎内打点（D3 天然挂载点） |
"""
import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable

from jsonschema import Draft7Validator

from mcpGateway.audit_recorder import write_audit_log
from mcpGateway.rbac_config import get_agent_permission
from mcpGateway.tool_model import (
    CONNECTION_EXC, InvokeContext, ToolError, ToolErrorKind, ToolResult, ToolSpec,
)

if TYPE_CHECKING:  # pragma: no cover
    from mcpGateway.registry import ToolRegistry


def _validate_params(schema: dict, args: dict) -> str | None:
    """
    函数功能与逻辑描述：
        用 Draft7Validator 按工具契约的 params_schema 校验入参，是引擎流水线 ③ 步的
        实现；刻意只取「按错误路径字典序排序后的第一条」错误上报——JSON Schema 的全量
        错误列表对调用方与 LLM 都无法消化，只回传最靠前的一条更利于定位。
        纯函数、无副作用、不访问数据库；校验失败不抛异常，而是把错误文本回传给调用方决定包装方式。
    入参说明：
        schema (dict)：JSON Schema，取自 `ToolSpec.params_schema`。
        args (dict)：本次调用的待校验入参。
    返回值说明：
        str | None：校验不通过时返回错误文本——有路径时为 "path: message"（路径按字面量点分拼接），
            字段级错误无路径时退化为 message 本身；全部通过返回 None。
    """
    errors = sorted(Draft7Validator(schema).iter_errors(args), key=lambda e: list(e.path))
    if not errors:
        return None
    err = errors[0]
    path = ".".join(str(p) for p in err.path)
    return f"{path}: {err.message}" if path else err.message


class ExecutionEngine:
    """
    函数/类功能与逻辑描述：
        工具调用的统一执行引擎，把原本散落在各调用点手写的治理逻辑收拢成一条固定流水线：
        ①解析 ②鉴权 ③参数校验 ④前置审计 ⑤执行 ⑥重试 ⑦结果 ⑧后置审计 ⑨可观测打点。
        协议差异（mcp / local / http）全部由 `registry.get_adapter(protocol)` 承担，
        引擎内不存在任何"MCP 特判"，这是工具接入平台成立的前提。
    构造入参说明：
        registry (ToolRegistry)：工具注册表，提供 get / get_adapter，必填。
        audit (Callable | None)：审计落盘函数，签名 (agent_id, name, payload, ok)，
            默认 None → 回退 `write_audit_log`。
        tracer (Any | None)：可观测打点对象，需提供 .span(...)；默认 None →
            引擎内回退 `utils.tracer.record_tool_span`。
    返回值说明：
        构造返回 ExecutionEngine 实例，无副作用；调用入口见 `invoke`。
    """

    def __init__(self, registry: "ToolRegistry",
                 audit: Callable | None = None,
                 tracer: Any | None = None):
        """
        函数功能与逻辑描述：
            装配三项依赖并挂到实例属性：工具注册表（必填）、审计函数、打点对象；
            审计与打点均为「可替换依赖」，便于单测注入假实现。不做任何 I/O，不抛异常。
        入参说明：
            registry (ToolRegistry)：工具注册表，必填。
            audit (Callable | None)：审计函数，默认 None → 用 `write_audit_log`。
            tracer (Any | None)：打点对象，默认 None → 走内置 `record_tool_span`。
        返回值说明：
            无（仅初始化 self._reg / self._audit / self._tracer）。
        """
        self._reg = registry
        self._audit = audit or write_audit_log   # 默认接现有审计
        self._tracer = tracer                    # D3（M7）挂载点，可为 None

    async def invoke(self, name: str, args: dict, *, agent_id: str,
                     trace_id: str | None = None,
                     session_id: str | None = None,
                     task_id: str | None = None) -> ToolResult:
        """
        函数功能与逻辑描述：
            引擎唯一对外入口，按 9 步流水线执行一次工具调用：①按名查契约，未注册即返回
            NOT_FOUND；②按 `ToolSpec.required_perm` 做静态 RBAC 判定，权限不足返回 PERMISSION；
            ③JSON Schema 校验参数，失败返回 VALIDATION；④仅「auditable 且有副作用」的工具
            写前置审计（读操作不写，避免刷屏）；⑤⑥交 `_execute` 执行与重试；⑦汇总结果；
            ⑧`auditable` 为真即写后置审计（含读操作，全量口径）；⑨打点 span。
            边界与异常处理：全流程以 `ToolResult` 表达失败而**不向上抛异常**；trace_id 缺省时
            尝试从 `utils.tracer.current_run_id()` 兜底（外部独立直调且无 run_scope 时得到 None）；
            打点（含兜底 tracer）异常一律吞掉，绝不阻断主链路。
        入参说明：
            name (str)：工具名（点分命名，如 "sql.read"）。
            args (dict)：工具入参，须符合该工具的 params_schema。
            agent_id (str)：调用方 Agent 标识，关键字必填，用于鉴权与审计归属。
            trace_id (str | None)：链路追踪 ID，默认 None（自动兜底取当前 run_id）。
            session_id (str | None)：会话标识，默认 None。
        返回值说明：
            ToolResult：成功时 ok=True、data 为适配器返回结果，并回填 attempts / cost_ms / trace_id；
                失败时 ok=False、error 为分类后的 `ToolError`（NOT_FOUND / PERMISSION / VALIDATION /
                TIMEOUT / CONNECTION / REMOTE / INTERNAL）。
        """
        # ── ① 解析 ──
        spec: ToolSpec | None = self._reg.get(name)
        if spec is None:
            return ToolResult(ok=False, error=ToolError(
                ToolErrorKind.NOT_FOUND, f"工具不存在: {name}", tool_name=name))

        # ── ② 鉴权 ──
        perms = set(get_agent_permission(agent_id))
        if spec.required_perm and spec.required_perm not in perms:
            return ToolResult(ok=False, error=ToolError(
                ToolErrorKind.PERMISSION,
                f"Agent[{agent_id}] 无 {spec.required_perm.value} 权限",
                tool_name=name))

        # ── ③ 参数校验（JSON Schema）──
        err_msg = _validate_params(spec.params_schema, args)
        if err_msg:
            return ToolResult(ok=False, error=ToolError(
                ToolErrorKind.VALIDATION, f"参数校验失败: {err_msg}", tool_name=name))

        if not trace_id:
            # M7（D3）：引擎兜底把工具调用归入当前 run 上下文（registry 独立直调且无
            # run_scope 时得到 None → record_tool_span 静默跳过，不产生噪音）
            try:
                from utils.tracer import current_run_id
                trace_id = current_run_id()
            except Exception:  # noqa: BLE001
                pass
        ctx = InvokeContext(agent_id=agent_id, trace_id=trace_id, session_id=session_id,
                            task_id=task_id)
        t0 = time.perf_counter()

        # ── ④ 前置审计（仅副作用工具，避免读操作刷屏）──
        if spec.auditable and spec.side_effect:
            self._audit(agent_id, name, str(args), False)

        # ── ⑤⑥ 执行 + 重试 ──
        result = await self._execute(spec, args, ctx, t0)

        # ── ⑧ 后置审计（★全量：读操作也审，与 M1 前 client.py 内联审计行为一致，
        #   见 client.py `call_bill_sql` / `call_llm_finance` 的「回滚参考」）──
        if spec.auditable:
            self._audit(agent_id, name, str(args), result.ok)

        # ── ⑨ 可观测打点（D3 接入点③：M7 起默认落 trace_span 表；打点失败不阻断
        #   主链路，铁律见 `03` §9.3.1）──
        if self._tracer:
            # 显式注入的自定义 tracer 优先（单测/未来替换），默认走内置实现
            self._tracer.span(tool=name, agent=agent_id, ok=result.ok,
                              cost_ms=result.cost_ms, trace_id=trace_id)
        else:
            try:
                from utils.tracer import record_tool_span
                record_tool_span(
                    tool_name=name, agent_id=agent_id,
                    ok=result.ok, duration_ms=int(result.cost_ms),
                    error=None if result.ok else str(result.error),
                    run_id=trace_id,
                )
            except Exception:  # noqa: BLE001
                pass
        return result

    async def _execute(self, spec: ToolSpec, args: dict, ctx: InvokeContext,
                       t0: float) -> ToolResult:
        """
        函数功能与逻辑描述：
            引擎层的重试与超时核心循环，负责把「一次工具调用」放大为「一次受控调用」：
            ① 取协议适配器（protocol 未注册会抛 ToolError，不重试）；
            ② 最多执行 spec.max_retry + 1 次，每次调用统一包在 asyncio.wait_for 里做超时护栏——
            ★timeout <= 0 表示**显式关闭超时**（M4 要求超时阈值可配置关闭），
            否则把阈值配成 0 时 wait_for(timeout=0) 会立即抛超时；
            ③ 异常分流为四类并统一转成 ToolError：TimeoutError → TIMEOUT、
            ToolError → 原样保留、CONNECTION_EXC → CONNECTION、其余 → INTERNAL（不重试）；
            ④ 重试决策只看「是否允许重试」与「是否已达上限」：**有副作用（side_effect）
            的工具永不重试**（D2 防重复记账，机制层面而非业务层面）；
            已达上限时不再退避等待（否则最后一次失败会白等退避时长，M4 实测过该问题）；
            ⑤ 重试间隔为指数退避 min(2^attempt, 5)，即 2/4/5/5...
            异常顺序敏感：Python 3.11+ 的 asyncio.TimeoutError 是 OSError 子类，
            必须排在 CONNECTION_EXC（含 OSError）之前，否则超时会被误判为连接故障。
            本函数不抛异常，失败一律收敛为 ok=False 的 ToolResult。
        入参说明：
            spec (ToolSpec)：工具元数据，决定协议、超时阈值、重试上限与副作用标记。
            args (dict)：业务参数（已通过 params_schema 校验）。
            ctx (InvokeContext)：调用上下文（agent_id / trace_id / session_id）。
            t0 (float)：调用起始时间戳（time.perf_counter 口径），用于回填 cost_ms。
        返回值说明：
            ToolResult：成功时为 ok=True 且 data 为工具原始结果、attempts 为实际尝试次数；
                失败时为 ok=False 且 error 为最后一次的 ToolError；
                两者的 cost_ms 与 trace_id 均已回填。
        """
        adapter = self._reg.get_adapter(spec.protocol)
        attempts = 0
        last_err: ToolError | None = None

        for attempt in range(spec.max_retry + 1):
            attempts = attempt + 1
            try:
                coro = adapter.invoke(spec, args, ctx)
                # ⑤ 统一超时（★M4：timeout<=0 = 关闭超时——M4 回滚要求"超时阈值可配置关闭"，
                #   否则把阈值配成 0 时 wait_for(timeout=0) 会立即抛超时）
                if spec.timeout and spec.timeout > 0:
                    out = await asyncio.wait_for(coro, timeout=spec.timeout)
                else:
                    out = await coro
                return ToolResult(ok=True, data=out, attempts=attempts,
                                  cost_ms=(time.perf_counter() - t0) * 1000,
                                  trace_id=ctx.trace_id)
            # ⚠️ 顺序敏感：Python 3.11+ 的 TimeoutError 是 OSError 子类，
            #    必须排在 CONNECTION_EXC（含 OSError）之前，否则超时会被误判为连接故障。
            except asyncio.TimeoutError:
                last_err = ToolError(ToolErrorKind.TIMEOUT,
                                     f"工具[{spec.name}] 超时 {spec.timeout}s",
                                     tool_name=spec.name,
                                     retriable=not spec.side_effect)
            except ToolError as e:
                last_err = e
                # ★重试安全性由「副作用」决定，而非「幂等性」——见 `08` §1.7 修正说明
                last_err.retriable = e.retriable and not spec.side_effect
            except CONNECTION_EXC as e:
                last_err = ToolError(ToolErrorKind.CONNECTION, str(e), origin=e,
                                     tool_name=spec.name,
                                     retriable=not spec.side_effect)
            except Exception as e:                     # 兜底：内部错误，不重试
                last_err = ToolError(ToolErrorKind.INTERNAL, str(e), origin=e,
                                     tool_name=spec.name)

            # ★重试决策：有副作用 → 永不重试（D2 防重复记账，机制层面而非业务层面）；
            #   ★已达 max_retry 上限也不 sleep——否则最后一次失败会**白等**退避时长
            #   （M4 实测：max_retry=0 的读工具超时后仍 sleep 1s，超时护栏被拖慢）。
            if not last_err.retriable or attempt >= spec.max_retry:
                break
            await asyncio.sleep(min(2 ** attempt, 5))  # 指数退避 1,2,4,5,5...

        return ToolResult(ok=False, error=last_err, attempts=attempts,
                          cost_ms=(time.perf_counter() - t0) * 1000,
                          trace_id=ctx.trace_id)


__all__ = ["ExecutionEngine", "_validate_params"]
