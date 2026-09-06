# -*- coding: utf-8 -*-
"""ExecutionEngine：把散落各处的治理逻辑收拢成一条固定的 9 步流水线。

设计依据：`设计/08_架构演进方案与决策记录.md` §1.7。

现状对照（这些能力原本要么没有、要么散落在调用点手写）：

| 能力 | 现状 | 收拢后 |
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
    """JSON Schema 参数校验。返回错误文本，或 None 表示通过。"""
    errors = sorted(Draft7Validator(schema).iter_errors(args), key=lambda e: list(e.path))
    if not errors:
        return None
    err = errors[0]
    path = ".".join(str(p) for p in err.path)
    return f"{path}: {err.message}" if path else err.message


class ExecutionEngine:
    """固定 9 步流水线：①解析 ②鉴权 ③参数校验 ④前置审计 ⑤执行 ⑥重试
    ⑦结果 ⑧后置审计 ⑨可观测打点。"""

    def __init__(self, registry: "ToolRegistry",
                 audit: Callable | None = None,
                 tracer: Any | None = None):
        self._reg = registry
        self._audit = audit or write_audit_log   # 默认接现有审计
        self._tracer = tracer                    # D3（M7）挂载点，可为 None

    async def invoke(self, name: str, args: dict, *, agent_id: str,
                     trace_id: str | None = None,
                     session_id: str | None = None) -> ToolResult:
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
        ctx = InvokeContext(agent_id=agent_id, trace_id=trace_id, session_id=session_id)
        t0 = time.perf_counter()

        # ── ④ 前置审计（仅副作用工具，避免读操作刷屏）──
        if spec.auditable and spec.side_effect:
            self._audit(agent_id, name, str(args), False)

        # ── ⑤⑥ 执行 + 重试 ──
        result = await self._execute(spec, args, ctx, t0)

        # ── ⑧ 后置审计（★全量：读操作也审，与现状 client.py L115 行为一致）──
        if spec.auditable:
            self._audit(agent_id, name, str(args), result.ok)

        # ── ⑨ 可观测打点（D3 接入点③：M7 起默认落 trace_span 表；失败不阻断 03:534）──
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
