# -*- coding: utf-8 -*-
"""工具统一模型：元数据 / 入参 / 返回 / 错误 —— 四件套。

设计依据：`设计/08_架构演进方案与决策记录.md` §1.4（D1 工具接入平台 · M1）。

★硬规则：任何一层都不允许出现"MCP 特判"或"本地特判"——
  统一模型不存在，工具接入平台就不成立。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

# 复用现有权限枚举（不重复造），并作为统一出口供上层一处导入
from mcpGateway.rbac_config import DBPerm, LLMPerm

# 连接类异常集合：平台层判定"是否可重连"的唯一依据。
# 2026-08-30：由 client.py 上移到统一模型层，供 adapters / executor / client 共用——
# 若留在 client.py，会形成导入环 client → registry → adapters → client。
# ★M3：加入 httpx.TransportError —— Streamable HTTP 常驻服务的连接失败 / 网络超时
#   均为此类（`03` §9.8.2），否则 `_call_server` 不识别为可重连、executor 会误判为内部错误。
CONNECTION_EXC = (IOError, ValueError, OSError, httpx.TransportError)


# ════════ ① 元数据：一个工具"是什么" ════════
@dataclass(frozen=True)
class ToolSpec:
    name: str                        # 全局唯一，点分命名："sql.read" / "bill.parse"
    description: str                 # ★给 LLM 看的，直接决定它选不选这个工具
    params_schema: dict              # JSON Schema（同时驱动"参数校验"与"LLM 可见性"）
    version: str = "1.0"
    result_schema: dict | None = None          # 可选：返回结构声明
    required_perm: Enum | None = None          # DBPerm.BILL_READ / LLMPerm.LLM_FINANCE / None
    side_effect: bool = False        # 写副作用 → 前置审计 + ★永不自动重试（D2 防重复记账）
    auditable: bool = True           # 是否记审计；llm.chat 设 False 以对齐现状（现状不审 LLM）
    idempotent: bool = True          # 结果是否确定（语义标注，★不参与重试闸门判定）
    timeout: float = 120.0           # 单次调用超时（秒）
    max_retry: int = 0               # 引擎层自动重试上限
    protocol: str = "local"          # "mcp" / "local" / "http"（需已注册对应适配器）
    binding: dict = field(default_factory=dict)
    #   mcp:   {"server_key": "sql_bill", "tool": "exec_sql", "inject_agent_id": True}
    #   local: {"func": <callable>, "to_thread": True}
    tags: tuple[str, ...] = ()       # 能力标签，发现层可按标签过滤
    hidden: bool = False             # True → 不出现在 list_tools（内部工具，不暴露给 LLM）

    def to_llm_schema(self) -> dict:
        """导出为 OpenAI function-calling 的单工具描述"""
        return {"name": self.name, "description": self.description,
                "parameters": self.params_schema}


# ════════ ② 入参：一次调用"带什么" ════════
@dataclass(frozen=True)
class InvokeContext:
    """引擎注入的上下文。

    ★不进 params_schema —— agent_id/trace_id 是"谁在调用"而非"调用什么"，
    混在一起会污染工具契约，且 agent_id 一旦暴露给 LLM 就有被伪造提权的风险
    （详见 `08` §1.10「agent_id 归属」）。
    """
    agent_id: str
    trace_id: str | None = None
    session_id: str | None = None


# ════════ ③ 返回：一次调用"得什么" ════════
@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: "ToolError | None" = None
    attempts: int = 1                # 实际尝试次数（含重试）
    cost_ms: float = 0.0
    trace_id: str | None = None

    def unwrap(self) -> Any:
        """★兼容旧语义：成功返回 data；失败抛**调用方熟知的原生异常类型**。

        ⚠️ 为什么必须做类型映射（2026-08-30 动工时发现的兼容性陷阱）：
        现状 `call_bill_sql` / `call_llm_finance` 无权限时抛 **`PermissionError`**，
        且 `tests/integration/test_mcp_gateway.py` 有 2 处断言
        `pytest.raises(PermissionError)`。若失败一律抛 `ToolError`，
        这层契约就断了 —— 与 M1 验收「~40 处测试零改动」直接冲突。
        """
        if self.ok:
            return self.data
        err = self.error or ToolError(ToolErrorKind.INTERNAL, "未知工具错误")
        raise _as_native(err)


# ════════ ④ 错误：平台层统一分类 ════════
class ToolErrorKind(Enum):
    NOT_FOUND = "not_found"          # 工具不存在
    PERMISSION = "permission"        # 无权限（RBAC 拒绝）
    VALIDATION = "validation"        # 参数不符合 params_schema
    TIMEOUT = "timeout"              # 超过 spec.timeout
    CONNECTION = "connection"        # 连接类故障
    REMOTE = "remote"                # 远端业务错误（MCP server 明确返回错误）
    INTERNAL = "internal"            # 适配器 / 引擎内部错误


class ToolError(Exception):
    """平台层统一错误。

    ⚠️ 刻意**不用** `@dataclass(frozen=True)`：dataclass 生成的 `__init__` 不会调用
    `super().__init__(message)`，导致 `Exception.args` 为空 → `str(e)` 返回空字符串，
    错误信息在日志与异常冒泡中**全部丢失**。此处手写 `__init__` 以保证 `str(e)` 可用。
    """

    def __init__(self, kind: ToolErrorKind, message: str,
                 origin: Exception | None = None,
                 tool_name: str | None = None,
                 retriable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.origin = origin
        self.tool_name = tool_name
        self.retriable = retriable

    def __str__(self) -> str:
        return f"[{self.kind.value}] {self.message}"


# 平台错误 → 原生异常类型映射（供 ToolResult.unwrap 保持旧调用契约）
_NATIVE_EXC_MAP = {
    ToolErrorKind.PERMISSION: PermissionError,
    ToolErrorKind.VALIDATION: ValueError,
    ToolErrorKind.TIMEOUT: TimeoutError,
    ToolErrorKind.CONNECTION: ConnectionError,
    ToolErrorKind.NOT_FOUND: ValueError,
    ToolErrorKind.REMOTE: RuntimeError,
    ToolErrorKind.INTERNAL: RuntimeError,
}


def _as_native(err: "ToolError") -> Exception:
    """把平台错误映射为原生异常类型，保留 kind 语义与底层原始异常上下文。"""
    # 远端业务错误：直接抛底层原始异常，保留真实报错上下文（对齐现状 _call_server 的 raise e）
    if err.kind is ToolErrorKind.REMOTE and err.origin is not None:
        return err.origin
    cls = _NATIVE_EXC_MAP.get(err.kind, ToolError)
    if cls is ToolError:
        return err
    exc = cls(str(err))
    if err.origin is not None:
        exc.__cause__ = err.origin
    return exc


__all__ = ["DBPerm", "LLMPerm", "CONNECTION_EXC", "ToolSpec", "InvokeContext",
           "ToolResult", "ToolErrorKind", "ToolError"]
