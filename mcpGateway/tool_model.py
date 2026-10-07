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
    """
    函数/类功能与逻辑描述：
        工具元数据模型（不可变），是工具接入平台的单一事实来源：既描述工具"是什么"，
        又同时驱动参数校验、LLM 可见性（to_llm_schema）、权限校验与执行策略（重试 / 超时）。
        字段 design 要点：side_effect=True 的工具（写操作）永不自动重试以避免重复副作用；
        auditable=False 用于对齐「现状不审计 LLM 调用」的历史行为；idempotent 仅为语义标注，
        不参与重试闸门判定；hidden=True 的内部工具不出现在 list_tools 中、不暴露给 LLM。
    构造入参说明：
        name (str)：全局唯一工具名，点分命名如 "sql.read" / "bill.parse"。
        description (str)：给 LLM 看的工具说明，直接决定模型是否选用该工具。
        params_schema (dict)：JSON Schema，同时驱动「参数校验」与「LLM 可见性」。
        version (str)：工具版本，默认 "1.0"。
        result_schema (dict | None)：可选的返回结构声明，默认 None。
        required_perm (Enum | None)：所需权限枚举（如 DBPerm.BILL_READ / LLMPerm.LLM_FINANCE），默认 None。
        side_effect (bool)：是否写副作用，默认 False；为 True 时永不自动重试（D2 防重复记账）。
        auditable (bool)：是否记审计日志，默认 True。
        idempotent (bool)：结果是否确定（语义标注，不参与重试判定），默认 True。
        timeout (float)：单次调用超时秒数，默认 120.0。
        max_retry (int)：引擎层自动重试上限，默认 0。
        protocol (str)：协议标识，取值 "mcp" / "local" / "http"，默认 "local"，须已注册对应适配器。
        binding (dict)：协议相关绑定信息，默认空字典。mcp 形如
            {"server_key": "sql_bill", "tool": "exec_sql", "inject_agent_id": True}；
            local 形如 {"func": <callable>, "to_thread": True}。
        tags (tuple[str, ...])：能力标签，发现层可按标签过滤，默认空元组。
        hidden (bool)：是否隐藏，默认 False；为 True 时不出现在 list_tools。
    返回值说明：
        ToolSpec：不可变（frozen）实例，创建后字段不可修改，可安全跨协程共享。
    """
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
        """
        函数功能与逻辑描述：
            把工具元数据投影为 OpenAI function-calling 规范的单工具描述，
            仅导出 name / description / parameters 三项。**刻意不导出 binding、
            required_perm、side_effect 等执行侧字段**——它们属于平台内部实现，
            暴露给模型既无用又会扩大伪造面。
        入参说明：
            无（隐式 self）。
        返回值说明：
            dict：{"name": str, "description": str, "parameters": dict}，
                parameters 即 params_schema 原样透传（不做拷贝）。
        """
        return {"name": self.name, "description": self.description,
                "parameters": self.params_schema}


# ════════ ② 入参：一次调用"带什么" ════════
@dataclass(frozen=True)
class InvokeContext:
    """
    函数/类功能与逻辑描述：
        引擎注入的调用上下文（不可变），承载「谁在调用」而非「调用什么」。
        ★设计红线：这些字段不进 params_schema —— agent_id/trace_id 属于调用方身份，
        与业务参数混在一起会污染工具契约；且 agent_id 一旦暴露给 LLM 就有被伪造提权的风险
        （详见 `08` §1.10「agent_id 归属」）。写入 payload 的动作只在适配器层完成。
    构造入参说明：
        agent_id (str)：调用方 Agent 标识，强制必填，供远端二次鉴权与审计归属。
        trace_id (str | None)：链路追踪 ID，用于跨进程归因 LLM 调用，默认 None。
        session_id (str | None)：会话标识，供会话隔离场景使用，默认 None。
    返回值说明：
        InvokeContext：不可变实例，创建后不可修改。
    """
    agent_id: str
    trace_id: str | None = None
    session_id: str | None = None
    # ★M15：单次任务标识（子 Agent 从 A2A 消息取得的 task_id）。用途与 agent_id 同类——
    #   由引擎注入到**需要它的工具**（经 binding.inject_ctx 声明，如 bill.add 写对账字段），
    #   **不进 params_schema**（红线：防模型伪造，保住对账依据的可信度）。
    task_id: str | None = None


# ════════ ③ 返回：一次调用"得什么" ════════
@dataclass
class ToolResult:
    """
    函数/类功能与逻辑描述：
        工具调用的统一返回包，把成功数据与失败错误统一在同一结构中，
        使调用方既能用 unwrap() 走「成功取值 / 失败抛异常」的简洁路径，
        也能直接检查 ok 字段做分支处理；attempts 与 cost_ms 供可观测与重试统计使用。
        本类不限冻结（可变），因为 attempts / cost_ms 会在执行过程中被回填。
    构造入参说明：
        ok (bool)：调用是否成功。
        data (Any)：成功时的结果对象，默认 None。
        error (ToolError | None)：失败时的错误对象，默认 None。
        attempts (int)：实际尝试次数（含重试），默认 1。
        cost_ms (float)：本次调用耗时毫秒，默认 0.0。
        trace_id (str | None)：链路追踪 ID，默认 None。
    返回值说明：
        ToolResult：可变实例；取值请用 unwrap()。
    """
    ok: bool
    data: Any = None
    error: "ToolError | None" = None
    attempts: int = 1                # 实际尝试次数（含重试）
    cost_ms: float = 0.0
    trace_id: str | None = None

    def unwrap(self) -> Any:
        """
        函数功能与逻辑描述：
            ★兼容旧语义的取值入口：成功直接返回 data；失败则抛出**调用方熟知的原生异常类型**
            （而非平台自有的 ToolError），以保住既有上层契约。
            ⚠️ 为什么必须做类型映射（2026-08-30 动工时发现的兼容性陷阱）：
            现状 `call_bill_sql` / `call_llm_finance` 无权限时抛 **`PermissionError`**，
            且 `tests/integration/test_mcp_gateway.py` 有 2 处断言
            `pytest.raises(PermissionError)`。若失败一律抛 `ToolError`，这层契约就断了，
            与 M1 验收「~40 处测试零改动」直接冲突。
            映射规则见 _as_native：kind=REMOTE 且带 origin 时直接抛底层原始异常以保留真实报错上下文；
            其余按 _NATIVE_EXC_MAP 映射；error 为空时按 INTERNAL 兜底。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Any：self.ok 为 True 时返回 self.data（可能为 None）。
        异常说明：
            按 self.error.kind 映射出的原生异常类型（PermissionError / ValueError /
            TimeoutError / ConnectionError / RuntimeError，或 REMOTE 场景下的原始异常）；
            self.error 为 None 时抛 ToolError(INTERNAL, "未知工具错误")。
        """
        if self.ok:
            return self.data
        err = self.error or ToolError(ToolErrorKind.INTERNAL, "未知工具错误")
        raise _as_native(err)


# ════════ ④ 错误：平台层统一分类 ════════
class ToolErrorKind(Enum):
    """
    函数/类功能与逻辑描述：
        平台层错误分类枚举，是「是否可重试 / 是否可笼统归类」等策略判定的唯一依据，
        也决定 ToolResult.unwrap 抛出的原生异常类型（见 _NATIVE_EXC_MAP）。
    构造入参说明：
        无（枚举成员固定）。
    返回值说明：
        无（枚举类，成员为 ToolErrorKind 实例）。
    """
    NOT_FOUND = "not_found"          # 工具不存在
    PERMISSION = "permission"        # 无权限（RBAC 拒绝）
    VALIDATION = "validation"        # 参数不符合 params_schema
    TIMEOUT = "timeout"              # 超过 spec.timeout
    CONNECTION = "connection"        # 连接类故障
    REMOTE = "remote"                # 远端业务错误（MCP server 明确返回错误）
    INTERNAL = "internal"            # 适配器 / 引擎内部错误


class ToolError(Exception):
    """
    函数/类功能与逻辑描述：
        平台层统一错误类型，承载分类信息（kind）、原始异常（origin）、工具名与可重试标记。
        ⚠️ 刻意**不用** `@dataclass(frozen=True)`：dataclass 生成的 `__init__` 不会调用
        `super().__init__(message)`，导致 `Exception.args` 为空 → `str(e)` 返回空字符串，
        错误信息在日志与异常冒泡中**全部丢失**。因此此处手写 `__init__` 以保证 `str(e)` 可用。
    构造入参说明：
        kind (ToolErrorKind)：错误分类。
        message (str)：人类可读的错误描述。
        origin (Exception | None)：底层原始异常，用于保留报错上下文与异常链，默认 None。
        tool_name (str | None)：出错工具名，默认 None。
        retriable (bool)：是否允许重试，默认 False。
    返回值说明：
        ToolError：异常实例；字符串表现为 "[{kind.value}] {message}"。
    """

    def __init__(self, kind: ToolErrorKind, message: str,
                 origin: Exception | None = None,
                 tool_name: str | None = None,
                 retriable: bool = False):
        """
        函数功能与逻辑描述：
            手写构造：先调用 super().__init__(message) 把消息写入 Exception.args
            （这是 str(e) 能返回内容的前提），再把各扩展字段挂到实例属性上。
            不抛异常、无副作用。
        入参说明：
            kind (ToolErrorKind)：错误分类。
            message (str)：错误描述，将被写入 Exception.args。
            origin (Exception | None)：底层原始异常，默认 None。
            tool_name (str | None)：出错工具名，默认 None。
            retriable (bool)：是否允许重试，默认 False。
        返回值说明：
            无（构造器，就地初始化实例属性）。
        """
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.origin = origin
        self.tool_name = tool_name
        self.retriable = retriable

    def __str__(self) -> str:
        """
        函数功能与逻辑描述：
            统一错误字符串表现，把分类与描述拼成 "[{kind.value}] {message}"，
            便于日志检索与上层按前缀识别错误类别。
        入参说明：
            无（隐式 self）。
        返回值说明：
            str：形如 "[permission] 权限不足：..." 的文本。
        """
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
    """
    函数功能与逻辑描述：
        把平台 ToolError 映射为上层熟知的原生异常实例，以维持历史调用契约。
        特例：kind=REMOTE 且存在 origin 时，直接返回底层原始异常对象，
        以保留真实报错类型与上下文（对齐现状 _call_server 中 `raise e` 的行为）。
        其余情况按 _NATIVE_EXC_MAP 取目标类型并构造新实例；若 origin 存在，
        通过 __cause__ 挂上原始异常形成异常链。映射表未覆盖的 kind 回退为原 ToolError 对象。
    入参说明：
        err (ToolError)：待转换的平台错误对象。
    返回值说明：
        Exception：可直接 raise 的异常实例。REMOTE + origin 时为 origin 本身；
            其余为映射类型的新实例；无匹配映射时返回传入的 err 自身。
    """
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
