# -*- coding: utf-8 -*-
"""协议适配层：把"怎么连上去、怎么传参、怎么收回来"隔离在平台最底层。

设计依据：`设计/08_架构演进方案与决策记录.md` §1.8。

★★ 连接模式（短连接 / 长连接 / 连接池 / 进程复用）是 MCPAdapter 的**内部实现细节**——
   上层、ToolSpec、调用方全部无感知。这正是「MCP 只是众多可接入协议中的一种」的工程落地。
   当前实现复用现有 MCPClient（短连接 + 内部重试 + 120s 握手超时）；
   未来 M3 改 Streamable HTTP 常驻，**只改这一个类，四层零改动**。
"""
import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mcpGateway.tool_model import (
    CONNECTION_EXC, InvokeContext, ToolError, ToolErrorKind, ToolSpec,
)

# ★打破导入环 client → registry → adapters → client：
#   仅在类型检查期引用 MCPClient，运行时零依赖（实例由启动装配注入）。
if TYPE_CHECKING:  # pragma: no cover
    from mcpGateway.client import MCPClient


def _apply_ctx_injection(spec: ToolSpec, payload: dict, ctx: InvokeContext) -> dict:
    """
    函数功能与逻辑描述：
        按 `spec.binding["inject_ctx"]` 声明的字段名，从调用上下文（InvokeContext）取值注入到
        本次调用的参数里，供**协议无关**地传递"调用方身份类"参数（M15 引入，首个消费方是
        `bill.add` 的 `task_id`——崩溃恢复对账依据必须由代码写入，不能由模型提供）。
        与 MCPAdapter 里既有的 `inject_agent_id` / `inject_run_id` 同源同理由：
        这些字段**不进 params_schema**（否则会被 `to_openai_schema` 暴露给模型 → 伪造提权），
        故引擎的参数校验（发生在注入之前）看不到它们，工具函数以附加形参接收。
        取值为 None 时**不注入**——避免用 None 覆盖工具函数的默认值（如 task_id 缺省时
        `add_bill` 走 MAX(id) 回读分支）。返回新字典，不改动调用方传入的 args。
    入参说明：
        spec (ToolSpec)：工具契约，可选 `binding["inject_ctx"]` 为字段名元组/列表（如 ("task_id",)）。
        payload (dict)：本次调用的业务参数（适配器已做的浅拷贝）。
        ctx (InvokeContext)：调用上下文，字段名须与其属性同名（agent_id / trace_id / session_id / task_id）。
    返回值说明：
        dict：注入后的参数字典；未声明 inject_ctx 或取值全为 None 时，返回与入参内容相同的字典。
    """
    keys = spec.binding.get("inject_ctx") or ()
    if not keys:
        return payload
    merged = dict(payload)
    for key in keys:
        value = getattr(ctx, key, None)
        if value is not None:
            merged[key] = value
    return merged


class ProtocolAdapter(ABC):
    """
    函数/类功能与逻辑描述：
        协议适配器抽象基类，定义「执行一次调用」与「列出远端工具定义」两个必需能力。
        新增一种协议 = 实现本类 + `registry.register_adapter(protocol, adapter)`，
        注册 / 发现 / 执行三层零改动，平台不为任何具体协议开后门。
    构造入参说明：
        无（抽象基类，子类自行定义构造，如 MCPAdapter 需要注入 MCPClient）。
    返回值说明：
        无（抽象基类不可直接实例化）。
    """
    protocol: str = ""

    @abstractmethod
    async def invoke(self, spec: ToolSpec, args: dict, ctx: InvokeContext) -> Any:
        """
        函数功能与逻辑描述：
            抽象方法：执行一次工具调用。子类实现负责连接建立、参数投递与结果回收，
            并把所有失败统一抛成 `ToolError`（由执行引擎负责最终兜底与分类），
            不得向上抛出底层库的原生异常类型。
        入参说明：
            spec (ToolSpec)：工具元数据，含 binding（协议相关连接信息）、side_effect、name 等。
            args (dict)：业务参数（已通过 params_schema 校验）。
            ctx (InvokeContext)：引擎注入的调用上下文（agent_id / trace_id / session_id）。
        返回值说明：
            Any：工具的原始返回结果，结构由工具自身决定，平台不做规范化。
        """

    @abstractmethod
    async def list_remote_tools(self) -> list[dict]:
        """
        函数功能与逻辑描述：
            抽象方法：返回远端工具定义列表（原始 schema），供 `registry.verify_schemas()`
            做离线自检以防 schema 漂移。无远端概念的协议（如本地函数）应返回空列表。
        入参说明：
            无。
        返回值说明：
            list[dict]：远端工具定义列表，每项为工具名 + 入参 schema 等原始描述；
                无远端定义时返回 []。
        """


class MCPAdapter(ProtocolAdapter):
    """
    函数/类功能与逻辑描述：
        MCP 协议适配器，是当前唯一的远端协议实现。持有 MCPClient 以复用其短连接、
        内部重试与 120s 握手超时能力，自身只负责「按 binding 组装 payload」与「异常分类」。
        关键职责包括：注入 agent_id 与 run_id（均不进 params_schema，防 LLM 伪造提权）、
        对写副作用工具关闭底层重试（D2 防重复记账）。
    构造入参说明：
        client (MCPClient)：MCP 客户端实例，由启动装配阶段注入，不在此处创建。
    返回值说明：
        构造返回 MCPAdapter 实例；调用能力见 invoke / list_remote_tools。
    """

    protocol = "mcp"

    def __init__(self, client: "MCPClient"):
        """
        函数功能与逻辑描述：
            保存外部注入的 MCPClient 引用，不做连接建立或健康检查（连接时机由
            MCPClient 自行掌控）。不重写客户端实现，避免与既有链路产生第二套连接管理。
        入参说明：
            client (MCPClient)：已装配好的 MCP 客户端实例。
        返回值说明：
            无（仅赋值实例属性）。
        """
        self._client = client        # 复用现有 MCPClient，不重写

    async def invoke(self, spec: ToolSpec, args: dict, ctx: InvokeContext) -> Any:
        """
        函数功能与逻辑描述：
            按 ToolSpec.binding 组装实际投递给 MCP server 的 payload，并调用客户端执行：
            ① 复制业务参数（不改动调用方传入的 args）；
            ② 依 binding.inject_agent_id 注入调用方身份，取值语义为 True → 注入为 "agent_id"、
            传入字符串 → 注入为该名称的远端参数（如 llm_chat 侧叫 agent_tag，用于差异化推理参数）、
            False → 不注入；
            ③ 依 binding.inject_run_id 注入 trace_id 以贯通链路归因；
            ④ 对写副作用工具把底层重试次数置 0——INSERT 已落库但连接断开时，
            底层重试会再执行一次造成重复记账（D2 要防的核心场景）；
            非写工具传 None 表示沿用 MCPClient 默认重试次数。
            异常按类型分流：ToolError 原样透传；连接类异常转 ToolError(CONNECTION)，
            且仅在非写副作用时标记 retriable=True；其余异常一律归为 ToolError(REMOTE)。
        入参说明：
            spec (ToolSpec)：工具元数据，必须含 binding["server_key"] 与 binding["tool"]。
            args (dict)：业务参数，函数内会做浅拷贝，原字典不会被修改。
            ctx (InvokeContext)：调用上下文，提供 agent_id 与 trace_id 用于注入。
        返回值说明：
            Any：MCP server 返回的原始结果（由 client._call_server 提取后的形态）。
        异常说明：
            ToolError：kind 为 CONNECTION（连接类故障，非写操作时 retriable=True）、
                REMOTE（远端业务错误）或上游本就抛出的 ToolError（原样透传）。
        """
        payload = dict(args)
        # ★agent_id 由引擎注入，**不进 params_schema**：
        #   MCP server 侧 exec_sql / finance_chat 需要它做二次鉴权，
        #   但若写进 schema 就会被 to_openai_schema 暴露给 LLM → 模型可伪造 agent_id 提权。
        #   见 `08` §1.10「agent_id 归属」。
        #   inject_agent_id 取值：True→注入为 "agent_id"；"agent_tag"→注入为该远端参数名
        #   （llm_chat 侧叫 agent_tag，用于差异化推理参数）；False→不注入。
        inject = spec.binding.get("inject_agent_id", True)
        if inject:
            key = inject if isinstance(inject, str) else "agent_id"
            payload[key] = ctx.agent_id

        # M7（D3 阶段2）：run_id 跨进程透传——仅对 binding 显式开启的 LLM 工具注入
        # （server 侧签名带可选 run_id，函数内 bind 到本进程 context → llm_call_stat 归因）。
        # 不进 params_schema（防 LLM 伪造），计入 _schema_diff 豁免集，见 `11` §3 M7。
        if spec.binding.get("inject_run_id") and ctx.trace_id:
            payload["run_id"] = ctx.trace_id

        # M15：上下文注入面（协议无关，当前用于 bill.add 的 task_id；本地/远端一致处理）
        payload = _apply_ctx_injection(spec, payload, ctx)

        # ★写副作用工具关闭底层连接重试：
        #   INSERT 已落库但连接断开时，底层重试会再执行一次 → 重复记账（D2 要防的核心场景）。
        #   None = 沿用 MCPClient 默认重试次数；0 = 一次即止。
        inner_retry = 0 if spec.side_effect else None

        try:
            return await self._client._call_server(
                spec.binding["server_key"], spec.binding["tool"], payload,
                max_retry=inner_retry)
        except ToolError:
            raise
        except CONNECTION_EXC as e:
            raise ToolError(ToolErrorKind.CONNECTION, str(e), origin=e,
                            tool_name=spec.name, retriable=not spec.side_effect) from e
        except Exception as e:
            raise ToolError(ToolErrorKind.REMOTE, str(e), origin=e,
                            tool_name=spec.name) from e

    async def list_remote_tools(self) -> list[dict]:
        """
        函数功能与逻辑描述：
            拉取全部 MCP server 的 tools/list，聚合为一个扁平列表返回。
            ⚠️ 每次调用都会拉起子进程（握手约 26~29s），因此**不进启动路径**，
            仅在 `registry.verify_schemas()` 自检时调用，避免拖慢冷启动。
        入参说明：
            无。
        返回值说明：
            list[dict]：全部 server 的工具定义列表（已聚合，不含 server_key 分组）。
        """
        return await self._client.list_all_server_tools()


class LocalAdapter(ProtocolAdapter):
    """
    函数/类功能与逻辑描述：
        本地函数适配器：把普通 Python 可调用对象包装成与 MCP 工具同构的协议实现，
        从而让上层 ToolRegistry / ExecutionEngine 对本地工具与远端工具零差别对待。
        同步函数默认用 asyncio.to_thread 卸载到线程池，避免阻塞事件循环；
        协程函数（如 fact_tools.upsert_habit）则被直接 await，以保证协程级锁语义有效。
    构造入参说明：
        无（binding 中直接携带 func，不需要额外依赖注入）。
    返回值说明：
        构造返回 LocalAdapter 实例；调用能力见 invoke / list_remote_tools。
    """

    protocol = "local"

    async def invoke(self, spec: ToolSpec, args: dict, ctx: InvokeContext) -> Any:
        """
        函数功能与逻辑描述：
            以关键字参数方式调用 binding["func"]。是否卸载到线程池由两层条件共同决定：
            binding.to_thread（默认 True）为真 **且** 目标函数不是协程函数时才走
            asyncio.to_thread；协程函数必须直接 await，否则 to_thread 会返回未 await 的协程对象，
            且外层协程级锁（如 db_lock）会失效。不做参数校验与异常包装——校验由执行引擎
            依 params_schema 完成，异常由引擎统一转成 ToolError。
        入参说明：
            spec (ToolSpec)：需含 binding["func"]（可调用对象），可选 binding["to_thread"]（默认 True）。
            args (dict)：以关键字形式透传给 func 的参数。
            ctx (InvokeContext)：调用上下文；本地适配器当前不使用，仅为接口一致而保留。
        返回值说明：
            Any：func 的返回值（同步函数经线程池返回，协程函数 await 后的结果）。
        """
        fn = spec.binding["func"]
        kwargs = _apply_ctx_injection(spec, args, ctx)   # ★M15：inject_ctx 声明的字段在此注入
        if spec.binding.get("to_thread", True) and not inspect.iscoroutinefunction(fn):
            return await asyncio.to_thread(fn, **kwargs)
        return await fn(**kwargs)

    async def list_remote_tools(self) -> list[dict]:
        """
        函数功能与逻辑描述：
            本地函数没有远端工具定义，固定返回空列表；`registry.verify_schemas()`
            据此自动跳过本地工具，不会误报 schema 漂移。
        入参说明：
            无。
        返回值说明：
            list[dict]：恒为空列表 []。
        """
        return []      # 本地函数无远端定义，verify_schemas 自动跳过


__all__ = ["ProtocolAdapter", "MCPAdapter", "LocalAdapter"]
