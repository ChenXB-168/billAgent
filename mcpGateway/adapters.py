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


class ProtocolAdapter(ABC):
    """协议适配器基类。

    新增一种协议 = 实现本类 + `registry.register_adapter(protocol, adapter)`，
    注册 / 发现 / 执行三层零改动。
    """
    protocol: str = ""

    @abstractmethod
    async def invoke(self, spec: ToolSpec, args: dict, ctx: InvokeContext) -> Any:
        """执行一次工具调用。异常统一抛 `ToolError`，由执行引擎负责最终兜底分类。"""

    @abstractmethod
    async def list_remote_tools(self) -> list[dict]:
        """返回远端工具定义列表，供 `registry.verify_schemas()` 离线自检防 schema 漂移。"""


class MCPAdapter(ProtocolAdapter):
    """MCP 协议适配器（当前唯一远端协议）。"""

    protocol = "mcp"

    def __init__(self, client: "MCPClient"):
        self._client = client        # 复用现有 MCPClient，不重写

    async def invoke(self, spec: ToolSpec, args: dict, ctx: InvokeContext) -> Any:
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
        """拉取全部 MCP server 的 tools/list。

        ⚠️ 每次调用都会拉起子进程（握手约 26~29s），**不进启动路径**，
        仅在 `registry.verify_schemas()` 自检时调用。
        """
        return await self._client.list_all_server_tools()


class LocalAdapter(ProtocolAdapter):
    """本地函数适配器：同步函数默认 `to_thread` 卸载，避免阻塞事件循环。"""

    protocol = "local"

    async def invoke(self, spec: ToolSpec, args: dict, ctx: InvokeContext) -> Any:
        fn = spec.binding["func"]
        if spec.binding.get("to_thread", True) and not inspect.iscoroutinefunction(fn):
            return await asyncio.to_thread(fn, **args)
        return await fn(**args)

    async def list_remote_tools(self) -> list[dict]:
        return []      # 本地函数无远端定义，verify_schemas 自动跳过


__all__ = ["ProtocolAdapter", "MCPAdapter", "LocalAdapter"]
