import sys
import os
import time
import warnings
import asyncio
import subprocess
import traceback
from loguru import logger
from typing import Optional, Dict, Tuple

# Python 3.10 用 exceptiongroup 移植包（anyio 依赖）；3.11+ 为内置。
# mcp SDK 的 streamablehttp_client 会把连接异常包装进 ExceptionGroup，判定需递归展开。
try:
    from exceptiongroup import BaseExceptionGroup  # noqa: F401
except ImportError:
    pass  # Python 3.11+ 内置

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from config.config import (
    BASE_DIR, MCP_HOST, MCP_SERVER_PORTS, MCP_STDIO_FALLBACK, MCP_HTTP_PATH,
)
from mcpGateway.rbac_config import get_agent_permission, DBPerm, LLMPerm
from mcpGateway.audit_recorder import write_audit_log
# 2026-08-30（M1）：连接异常集合上移到统一模型层，供 adapters/executor/client 共用。
# 若继续留在 client.py，会形成 client → registry → adapters → client 的模块级导入环。
from mcpGateway.tool_model import CONNECTION_EXC
from mcpGateway.registry import get_registry

# 屏蔽底层无用资源告警
warnings.filterwarnings("ignore", category=ResourceWarning)


# ===================== 全局常量：MCP服务启动命令 =====================
SERVER_CMD_MAP = {
    "sql_bill": [sys.executable, "-u", os.path.join(BASE_DIR, "mcpGateway", "sql_server", "server_bill.py")],
    "llm_base": [sys.executable, "-u", os.path.join(BASE_DIR, "mcpGateway", "server_llm_base.py")],
    "llm_finance": [sys.executable, "-u", os.path.join(BASE_DIR, "mcpGateway", "server_llm_finance.py")]
}

MAX_CONNECT_RETRY = 2   # 底层连接重试次数上限；引擎层 max_retry=0，避免两者叠加成乘数效应


class MCPClient:
    """统一MCP客户端，所有Agent必须通过此处访问底层资源

    连接模式说明（联调踩坑修复）：
      原实现用 __aenter__ 建立长连接并跨 task 复用（心跳/调用/shutdown 不同 task），
      stdio_client 的 cancel scope 被绑定在首个进入的 task，其它 task 退出时报
      "Attempted to exit cancel scope in a different task" 直接崩溃。
      现改为短连接模式：每次调用在同一 task 内 async with 建立并释放连接，
      从根上消除跨 task 生命周期问题；调用频率低，子进程启停开销可接受。
    """

    def __init__(self):
        self._shutdown_flag: bool = False
        # ★M3：删除全局串行锁 self._lock（`03` §9.8.3）——常驻 HTTP 服务天然并发，
        #   锁是"派发层并行、执行层排队"的直接原因；stdio 回滚路径也无需锁（每次独立子进程）。

    def _server_url(self, server_key: str) -> str:
        """M3：常驻服务地址（`03` §9.8.2）——host/port/path 均出自 config，单一来源"""
        return f"http://{MCP_HOST}:{MCP_SERVER_PORTS[server_key]}{MCP_HTTP_PATH}"

    def _is_connection_error(self, e: Exception) -> bool:
        """M3：判定连接类异常（ExceptionGroup 递归展开）。

        mcp SDK 的 streamablehttp_client 会把 httpx.ConnectError / OSError 等
        底层连接异常包装进 **ExceptionGroup**（anyio 任务组机制），直接
        isinstance(CONNECTION_EXC) 会漏判 → 连接失败不重试、adapters 误判为 REMOTE。
        这里递归展开子异常逐一判定。
        """
        if isinstance(e, CONNECTION_EXC):
            return True
        if isinstance(e, BaseExceptionGroup):
            return any(self._is_connection_error(sub) for sub in e.exceptions)
        return False

    async def _call_server(self, server_key: str, tool_name: str, arguments: dict,
                           max_retry: int | None = None):
        """通用MCP工具调用，统一异常处理

        ★M3 传输层改造（`03` §9.8.2 / §9.8.3）：
          - **主路径**：`streamablehttp_client(url)` 连**常驻 HTTP 服务**（随 bootstrap 拉起），
            无状态请求、天然并发、只付一次冷启动；
          - **回滚路径**：`BILLAGENT_MCP_STDIO=1` 时切回旧 **stdio 短连接**（M3 回滚要求）；
          - **删除全局锁**：HTTP 服务天然并发，客户端无需 `asyncio.Lock`。

        :param max_retry: 连接类异常的重试次数上限（**不含首次调用**）。
            None → 沿用 MAX_CONNECT_RETRY（现状行为）；0 → 一次即止，不重试。
            ★M1 新增：写副作用工具（sql.write）由 MCPAdapter 传 0——
              "INSERT 已落库但连接随后断开"时若重试，会造成重复记账（D2 要防的核心场景）。
        """
        retry = MAX_CONNECT_RETRY if max_retry is None else max_retry
        cmd = SERVER_CMD_MAP[server_key]
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, retry)):     # 至少执行 1 次；max_retry=0 即"不重试"
            t0 = time.time()
            try:
                if MCP_STDIO_FALLBACK:
                    # ★回滚路径：stdio 短连接（M3 回滚要求）
                    #   透传完整环境变量（M1 发现的 mcp SDK 默认过滤 BILLAGENT_* 问题）
                    params = StdioServerParameters(
                        command=cmd[0], args=cmd[1:],
                        env=dict(os.environ), timeout=120,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    print(f"[MCP] {server_key}.{tool_name} 第{attempt+1}次调用(stdio回滚): 启动子进程...",
                          file=sys.stderr, flush=True)
                    async with stdio_client(params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()  # MCP 握手
                            resp = await session.call_tool(name=tool_name, arguments=arguments)
                else:
                    # ★M3 主路径：Streamable HTTP 常驻服务（服务随 bootstrap 拉起，无锁）
                    url = self._server_url(server_key)
                    print(f"[MCP] {server_key}.{tool_name} 第{attempt+1}次调用(HTTP常驻): {url}",
                          file=sys.stderr, flush=True)
                    async with streamablehttp_client(url) as (read, write, _get_session_id):
                        async with ClientSession(read, write) as session:
                            await session.initialize()  # MCP 握手
                            resp = await session.call_tool(name=tool_name, arguments=arguments)
                # MCP 返回的是 CallToolResult，content 是一个列表（可能是文本、可能是图片）。
                # 这里取第一条内容的 text 字段——统一把响应变成字符串返回给上层。这一步很关键：业务层拿到的永远是 str
                print(f"[MCP] {server_key}.{tool_name} 返回，总耗时 {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
                return resp.content[0].text
            except Exception as e:
                last_exc = e
                if not self._is_connection_error(e):
                    print(f"[{server_key}] 业务调用异常，不重试：{e}", file=sys.stderr)
                    raise e
                print(f"[{server_key}] 连接/管道异常，第{attempt + 1}次重试：{e}", file=sys.stderr)
        raise ConnectionError(f"{server_key} 连接失败，已尝试 {max(1, retry)} 次") from last_exc

    async def shutdown_all(self):
        """优雅停机：客户端无长连接需释放（常驻服务进程由 bootstrap.shutdown_system 关闭）"""
        self._shutdown_flag = True
        print("MCP客户端已标记关闭（无状态请求，无连接需释放）", file=sys.stderr)

    # ===================== 远端工具清单（供 registry.verify_schemas 自检） =====================
    async def list_server_tools(self, server_key: str) -> list[dict]:
        """拉取指定 MCP server 的 `tools/list`。

        ★M3：默认连**常驻 HTTP 服务**（需 bootstrap 已拉起）；`MCP_STDIO_FALLBACK=1` 时走 stdio 短连接。
        ⚠️ **不进启动路径**，仅供 `registry.verify_schemas()` 在 CI / 手动自检时调用。
        """
        cmd = SERVER_CMD_MAP[server_key]
        if MCP_STDIO_FALLBACK:
            params = StdioServerParameters(
                command=cmd[0], args=cmd[1:], timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
        else:
            async with streamablehttp_client(self._server_url(server_key)) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
        return [{"name": t.name, "description": t.description,
                 "inputSchema": t.inputSchema} for t in tools.tools]

    async def list_all_server_tools(self) -> list[dict]:
        """遍历全部 MCP server，合并返回工具清单。"""
        out: list[dict] = []
        for key in SERVER_CMD_MAP:
            out.extend(await self.list_server_tools(key))
        return out

    # ===================== 对外业务接口 =====================
    async def call_bill_sql(self, agent_id: str, sql: str, params: list = None):
        """账单数据库SQL调用。

        ★M1 改造：**签名与返回类型完全不变**，内部转调工具平台执行引擎——
        鉴权 / 参数校验 / 超时 / 审计全部由引擎统一治理（见 `08` §1.9）。
        ~18 处生产调用与 ~40 处测试 patch 因此零改动。

        Q1 决策：动态权限判定（SELECT→READ，否则→WRITE）**下沉为两个静态工具**
        `sql.read` / `sql.write`，使 required_perm 语义纯净、`list_tools` 可静态过滤；
        兼容层保留按 SQL 前缀分流的旧行为，调用方无感。

        ★agent_id 不再手工塞进 arguments —— 改由 MCPAdapter 从调用上下文注入，
        这样它不会出现在 params_schema 里，也就不会被 LLM 看到或伪造（`08` §1.10）。

        回滚参考（M1 前的原实现）：
            perms = get_agent_permission(agent_id)
            required_perm = DBPerm.BILL_READ if sql.strip().upper().startswith("SELECT") else DBPerm.BILL_WRITE
            if required_perm not in perms:
                write_audit_log(agent_id, "SQL_BILL", sql, False)
                raise PermissionError(f"Agent[{agent_id}]无账单数据库{required_perm.value}权限")
            res = await self._call_server("sql_bill", "exec_sql",
                                          {"agent_id": agent_id, "sql": sql, "params": params or []})
            write_audit_log(agent_id, "SQL_BILL", sql, True)
            return res
        """
        tool = "sql.read" if sql.strip().upper().startswith("SELECT") else "sql.write"
        res = await get_registry().invoke(
            tool, {"sql": sql, "params": params or []}, agent_id=agent_id)
        return res.unwrap()      # 成功返回 str；失败抛原生异常（无权限→PermissionError）

    async def call_llm_base(self, sys_prompt: str, user_text: str, agent_tag: str = "unknown"):
        """通用基座模型调用，编排器与业务Agent通用。

        ★M1 改造：内部转调 `llm.chat` 工具，签名与返回类型不变；
        `agent_id` 由适配器映射为远端 `agent_tag` 参数（差异化推理参数，见踩坑记录坑 12）。

        ⚠️ **`agent_tag` 必须传真实 Agent 标识**：它同时是鉴权主体
        （`llm.chat` 的 required_perm = LLMPerm.LLM_BASE）。
        沿用默认值 `"unknown"` 会因无任何权限被网关拒绝 —— 这是 M1 的**治理目标而非缺陷**：
        改造前 `call_llm_base` 完全不校验权限，网关形同虚设。
        4 个业务 Agent 均已显式传自身标识，主链路不受影响。
        """
        logger.info(f"[LLM_GLOBAL_IN] agent={agent_tag}, system_prompt_len={len(sys_prompt)}, user_input={user_text}")
        logger.debug(f"[LLM_GLOBAL_FULL_PROMPT] agent={agent_tag}, system={sys_prompt[:800]}...")

        try:
            res = await get_registry().invoke(
                "llm.chat",
                {"system_prompt": sys_prompt, "user_input": user_text},
                agent_id=agent_tag)
            logger.info(f"[LLM_GLOBAL_OUT] agent={agent_tag}, llm_raw_response={res.data}")
            return res.unwrap()
        except Exception as e:
            err_stack = traceback.format_exc()
            logger.error(f"[LLM_GLOBAL_ERROR] agent={agent_tag}, LLM调用失败, err={str(e)}, stack={err_stack}")
            raise

    async def call_llm_finance(self, agent_id: str, sys_prompt: str, user_text: str):
        """理财专属模型调用。

        ★M1 改造：权限校验交由执行引擎（`llm.finance` 的 required_perm = LLMPerm.LLM_FINANCE），
        审计由引擎统一记录；签名、返回类型与**无权限时抛 PermissionError 的行为**均保持不变。

        回滚参考（M1 前的原实现）：
            perms = get_agent_permission(agent_id)
            if LLMPerm.LLM_FINANCE not in perms:
                write_audit_log(agent_id, "LLM_FINANCE", user_text, False)
                raise PermissionError(f"Agent[{agent_id}]无理财模型调用权限")
            res = await self._call_server("llm_finance", "finance_chat",
                                          {"agent_id": agent_id, "system_prompt": sys_prompt,
                                           "user_input": user_text})
            write_audit_log(agent_id, "LLM_FINANCE", user_text, True)
            return res
        """
        logger.info(f"[LLM_FIN_IN] agent={agent_id}, sys_len={len(sys_prompt)}, user_input={user_text}")
        logger.debug(f"[LLM_FIN_FULL_PROMPT] agent={agent_id}, system={sys_prompt[:800]}...")

        try:
            res = await get_registry().invoke(
                "llm.finance",
                {"system_prompt": sys_prompt, "user_input": user_text},
                agent_id=agent_id)
            logger.info(f"[LLM_FIN_OUT] agent={agent_id}, llm_raw_response={res.data}")
            return res.unwrap()
        except Exception as e:
            err_stack = traceback.format_exc()
            logger.error(f"[LLM_FIN_ERROR] agent={agent_id}, 理财LLM调用失败, err={str(e)}\n完整堆栈:{err_stack}")
            raise


# 全局单例
mcp_client = MCPClient()

__all__ = ["mcp_client"]
