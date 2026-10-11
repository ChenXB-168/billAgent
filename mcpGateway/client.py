import sys
import os
import json
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

MAX_CONNECT_RETRY = 2   # 底层调用尝试次数上限（含首次；循环按 max(1, retry) 归一）；引擎层 max_retry=0，避免两者叠加成乘数效应


class MCPClient:
    """
    函数/类功能与逻辑描述：
        统一 MCP 客户端，是 Agent 访问 MCP 底层资源（账单 SQL / 基座模型 / 理财模型）
        的唯一出口：对外提供 call_bill_sql / call_llm_base / call_llm_finance 三个兼容
        业务入口与 list_server_tools / list_all_server_tools 自检入口，实际投递统一收敛到
        `_call_server`，由其在内部按传输模式分流。
        传输模式（M1 短连接修复 + M3 常驻 HTTP 双轨，联调踩坑修复）：
          原实现用 __aenter__ 建立长连接并跨 task 复用（心跳/调用/shutdown 不同 task），
          stdio_client 的 cancel scope 被绑定在首个进入的 task，其它 task 退出时报
          "Attempted to exit cancel scope in a different task" 直接崩溃。现所有路径都改为
          **单次调用在同一 task 内 async with 建立并释放连接**，从根上消除跨 task 生命周期问题：
            - 主路径：`streamablehttp_client` 连常驻 HTTP 服务（随 bootstrap 拉起），
              无状态请求、天然并发、只付一次冷启动；
            - 回滚路径：`BILLAGENT_MCP_STDIO=1`（config.MCP_STDIO_FALLBACK）时切回旧 stdio
              短连接，此时每次调用付子进程启停开销，调用频率低可接受。
        协作口径：M1 起三个业务入口内部转调工具平台执行引擎（Registry / ExecutionEngine），
        鉴权、参数校验、超时、审计与重试均由引擎治理，本类只负责建连、取回文本结果。
    构造入参说明：
        无。
    返回值说明：
        构造返回 MCPClient 实例；模块底部以 `mcp_client` 全局单例形式对外提供。
    """

    # 本类为**无状态**实现：无实例属性，每次调用按需建连（见 `_call_server`），
    #   无资源需要初始化或释放。常驻 HTTP 服务天然并发，故不设全局串行锁（`03` §9.8.3）
    #   ——锁是"派发层并行、执行层排队"的直接原因；stdio 回滚路径每次独立子进程，亦无需锁。

    def _server_url(self, server_key: str) -> str:
        """
        函数功能与逻辑描述：
            拼装指定 MCP server 的常驻 HTTP 服务地址（M3，依据 `03` §9.8.2）。
            host / port / path 全部取自 config 单一来源，不在本模块硬编码：host 用 MCP_HOST
            （127.0.0.1，仅本机可见不对外暴露）、port 按 MCP_SERVER_PORTS[server_key] 取
            （sql_bill=8001 / llm_base=8002 / llm_finance=8003，基址 MCP_PORT_BASE 可由环境变量
            BILLAGENT_MCP_PORT 覆盖）、path 用 MCP_HTTP_PATH（/mcp）。
            纯字符串拼接：不做连通性探测、不缓存结果；server_key 非 MCP_SERVER_PORTS 的键时
            直接抛 KeyError。
        入参说明：
            server_key (str)：server 标识，须为 SERVER_CMD_MAP / MCP_SERVER_PORTS 的键，
                当前取值 sql_bill / llm_base / llm_finance。
        返回值说明：
            str：形如 http://127.0.0.1:8001/mcp 的完整服务地址。
        """
        return f"http://{MCP_HOST}:{MCP_SERVER_PORTS[server_key]}{MCP_HTTP_PATH}"

    def _is_connection_error(self, e: Exception) -> bool:
        """
        函数功能与逻辑描述：
            判定异常是否属「连接类」故障（M3）——该判定是 `_call_server` 是否重试、以及
            MCPAdapter 归类为 ToolError(CONNECTION) 还是 REMOTE 的唯一依据。
            mcp SDK 的 streamablehttp_client 会把 httpx.ConnectError / OSError 等底层连接异常
            包装进 **ExceptionGroup**（anyio 任务组机制），只做 isinstance(e, CONNECTION_EXC)
            会漏判 → 连接失败不重试、adapters 误判为 REMOTE。这里对 BaseExceptionGroup
            递归展开子异常逐一判定，任一子异常命中即视为连接类。
            判定顺序为先自身后分组，纯判定无副作用、不吞异常。
        入参说明：
            e (Exception)：待判定的异常对象；若为 BaseExceptionGroup 则递归遍历其 .exceptions。
        返回值说明：
            bool：True = 连接/管道类异常；False = 非连接类（含普通业务异常）或分组内均不命中。
        """
        if isinstance(e, CONNECTION_EXC):
            return True
        if isinstance(e, BaseExceptionGroup):
            return any(self._is_connection_error(sub) for sub in e.exceptions)
        return False

    async def _call_server(self, server_key: str, tool_name: str, arguments: dict,
                           max_retry: int | None = None):
        """
        函数功能与逻辑描述：
            通用 MCP 工具调用底层实现，统一封装「建连 → 握手 → call_tool → 取文本 → 异常分流」，
            是业务入口与 MCPAdapter 共用的执行出口。
            传输分流（★M3 传输层改造，`03` §9.8.2 / §9.8.3）：
              - **主路径**：`streamablehttp_client(url)` 连**常驻 HTTP 服务**（随 bootstrap 拉起），
                无状态请求、天然并发、只付一次冷启动；
              - **回滚路径**：`BILLAGENT_MCP_STDIO=1`（config.MCP_STDIO_FALLBACK）时切回旧
                **stdio 短连接**（M3 回滚要求），并显式 env=dict(os.environ) 透传完整环境变量
                （M1 发现的 mcp SDK 默认过滤 BILLAGENT_* 问题），子进程握手超时 120s；
              - **删除全局锁**：HTTP 服务天然并发，客户端无需 `asyncio.Lock`。
            两条路径均在本次调用同一 task 内 async with，先 session.initialize() 握手再
            call_tool；成功统一取 `resp.content[0].text`（CallToolResult 的 content 为列表，
            业务层永远拿到 str）。
            异常分流与重试：非连接类异常立即抛出、不重试；连接类异常（`_is_connection_error`）
            在尝试次数耗尽前继续循环，全败则抛 ConnectionError 并链上最后一次异常。
            写副作用工具（如 sql.write）由 MCPAdapter 传 max_retry=0——"INSERT 已落库但连接
            随后断开"时若重试会再执行一次造成重复记账（D2 要防的核心场景）。
        入参说明：
            server_key (str)：目标 server 标识，取值 sql_bill / llm_base / llm_finance；
                同时用于取 SERVER_CMD_MAP 的 stdio 启动命令与 HTTP 端口。
            tool_name (str)：远端工具名，如 exec_sql / llm_chat / finance_chat。
            arguments (dict)：投递给远端工具的业务参数（agent_id / run_id 已由适配器注入）。
            max_retry (int | None)：连接类异常场景下的**总尝试次数上限（含首次）**，代码以
                max(1, retry) 归一、下限为 1。★M1 新增该形参：None → 沿用 MAX_CONNECT_RETRY
                （=2，即首次失败后最多再试 1 次）；0 → 归一为 1 次，一次即止不重试
                （写副作用工具走此值）。
        返回值说明：
            str：远端工具首条 content 的 text 字段，业务层据此按字符串解析。
                失败形态：非连接类异常原样抛出；连接类异常在尝试耗尽后抛
                ConnectionError(f"{server_key} 连接失败，已尝试 N 次")。
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
        """
        函数功能与逻辑描述：
            兼容性停机钩子（调用方：`smoke_test.py`）：无状态请求模型下本类不持有
            任何长连接或子进程，故**无资源需要释放**，本方法只往 stderr 打一行提示；
            3 个常驻 MCP 服务进程由 startup/bootstrap.shutdown_system 统一关闭，本方法不负责杀进程。
            方法保留以兼容既有调用方。
        入参说明：
            无。
        返回值说明：
            无（仅写 stderr 日志，无副作用）。
        """
        print("MCP客户端已标记关闭（无状态请求，无连接需释放）", file=sys.stderr)

    # ===================== 远端工具清单（供 registry.verify_schemas 自检） =====================
    async def list_server_tools(self, server_key: str) -> list[dict]:
        """
        函数功能与逻辑描述：
            拉取指定 MCP server 的 `tools/list`，并把每个工具归一为 name / description /
            inputSchema 三字段的 dict（远端 schema 原始形态，供与本地 ToolSpec 契约比对）。
            M3 传输分流：默认连**常驻 HTTP 服务**（需 bootstrap 已拉起）；MCP_STDIO_FALLBACK=1
            时走 stdio 短连接（建 StdioServerParameters，握手超时 120s）。两路均先
            session.initialize() 再 list_tools。
            ⚠️ **不进启动路径**：每次调用都要建连握手（stdio 回滚路径还需拉起子进程），
            仅供 `registry.verify_schemas()` 在 CI / 手动自检时调用，以免拖慢冷启动。
        入参说明：
            server_key (str)：server 标识，取值 sql_bill / llm_base / llm_finance。
        返回值说明：
            list[dict]：每项含 name / description / inputSchema；server 无工具时为空列表 []。
        """
        cmd = SERVER_CMD_MAP[server_key]
        if MCP_STDIO_FALLBACK:
            # creationflags 为 Windows 专有语义，非 Windows 平台传 0
            #   （同 bootstrap.start_mcp_servers 的处理）
            params = StdioServerParameters(
                command=cmd[0], args=cmd[1:], timeout=120,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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
        """
        函数功能与逻辑描述：
            遍历 SERVER_CMD_MAP 中的全部 MCP server，逐个调 `list_server_tools` 后顺序
            extend 成扁平列表（MCPAdapter.list_remote_tools 据此做 schema 漂移自检）。
            串行遍历、不做并发与去重；任一 server 抛异常即中断并向上抛出，不吞异常。
        入参说明：
            无。
        返回值说明：
            list[dict]：全部 server 工具定义的合并列表（已扁平化，不含 server_key 分组）。
        """
        out: list[dict] = []
        for key in SERVER_CMD_MAP:
            out.extend(await self.list_server_tools(key))
        return out

    # ===================== 对外业务接口 =====================
    async def call_bill_sql(self, agent_id: str, sql: str, params: list = None):
        """
        函数功能与逻辑描述：
            账单数据库 SQL 调用兼容入口（★M1 改造）：**签名与返回类型完全不变**，内部不再
            直连 server，而是转调工具平台执行引擎——鉴权 / 参数校验 / 超时 / 审计全部由引擎
            统一治理（见 `08` §1.9），现有生产调用与测试 patch 因此零改动（当前生产侧 6 处
            调用：agents/ 下 3 处 + evaluation/ 下 3 处；tests/ 下 49 处引用）。
            SQL 分流：`sql.strip().upper().startswith("SELECT")` → 走只读工具 `sql.read`
            （required_perm=DBPerm.BILL_READ），否则走 `sql.write`（DBPerm.BILL_WRITE，
            side_effect=True → 引擎层与适配器层均不自动重试，防 D2 重复记账）。
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
        入参说明：
            agent_id (str)：调用方 Agent 标识，作为鉴权主体与审计主体交由引擎处理。
            sql (str)：待执行 SQL；去掉首尾空白并转大写后以 "SELECT" 开头走读工具，否则走写工具。
            params (list | None)：SQL 占位符参数列表，默认 None，入引擎前归一为 []。
        返回值说明：
            str：`ToolResult.unwrap()` 成功时返回的工具结果文本。
                失败形态：unwrap 抛原生异常以保住旧契约——无权限 → PermissionError、
                参数非法 → ValueError、超时 → TimeoutError，其余为远端/Runtime 异常。
        """
        tool = "sql.read" if sql.strip().upper().startswith("SELECT") else "sql.write"
        res = await get_registry().invoke(
            tool, {"sql": sql, "params": params or []}, agent_id=agent_id)
        return res.unwrap()      # 成功返回 str；失败抛原生异常（无权限→PermissionError）

    async def call_llm_base(self, sys_prompt: str, user_text: str, agent_tag: str = "unknown"):
        """
        函数功能与逻辑描述：
            通用基座模型调用入口，编排器与各业务 Agent 共用。
            ★M1 改造：内部转调工具 `llm.chat`（经 Registry + ExecutionEngine），
            签名与返回类型保持不变；`agent_id` 由适配器映射为远端 `agent_tag` 参数
            （差异化推理参数，见踩坑记录坑 12）。日志分三段打点：
            入口（提示词长度 + 用户输入）、出口（模型原始返回）、异常（含完整堆栈）。
            ⚠️ **`agent_tag` 必须传真实 Agent 标识**：它同时是鉴权主体
            （`llm.chat` 的 required_perm = LLMPerm.LLM_BASE）。
            沿用默认值 `"unknown"` 会因无任何权限被网关拒绝 —— 这是 M1 的**治理目标而非缺陷**：
            改造前 call_llm_base 完全不校验权限，网关形同虚设。
            4 个业务 Agent 均已显式传自身标识，主链路不受影响。
        入参说明：
            sys_prompt (str)：系统提示词。
            user_text (str)：用户输入原文。
            agent_tag (str)：调用方 Agent 标识，既作为鉴权主体，也作为模型侧差异化推理参数；
                默认 "unknown"（无权限，会被网关拒绝）。
        返回值说明：
            str：模型生成的文本（unwrap 成功路径）。
            失败形态：unwrap 抛原生异常以保住旧契约——无权限 → PermissionError、
                参数非法 → ValueError、超时 → TimeoutError，其余为远端/Runtime 异常；
                本函数捕获异常后仅补打错误日志（含堆栈），随后**原样重新抛出**，不做降级或吞异常。
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

    async def call_llm_fc(self, agent_tag: str, sys_prompt: str, user_text: str,
                          tools: list = None, extra_messages: list = None) -> dict:
        """
        函数功能与逻辑描述：
            **FC 通道**（function calling）的基座模型调用入口（M15 新增），供 bill_agent 的
            目标消解循环使用。与 `call_llm_base` 的关系：同一个基座后端、同一条网关路径
            （`llm.chat_fc` → 引擎鉴权/超时/打点 → MCP server `llm_chat_fc` → Router），
            差别只在**返回形态**——本方法返回**结构化 dict**（正文 + 工具调用），
            因为 FC 场景下模型可能"只调工具、不产出正文"，纯文本契约承载不了 tool_calls。
            ★为什么不改 `call_llm_base`/`llm.chat`：那是 18+ 处生产调用与 ~40 处测试 patch 依赖的
            纯文本契约，按参数分叉返回形态会让契约二义（`设计/12` §4.3 的取舍）；新增工具则可
            单独下线回退（`registry.unregister("llm.chat_fc")`）。
            `tools` 由调用方经 `registry.to_openai_schema(agent_id, tags=...)` 取得——仍走网关，
            不新增直连 provider 的旁路（`09` §0.3 原则 2）。
            错误处理：server 侧沿用 `FINANCE_ERROR:` 前缀字符串协议承载业务失败，本方法识别该前缀
            后抛 RuntimeError；返回非法 JSON 同样抛 RuntimeError——两种都属"通道不可用"，由上层
            （循环）决定降级或转追问，而不是静默返回空 tool_calls 让模型"凭空想象"。
        入参说明：
            agent_tag (str)：调用方 Agent 标识，既是鉴权主体（`llm.chat_fc` 要求 LLMPerm.LLM_BASE），
                也会被适配器映射为远端 `agent_tag` 用于差异化推理参数与埋点归因。
            sys_prompt (str)：系统提示词。
            user_text (str)：用户输入原文。
            tools (list | None)：function-calling 工具清单（OpenAI 格式），默认 None；
                为空时不向请求体加 tools 字段（等价于普通对话）。
            extra_messages (list | None)：多轮 tool 消息（assistant 的 tool_calls 与 tool 结果回填），
                默认 None；由 server 侧按 system → user → extra 顺序组装。
        返回值说明：
            dict：固定两键——`text`（str，模型正文，可能为空串）与
                `tool_calls`（list | None，归一形态 `[{"id","name","arguments":dict}]`）；
                无工具调用时为 None（循环据此判定"该结束了"）。
        异常说明：
            RuntimeError：server 返回 `FINANCE_ERROR:` 前缀（模型侧业务失败）或返回体不是合法 JSON。
            其余异常（PermissionError / ValueError / TimeoutError / ConnectionError 等）
                由 `unwrap()` 按既有契约抛出并原样向上传播。
        """
        args: Dict = {"system_prompt": sys_prompt, "user_input": user_text}
        if tools:
            args["tools"] = tools
        if extra_messages:
            args["extra_messages"] = extra_messages
        logger.info(f"[LLM_FC_IN] agent={agent_tag}, tools={len(tools or [])} 个, "
                    f"extra_messages={len(extra_messages or [])} 条")
        try:
            res = await get_registry().invoke("llm.chat_fc", args, agent_id=agent_tag)
            raw = res.unwrap()
            if isinstance(raw, str) and raw.startswith("FINANCE_ERROR"):
                raise RuntimeError(raw)
            if isinstance(raw, dict):
                data = raw
            else:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else {}
                except (ValueError, TypeError) as e:
                    raise RuntimeError(f"llm.chat_fc 返回非法 JSON：{str(raw)[:200]}") from e
            out = {"text": str((data or {}).get("text") or ""),
                   "tool_calls": (data or {}).get("tool_calls") or None}
            logger.info(f"[LLM_FC_OUT] agent={agent_tag}, "
                        f"tool_calls={[c.get('name') for c in (out['tool_calls'] or [])]}")
            return out
        except Exception as e:
            err_stack = traceback.format_exc()
            logger.error(f"[LLM_FC_ERROR] agent={agent_tag}, FC调用失败, err={str(e)}, stack={err_stack}")
            raise

    async def call_llm_finance(self, agent_id: str, sys_prompt: str, user_text: str):
        """
        函数功能与逻辑描述：
            理财专属模型调用入口。★M1 改造：权限校验交由执行引擎
            （`llm.finance` 的 required_perm = LLMPerm.LLM_FINANCE），审计由引擎统一记录；
            签名、返回类型与**无权限时抛 PermissionError 的行为**均保持不变。
            同样按入口 / 出口 / 异常三段打日志，异常补打完整堆栈后原样重抛。
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
        入参说明：
            agent_id (str)：调用方 Agent 标识，是鉴权主体（须持 LLMPerm.LLM_FINANCE）。
            sys_prompt (str)：理财助手系统提示词。
            user_text (str)：用户输入原文。
        返回值说明：
            str：理财模型生成的文本（unwrap 成功路径）。
            失败形态：unwrap 抛原生异常以保持旧契约——无权限 → PermissionError、
                超时 → TimeoutError、参数非法 → ValueError，其余为远端/Runtime 异常；
                本函数捕获后仅补打错误日志（含堆栈），随后原样重新抛出。
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
