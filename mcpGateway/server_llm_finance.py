# 强制统一编码，根治Windows stdio管道 GBK/UTF-8 乱码报错
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

import anyio
from mcp.types import TextContent
from mcp.server.fastmcp import FastMCP
from config.config import (
    FINANCE_LORA_MODEL, EXTERNAL_FINANCE_MODEL,
    MCP_HOST, MCP_SERVER_PORTS, MCP_HTTP_PATH, MCP_STDIO_FALLBACK,
)
# M14（D17）：finance 通道 Router = prefer_external（外部可用即外部，不查 EXTERNAL_LLM_AGENTS——
#   `08` §8.6 决策表未覆盖的通道差异）+ allow_fallback（外部失败回退本地 QLoRA 产物，`08` §8.7）
from modelService.chat_model import ChatError
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider
from modelService.router import ModelRouter
from mcpGateway.rbac_config import get_agent_permission, LLMPerm
from mcpGateway.audit_recorder import record_operate

# 初始化MCP服务（M3：常驻 HTTP，`03` §9.8.2）
finance_mcp = FastMCP(
    "llm-finance-mcp",
    host=MCP_HOST,
    port=MCP_SERVER_PORTS["llm_finance"],
    streamable_http_path=MCP_HTTP_PATH,
)

# finance 通道：外部强模型优先，失败自动回退本地（原 L56-86 的 if/else + startswith fallback 收口进 Router）
_finance_router = ModelRouter(
    local=OllamaProvider(FINANCE_LORA_MODEL),
    external=OpenAICompatProvider(EXTERNAL_FINANCE_MODEL),
    prefer_external=True,
    allow_fallback=True,
)


@finance_mcp.tool()
async def finance_chat(agent_id: str, system_prompt: str, user_input: str,
                       run_id: str | None = None):
    """
    函数功能与逻辑描述：
        理财专属模型通道的 MCP 工具入口，流程为「RBAC 校验 → 线程池推理 → 审计落盘」：
        1) 先查权限表确认 agent_id 持有 LLMPerm.LLM_FINANCE，否则记一条失败审计并返回权限不足；
        2) 通过后把同步推理丢进 anyio 线程池执行，避免阻塞事件循环；模型路由交由
        _finance_router 收口（prefer_external=True + allow_fallback=True，即外部强模型优先、
        失败回退本地 QLoRA 产物，重试与埋点也在 Router 内统一处理）；
        3) 无论推理成功或抛 ChatError，都记一条 success=True 的审计并返回文本结果。
        若传入 run_id，则先绑定到本进程 tracing context，使 record_llm_call 能自动归因到该 run。
    入参说明：
        agent_id (str)：调用方 Agent 标识，既用于权限校验，也作为模型调用的归因标签（agent_tag）。
        system_prompt (str)：系统提示词，定义理财助手的角色与输出约束。
        user_input (str)：用户输入原文，同时作为审计内容留痕。
        run_id (str | None)：链路追踪 ID，由引擎经参数注入（不在 params_schema 中暴露以防伪造）；
            默认 None 表示不参与归因。
    返回值说明：
        list[TextContent]：长度为 1 的文本结果列表
            - 权限不足："权限不足：无理财模型调用权限"。
            - 正常：模型生成文本；推理失败时为 "FINANCE_ERROR: {错误信息}"（上游按此前缀判定失败）。
    """
    # M7（D3 阶段2）：run_id 由引擎注入 → 本进程 context → record_llm_call 归因（同 llm_base）
    if run_id:
        from utils.tracer import bind_run_id
        bind_run_id(run_id)
    # 1. 权限校验
    perms = get_agent_permission(agent_id)
    if LLMPerm.LLM_FINANCE not in perms:
        record_operate(agent_id, LLMPerm.LLM_FINANCE.value, user_input, False)
        return [TextContent(type="text", text="权限不足：无理财模型调用权限")]

    # 2. 同步推理丢入异步线程，避免阻塞事件循环
    # M14（D17）：外部优先 + 失败回退本地 + 统一重试/埋点全在 Router（prefer_external + allow_fallback）
    # M7（D3）：finance 通道归因标签沿用同一主体（financial_agent）
    try:
        # ★2026-09-17 修复（真链路实测缺陷，finance 全通道不可用）：
        #   anyio 的 `run_sync(func, *args)` **只把位置参数透传给 func**，写成关键字会被当作
        #   run_sync 自身的参数 → 抛 `run_sync() got an unexpected keyword argument 'system'`，
        #   于是整条 finance 调用失败，用户收到
        #   「理财分析完成：Error executing tool finance_chat: run_sync() got an unexpected …」。
        #   与 `sql_common/sql_mcp_base.py` 同口径，改用 functools.partial 承载关键字参数。
        from functools import partial
        resp = await anyio.to_thread.run_sync(
            partial(_finance_router.generate,
                    system=system_prompt, user=user_input, agent_tag=agent_id)
        )
        text = resp.text
    except ChatError as err:
        text = f"FINANCE_ERROR: {err.message}"

    # 3. 写入审计日志
    record_operate(agent_id, LLMPerm.LLM_FINANCE.value, user_input, True)
    return [TextContent(type="text", text=text)]

if __name__ == "__main__":
    # ★M4（D8 沙箱）：进程 self-prison（POSIX RLIMIT_AS/RLIMIT_CORE；Windows no-op，由 bootstrap Job 兜底）
    from mcpGateway.sandbox import apply_process_limits
    apply_process_limits()
    if MCP_STDIO_FALLBACK:
        # ★M3 回滚：BILLAGENT_MCP_STDIO=1 时 server 走 stdio（与客户端回滚开关配对）
        print("【理财MCP服务】已启动（stdio 回滚模式），当前使用理财专属模型", file=sys.stderr)
        finance_mcp.run()
    else:
        print(f"【理财MCP服务】已启动 http://{MCP_HOST}:{MCP_SERVER_PORTS['llm_finance']}{MCP_HTTP_PATH}，当前使用理财专属模型", file=sys.stderr)
        finance_mcp.run(transport="streamable-http")
