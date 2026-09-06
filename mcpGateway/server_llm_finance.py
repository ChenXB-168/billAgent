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
    理财专属模型调用接口
    带RBAC权限校验 + 全链路审计日志
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
        resp = await anyio.to_thread.run_sync(
            _finance_router.generate,
            system=system_prompt,
            user=user_input,
            agent_tag=agent_id,
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
