# 文件顶部强制统一编码，根治Windows GBK/UTF8冲突
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from mcp.server.fastmcp import FastMCP
from config.config import (
    OLLAMA_MODEL_NAME, EXTERNAL_ORCH_MODEL,
    MCP_HOST, MCP_SERVER_PORTS, MCP_HTTP_PATH, MCP_STDIO_FALLBACK,
)
# M14（D17）：路由集中——本服务不再散落 use_external if/else，统一走 ModelRouter.resolve
# （开关 DISABLE/FORCE/AGENTS 决策表见 `08` §8.6 / `modelService/router.py`）
from modelService.chat_model import ChatError
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider
from modelService.router import ModelRouter

# M3：常驻 HTTP 服务（`03` §9.8.2）——host/port 由 config 统一定义，随 bootstrap 拉起
llm_mcp = FastMCP(
    "llm-base-server",
    host=MCP_HOST,
    port=MCP_SERVER_PORTS["llm_base"],
    streamable_http_path=MCP_HTTP_PATH,
)

# base 通道 Router：标准路由（orchestrator_agent ∈ EXTERNAL_LLM_AGENTS → 外部；其余本地）
_base_router = ModelRouter(
    local=OllamaProvider(OLLAMA_MODEL_NAME),
    external=OpenAICompatProvider(EXTERNAL_ORCH_MODEL),
)


@llm_mcp.tool()
def llm_chat(system_prompt: str, user_input: str, agent_tag: str = "unknown",
             run_id: str | None = None):
    """通用大模型对话工具"""
    # M7（D3 阶段2）：run_id 由引擎经 binding["inject_run_id"] 注入（不进 params_schema，
    #    防 LLM 伪造）→ 写**本进程** context → record_llm_call 自动归因，见 `11` §3 M7
    if run_id:
        from utils.tracer import bind_run_id
        bind_run_id(run_id)
    # M14（D17）：模型路由 = resolve 决策表（外部/本地/参数归一/重试/埋点全在 Router 收口）；
    #   FORCE_EXTERNAL_LLM=1（测试/联调）时**所有**Agent一律走外部，绕开本地CPU推理；
    #   未配 API 时自动回退本地，链路不中断。
    try:
        result = _base_router.generate(
            system=system_prompt, user=user_input, agent_tag=agent_tag)
        return result.text
    except ChatError as err:
        # MCP 边界字符串化（错误字符串化留最外层，`08` §8.9）；业务层 startswith 判断语义不变
        return f"FINANCE_ERROR: {err.message}"
    # RuntimeError（如 DISABLE_LOCAL_LLM 且外部未启用 = 配置矛盾）向上抛，暴露配置问题

if __name__ == "__main__":
    # ★M4（D8 沙箱）：进程 self-prison（POSIX RLIMIT_AS/RLIMIT_CORE；Windows no-op，由 bootstrap Job 兜底）
    from mcpGateway.sandbox import apply_process_limits
    apply_process_limits()
    if MCP_STDIO_FALLBACK:
        # ★M3 回滚：BILLAGENT_MCP_STDIO=1 时 server 走 stdio（与客户端回滚开关配对）
        print("[LLM MCP 基座服务启动成功]（stdio 回滚模式）", file=sys.stderr)
        llm_mcp.run()
    else:
        print(f"[LLM MCP 基座服务启动成功] http://{MCP_HOST}:{MCP_SERVER_PORTS['llm_base']}{MCP_HTTP_PATH}", file=sys.stderr)
        llm_mcp.run(transport="streamable-http")
