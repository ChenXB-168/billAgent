# 文件顶部强制统一编码，根治Windows GBK/UTF8冲突
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
import json
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
    """
    函数功能与逻辑描述：
        通用基座模型通道的 MCP 工具入口，把请求交给模块级 _base_router 统一决策：
        agent_tag 命中 EXTERNAL_LLM_AGENTS 则走外部模型，否则走本地 Ollama 基座；
        参数归一、重试与埋点均在 ModelRouter 内部完成，本函数不做分支判断
        （M14/D17 路由集中化，决策表见 `08` §8.6 / modelService/router.py）。
        若传入 run_id，则先绑定到本进程 tracing context，供 record_llm_call 自动归因。
        本工具为同步函数（非 async），由 MCP 框架安排执行；不写审计日志（与 finance 通道不同）。
        抛出的 RuntimeError 不在此捕获，用于暴露「禁用本地模型且外部未启用」这类配置矛盾。
    入参说明：
        system_prompt (str)：系统提示词，定义模型角色与输出约束。
        user_input (str)：用户输入原文。
        agent_tag (str)：调用方 Agent 标签，既参与外部/本地路由判定，也作为埋点归因标签；
            默认 "unknown"。
        run_id (str | None)：链路追踪 ID，由引擎注入（不进 params_schema 以防 LLM 伪造）；
            默认 None 表示不参与归因。
    返回值说明：
        str：模型生成的文本；推理失败时返回 "FINANCE_ERROR: {错误信息}" 形式的错误串
            （错误字符串化统一收口在最外层，业务层按前缀判断语义不变）。
    """
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


@llm_mcp.tool()
def llm_chat_fc(system_prompt: str, user_input: str,
                tools: list | None = None, extra_messages: list | None = None,
                agent_tag: str = "unknown", run_id: str | None = None):
    """
    函数功能与逻辑描述：
        **FC 通道**（M15）：与 `llm_chat` 同后端、同路由，但支持 function-calling 并以
        **JSON 字符串**返回结构化结果——`{"text": <正文>, "tool_calls": [{"id","name","arguments"}...]}`。
        为什么另开一个工具而不是给 `llm_chat` 加参数：后者的返回契约是**纯文本 str**，
        18+ 处生产调用与 ~40 处测试 patch 依赖它；按参数分叉返回形态会让契约二义
        （`设计/12` §4.3 的取舍），且无法单独回退。本工具可独立下线。
        返回必须是 str（MCP content 只取第一条 text，客户端再 `json.loads` 还原结构）——
        这是"结构化结果过 MCP 文本边界"的既定形态，与错误串 `FINANCE_ERROR:` 前缀协议共存：
        客户端先按前缀判错、再按 JSON 解析，两条路径互不干扰。
        路由与重试仍在 `_base_router` 内统一（含 fallback / 埋点），本函数不做分支判断。
    入参说明：
        system_prompt (str)：系统提示词。
        user_input (str)：用户输入原文。
        tools (list | None)：function-calling 工具清单（OpenAI 格式），默认 None；
            由调用方经网关 `registry.to_openai_schema(agent_id, tags=...)` 取得。
        extra_messages (list | None)：多轮 tool 消息（assistant tool_calls / tool 结果回填），默认 None。
        agent_tag (str)：调用方 Agent 标签，参与路由判定与埋点归因，默认 "unknown"。
        run_id (str | None)：链路追踪 ID，由引擎注入（不进 params_schema 以防 LLM 伪造），默认 None。
    返回值说明：
        str：JSON 字符串 `{"text": str, "tool_calls": list | None}`；模型侧业务失败时返回
            `"FINANCE_ERROR: {错误信息}"` 前缀串（与 llm_chat 同一协议）。
    """
    if run_id:
        from utils.tracer import bind_run_id
        bind_run_id(run_id)
    try:
        result = _base_router.generate(
            system=system_prompt, user=user_input, agent_tag=agent_tag,
            tools=tools, extra_messages=extra_messages)
        return json.dumps({"text": result.text, "tool_calls": result.tool_calls},
                          ensure_ascii=False)
    except ChatError as err:
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
