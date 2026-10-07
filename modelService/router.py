"""M14（D17）路由层：ModelRouter —— 开关集中 + 统一重试/fallback + 统一埋点。

设计出处：`08` 第 8 章 §8.6/§8.7/§8.8。本文件为执行时行为权威（回写 `10` §5.2.7）。

- 路由集中：三个开关（EXTERNAL_LLM_ENABLED / FORCE_EXTERNAL_LLM / DISABLE_LOCAL_LLM）**只在此处判断**，
  替代散落在 server_llm_base / server_llm_finance 的 if/else（`08` §8.6 决策表）。
- 统一重试（§8.7）：可重试（429/5xx/连接/空 content）共用一个 attempt 预算池（默认 3 次）；
  不可重试（401/400/…/Timeout/未知）立即短路；每次尝试**单独埋点**（重试是真实计费）。
- fallback（§8.7）：外部失败 → 可选回退本地，**仅** prefer_external + allow_fallback 的通道（finance）；
  DISABLE_LOCAL_LLM=1 时**禁止回退**（本地被封死）；本地失败不回退外部（免费基线，回退不解决根因）。
- 统一埋点（§8.8）：消除 llm_loader 的 _record_local_usage / _record_external_usage 双 helper，
  全部经 utils.tracer.record_llm_call 单点记账（可观测失败不阻断主链路——埋点自身吞异常）。
- context_window_for：供 D6 ContextManager 按**实际所选后端窗口**换算（`08` §8.3 铁律3）。
"""
import time
from typing import Dict, Optional, Tuple

from config.config import (
    EXTERNAL_LLM_ENABLED, EXTERNAL_LLM_AGENTS,
    FORCE_EXTERNAL_LLM, DISABLE_LOCAL_LLM,
    BASE_LLM_OPTIONS, AGENT_LLM_CONFIG,
    EXTERNAL_BASE_LLM_OPTIONS, AGENT_EXTERNAL_LLM_CONFIG,
    EXTERNAL_ORCH_MODEL,
)
from modelService.chat_model import ChatError, ChatModel, ChatResult
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider

# 统一重试预算池（含首次；本地快速失败不重试 = 1）
MAX_REQUEST_ATTEMPTS: int = 3
RETRY_BACKOFF_BASE: float = 1.0
# fallback / 单点重试的场景标识
BACKEND_CHANNEL = {"ollama": "local", "openai-compat": "external"}


class ModelRouter:
    """
    函数/类功能与逻辑描述：
        模型路由层，负责「隔离 + 契约 + 路由」，**绝不跨端转译参数**。
        把三个路由开关的判定集中到 resolve 一处，把重试预算池与埋点集中到 execute 一处，
        把可选 fallback 集中到 generate 一处，从而让各 MCP server 只剩一行调用。
        通道差异由构造参数表达：base 类通道（orchestrator 等按 EXTERNAL_LLM_AGENTS 路由）用默认值，
        finance 类通道（外部优先、本地兜底）传 prefer_external=True + allow_fallback=True。
    构造入参说明：
        local (ChatModel)：本地后端（OllamaProvider）。免费基线——不配 API Key 也能跑（`01` §7）。
        external (ChatModel)：外部后端（OpenAICompatProvider，强模型）。
        prefer_external (bool)：finance 类通道差异——**外部可用即外部**（不查 EXTERNAL_LLM_AGENTS），
            失败回退本地；base 类通道为 False，默认 False。
        allow_fallback (bool)：外部失败是否允许回退本地（仅"外部优先、本地兜底"通道需要）；
            DISABLE_LOCAL_LLM=1 时自动失效，默认 False。
    返回值说明：
        构造返回 ModelRouter 实例；对外能力见 resolve / build_options / context_window_for / generate。
    """

    def __init__(self, local: ChatModel, external: ChatModel, *,
                 prefer_external: bool = False,
                 allow_fallback: bool = False):
        """
        函数功能与逻辑描述：
            保存两个后端与两个通道开关，不做连通性检查、不加载任何模型
            （Provider 构造本身就是纯对象，因此模块级可安全实例化）。
        入参说明：
            local (ChatModel)：本地后端实例。
            external (ChatModel)：外部后端实例。
            prefer_external (bool)：是否外部优先，默认 False。
            allow_fallback (bool)：是否允许外部失败后回退本地，默认 False。
        返回值说明：
            无（仅初始化实例属性）。
        """
        self._local = local
        self._external = external
        self._prefer_external = prefer_external
        self._allow_fallback = allow_fallback

    # ------------------------------------------------------------------
    def resolve(self, agent_tag: str) -> Tuple[ChatModel, Dict]:
        """
        函数功能与逻辑描述：
            路由决策的唯一出口（`08` §8.6 决策表），按优先级自上而下短路命中，返回
            「选中的后端 + 该 agent 在该后端的差异化参数」。决策表：
            1. DISABLE_LOCAL_LLM=1 → 外部；若外部也未启用则说明配置矛盾，
               立即抛 RuntimeError 暴露问题（config.py L86 语义：走到本地 = 路由缺陷）；
            2. prefer_external（finance）→ 外部可用即外部，否则本地（不查 EXTERNAL_LLM_AGENTS）；
            3. FORCE_EXTERNAL_LLM=1 且已启用 → 外部（联调/测试全量走强模型）；
            4. 已启用且 agent_tag ∈ EXTERNAL_LLM_AGENTS → 外部（默认：主调度走外部）；
            5. 其余 → 本地（1.8B 兜底）。
            参数取自 AGENT_EXTERNAL_LLM_CONFIG（外部通道）或 AGENT_LLM_CONFIG（本地通道），
            查不到时返回空字典而非 None，保证调用方可直接 update。
        入参说明：
            agent_tag (str)：调用方 Agent 标识，是决策表第 3/4 步的判定依据，
                也是取差异化参数配置的键。
        返回值说明：
            Tuple[ChatModel, Dict]：二元组 (选中的后端实例, 该 agent 的差异化参数)。
                参数为已解引用后的字典，可能为空字典。
        异常说明：
            RuntimeError：当 DISABLE_LOCAL_LLM=1 且 EXTERNAL_LLM_ENABLED 为假时抛出，
                文案以 "[M14]" 开头并提示检查环境变量（配置矛盾，无法路由到任何可用后端）。
        """
        if DISABLE_LOCAL_LLM:
            # config.py L86 语义：走到本地 = 路由缺陷
            if not EXTERNAL_LLM_ENABLED:
                raise RuntimeError(
                    "[M14] DISABLE_LOCAL_LLM=1 但外部 API 未启用（缺 base_url/api_key）——"
                    "本地已被封死，配置矛盾，请检查环境变量")
            return self._external, (AGENT_EXTERNAL_LLM_CONFIG.get(agent_tag) or {})
        if self._prefer_external:
            if EXTERNAL_LLM_ENABLED:
                return self._external, (AGENT_EXTERNAL_LLM_CONFIG.get(agent_tag) or {})
            return self._local, (AGENT_LLM_CONFIG.get(agent_tag) or {})
        if FORCE_EXTERNAL_LLM and EXTERNAL_LLM_ENABLED:
            return self._external, (AGENT_EXTERNAL_LLM_CONFIG.get(agent_tag) or {})
        if EXTERNAL_LLM_ENABLED and agent_tag in EXTERNAL_LLM_AGENTS:
            return self._external, (AGENT_EXTERNAL_LLM_CONFIG.get(agent_tag) or {})
        return self._local, (AGENT_LLM_CONFIG.get(agent_tag) or {})

    def build_options(self, provider: ChatModel, agent_options: Optional[Dict]) -> Dict:
        """
        函数功能与逻辑描述：
            参数归一（**非转译**）：以所选后端的默认参数层为底
            （ollama 用 BASE_LLM_OPTIONS，其它用 EXTERNAL_BASE_LLM_OPTIONS），
            再叠加 agent 差异化参数。同义参数（temperature）两端共用同一键；
            异义参数（num_predict vs max_tokens）本就分属两端默认层，
            跨端越界项由各 provider 的 capabilities 在 generate 时过滤并告警，此处不做映射。
            返回新字典，不修改入参 agent_options，也不修改 config 中的默认参数字典。
        入参说明：
            provider (ChatModel)：已选定的后端，据其 capabilities.backend 决定默认参数层。
            agent_options (Optional[Dict])：agent 差异化参数；None 或非 dict 时跳过叠加。
        返回值说明：
            Dict：合并后的完整参数集（可直接作为 options 传入 execute）。
        """
        if provider.capabilities.backend == "ollama":
            merged: Dict = dict(BASE_LLM_OPTIONS)
        else:
            merged = dict(EXTERNAL_BASE_LLM_OPTIONS)
        if agent_options and isinstance(agent_options, dict):
            merged.update(agent_options)
        return merged

    def context_window_for(self, agent_tag: str) -> int:
        """
        函数功能与逻辑描述：
            查询某 agent 按当前路由决策实际命中的后端上下文窗口，
            供 D6 ContextManager 做分层预算换算（`08` §8.3 铁律3：预算必须按**实际窗口**换算，
            不能写死 4096——orchestrator_agent 默认走外部，窗口远大于本地）。
            内部复用 resolve，因此与真实调用路径使用完全相同的决策逻辑，不会出现口径偏差。
        入参说明：
            agent_tag (str)：调用方 Agent 标识。
        返回值说明：
            int：命中后端的上下文窗口（token 数）。
        异常说明：
            RuntimeError：当 resolve 判定配置矛盾（DISABLE_LOCAL_LLM=1 且外部未启用）时透传。
        """
        provider, _ = self.resolve(agent_tag)
        return provider.capabilities.context_window

    # ------------------------------------------------------------------
    def execute(self, provider: ChatModel, *, system: str, user: str,
                options: Dict, agent_tag: str = "unknown",
                max_attempts: int = MAX_REQUEST_ATTEMPTS,
                timeout: Optional[int] = None,
                tools: Optional[list] = None,
                extra_messages: Optional[list] = None) -> ChatResult:
        """
        函数功能与逻辑描述：
            单后端点对点执行：统一重试池 + **统一埋点**。每轮尝试都单独记一次 llm_call_stat
            （成功记真实 token 与耗时，失败记 0 token 与错误类型），因为重试是真实计费，
            只记最终结果会漏计成本。重试决策只依据 ChatError.retryable：
            不可重试或已耗尽预算（attempt >= max_attempts - 1）立即重新抛出；
            可重试则按 RETRY_BACKOFF_BASE * 2^attempt 指数退避后再试。
            注意本方法用**同步 time.sleep**退避（Router 整体是同步实现），
            由调用方置于线程池中执行以避免阻塞事件循环。
            循环末尾的 raise 是防御性兜底（理论上不可达，因为循环内必 return 或 raise）。
        入参说明：
            provider (ChatModel)：已选定的后端实例（本方法不做路由）。
            system (str)：系统提示词。
            user (str)：用户输入原文。
            options (Dict)：已归一的参数集。
            agent_tag (str)：调用方 Agent 标识，用于埋点归因，默认 "unknown"。
            max_attempts (int)：重试预算**含首次**，默认 MAX_REQUEST_ATTEMPTS（3）；
                本地快速失败场景应显式传 1。
            timeout (Optional[int])：本次调用的超时覆盖值，默认 None。
            tools (Optional[list])：function-calling 工具清单，**原样透传**给 Provider——
                不 merge、不转译、更不放进 options（协议结构与调参项必须分开，`设计/12` §4.3）。
            extra_messages (Optional[list])：tool 循环的追加消息，同样原样透传。
        返回值说明：
            ChatResult：后端成功返回的结果（埋点已按成功记账）。
        异常说明：
            ChatError：不可重试或预算耗尽时抛出最后一轮的错误对象（失败埋点已记账）。
            ChatError("模型调用失败（未知）")：仅防御性兜底路径可能产生，
                使 last_error 为空时仍能抛出明确错误。
        """
        channel = BACKEND_CHANNEL.get(provider.capabilities.backend, "local")
        last_error: Optional[ChatError] = None
        for attempt in range(max_attempts):
            _t0 = time.perf_counter()
            try:
                result = provider.generate(
                    system=system, user=user, options=options,
                    agent_tag=agent_tag, timeout=timeout,
                    tools=tools, extra_messages=extra_messages)
                _record(agent_tag, result.model, channel,
                        result.prompt_tokens, result.completion_tokens,
                        result.reasoning_tokens, result.total_tokens,
                        result.latency_ms, True, None)
                return result
            except ChatError as err:
                last_error = err
                _record(agent_tag, getattr(provider, "model", "") or "",
                        channel, 0, 0, 0, 0,
                        int((time.perf_counter() - _t0) * 1000),
                        False, err.error_type)
                if not err.retryable or attempt >= max_attempts - 1:
                    raise
                time.sleep(RETRY_BACKOFF_BASE * (2 ** attempt))
        # 理论不可达（for 内必 return / raise），防御性兜底
        raise last_error or ChatError("模型调用失败（未知）",
                                      retryable=False, error_type="Unknown")

    # ------------------------------------------------------------------
    def generate(self, *, system: str, user: str, agent_tag: str,
                 options: Optional[Dict] = None,
                 tools: Optional[list] = None,
                 extra_messages: Optional[list] = None) -> ChatResult:
        """
        函数功能与逻辑描述：
            对外统一入口（由 MCP server 层调用），串起完整链路：
            resolve 决策后端 → 归一参数 → execute 执行 → 可选 fallback。
            fallback（`08` §8.7）仅在三个条件同时满足时触发：allow_fallback 为真、
            DISABLE_LOCAL_LLM 为假（本地未被封禁）、且当前后端不是 ollama（即确实是从外部失败回落）；
            回退时重新按本地方案 build_options，并把预算收敛为 1 次
            （本地是免费基线，重试不解决根因）。外部通道预算为 MAX_REQUEST_ATTEMPTS，
            本地通道为 1（快速失败）。若显式传入 options，则完全采用它、跳过 build_options
            （给调用方一个精确控制参数的逃生口）。
        入参说明：
            system (str)：系统提示词。
            user (str)：用户输入原文。
            agent_tag (str)：调用方 Agent 标识，驱动路由决策与埋点归因。
            options (Optional[Dict])：显式指定的完整参数集；None（默认）表示由
                build_options 按选定后端与 agent 差异化配置自动合成。
            tools (Optional[list])：function-calling 工具清单（M15），默认 None；原样透传，
                对既有调用零影响（不传即完全等价于 M14 形态）。
            extra_messages (Optional[list])：tool 循环的追加消息（M15），默认 None；原样透传。
        返回值说明：
            ChatResult：成功时的模型结果；若外部失败且允许 fallback，则为本地兜底的结果。
        异常说明：
            ChatError：不允许 fallback（或 fallback 也失败）时抛出外部失败的错误对象；
            RuntimeError：resolve 判定配置矛盾时透传。
        """
        provider, agent_options = self.resolve(agent_tag)
        full_options = (options if options is not None
                        else self.build_options(provider, agent_options))
        attempts = (MAX_REQUEST_ATTEMPTS
                    if provider.capabilities.backend == "openai-compat" else 1)
        try:
            return self.execute(provider, system=system, user=user,
                                options=full_options, agent_tag=agent_tag,
                                max_attempts=attempts,
                                tools=tools, extra_messages=extra_messages)
        except ChatError:
            if (self._allow_fallback and not DISABLE_LOCAL_LLM
                    and provider.capabilities.backend != "ollama"):
                # 外部失败 → 回退本地（仅"外部优先、本地兜底"通道；本地被封禁不回退）
                local_options = self.build_options(
                    self._local, AGENT_LLM_CONFIG.get(agent_tag))
                return self.execute(self._local, system=system, user=user,
                                    options=local_options, agent_tag=agent_tag,
                                    max_attempts=1,
                                    tools=tools, extra_messages=extra_messages)
            raise


# ── 模块级便捷入口（base 语义：按 EXTERNAL_LLM_AGENTS 决策，窗口查询/轻量场景用）──
# Provider 构造为纯对象（不连网、不加载模型），import 无副作用。
_default_router = ModelRouter(
    local=OllamaProvider(),
    external=OpenAICompatProvider(EXTERNAL_ORCH_MODEL),
)


def context_window_for(agent_tag: str) -> int:
    """
    函数功能与逻辑描述：
        模块级便捷入口：查询某 agent 按 **base 语义路由决策**（即按 EXTERNAL_LLM_AGENTS 判定，
        非 finance 的 prefer_external 语义）命中后端的上下文窗口，供 D6 ContextManager 做预算换算。
        之所以需要它：context_manager 里写死的默认窗口 4096 只代表本地后端，
        而 orchestrator_agent ∈ EXTERNAL_LLM_AGENTS，默认走外部（8k+），
        分层预算必须按实际窗口换算而非写死（`08` §8.3 铁律 3）。
        委托给模块级 _default_router，与运行期真实链路共用同一套决策逻辑。
    入参说明：
        agent_tag (str)：调用方 Agent 标识。
    返回值说明：
        int：命中后端的上下文窗口（token 数）。
    异常说明：
        RuntimeError：DISABLE_LOCAL_LLM=1 且外部未启用（配置矛盾）时由 resolve 抛出并透传。
    """
    return _default_router.context_window_for(agent_tag)


def _record(agent_tag: str, model: str, channel: str,
            prompt_tokens: int, completion_tokens: int, reasoning_tokens: int,
            total_tokens: int, latency_ms: int, success: bool,
            error_type: Optional[str]) -> None:
    """
    函数功能与逻辑描述：
        统一埋点入口（消除 llm_loader 的 _record_local_usage / _record_external_usage 双 helper），
        把一次模型调用记入 llm_call_stat，完成 §8.8 的单点记账。
        ★可观测失败**绝不阻断主链路**：整个记账过程被 try/except 包裹并吞掉一切异常
        （tracer 依赖数据库、context 绑定等外部条件，任一条件不满足都不应影响业务返回）。
        `from utils.tracer import record_llm_call` 刻意放在函数内做延迟导入，
        避免 modelService 在 import 期就把 tracer 及其依赖链拉起。
        所有计数字段统一 int() 转换并做 None 兜底，避免上游传 None 导致记账表写入失败。
    入参说明：
        agent_tag (str)：调用方 Agent 标识；空串会被归为 "unknown"。
        model (str)：模型名，失败路径下可能为空串。
        channel (str)：通道标识，取值 local / external（由 BACKEND_CHANNEL 映射得到）。
        prompt_tokens (int)：输入 token 数；失败路径传 0。
        completion_tokens (int)：输出 token 数；失败路径传 0。
        reasoning_tokens (int)：推理 token 数（非推理模型为 0）；失败路径传 0。
        total_tokens (int)：总 token 数；失败路径传 0。
        latency_ms (int)：本次尝试耗时毫秒（每次重试单独计）。
        success (bool)：本次尝试是否成功。
        error_type (Optional[str])：失败时的错误类型（如 Timeout / HTTP_429）；成功时为 None。
    返回值说明：
        无（仅写 llm_call_stat；任何失败都被静默吞掉，不抛出、不返回错误）。
    """
    try:
        from utils.tracer import record_llm_call
        record_llm_call(
            agent_tag=agent_tag or "unknown",
            model=model,
            channel=channel,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
            total_tokens=int(total_tokens or 0),
            latency_ms=int(latency_ms or 0),
            success=success,
            error_type=error_type,
        )
    except Exception:  # noqa: BLE001 —— 可观测层绝不阻断主链路
        pass
