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
    """模型路由（隔离 + 契约 + 路由；**绝不转译**跨端参数）。

    :param local: 本地后端（OllamaProvider，免费基线——不配 API 也能跑，`01` §7）
    :param external: 外部后端（OpenAICompatProvider，强模型）
    :param prefer_external: finance 类通道差异——**外部可用即外部**（不查 EXTERNAL_LLM_AGENTS），
        失败回退本地。base 类通道（orchestrator 等按 EXTERNAL_LLM_AGENTS 路由）为 False。
    :param allow_fallback: 外部失败是否允许回退本地（仅 finance 等"外部优先、本地兜底"通道；
        DISABLE_LOCAL_LLM=1 时自动失效）。
    """

    def __init__(self, local: ChatModel, external: ChatModel, *,
                 prefer_external: bool = False,
                 allow_fallback: bool = False):
        self._local = local
        self._external = external
        self._prefer_external = prefer_external
        self._allow_fallback = allow_fallback

    # ------------------------------------------------------------------
    def resolve(self, agent_tag: str) -> Tuple[ChatModel, Dict]:
        """开关集中判断（`08` §8.6 决策表），返回 (后端, agent 差异化 options)。

        决策表：
        1. DISABLE_LOCAL_LLM=1        → 外部（本地被封；外部也未启用 = 配置矛盾 → 立即抛错暴露）
        2. prefer_external（finance） → 外部可用即外部，否则本地（不查 EXTERNAL_LLM_AGENTS）
        3. FORCE_EXTERNAL_LLM=1 且已启用 → 外部（联调/测试全量走强模型）
        4. 已启用且 agent_tag ∈ EXTERNAL_LLM_AGENTS → 外部（默认：主调度走外部）
        5. 其余                            → 本地（1.8B 兜底）
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
        """参数归一（非转译）：merge 后端默认参数 + agent 差异化参数。

        同义参数（temperature）两端共用同一键；异义参数（num_predict vs max_tokens）
        本就分属两端默认层，各 provider 的 capabilities 会过滤越界项并告警。
        """
        if provider.capabilities.backend == "ollama":
            merged: Dict = dict(BASE_LLM_OPTIONS)
        else:
            merged = dict(EXTERNAL_BASE_LLM_OPTIONS)
        if agent_options and isinstance(agent_options, dict):
            merged.update(agent_options)
        return merged

    def context_window_for(self, agent_tag: str) -> int:
        """当前 agent_tag 按路由决策命中的后端窗口（喂 D6 ContextManager 预算换算）。"""
        provider, _ = self.resolve(agent_tag)
        return provider.capabilities.context_window

    # ------------------------------------------------------------------
    def execute(self, provider: ChatModel, *, system: str, user: str,
                options: Dict, agent_tag: str = "unknown",
                max_attempts: int = MAX_REQUEST_ATTEMPTS,
                timeout: Optional[int] = None) -> ChatResult:
        """单后端点对点执行：统一重试池 + **统一埋点**（成功/失败/每次重试都记）。

        :param max_attempts: 重试预算（含首次）。本地快速失败传 1；外部可重试传 3。
        返回 ChatResult；不可重试 / 预算耗尽 → 抛 ChatError。
        """
        channel = BACKEND_CHANNEL.get(provider.capabilities.backend, "local")
        last_error: Optional[ChatError] = None
        for attempt in range(max_attempts):
            _t0 = time.perf_counter()
            try:
                result = provider.generate(
                    system=system, user=user, options=options,
                    agent_tag=agent_tag, timeout=timeout)
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
                 options: Optional[Dict] = None) -> ChatResult:
        """统一入口（server 层调用）：resolve 决策 → 归一参数 → 执行 → 可选 fallback。

        fallback（`08` §8.7）：外部失败且允许（allow_fallback && 非本地封禁）→ 本地兜底一次。
        """
        provider, agent_options = self.resolve(agent_tag)
        full_options = (options if options is not None
                        else self.build_options(provider, agent_options))
        attempts = (MAX_REQUEST_ATTEMPTS
                    if provider.capabilities.backend == "openai-compat" else 1)
        try:
            return self.execute(provider, system=system, user=user,
                                options=full_options, agent_tag=agent_tag,
                                max_attempts=attempts)
        except ChatError:
            if (self._allow_fallback and not DISABLE_LOCAL_LLM
                    and provider.capabilities.backend != "ollama"):
                # 外部失败 → 回退本地（仅"外部优先、本地兜底"通道；本地被封禁不回退）
                local_options = self.build_options(
                    self._local, AGENT_LLM_CONFIG.get(agent_tag))
                return self.execute(self._local, system=system, user=user,
                                    options=local_options, agent_tag=agent_tag,
                                    max_attempts=1)
            raise


# ── 模块级便捷入口（base 语义：按 EXTERNAL_LLM_AGENTS 决策，窗口查询/轻量场景用）──
# Provider 构造为纯对象（不连网、不加载模型），import 无副作用。
_default_router = ModelRouter(
    local=OllamaProvider(),
    external=OpenAICompatProvider(EXTERNAL_ORCH_MODEL),
)


def context_window_for(agent_tag: str) -> int:
    """当前 agent_tag 按 base 路由决策命中后端的上下文窗口（D6 ContextManager 预算换算）。

    context_manager 的默认窗口 4096 只代表本地；orchestrator_agent ∈ EXTERNAL_LLM_AGENTS
    默认走外部（8k+），分层预算按实际窗口换算，而非写死（`08` §8.3 铁律 3）。
    """
    return _default_router.context_window_for(agent_tag)


def _record(agent_tag: str, model: str, channel: str,
            prompt_tokens: int, completion_tokens: int, reasoning_tokens: int,
            total_tokens: int, latency_ms: int, success: bool,
            error_type: Optional[str]) -> None:
    """统一埋点入口（消除 llm_loader 双 helper）。可观测失败不得阻断主链路——吞一切异常。"""
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
