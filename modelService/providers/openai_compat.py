"""M14（D17）外部 Provider：OpenAI 兼容协议（POST {base_url}/chat/completions）。

职责边界（`08` §8.5）：
- reasoning 预算告警：reasoning=True 且 max_tokens < 下限 → warning（R10 不静默）——
  reasoning 与 content 共享 max_tokens 预算，过小会被 thinking 挤空 content（M1 踩坑根源）；
- 异常分类（T5 结构化落点）：可重试（429/5xx/连接/空 content → retryable=True，进 Router 预算池）
  vs 不可重试（401/402/400/403/404/Timeout/未知 → retryable=False，短路）；
- usage 解析（含 reasoning_tokens）收口为模块函数 parse_usage——原 llm_loader._record_external_usage
  的解析逻辑（M14 验收#4：双埋点 helper 消除）。
"""
import time
from typing import Dict, Optional

import requests

from config.config import (
    EXTERNAL_LLM_BASE_URL, EXTERNAL_LLM_API_KEY,
    EXTERNAL_LLM_TIMEOUT, EXTERNAL_CONTEXT_WINDOW,
)
from modelService.chat_model import (
    Capabilities, ChatError, ChatModel, ChatResult, filter_options,
)

# 外部 API 认识的参数（契约边界 = 原 EXTERNAL_OPTION_WHITELIST，M14 收口进 capabilities）
_SUPPORTED = frozenset({"temperature", "max_tokens"})

# 可重试（占池）：瞬时/服务端/限流/连接 —— 进 Router 统一预算池
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
# 不可重试（短路）：鉴权/配额/参数/权限 —— 重试只烧钱
NON_RETRYABLE_HTTP_STATUS = frozenset({400, 401, 402, 403, 404})
# reasoning 模型 max_tokens 下限（M1 实测 400 会被 thinking 挤空 content → 取 512 兜底告警）
REASONING_MIN_OUTPUT_BUDGET: int = 512


def parse_usage(data: Optional[dict]) -> Dict[str, int]:
    """OpenAI 兼容 usage 解析（含 completion_tokens_details.reasoning_tokens）。
    返回 {prompt_tokens, completion_tokens, reasoning_tokens, total_tokens}——
    total 优先取原始 usage.total_tokens（≠ prompt+completion 时尊重服务端口径）。
    原 llm_loader._record_external_usage 的解析逻辑（M14 收口于此，单测沿用）。"""
    usage = (data or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(details.get("reasoning_tokens") or 0)
    total = int(usage["total_tokens"]) if usage.get("total_tokens") is not None \
        else (prompt + completion + reasoning)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
    }


class OpenAICompatProvider(ChatModel):
    """外部 OpenAI 兼容后端（deepseek / glm-4-flash 等）。

    :param reasoning: 是否推理模型（thinking 与 content 共享 max_tokens 预算）。按所选模型配置传入，
        不写死——默认 True（当前 hy3/DeepSeek 均为推理模型，`08` §8.10 R2）。
    :param context_window: 所选模型上下文窗口（保守默认 8k，可用 BILLAGENT_EXT_CONTEXT_WINDOW 覆盖）。
    """

    def __init__(self, model: Optional[str] = None, *,
                 reasoning: bool = True,
                 context_window: int = EXTERNAL_CONTEXT_WINDOW,
                 timeout: Optional[int] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._model = model or ""
        self._reasoning = reasoning
        self._timeout = timeout if timeout is not None else EXTERNAL_LLM_TIMEOUT
        self._base_url = (base_url if base_url is not None
                          else EXTERNAL_LLM_BASE_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else EXTERNAL_LLM_API_KEY
        self._capabilities = Capabilities(
            backend="openai-compat",
            supported_params=frozenset(_SUPPORTED),
            reasoning=reasoning,
            context_window=context_window,
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def model(self) -> str:
        return self._model

    def generate(self, *, system: str, user: str, options: Dict,
                 agent_tag: str, timeout: Optional[int] = None) -> ChatResult:
        # 1. 参数过滤（契约边界）：越界 → warning（R10 不静默），绝不转译
        opts = filter_options(options, self._capabilities.supported_params,
                              self._capabilities.backend)

        # 2. reasoning 预算告警（结构上杜绝"thinking 挤空 content"）
        if self._reasoning:
            mt = opts.get("max_tokens")
            if mt is not None and int(mt) < REASONING_MIN_OUTPUT_BUDGET:
                import logging
                logging.getLogger(__name__).warning(
                    "[M14] reasoning 模型 max_tokens=%d < 下限 %d——"
                    "thinking 与 content 共享预算，过小会被挤空 content（R10 不静默）",
                    int(mt), REASONING_MIN_OUTPUT_BUDGET)

        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": user})

        payload = {"model": self._model, "messages": messages,
                   "stream": False}
        payload.update(opts)
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        use_timeout = timeout if timeout is not None else self._timeout
        _t0 = time.perf_counter()
        try:
            resp = requests.post(url=url, json=payload, headers=headers,
                                 timeout=use_timeout)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = int((time.perf_counter() - _t0) * 1000)
            content = ""
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                content = ""
            text = (content or "").strip()
            usage = parse_usage(data)
            if not text:
                # 请求成功但内容为空：可重试（进 Router 预算池）——通常是 reasoning 吃满预算
                raise ChatError(
                    "外部API返回空 content（可能 reasoning 吃满 max_tokens 预算）",
                    retryable=True, error_type="EMPTY_CONTENT")
            return ChatResult(
                text=text,
                model=self._model,
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                total_tokens=usage["total_tokens"],
                latency_ms=latency_ms,
                channel="external",
            )
        except ChatError:
            raise
        except requests.exceptions.Timeout:
            # 不可重试：300s 已足够长，重试大概率仍超时，代价高价值低
            raise ChatError("外部API请求超时，请检查网络或调大EXTERNAL_LLM_TIMEOUT",
                            retryable=False, error_type="Timeout")
        except requests.exceptions.HTTPError as http_err:
            status = (http_err.response.status_code
                      if http_err.response is not None else None)
            detail = ""
            if http_err.response is not None and http_err.response.text:
                detail = f" 外部API返回详情：{http_err.response.text}"
            raise ChatError(
                f"外部API HTTP异常{str(http_err)}{detail}",
                retryable=(status in RETRYABLE_HTTP_STATUS),
                error_type=f"HTTP_{status}" if status else "HTTPError")
        except requests.exceptions.ConnectionError:
            # 可重试：网络瞬时抖动，落池重试
            raise ChatError("外部API连接失败（网络瞬时抖动？）",
                            retryable=True, error_type="ConnectionError")
        except Exception as e:  # noqa: BLE001 —— 网络层兜底
            # 未知异常：不重试（不赌未知异常可重试）
            raise ChatError(f"外部API调用异常：{str(e)}",
                            retryable=False, error_type=type(e).__name__)
