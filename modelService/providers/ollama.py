"""M14（D17）本地 Provider：Ollama（/api/chat）。

职责边界（`08` §8.5）：组装请求体 → POST → 响应归一为 ChatResult / ChatError。
- 快速失败、**不重试**（本地单次 ~189s，重试价值低、延迟灾难，`08` 7.2.2 原则3）；
- 越界参数由 filter_options 过滤 + warning（R10），绝不转译。
"""
import time
from typing import Dict, Optional

import requests

from config.config import (
    OLLAMA_API_URL, OLLAMA_MODEL_NAME, MODEL_TIMEOUT, OLLAMA_CONTEXT_WINDOW,
)
from modelService.chat_model import (
    Capabilities, ChatError, ChatModel, ChatResult, filter_options,
)

# 本地 Ollama 认识的参数（契约边界 = 原 LOCAL_OPTION_WHITELIST，M14 收口进 capabilities）
_SUPPORTED = frozenset({
    "temperature", "num_predict", "num_ctx",
    "top_p", "top_k", "repeat_penalty", "stop",
})


class OllamaProvider(ChatModel):
    """本地 Ollama 后端（1.8B QLoRA 微调产物 / qwen-bill-1.5-1.8b-q4km）。"""

    def __init__(self, model: Optional[str] = None,
                 timeout: Optional[int] = None):
        self._model = model or OLLAMA_MODEL_NAME
        self._timeout = timeout if timeout is not None else MODEL_TIMEOUT
        self._capabilities = Capabilities(
            backend="ollama",
            supported_params=frozenset(_SUPPORTED),
            reasoning=False,
            context_window=OLLAMA_CONTEXT_WINDOW,   # 对齐 num_ctx=4096
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

        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        use_timeout = timeout if timeout is not None else self._timeout
        _t0 = time.perf_counter()
        try:
            resp = requests.post(url=OLLAMA_API_URL, json=payload,
                                 timeout=use_timeout)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = int((time.perf_counter() - _t0) * 1000)
            content = (data.get("message") or {}).get("content") or ""
            text = str(content).strip()
            # Ollama 以 prompt_eval_count / eval_count 计量（非严格 token，作等价指标）
            return ChatResult(
                text=text,
                model=self._model,
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
                total_tokens=(int(data.get("prompt_eval_count") or 0)
                              + int(data.get("eval_count") or 0)),
                latency_ms=latency_ms,
                channel="local",
            )
        except requests.exceptions.Timeout:
            raise ChatError(
                "模型请求超时，请检查Ollama服务状态或调大超时时间",
                retryable=False, error_type="Timeout")
        except requests.exceptions.HTTPError as http_err:
            detail = ""
            if http_err.response is not None and http_err.response.text:
                detail = f" Ollama返回详情：{http_err.response.text}"
            raise ChatError(
                f"HTTP异常{str(http_err)}{detail}",
                # 本地快速失败不重试（Router 侧 attempt 预算 1）
                retryable=False, error_type=f"HTTP_{http_err.response.status_code}"
                if http_err.response is not None else "HTTPError")
        except ChatError:
            raise
        except Exception as e:  # noqa: BLE001 —— 网络层兜底，统一进 ChatError
            raise ChatError(f"模型调用异常：{str(e)}",
                            retryable=False, error_type=type(e).__name__)
