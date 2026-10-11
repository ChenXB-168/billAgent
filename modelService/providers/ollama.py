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
    Capabilities, ChatError, ChatModel, ChatResult, build_messages,
    filter_options, normalize_tool_calls,
)

# 本地 Ollama 认识的参数（契约边界 = 原 LOCAL_OPTION_WHITELIST，M14 收口进 capabilities）
_SUPPORTED = frozenset({
    "temperature", "num_predict", "num_ctx",
    "top_p", "top_k", "repeat_penalty", "stop",
})


class OllamaProvider(ChatModel):
    """
    函数/类功能与逻辑描述：
        本地 Ollama 后端实现（1.8B QLoRA 微调产物 / qwen-bill-1.5-1.8b-q4km）。
        只负责三件事：过滤越界参数、组装 /api/chat 请求体、把响应归一为 ChatResult / ChatError。
        ❌ 不决定走哪个后端（路由在 ModelRouter）；❌ 不做跨端参数转译；❌ 不处理 fallback。
        支持参数集 `_SUPPORTED` 即本后端的契约边界，越界参数会被丢弃并告警（绝不转译）。
    构造入参说明：
        model (Optional[str])：模型名，默认 None 表示取 config.OLLAMA_MODEL_NAME。
        timeout (Optional[int])：请求超时秒数，默认 None 表示取 config.MODEL_TIMEOUT。
    返回值说明：
        构造返回 OllamaProvider 实例；核心能力见 generate。
    """

    def __init__(self, model: Optional[str] = None,
                 timeout: Optional[int] = None):
        """
        函数功能与逻辑描述：
            初始化本地后端配置：解析模型名与超时（均支持「显式传参优先、否则回落 config」），
            并构建本端能力声明 Capabilities（后端标识 ollama、支持参数集、reasoning=False、
            上下文窗口对齐 num_ctx=4096）。构造阶段不发任何网络请求，不做连通性检查。
            注意：timeout 用 `is not None` 判定而非 `or`，因此显式传 0 不会被 config 覆盖
            （0 会被 requests 解释为不超时，需由调用方自行保证语义）。
        入参说明：
            model (Optional[str])：模型名；None 时回落 OLLAMA_MODEL_NAME。
            timeout (Optional[int])：请求超时秒数；None 时回落 MODEL_TIMEOUT。
        返回值说明：
            无（仅初始化实例属性）。
        """
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
        """
        函数功能与逻辑描述：
            暴露本后端的能力声明（只读属性），供 Router 在派发前判定参数是否可下发、
            上下文窗口是否足够，避免把 Ollama 不认识的参数传下去。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Capabilities：构造时构建的能力对象（后端标识 / 支持参数集 / reasoning 标记 / 上下文窗口）。
        """
        return self._capabilities

    @property
    def model(self) -> str:
        """
        函数功能与逻辑描述：
            暴露当前实际使用的模型名（只读属性），供 Router 埋点与 llm_call_stat 记录使用。
        入参说明：
            无（隐式 self）。
        返回值说明：
            str：模型名（构造时已按「显式传参优先、否则 config」解析完毕）。
        """
        return self._model

    def generate(self, *, system: str, user: str, options: Dict,
                 agent_tag: str, timeout: Optional[int] = None,
                 tools: Optional[list] = None,
                 extra_messages: Optional[list] = None) -> ChatResult:
        """
        函数功能与逻辑描述：
            执行一次本地推理：① 先用 filter_options 按本端契约边界过滤 options，
            越界参数只告警（R10 不静默）而不转译；② 组装 messages（system 为空或全空白时
            不追加 system 消息，避免下发空 system 干扰小模型）；③ POST {OLLAMA_API_URL}，
            stream=False 一次性取回；④ 归一为 ChatResult，token 计量取 Ollama 口径的
            prompt_eval_count / eval_count（非严格 token，作为等价指标），channel 固定 "local"。
            全过程不重试：所有失败路径统一抛 ChatError 且 retryable=False
            （本地单次耗时长，重试延迟代价远大于收益）。agent_tag 本后端不使用，仅为接口一致而保留。
        入参说明：
            system (str)：系统提示词；空串或纯空白时不写入请求体。
            user (str)：用户输入原文，始终作为 user 消息写入。
            options (Dict)：待下发参数（temperature / num_predict / num_ctx / top_p / top_k /
                repeat_penalty / stop 等）；越界项会在过滤阶段被丢弃并告警。
            agent_tag (str)：调用方 Agent 标识，本端不参与推理参数差异化，仅保留签名一致性。
            timeout (Optional[int])：本次调用的超时覆盖值；None 表示用构造时的超时。
            tools (Optional[list])：function-calling 工具清单（OpenAI 格式），默认 None。
                ★协议结构，不走 options 白名单；本地 1.8B 是否真支持 FC 由实测决定，
                透传本身不改变既有行为（不传即完全等价于 M14 形态）。
            extra_messages (Optional[list])：追加消息（tool 循环回填），顺序由 `build_messages` 统一约定。
        返回值说明：
            ChatResult：含 text（已 strip 的生成文本）、model、prompt_tokens、
                completion_tokens、total_tokens（prompt + completion）、latency_ms、channel="local"、
                tool_calls（归一为 `[{"id","name","arguments":dict}]`，无调用或后端不支持时为 None）。
                ⚠️ Ollama 的 tool_call **不带 id**，归一阶段按位置补 `call_{idx}`。
        异常说明：
            ChatError：所有失败统一收敛，且恒为 retryable=False。
                error_type 取值：Timeout（请求超时）、HTTP_{状态码}（HTTP 错误，带后端返回详情）、
                其它异常类名（网络层兜底）。上游 ChatError 原样透传。
        """
        # 1. 参数过滤（契约边界）：越界 → warning（R10 不静默），绝不转译
        opts = filter_options(options, self._capabilities.supported_params,
                              self._capabilities.backend)

        messages = build_messages(system, user, extra_messages)

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": opts,
        }
        if tools:
            # ★tools 是协议结构（与 messages 同级），**不进 options**——否则会被 filter_options 滤掉（M14 铁律）
            payload["tools"] = tools
        use_timeout = timeout if timeout is not None else self._timeout
        _t0 = time.perf_counter()
        try:
            resp = requests.post(url=OLLAMA_API_URL, json=payload,
                                 timeout=use_timeout)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = int((time.perf_counter() - _t0) * 1000)
            message = data.get("message") or {}
            text = str(message.get("content") or "").strip()
            tool_calls = normalize_tool_calls(message.get("tool_calls"))
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
                tool_calls=tool_calls or None,
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
