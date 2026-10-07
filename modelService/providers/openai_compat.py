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
    Capabilities, ChatError, ChatModel, ChatResult, build_messages,
    filter_options, normalize_tool_calls,
)

# 外部 API 认识的参数（契约边界 = 原 EXTERNAL_OPTION_WHITELIST，M14 收口进 capabilities）
_SUPPORTED = frozenset({"temperature", "max_tokens"})

# ★2026-09-15（M15）：外部通道**显式直连**——本项目不需要代理支持（单机应用 + 国内 API）。
#   背景：requests 默认会读环境变量**以及 Windows 注册表的系统代理**（urllib.getproxies），
#   一旦本机留有失效的代理残留（如 Clash 异常退出留下的 127.0.0.1:7892，端口已无人监听），
#   所有外部调用都会以 ProxyError/10061 失败，且报错与"模型不可用"难以区分。
#   trust_env=False → 同时忽略环境变量与注册表代理，网络行为确定、不依赖运行环境。
#   ⚠️ 若部署环境**必须**经代理出网，请在此显式传 proxies（不要退回依赖隐式系统代理）。
_SESSION = requests.Session()
_SESSION.trust_env = False

# 可重试（占池）：瞬时/服务端/限流/连接 —— 进 Router 统一预算池
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
# 不可重试（短路）：鉴权/配额/参数/权限 —— 重试只烧钱
NON_RETRYABLE_HTTP_STATUS = frozenset({400, 401, 402, 403, 404})
# reasoning 模型 max_tokens 下限（M1 实测 400 会被 thinking 挤空 content → 取 512 兜底告警）
REASONING_MIN_OUTPUT_BUDGET: int = 512


def parse_usage(data: Optional[dict]) -> Dict[str, int]:
    """
    函数功能与逻辑描述：
        解析 OpenAI 兼容协议的 usage 字段，是 token 埋点的唯一解析出口
        （原 llm_loader._record_external_usage 的逻辑，M14 收口于此，单测沿用）。
        除常规 prompt/completion 外，额外提取 reasoning_tokens
        （位于 usage.completion_tokens_details.reasoning_tokens，仅推理模型返回）。
        total_tokens 的取值优先级：**尊重服务端原始 usage.total_tokens**（当其显式存在时），
        否则回退为 prompt + completion + reasoning 之和——因为部分服务商的 total
        并不等于三段之和，强行自算会导致埋点与账单口径不一致。
        对 data 为 None、usage 缺失、details 缺失等情形全部按 0 兜底，不抛异常。
    入参说明：
        data (Optional[dict])：外部 API 的完整 JSON 响应体；None 或结构缺失时各项按 0 处理。
    返回值说明：
        Dict[str, int]：固定四键 —— prompt_tokens、completion_tokens、
            reasoning_tokens（非推理模型恒为 0）、total_tokens；全部为非负整数。
    """
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
    """
    函数/类功能与逻辑描述：
        外部 OpenAI 兼容后端实现（deepseek / glm-4-flash 等）。
        负责参数过滤、reasoning 预算告警、组装 {base_url}/chat/completions 请求体、
        把响应归一为 ChatResult，并做**可重试性分类**（异常分类是 T5 的结构化落点，
        Router 依据 retryable 决定是否占重试预算）。
        ❌ 不决定走哪个后端（路由在 ModelRouter）；❌ 不做跨端参数转译；❌ 不处理 fallback。
    构造入参说明：
        model (Optional[str])：模型名，默认 None 表示空串（由调用方按所选模型显式指定）。
        reasoning (bool)：是否推理模型（thinking 与 content 共享 max_tokens 预算）。
            按所选模型配置传入，不写死——默认 True（当前 hy3/DeepSeek 均为推理模型，`08` §8.10 R2）。
        context_window (int)：所选模型上下文窗口，保守默认 8k（EXTERNAL_CONTEXT_WINDOW），
            可用环境变量 BILLAGENT_EXT_CONTEXT_WINDOW 覆盖。
        timeout (Optional[int])：请求超时秒数，默认 None 表示取 EXTERNAL_LLM_TIMEOUT。
        base_url (Optional[str])：API 基址，默认 None 表示取 EXTERNAL_LLM_BASE_URL；
            构造时会 rstrip("/") 以免拼出双斜杠路径。
        api_key (Optional[str])：API Key，默认 None 表示取 EXTERNAL_LLM_API_KEY。
    返回值说明：
        构造返回 OpenAICompatProvider 实例；核心能力见 generate。
    """

    def __init__(self, model: Optional[str] = None, *,
                 reasoning: bool = True,
                 context_window: int = EXTERNAL_CONTEXT_WINDOW,
                 timeout: Optional[int] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        """
        函数功能与逻辑描述：
            初始化外部后端配置：解析模型名、推理标记、超时、base_url（去掉尾部斜杠）
            与 api_key（均支持「显式传参优先、否则回落 config」），并构建能力声明 Capabilities
            （后端标识 openai-compat、支持参数集 {temperature, max_tokens}、
            reasoning 标记、上下文窗口）。构造阶段不发任何网络请求。
            reasoning=False 时后端内部会关闭 max_tokens 下限告警。
        入参说明：
            model (Optional[str])：模型名；None 时置空串，由调用方保证后续显式指定。
            reasoning (bool)：是否推理模型，默认 True。
            context_window (int)：上下文窗口，默认 EXTERNAL_CONTEXT_WINDOW。
            timeout (Optional[int])：超时秒数；None 时回落 EXTERNAL_LLM_TIMEOUT。
            base_url (Optional[str])：API 基址；None 时回落 EXTERNAL_LLM_BASE_URL。
            api_key (Optional[str])：API Key；None 时回落 EXTERNAL_LLM_API_KEY。
        返回值说明：
            无（仅初始化实例属性，不含副作用）。
        """
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
        """
        函数功能与逻辑描述：
            暴露本后端的能力声明（只读属性），供 Router 判定参数是否可下发与上下文窗口是否足够；
            其中 reasoning 标记也会决定 generate 内部是否启用 max_tokens 下限告警。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Capabilities：构造时构建的能力对象。
        """
        return self._capabilities

    @property
    def model(self) -> str:
        """
        函数功能与逻辑描述：
            暴露当前模型名（只读属性），供 Router 埋点与 llm_call_stat 记录；
            未显式指定时为空串。
        入参说明：
            无（隐式 self）。
        返回值说明：
            str：模型名，可能为空串（表示由调用方在别处指定）。
        """
        return self._model

    def generate(self, *, system: str, user: str, options: Dict,
                 agent_tag: str, timeout: Optional[int] = None,
                 tools: Optional[list] = None,
                 extra_messages: Optional[list] = None) -> ChatResult:
        """
        函数功能与逻辑描述：
            执行一次外部推理：① filter_options 按契约边界过滤参数（越界只告警不转译）；
            ② reasoning 模型下校验 max_tokens 是否低于 REASONING_MIN_OUTPUT_BUDGET（512），
            过低则 warning——这是从结构上杜绝「thinking 挤空 content」的护栏
            （M1 实测 400 会被吃空）；③ 组装 messages（system 为空或纯空白时不追加）
            与 payload，POST {base_url}/chat/completions，stream=False；
            ④ 解析 choices[0].message.content，用 parse_usage 取用量，
            channel 固定 "external"。
            异常分类是本函数的重点：请求成功但 content 为空视为**可重试**
            （error_type=EMPTY_CONTENT，通常是 reasoning 吃满预算，重试有实际收益）；
            HTTP 状态码按 RETRYABLE_HTTP_STATUS（429/5xx）与 NON_RETRYABLE_HTTP_STATUS
            （400/401/402/403/404）分流；连接错误可重试（网络瞬时抖动）；
            超时不可重试（超时阈值已足够长，重试代价高）；未知异常不可重试（不赌未知可重试）。
            agent_tag 本端不使用，仅为接口一致性保留。
        入参说明：
            system (str)：系统提示词；空串或纯空白时不写入请求体。
            user (str)：用户输入原文，始终作为 user 消息写入。
            options (Dict)：待下发参数，仅 temperature 与 max_tokens 会被保留，其余过滤告警。
            agent_tag (str)：调用方 Agent 标识，本端不参与推理参数差异化。
            timeout (Optional[int])：本次调用的超时覆盖值；None 表示用构造时的超时。
            tools (Optional[list])：function-calling 工具清单（OpenAI 格式），默认 None。
                ★它是**协议结构**（与 messages 同级），不放进 options、也不走 filter_options（`设计/12` §4.3）。
            extra_messages (Optional[list])：追加消息（tool 循环的 assistant tool_calls 与 tool 结果），
                经 `build_messages` 按 system → user → extra 的顺序组装，默认 None。
        返回值说明：
            ChatResult：含 text（已 strip）、model、prompt_tokens、completion_tokens、
                reasoning_tokens、total_tokens、latency_ms、channel="external"、
                tool_calls（归一为 `[{"id","name","arguments":dict}]`，无调用时为 None）。
                ★`text` 与 `tool_calls` 至少有一个非空；**有 tool_calls 时 text 可为空串**
                （FC 的正常形态，不算 EMPTY_CONTENT）。
        异常说明：
            ChatError：所有失败统一收敛，retryable 取决于错误类别。
                error_type 取值：EMPTY_CONTENT（空内容，可重试）、Timeout（不可重试）、
                HTTP_{状态码}（按状态码是否在可重试集合中决定）、ConnectionError（可重试）、
                其它异常类名（不可重试）。上游 ChatError 原样透传。
        """
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

        messages = build_messages(system, user, extra_messages)

        payload = {"model": self._model, "messages": messages,
                   "stream": False}
        if tools:
            # ★tools 是**协议结构**（与 messages 同级），不是调参项：
            #   既不塞进 options/opts，也绝不进 filter_options 白名单（M14 铁律，`设计/12` §4.3）。
            payload["tools"] = tools
        payload.update(opts)
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        use_timeout = timeout if timeout is not None else self._timeout
        _t0 = time.perf_counter()
        try:
            resp = _SESSION.post(url=url, json=payload, headers=headers,
                                 timeout=use_timeout)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = int((time.perf_counter() - _t0) * 1000)
            message: dict = {}
            try:
                message = data["choices"][0]["message"] or {}
            except (KeyError, IndexError, TypeError):
                message = {}
            text = str(message.get("content") or "").strip()
            tool_calls = normalize_tool_calls(message.get("tool_calls"))
            usage = parse_usage(data)
            if not text and not tool_calls:
                # 请求成功但**既无正文也无工具调用**：可重试（进 Router 预算池）——通常是
                # reasoning 吃满预算。★M15：判定条件已从"text 为空"改为"text 与 tool_calls 双空"，
                # 因为 FC 场景下模型本就用 tool_calls 代替自然语言回复，此时 text 为空是**正常**的。
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
                tool_calls=tool_calls or None,
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
