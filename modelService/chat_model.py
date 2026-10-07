"""M14（D17）模型接入层抽象：能力契约 Capabilities + ChatResult / ChatError + ChatModel(ABC)。

设计出处：`08` 第 8 章（§8.3/§8.4）。本文件为**契约权威**（`10` §5.2.7 依此收敛）。
核心判据（`08` §8.0）：适配层职责 = 隔离 + 契约 + 路由，**绝不转译**——参数适配 = 归一（同义）
+ 过滤告警（异义）；跨端参数（num_predict vs max_tokens）语义不同，禁止互转。

三条铁律（`08` §8.3）：
1. supported_params 是参数边界：Provider 只认自己认识的名字，越界 → 过滤 + warning（R10 不静默）。
2. reasoning=True 时输出预算必须覆盖 thinking + content（max_tokens 硬上限含 reasoning 消耗）。
3. context_window 供 D6 ContextManager 分层预算换算（按实际所选后端窗口，而非写死）。

关键不变量（`08` §8.4）：generate() 只返回 ChatResult 或抛 ChatError——**不再用
"FINANCE_ERROR: ..." 字符串承载错误**；错误语义由 ChatError.error_type 结构化承载，
字符串化留给最外层（兼容层 / MCP server 边界）。
"""
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capabilities:
    """
    函数/类功能与逻辑描述：
        能力契约：一个后端的「参数边界 + 推理特性 + 窗口」显式声明。适配行为完全由本契约决定
        ——这是"隔离"的结构化落点，不是靠注释/if-else 记忆；与 filter_options 协作，其
        supported_params 即参数过滤边界。frozen 数据类，实例不可变、可哈希、可跨模块共享。
    构造入参说明：
        backend (str)：后端类型标识，取值 "ollama" | "openai-compat"。
        supported_params (frozenset)：本后端认识的参数名集合（契约边界，越界即过滤+告警）。
        reasoning (bool)：是否推理模型；True 时 reasoning 与 content 共享 max_tokens 预算。
        context_window (int)：上下文窗口 token 数（喂 D6 ContextManager 分层预算换算）。
    返回值说明：
        构造返回 Capabilities 实例（不可变）；无业务方法，仅承载上述契约字段。
    """
    backend: str                      # "ollama" | "openai-compat"
    supported_params: frozenset       # 本后端认识的参数名（契约边界，越界即过滤+告警）
    reasoning: bool                   # 是否推理模型（reasoning 与 content 共享 max_tokens 预算）
    context_window: int               # 上下文窗口 token 数（喂 D6 ContextManager 预算表）


@dataclass
class ChatResult:
    """
    函数/类功能与逻辑描述：
        模型统一响应载体，屏蔽本地 / 外部后端的返回结构差异；非推理模型 reasoning_tokens 恒为 0。
        仅承载数据、无业务方法。
    构造入参说明：
        text (str)：模型生成的正文内容（已剥离 reasoning 思考）。
        model (str)：实际命中的模型名。
        prompt_tokens (int)：提示词消耗 token，默认 0。
        completion_tokens (int)：生成内容消耗 token，默认 0。
        reasoning_tokens (int)：推理模型的思考消耗，默认 0（非推理模型恒为 0）。
        total_tokens (int)：总消耗 token，默认 0。
        latency_ms (int)：请求耗时，单位毫秒，默认 0。
        channel (str)：调用通道，取值 "local" | "external"，默认空串。
        tool_calls (Optional[list])：function-calling 归一结果，默认 None；
            非 None 时为 `[{"id": str, "name": str, "arguments": dict}]`（M15）。
            ★语义提醒：**有 tool_calls 时 `text` 可以为空串**——这是 FC 的正常形态
            （模型用工具调用代替自然语言回复），不应被当作"空响应"错误。
    返回值说明：
        构造返回 ChatResult 实例；无业务方法。
    """
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0     # 推理模型的思考消耗（非推理模型恒为 0）
    total_tokens: int = 0
    latency_ms: int = 0
    channel: str = ""             # "local" | "external"
    tool_calls: Optional[list] = None   # M15：FC 归一结果 [{"id","name","arguments":dict}]


class ChatError(Exception):
    """
    函数/类功能与逻辑描述：
        统一错误类型：retryable 决定是否进入重试预算池（T5 分类的结构化落点）；error_type 承载
        结构化错误语义，业务层按类型分支，不做字符串 startswith。generate() 只返回 ChatResult
        或抛本异常，不再用 "FINANCE_ERROR: ..." 字符串承载错误。
    构造入参说明：
        message (str)：人类可读的错误描述。
        retryable (bool)：关键字参数，标识该错误是否可重试（进入重试预算池）。
        error_type (str)：关键字参数，结构化错误语义，如 "HTTP_429" / "HTTP_401" /
            "Timeout" / "EMPTY_CONTENT" / "ConnectionError" 等。
    返回值说明：
        无（构造异常实例，副作用为写入 self.message / self.retryable / self.error_type）。
    """

    def __init__(self, message: str, *, retryable: bool, error_type: str):
        """
        函数功能与逻辑描述：
            构造统一错误：先经 super().__init__(message) 把描述写入 Exception.args
            （这是 str(e) 与日志可读的前提），再把 retryable 与 error_type 挂到实例属性上。
            两个扩展字段刻意设计为**关键字参数且必填**，强制每个抛错点显式声明
            「这个错误能不能重试、属于哪一类」，避免遗漏导致重试池误判。
            不抛异常、无其它副作用。
        入参说明：
            message (str)：人类可读的错误描述。
            retryable (bool)：关键字参数，是否可重试（True → 进入 Router 的重试预算池）。
            error_type (str)：关键字参数，结构化错误语义，如 "HTTP_429" / "HTTP_401" /
                "Timeout" / "EMPTY_CONTENT" / "ConnectionError" 等。
        返回值说明：
            无（构造器，就地初始化 self.message / self.retryable / self.error_type）。
        """
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.error_type = error_type


class ChatModel(ABC):
    """
    函数/类功能与逻辑描述：
        模型后端抽象基类：业务层只认 generate()，不感知"本地还是外部"；子类必须实现
        capabilities 属性与 generate 方法，具体适配由 providers 下 Ollama / OpenAI 兼容后端落地。
    构造入参说明：
        无（抽象基类，不可直接实例化）。
    返回值说明：
        无（抽象基类，由子类实例化并实现上述契约）。
    """

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """
        函数功能与逻辑描述：
            抽象属性：暴露后端能力契约（参数边界 / 推理特性 / 窗口），供调用方在发请求前查询
            可传参数与窗口预算；具体实现由子类提供。
        入参说明：
            无。
        返回值说明：
            Capabilities：后端能力契约实例，含 backend / supported_params / reasoning /
                context_window 四字段。
        """

    @abstractmethod
    def generate(self, *, system: str, user: str,
                 options: Dict, agent_tag: str,
                 timeout: Optional[int] = None,
                 tools: Optional[list] = None,
                 extra_messages: Optional[list] = None) -> ChatResult:
        """
        函数功能与逻辑描述：
            统一调用入口。options 为**已归一**参数（调用方已合并后端默认）；实现方内部按
            capabilities.supported_params 过滤越界参数并告警（R10），绝不转译（num_predict 与
            max_tokens 语义不同，禁止互转）。只返回 ChatResult 或抛 ChatError——不返回错误字符串。
        入参说明：
            system (str)：关键字参数，系统提示词。
            user (str)：关键字参数，用户消息。
            options (Dict)：关键字参数，已归一化的模型参数（如温度、最大 token 等）。
            agent_tag (str)：关键字参数，调用方标识，用于日志与路由归因。
            timeout (Optional[int])：关键字参数，请求超时（秒）覆盖，None = 用 Provider 构造默认。
        返回值说明：
            ChatResult：统一响应对象；失败时抛 ChatError（不返回错误字符串）。
        """


def build_messages(system: str, user: str,
                   extra_messages: Optional[list] = None) -> list[dict]:
    """
    函数功能与逻辑描述：
        统一组装 messages 列表，是**消息顺序契约的唯一实现点**（M15 抽公共，防止两个 provider
        各写一份而漂移）。顺序固定为 `system → user → extra...`：
        ① system 为空或纯空白时不写入（避免空 system 干扰小模型，沿用 M14 行为）；
        ② user 恒写入（业务调用的主体输入）；
        ③ extra_messages 逐条追加——它是**多轮 tool 循环的载体**（`assistant` 的 tool_calls 消息
        与 `tool` 结果消息都从这里进来），顺序即对话顺序，不得重排、不得去重。
        非法元素（非 dict 或缺 role）静默跳过并告警，保证"模型侧收到脏消息"不会变成硬失败。
    入参说明：
        system (str)：系统提示词。
        user (str)：用户输入。
        extra_messages (Optional[list])：追加消息列表（每项为含 role 的 dict），默认 None。
    返回值说明：
        list[dict]：组装好的消息列表（每项 {"role": ..., "content": ...}）；
            system 为空且无 extra_messages 时返回仅含 user 一条的列表。
    """
    messages: list[dict] = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.append({"role": "user", "content": user})
    for msg in extra_messages or []:
        if isinstance(msg, dict) and msg.get("role"):
            messages.append(dict(msg))
        else:
            logger.warning("[M15] 忽略非法 extra_messages 元素：%r", msg)
    return messages


def normalize_tool_calls(raw) -> list[dict]:
    """
    函数功能与逻辑描述：
        把后端原生的 tool_calls 结构归一为平台统一形态 `[{"id", "name", "arguments": dict}]`
        （M15）。两个后端形态不同——OpenAI 兼容为 `{id, type, function:{name, arguments(JSON 字符串)}}`，
        Ollama 为 `{function:{name, arguments(dict)}}` 且**无 id**——归一后上层（bill_agent 循环）
        无需感知后端差异，这正是适配层"隔离"的职责。
        容错策略（宁可达上限也不硬失败）：非 dict 元素、无函数名的条目直接丢弃并告警；
        `arguments` 的 JSON 解析失败时置 `{}` 并告警——空参会被下游执行引擎的 JSON Schema
        判为参数不合规，进而作为 observation 回填给模型让它自我修正，链路不中断。
        缺 id 时按位置补 `call_{idx}`（Ollama 场景），保证循环回填 `tool` 消息时有可引用的 id。
    入参说明：
        raw：后端原生 tool_calls（通常为 list[dict]）；None / 非列表一律按空处理。
    返回值说明：
        list[dict]：归一后的工具调用列表；无有效调用时返回空列表 []（调用方据空列表判定"无 tool_calls"）。
    """
    calls: list[dict] = []
    for idx, item in enumerate(raw if isinstance(raw, list) else []):
        if not isinstance(item, dict):
            logger.warning("[M15] 忽略非法 tool_call 元素：%r", item)
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") or ""
        if not name:
            logger.warning("[M15] 忽略无函数名的 tool_call：%r", item)
            continue
        args = fn.get("arguments", item.get("arguments"))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (ValueError, TypeError):
                logger.warning("[M15] tool_call[%s] 的 arguments 非法 JSON，按空参处理：%r",
                               name, args)
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"id": str(item.get("id") or f"call_{idx}"),
                      "name": str(name), "arguments": args})
    return calls


def filter_options(options: Dict, supported: frozenset, backend: str) -> Dict:
    """
    函数功能与逻辑描述：
        按后端能力边界过滤参数：只保留 supported 内参数；越界参数 **过滤 + warning**（R10 不静默），
        绝不转译（num_predict↔max_tokens 语义不同，转译即踩坑根源，见 `08` 第 8 章）。
    入参说明：
        options (Dict)：待过滤的参数字典；为空时直接返回空字典。
        supported (frozenset)：后端支持参数名集合（通常取 Capabilities.supported_params）。
        backend (str)：后端标识，仅用于告警日志展示。
    返回值说明：
        Dict：仅含 supported 内参数的字典；options 为空或全部越界时返回 {}。
    """
    if not options:
        return {}
    kept = {}
    for key, value in options.items():
        if key in supported:
            kept[key] = value
        else:
            logger.warning(
                "[M14] %s 后端忽略非支持参数 %s=%r"
                "（跨端参数误传或未声明，见 Capabilities.supported_params）",
                backend, key, value,
            )
    return kept
