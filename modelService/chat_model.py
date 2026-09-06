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
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Capabilities:
    """能力契约：一个后端的「参数边界 + 推理特性 + 窗口」显式声明。

    适配行为完全由本契约决定——这是"隔离"的结构化落点，不是靠注释/if-else 记忆。
    """
    backend: str                      # "ollama" | "openai-compat"
    supported_params: frozenset       # 本后端认识的参数名（契约边界，越界即过滤+告警）
    reasoning: bool                   # 是否推理模型（reasoning 与 content 共享 max_tokens 预算）
    context_window: int               # 上下文窗口 token 数（喂 D6 ContextManager 预算表）


@dataclass
class ChatResult:
    """统一响应。非推理模型 reasoning_tokens 恒为 0。"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0     # 推理模型的思考消耗（非推理模型恒为 0）
    total_tokens: int = 0
    latency_ms: int = 0
    channel: str = ""             # "local" | "external"


class ChatError(Exception):
    """统一错误类型：retryable 决定是否进入重试预算池（T5 分类的结构化落点）。

    :param error_type: 结构化错误语义（"HTTP_429" / "HTTP_401" / "Timeout" /
        "EMPTY_CONTENT" / "ConnectionError" / ...），业务层按类型分支，不做字符串 startswith。
    """

    def __init__(self, message: str, *, retryable: bool, error_type: str):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.error_type = error_type


class ChatModel(ABC):
    """模型后端抽象：业务层只认 generate()，不感知"本地还是外部"。"""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        """后端能力契约（参数边界 / 推理特性 / 窗口）。"""

    @abstractmethod
    def generate(self, *, system: str, user: str,
                 options: Dict, agent_tag: str,
                 timeout: Optional[int] = None) -> ChatResult:
        """统一入口。options 为**已归一**参数（调用方已合并后端默认）；
        实现方内部按 capabilities.supported_params 过滤越界参数并告警（R10），绝不转译。
        只返回 ChatResult 或抛 ChatError——不返回错误字符串。
        :param timeout: 请求超时（秒）覆盖，None = 用 Provider 构造默认。
        """


def filter_options(options: Dict, supported: frozenset, backend: str) -> Dict:
    """按后端能力边界过滤参数：只保留 supported 内参数；越界参数 **过滤 + warning**（R10 不静默），
    绝不转译（num_predict↔max_tokens 语义不同，转译即踩坑根源，见 `08` 第 8 章）。
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
