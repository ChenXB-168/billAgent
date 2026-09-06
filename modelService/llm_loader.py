"""M14（D17）模型接入层 **兼容层** —— ollama_base_call / external_llm_call 签名与返回不变。

设计出处：`08` 第 8 章 §8.9（过渡路径）。
- 阶段 A：保留两函数签名（约束 C5：直接删除会破坏 server_llm_* 与按名调用/评测），内部转调
  `ModelRouter`（统一重试 / fallback / 埋点）——**业务链路零改动，测试全绿**。
- 原双埋点 helper `_record_local_usage` / `_record_external_usage` 已**消除**（`08` §8.11 验收#4）：
  usage 解析收口到 `providers/openai_compat.parse_usage`，埋点收口到 `router.execute` 单点。
- 原双白名单 `LOCAL_OPTION_WHITELIST` / `EXTERNAL_OPTION_WHITELIST` 已**并入各 Provider 的
  `Capabilities.supported_params`**（`08` §8.9 过渡路径 ③），本文件不再散落。
- 阶段 B（后续，非 M14 必做）：server_llm_base / server_llm_finance 改直接调 `ModelRouter.generate()`
  后，本文件可整体删除（`08` §8.9 过渡路径 ② / R3 退役线）。
"""
import logging

from config.config import (
    DISABLE_LOCAL_LLM,
    EXTERNAL_ORCH_MODEL,
)
from modelService.chat_model import ChatError
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider
from modelService.router import ModelRouter, MAX_REQUEST_ATTEMPTS

logger = logging.getLogger(__name__)

# TODO(M14 阶段 B)：server_llm_* 改直接调 ModelRouter.generate() 后，本兼容层整体退役（`08` §8.9 / R3）。
# 共享 Router 仅用于 build_options（参数归一）；定点通道由各函数构造对应 Provider 执行。
_compat_router = ModelRouter(local=OllamaProvider(),
                             external=OpenAICompatProvider(EXTERNAL_ORCH_MODEL))


def _to_error_text(err: ChatError) -> str:
    """ChatError → 兼容层 FINANCE_ERROR 前缀字符串（业务层 startswith 判断语义不变）。"""
    return f"FINANCE_ERROR: {err.message}"


def ollama_base_call(
    custom_system: str,
    user_content: str,
    model_name: str = None,
    timeout: int = None,
    agent_custom_options: dict = None,
    agent_tag: str = None
) -> str:
    """
    Ollama 统一调用入口（兼容层：签名/返回不变，内部转调 ModelRouter → OllamaProvider）
    职责：仅完成HTTP请求、异常字符串化，无任何业务逻辑
    :param custom_system: 系统提示词（必填放第一位，避免传参颠倒）
    :param user_content: 用户输入文本
    :param model_name: 指定模型名，不传使用配置默认基座模型
    :param timeout: 超时时间（秒），不传使用全局配置MODEL_TIMEOUT
    :param agent_custom_options: 当前Agent专属生成参数，会覆盖全局BASE_LLM_OPTIONS
    :param agent_tag: 调用方Agent标识（D3 token归因用，可选，不影响既有调用）
    :return: 模型返回文本 / 异常提示（统一带FINANCE_ERROR前缀，业务层可直接识别）
    """
    # ★硬开关：排查/测试期封死本地通道。走到这里说明模型路由有遗漏，直接报错暴露问题，
    #   而不是默默用 1.8B 小模型跑出一个又慢又不可靠的结果。
    if DISABLE_LOCAL_LLM:
        raise RuntimeError(
            "[BILLAGENT_DISABLE_LOCAL_LLM=1] 本地模型已被禁用，但调用仍走到了 ollama_base_call。"
            "这属于模型路由缺陷（预期应走 external_llm_call）——"
            "请检查 server_llm_base 的路由条件，或该调用点传入的 agent_tag。"
        )

    provider = OllamaProvider(model_name)
    options = _compat_router.build_options(provider, agent_custom_options)
    try:
        result = _compat_router.execute(
            provider, system=custom_system, user=user_content,
            options=options, agent_tag=agent_tag or "unknown",
            max_attempts=1, timeout=timeout)   # 本地快速失败：不重试（单次 ~189s，重试价值低）
        return result.text
    except ChatError as err:
        return _to_error_text(err)


def external_llm_call(
    custom_system: str,
    user_content: str,
    model_name: str = None,
    timeout: int = None,
    agent_custom_options: dict = None,
    agent_tag: str = None
) -> str:
    """
    外部API强模型统一调用入口（兼容层：签名/返回不变，内部转调 ModelRouter → OpenAICompatProvider）
    :param custom_system: 系统提示词（必填放第一位，避免传参颠倒）
    :param user_content: 用户输入文本
    :param model_name: 指定模型名，不传使用配置默认强模型名
    :param timeout: 超时时间（秒），不传使用全局 EXTERNAL_LLM_TIMEOUT
    :param agent_custom_options: 外部差异化参数（temperature/max_tokens），merge EXTERNAL_BASE_LLM_OPTIONS
    :param agent_tag: 调用方Agent标识（D3 token归因用，可选，不影响既有调用）
    :return: 模型返回文本 / 异常提示（统一带FINANCE_ERROR前缀）
    """
    provider = OpenAICompatProvider(model_name)
    options = _compat_router.build_options(provider, agent_custom_options)
    try:
        result = _compat_router.execute(
            provider, system=custom_system, user=user_content,
            options=options, agent_tag=agent_tag or "unknown",
            max_attempts=MAX_REQUEST_ATTEMPTS, timeout=timeout)
        return result.text
    except ChatError as err:
        return _to_error_text(err)


# 对外暴露接口
__all__ = ["ollama_base_call", "external_llm_call"]
