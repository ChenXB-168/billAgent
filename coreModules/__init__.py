"""纯业务能力层统一导出（2026-09-04 单图遗留清理后：仅 price_compare 现役）

bill_parser / bill_analyzer / target_control / finance_advice 四个单图遗留模块
已随 LEGACY 链（agentCore 图栈 + offline_debug + task_memory）删除——
规则分别内联进 bill_agent / stat_agent / orchestrator；finance_advice 弃用（LLM 直出）。
"""
from .price_compare import price_compare, PriceCompare, calc_premium_evaluate
from .price_compare import (
    PREMIUM_THRESHOLD_LOW,
    PREMIUM_THRESHOLD_NORMAL,
    PREMIUM_THRESHOLD_HIGH,
)

__all__ = [
    "price_compare",
    "PriceCompare",
    "calc_premium_evaluate",
    "PREMIUM_THRESHOLD_LOW",
    "PREMIUM_THRESHOLD_NORMAL",
    "PREMIUM_THRESHOLD_HIGH",
]
