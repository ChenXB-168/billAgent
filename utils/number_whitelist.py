"""数字白名单共享工具（D16/M11 finance 防幻觉 —— 单一来源）

与 orchestrator/collect_node 的 LLM 汇总白名单闸门同一实现：
- orchestrator 侧：`agents/orchestrator/nodes.py` 在此模块中 re-export（`nodes._extract_numbers` /
  `nodes._numbers_inside_whitelist` 保持可用，collect_node 及既有单测引用不破）
- finance 侧：`agents/finance_agent/nodes.py`（M11 D16）输出白名单校验直接 import 本模块

背景（2026-08-30 修复历史）：`_ORDER_RE` 原仅剔行首序号，单行 "1.…2.…" 列举序号漏网
→ 被白名单闸门误判为幻觉数字；修复后序剔逻辑放宽到句子边界。
"""
import re

# 日期（"2026-07-15" / "7月15日" 等）不进数字集（白名单比对的是金额/百分比/数值）
_DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}[-/月]\d{1,2}日?")
# 数字抽取（金额/百分比/数值，剔除日期与列举序号）
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:%|％)?")
# 列举序号（"1." / "1、" / "1)"）：行首 或 句子边界（。；;：:！？ 空白）之后；
# 后继非数字（不误伤小数 1.5），限 1-2 位（不误伤金额）。
_ORDER_RE = re.compile(r"(?m)(?:^|(?<=[。；;：:！？\s]))\s*\d{1,2}[.)、](?=\s*[^\d\s])")


def extract_numbers(text: str) -> set[float]:
    """抽取文本中所有数值（日期剔除、序号剔除），返回绝对值集合用于白名单比对"""
    text = _DATE_RE.sub("", text)
    text = _ORDER_RE.sub("", text)
    out = set()
    for m in _NUM_RE.finditer(text):
        token = m.group(0)
        v = float(token.rstrip("%％"))
        if v != 0:
            out.add(abs(v))
    return out


def numbers_inside_whitelist(reply: str, whitelist: set[float]) -> bool:
    """回复中的每个数字都必须能在白名单中找到近似值（容忍0.5元内浮点误差）"""
    for v in extract_numbers(reply):
        if not any(abs(v - w) < 0.5 for w in whitelist):
            return False
    return True


__all__ = ["extract_numbers", "numbers_inside_whitelist"]
