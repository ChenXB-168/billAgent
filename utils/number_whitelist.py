"""数字白名单共享工具（D16/M11 finance 防幻觉 —— 单一来源）

与 orchestrator/collect_node 的 LLM 汇总白名单闸门同一实现：
- orchestrator 侧：`agents/orchestrator/nodes.py` 在此模块中 re-export（`nodes._extract_numbers` /
  `nodes._numbers_inside_whitelist` 保持可用，collect_node 及既有单测引用不破）
- finance 侧：`agents/finance_agent/nodes.py`（M11 D16）输出白名单校验直接 import 本模块

背景（2026-08-30 修复历史）：`_ORDER_RE` 原仅剔行首序号，单行 "1.…2.…" 列举序号漏网
→ 被白名单闸门误判为幻觉数字；修复后序剔逻辑放宽到句子边界。
"""
import re
from typing import Optional

# 日期（"2026-07-15" / "7月15日" 等）不进数字集（白名单比对的是金额/百分比/数值）
_DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}[-/月]\d{1,2}日?")
# 数字抽取（金额/百分比/数值，剔除日期与列举序号）
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:%|％)?")
# 列举序号（"1." / "1、" / "1)"）：行首 或 句子边界（。；;：:！？ 空白）之后；
# 后继非数字（不误伤小数 1.5），限 1-2 位（不误伤金额）。
_ORDER_RE = re.compile(r"(?m)(?:^|(?<=[。；;：:！？\s]))\s*\d{1,2}[.)、](?=\s*[^\d\s])")

# ---------------------------------------------------------------------------
# ★M20（D22）：中文数字支持（拷打 F5）——原实现只认阿拉伯数字，模型写"五十"即绕过白名单
# ---------------------------------------------------------------------------
# 个位字符（含大写与口语量词"两/俩"）
_CN_DIGITS = {
    "零": 0, "〇": 0,
    "一": 1, "壹": 1,
    "二": 2, "两": 2, "俩": 2, "贰": 2,
    "三": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}
# 十/百/千（段内单位）
_CN_UNITS = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
# 万/亿（段间单位）
_CN_MYRIAD = {"万": 10000, "亿": 100000000}
# 中文数字候选串：数字/单位字符连续段 + 可选口语后缀（多/余/来/几）
#   ★后缀**不参与取值** → "五十多" 天然取出**下界 50**（防幻觉取严不取宽）
#   ★★限定**长度 ≥ 2**：单字量词（"一句话/一方面/一般"的"一"、"两下"的"两"）**不抽**——
#      白名单是"全量包含"语义，抽单字会让**大量含"一/两/三"的正常回复被误判为幻觉**
#      （触发修正重试乃至回退本地渲染），**误报危害远大于漏检**。
#      代价：单价金额若写成单个中文数字（"花了三块"）会漏检——属**已知边界**，
#      与"模型汇总以阿拉伯数字为主"的现状相容（见 `设计/18` §七 R-3）。
_CN_NUM_RE = re.compile(
    r"[零〇一壹二两俩贰三叁四肆五伍六陆七柒八捌九玖十拾百佰千仟万亿]{2,}(?:多|余|来|几)?")


def _cn_to_number(s: str) -> Optional[float]:
    """
    函数功能与逻辑描述：
        中文数字 → float（★**取下界**："五十多" → 50）。实现分两步：
        ① 剔除口语后缀（多/余/来/几）——后缀**不参与取值**，故天然取下界；
        ② 扫描求和：段内（十/百/千）累加、段间（万/亿）进位；并处理**口语省略**
           （"一百二"=120、"三千五"=3500、"五万三"=53000）——末尾个位数字在
           **无"十"、无"零"**时，实为省略了"最近单位的下一个数量级"，
           故按 `最近单位 // 10` 放大（百→十、千→百、万→千）。
        不支持："半"、分数、成数（"八成"）、"廿/卅"。
    入参说明：
        s (str)：待解析的中文数字串（如 "五十" / "一百二" / "五十多"）。
    返回值说明：
        Optional[float]：解析值；无法解析（含未知字符）时返回 None。
    """
    s = re.sub(r"[多余来几]+$", "", s.strip())
    if not s:
        return None

    # ① 判口语省略：末尾是个位数字（非零）、且全文无"十"、无"零"
    #    "一百二" → 120；而"一百零二" → 102（有"零"，属精确表达，不省略）
    tail_scale = 1
    has_omission = (
        s[-1] in _CN_DIGITS and _CN_DIGITS[s[-1]] != 0
        and "十" not in s and "拾" not in s
        and "零" not in s and "〇" not in s
    )
    if has_omission:
        last_unit_val = 0
        for ch in s[:-1]:
            if ch in _CN_UNITS:
                last_unit_val = _CN_UNITS[ch]
            elif ch in _CN_MYRIAD:
                last_unit_val = _CN_MYRIAD[ch]
        if last_unit_val >= 10:
            tail_scale = last_unit_val // 10
        else:
            has_omission = False

    # ② 扫描求和
    total = 0.0
    section = 0.0
    number = 0.0
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            # 裸"十"（如"十五"）隐含 1
            section += (number if number > 0 else 1) * _CN_UNITS[ch]
            number = 0.0
        elif ch in _CN_MYRIAD:
            section = (section + number) * _CN_MYRIAD[ch]
            total += section
            section = 0.0
            number = 0.0
        else:
            return None
    if has_omission and number > 0:
        number *= tail_scale
    return total + section + number


def extract_numbers(text: str) -> set[float]:
    """
    函数功能与逻辑描述：
        从文本中抽取全部数值型 token，作为「LLM 汇总防幻觉白名单」的比对集合。
        实现分三步：先剔除日期串（避免 "2026-07-15" 被拆成 2026/07/15 三个假数字），
        再剔除行首或句子边界处的列举序号（"1." "2、" 等，限 1-2 位且后继非数字，
        避免误伤小数 1.5 与金额），最后正则匹配带可选负号与百分号的数字。
        分母结果做去重与取绝对值处理，且丢弃 0（0 恒在白名单语义内，无区分价值）。
    入参说明：
        text (str)：待抽取的原文，通常是 LLM 生成的汇总回复。
    返回值说明：
        set[float]：文本中出现的数值绝对值集合（已去重、已剔除 0）；文本无数值时返回空集合。
    """
    text = _DATE_RE.sub("", text)
    text = _ORDER_RE.sub("", text)
    out = set()
    for m in _NUM_RE.finditer(text):
        token = m.group(0)
        v = float(token.rstrip("%％"))
        if v != 0:
            out.add(abs(v))
    # ★M20（D22）：中文数字同口径纳入（防"用中文数字绕过白名单"）
    #   例："五十"→50、"一百二"→120、"五十多"→50（下界）
    #   已知边界：不做上下文消歧（"十点"的"十"会被抽取——与既有阿拉伯数字行为一致）
    for m in _CN_NUM_RE.finditer(text):
        v = _cn_to_number(m.group(0))
        if v is not None and v != 0:
            out.add(abs(v))
    return out


def numbers_inside_whitelist(reply: str, whitelist: set[float]) -> bool:
    """
    函数功能与逻辑描述：
        校验 LLM 回复中的每一个数字都能在可信白名单中找到近似值，用于拦截模型幻觉编造的数字。
        判定为「全量包含」语义：只要有一个数字落不进白名单即整体不通过并立即短路返回，
        不区分该数字出现在金额、百分比还是条数位置（避免漏检）。
        白名单本身为空时，任何含数字的回复都不会通过，调用方需自行决定该场景的兜底策略。
    入参说明：
        reply (str)：LLM 生成的待校验回复原文。
        whitelist (set[float])：可信数值白名单，通常来自数据库查询结果等确定性事实；
            比对时区分量纲（如 88.0 与 88% 视为同一数值 88.0）。
    返回值说明：
        bool：True 表示回复中所有数字均可溯源；False 表示存在无法溯源的数字（即疑似幻觉）。
    """
    for v in extract_numbers(reply):
        if not any(abs(v - w) < 0.5 for w in whitelist):
            return False
    return True


__all__ = ["extract_numbers", "numbers_inside_whitelist"]
