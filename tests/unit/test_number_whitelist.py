# -*- coding: utf-8 -*-
"""
utils/number_whitelist.py 单元测试 —— 防幻觉白名单（含 M20/D22 中文数字支持）。

覆盖：
  ① 阿拉伯数字（既有行为，零回归）
  ② 中文数字（M20 新增）：五十 / 一百二 / 两千 / 三千五 / 五万三 / 十五 / 一百零二
  ③ ★「五十多」→ 50（取下界）—— 防幻觉取严不取宽
  ④ 边界：无数字 / 混合文本 / 日期与列举序号不误抽
  ⑤ numbers_inside_whitelist 的「全量包含」语义（含"用中文数字绕过"的防护）
"""
import pytest
from utils.number_whitelist import extract_numbers, numbers_inside_whitelist


# ===================== ① 阿拉伯数字（零回归） =====================
def test_arabic_basic():
    assert extract_numbers("今天花了 35.5 元") == {35.5}


def test_arabic_percent_and_zero_dropped():
    """百分号同值；0 被丢弃（无区分价值）。"""
    assert extract_numbers("占比 88%") == {88.0}
    assert extract_numbers("总计 0 元") == set()


def test_date_not_extracted():
    """日期串整体剔除（避免 "2026-07-15" 被拆成三个假数字）。"""
    assert extract_numbers("2026-07-15 花了 50") == {50.0}


def test_order_number_not_extracted():
    """列举序号剔除（"1." "2、" 等）。"""
    assert extract_numbers("1. 吃饭 2. 打车") == set()


# ===================== ② 中文数字（M20 新增） =====================
@pytest.mark.parametrize("text,expected", [
    ("五十", 50.0),
    ("一百二", 120.0),          # ★口语省略：省略"十"（≠ 102）
    ("两百", 200.0),            # 口语量词
    ("两千", 2000.0),
    ("三千五", 3500.0),         # ★口语省略：省略"百"（≠ 3050）
    ("五万三", 53000.0),        # ★口语省略：省略"千"
    ("十五", 15.0),             # 裸"十"隐含 1
    ("二十", 20.0),
    ("一百零二", 102.0),        # 有"零" → 精确表达，**不**省略
])
def test_chinese_numbers(text, expected):
    assert expected in extract_numbers(text)


def test_single_char_quantifier_not_extracted():
    """★M20 边界：单字量词**不**抽取（长度 ≥ 2 才抽）——避免"一句话"被当成数字 1。

    理由：白名单是"全量包含"语义，抽单字会让大量含"一/两/三"的正常回复被误判为幻觉，
    误报危害远大于漏检。代价："花了三块"这类单字金额会漏检（已知边界）。
    """
    assert extract_numbers("一句话") == set()
    assert extract_numbers("这是一方面") == set()
    assert extract_numbers("比较一般") == set()


# ===================== ③ 下界语义（防幻觉取严） =====================
def test_chinese_approximate_takes_lower_bound():
    """★「五十多」→ 50（取下界）：防幻觉场景宁可误拦，不可漏放。"""
    assert extract_numbers("大概五十多块") == {50.0}


# ===================== ④ 边界 =====================
def test_no_number():
    assert extract_numbers("没有数字的一句话") == set()


def test_mixed_arabic_and_chinese():
    assert extract_numbers("大概五十块，实际 48") == {50.0, 48.0}


# ===================== ⑤ 白名单比对（含中文绕过防护） =====================
def test_whitelist_blocks_chinese_hallucination():
    """★M20 核心收益：模型用**中文数字**编造 → 现在能被检出（改造前会被绕过）。"""
    assert numbers_inside_whitelist("大约六十元", {50.0}) is False


def test_whitelist_allows_matching_chinese():
    """中文数字能回溯到白名单 → 放行（与阿拉伯数字同等对待）。"""
    assert numbers_inside_whitelist("大约五十元", {50.0}) is True


def test_whitelist_empty_blocks_any_number():
    """白名单为空时，任何含数字的回复都不通过。"""
    assert numbers_inside_whitelist("大约五十元", set()) is False
