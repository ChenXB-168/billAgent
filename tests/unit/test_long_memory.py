# ==============================================
# 长期消费记忆单元测试：写入/混合检索、空库与清理、top_k 上限、口语时间表达识别
# ==============================================
import pytest
import os
from datetime import date
# 长记忆模块位于项目包 memory 下（memory/long_memory.py）
from memory.long_memory import (
    LONG_MEM_PATH,
    save_consume_memory,
    search_history_consume,
    clear_all_long_mem,
    _extract_time_tokens,
)

# 每条用例执行前后清空长期记忆，用例之间数据隔离不干扰
@pytest.fixture(scope="function", autouse=True)
def fresh_long_memory():
    """
    函数功能与逻辑描述：
        为每条用例提供干净的长期记忆环境：function 作用域 + autouse，用例开始前与结束后
        各调用一次 clear_all_long_mem()（清空内存列表、重置 FAISS 索引并删除 pkl 文件），
        避免用例之间记忆数据互相污染。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（无产出值，仅提供用例前后清理，不向用例注入数据）。
    """
    clear_all_long_mem()
    yield
    clear_all_long_mem()


def test_save_and_vector_search():
    """
    函数功能与逻辑描述：
        验证写入 3 条不同月份月度总结后，混合检索（向量 + jieba 关键词融合）能正常召回：
        精确关键词查询「1月房租开销」优先命中「2026年1月」记录且返回行数不超过 top_k；
        模糊语义查询「每个月吃饭花多少钱」结果中含「餐饮」（不固定月份，避免随机失败）。
        覆盖场景：多条记忆的精确关键词召回 + 模糊语义召回。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参；记忆环境由 autouse fixture 隔离）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 写入3条不同主题月度总结
    save_consume_memory("2026年1月：总消费5000，房租3000，餐饮1200，购物800")
    save_consume_memory("2026年2月：总消费3500，餐饮2000，娱乐1000，交通500")
    save_consume_memory("2026年3月：总消费4200，房租3000，餐饮500，其他700")

    # 1. 精确关键词查询：1月房租，必须优先召回1月记录
    res1 = search_history_consume("1月房租开销", top_k=2)
    assert "2026年1月" in res1
    lines1 = res1.splitlines()
    assert len(lines1) <= 2

    # 2. 模糊语义查询：吃饭开销，只要包含餐饮相关内容即通过
    res2 = search_history_consume("每个月吃饭花多少钱", top_k=2)
    # 不固定月份，只校验核心关键词，不会随机失败
    assert "餐饮" in res2


def test_search_empty_memory():
    """
    函数功能与逻辑描述：
        验证空库检索兜底：无任何记忆时，search_history_consume 返回固定提示文本
        「暂无历史消费记录」。
        覆盖场景：记忆列表为空的分支。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    res = search_history_consume("随便查", top_k=3)
    assert res == "暂无历史消费记录"


def test_search_top_k_limit():
    """
    函数功能与逻辑描述：
        验证返回条数上限：写入 5 条含「月度消费」关键词的记录后，
        search_history_consume(top_k=2) 结果的行数恰为 2。
        覆盖场景：命中数多于 top_k 时的截断。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for i in range(5):
        save_consume_memory(f"2026年{i+1}月：月度消费总结{i}")

    res = search_history_consume("月度消费", top_k=2)
    lines = res.splitlines()
    assert len(lines) == 2


def test_clear_all_memory():
    """
    函数功能与逻辑描述：
        验证清空全部记忆：写入一条后调用 clear_all_long_mem()，
        再次检索返回「暂无历史消费记录」，且持久化文件 LONG_MEM_PATH 已被删除。
        覆盖场景：清空后的内存与磁盘双双无残留。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    save_consume_memory("2026年4月：测试数据，会被清空")
    clear_all_long_mem()
    res = search_history_consume("4月消费")
    assert res == "暂无历史消费记录"
    # 文件也被删除
    assert not os.path.exists(LONG_MEM_PATH)


# ===================== 时间表达识别（2026-09-10 用户口语兼容增强） =====================

def test_extract_time_tokens_current_month():
    """
    函数功能与逻辑描述：
        验证「这个月/本月/当月/这月」四种当月口语表达均能被 _extract_time_tokens
        识别为当月 token ("date", "YYYY-MM")——原实现仅覆盖「上个月」，这些表达恒漏。
        覆盖场景：四种当月说法逐一断言。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    cur = f"{date.today().year:04d}-{date.today().month:02d}"
    for q in ("这个月花了多少", "本月开销", "当月消费", "这月花了多少"):
        tokens = _extract_time_tokens(q)
        assert ("date", cur) in tokens, f"{q!r} 未识别当月: {tokens}"


def test_extract_time_tokens_numeric_date():
    """
    函数功能与逻辑描述：
        验证纯数字日期「2026-08」「2026/08」（用户不写「年」「月」汉字）能被识别为
        月份 token ("date", "2026-08")。
        覆盖场景：连字符与斜杠两种数字日期分隔符。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for q in ("2026-08 花了多少", "2026/08 的消费"):
        tokens = _extract_time_tokens(q)
        assert ("date", "2026-08") in tokens, f"{q!r} 未识别: {tokens}"


def test_extract_time_tokens_relative_years():
    """
    函数功能与逻辑描述：
        验证相对年份换算：去年 / 今年 / 前年 X月 分别换算为基准年的前一年、当年、
        前两年的月份 token ("date", "YYYY-MM")。
        覆盖场景：三种相对年份表达基于 date.today().year 动态断言。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    y = date.today().year
    cases = {
        "去年3月消费": f"{y - 1}-03",
        "今年3月消费": f"{y}-03",
        "前年3月消费": f"{y - 2}-03",
    }
    for q, want in cases.items():
        tokens = _extract_time_tokens(q)
        assert ("date", want) in tokens, f"{q!r} 期望 {want}, 实际 {tokens}"


def test_month_token_matches_standard_memory_format():
    """
    函数功能与逻辑描述：
        ★核心修复验证：month 分支须命中生产主格式「【YYYY-MM 消费习惯】」（MM 后是空格）。
        写入两条结构高度相似（关键词分、向量分接近）的月度习惯记忆，查询「8月餐饮花了多少」，
        由时间加权决定胜负，结果必须含「2026-08」。
        覆盖场景：生产格式（MM 后空格）与旧格式（含完整日期）的时间加权命中。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 两条记忆结构高度相似（关键词分、向量分接近），由时间加权决定胜负
    save_consume_memory("【2026-08 消费习惯】餐饮12笔1560.00元, 交通8笔420.00元")
    save_consume_memory("【2026-07 消费习惯】餐饮10笔1200.00元, 交通6笔380.00元")

    res = search_history_consume("8月餐饮花了多少", top_k=1)
    assert "2026-08" in res, f"「8月」查询未命中 8 月记忆（时间加权失效）: {res}"
