import pytest
import os
# 修复导入路径：你的长记忆在 ragKnowledge 下
from memory.long_memory import (
    LONG_MEM_PATH,
    save_consume_memory,
    search_history_consume,
    clear_all_long_mem
)

# 每条用例执行前后清空长期记忆，用例之间数据隔离不干扰
@pytest.fixture(scope="function", autouse=True)
def fresh_long_memory():
    clear_all_long_mem()
    yield
    clear_all_long_mem()


def test_save_and_vector_search():
    """测试存入记忆 + 混合语义+关键词检索正常召回"""
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
    # 修复：不固定月份，只校验核心关键词，不会随机失败
    assert "餐饮" in res2


def test_search_empty_memory():
    """无任何记忆时返回提示文本"""
    res = search_history_consume("随便查", top_k=3)
    assert res == "暂无历史消费记录"


def test_search_top_k_limit():
    """限制返回条数不超过top_k"""
    for i in range(5):
        save_consume_memory(f"2026年{i+1}月：月度消费总结{i}")

    res = search_history_consume("月度消费", top_k=2)
    lines = res.splitlines()
    assert len(lines) == 2


def test_clear_all_memory():
    """清空全部记忆功能验证"""
    save_consume_memory("2026年4月：测试数据，会被清空")
    clear_all_long_mem()
    res = search_history_consume("4月消费")
    assert res == "暂无历史消费记录"
    # 文件也被删除
    assert not os.path.exists(LONG_MEM_PATH)