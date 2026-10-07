"""
模块功能与逻辑描述：
    `agentCore/parsers/json_parser.py::_try_loads_merged` 的**分流语义**单测。

    该函数是"模型输出多个并列顶层 JSON 对象"的兼容入口。2026-09-19 由"一律浅合并"
    改为**按各段 key 集合是否一致分流**：
      - **key 集合一致** → 判为「同构并列记录」（典型：一次多笔记账）→ 返回 **list**，
        **绝不合并**。原实现走 `merged.update(obj)`，同名键互相覆盖 → 第一笔被静默丢弃
        （实测缺陷：两笔账只落一笔，且无任何告警）。
      - **key 集合不同** → 判为「同一逻辑对象被拆成多段」（典型：stat 的
        `{filter…}, {agg_ops…}`）→ 沿用**浅合并**。该行为是历史契约，**不可删除**，
        否则 stat 的 IR 解析会直接失败。

    背景：`agents/bill_agent/prompts/system.md` 早已写明"多条独立消费，输出多条独立JSON"，
    但正是这个浅合并让提示词的承诺无法落地。本文件即为此缺陷的防复发锁定。
"""

from agentCore.parsers.json_parser import _try_loads_merged


def test_single_object_unchanged():
    """
    函数功能与逻辑描述：
        整体可解析为单个对象时，原样返回 dict —— 既有行为不变（保证向后兼容）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    data, err = _try_loads_merged('{"amount": 30, "category": "餐饮"}')
    assert err is None
    assert isinstance(data, dict), f"单对象应返回 dict，实际 {type(data)}"
    assert data == {"amount": 30, "category": "餐饮"}


def test_homogeneous_parallel_records_return_list():
    """
    函数功能与逻辑描述：
        ★核心防复发用例：多个**同构**并列对象（多笔账）必须返回 **list**，不得浅合并。
        修复前 `merged.update` 会让两笔的 amount / category 互相覆盖，只剩最后一笔。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    raw = '{"amount": 30, "category": "餐饮"}, {"amount": 20, "category": "交通"}'
    data, err = _try_loads_merged(raw)
    assert err is None, err
    assert isinstance(data, list), f"同构并列记录必须返回 list，实际 {type(data)}: {data}"
    assert len(data) == 2, data
    assert data[0]["amount"] == 30, data
    assert data[1]["amount"] == 20, data
    assert data[0]["category"] == "餐饮" and data[1]["category"] == "交通", data


def test_three_homogeneous_records_keep_all():
    """
    函数功能与逻辑描述：
        三笔同构记录必须**全部保留**（排除"恰好只留两条"的巧合式通过）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    data, err = _try_loads_merged('{"amount": 1}, {"amount": 2}, {"amount": 3}')
    assert err is None, err
    assert isinstance(data, list) and len(data) == 3, data
    assert [d["amount"] for d in data] == [1, 2, 3], data


def test_heterogeneous_fragments_still_merged():
    """
    函数功能与逻辑描述：
        ★兼容性用例：key 集合**不同**的多段（stat 的 `{filter…}, {agg_ops…}`）
        必须仍**浅合并为 dict** —— 这是历史契约，删除会导致 stat IR 解析失败。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    raw = '{"filter": {"time_start": null}}, {"agg_ops": [{"op": "sum"}]}'
    data, err = _try_loads_merged(raw)
    assert err is None, err
    assert isinstance(data, dict), f"异构片段应浅合并为 dict，实际 {type(data)}: {data}"
    assert "filter" in data and "agg_ops" in data, data


def test_unparsable_multi_fragment_returns_error():
    """
    函数功能与逻辑描述：
        拆出多段但一段都解析不出时，必须走失败分支（data 为 None 且带错误摘要），
        不得把脏数据透传给调用方。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    data, err = _try_loads_merged("这不是JSON，也不是JSON")
    assert data is None
    assert err, "应返回非空错误摘要"
