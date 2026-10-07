# -*- coding: utf-8 -*-
"""账单数据库正确性集成测试（数据层，不依赖 LLM）

通过真实 MCP sql_bill server + bill.db 验证"数据真相"：
  1. 写读回环：INSERT → SELECT 金额/类别/日期一致
  2. 统计准确性：seed 已知数据 → SUM/COUNT/分类聚合 与期望一致
  3. 数据约束：负数金额、非法类别被 DB CHECK 拒绝且不落库

每个测试用唯一 marker 隔离，finally 清理，不污染共享 bill.db。
补充原因：unit 层全 mock（只验节点透传），L3 评测只查回复文本，
两者都无法回答"数据库真的写对/算对了吗"。
"""
import ast
import uuid
from datetime import date

import pytest

from mcpGateway.client import mcp_client

TEST_AGENT = "bill_agent"  # 唯一同时持有账单读+写权限的 agent（stat/price/finance 仅有 BILL_READ）


def _marker() -> str:
    """
    函数功能与逻辑描述：
        生成带随机后缀的唯一 remark 标记（前缀 it_db_），供每个用例把自己的 seed 数据
        与库中既有记录区分开，并在 finally 中按该标记精确清理，保证不污染共享 bill.db。
    入参说明：
        无。
    返回值说明：
        str：形如 "it_db_xxxxxxxx" 的唯一标记字符串。
    """
    return f"it_db_{uuid.uuid4().hex[:8]}"


def _as_rows(raw) -> list:
    """
    函数功能与逻辑描述：
        把 call_bill_sql 对 SELECT 的返回值统一解析为 list[dict]——兼容层透传的是
        str(list[dict]) 文本（也兼容已是 list 的情形）；解析失败 / 结果非列表时返回空列表。
    入参说明：
        raw (str | list)：工具返回的原始结果（字符串形式的行列表，或已是行列表）。
    返回值说明：
        list[dict]：解析后的行列表；无法解析或不是列表时返回 []。
    """
    if isinstance(raw, list):
        return raw
    try:
        data = ast.literal_eval(str(raw))
        return data if isinstance(data, list) else []
    except (ValueError, SyntaxError):
        return []


async def _clean(marker: str):
    """
    函数功能与逻辑描述：
        按 remark 模糊匹配删除本用例写入的全部账单行（走真实 MCP sql.write），
        实现用例级数据清理；清理自身异常被吞掉，不影响断言结论。
    入参说明：
        marker (str)：_marker() 生成的唯一标记，按 %marker% 作为 LIKE 条件。
    返回值说明：
        无（副作用：删除 bill 表匹配行；异常时静默跳过）。
    """
    try:
        await mcp_client.call_bill_sql(
            TEST_AGENT,
            "DELETE FROM bill WHERE remark LIKE ?",
            [f"%{marker}%"],
        )
    except Exception:
        pass  # 清理失败不影响断言结论


@pytest.mark.asyncio
async def test_write_read_roundtrip():
    """
    函数功能与逻辑描述：
        验证账单库「写读回环」：以 bill_agent 身份经真实 MCP sql_bill server 写入
        66.6 元 / 餐饮 / 今天的记录（断言返回文本含"成功"），再按 marker 查回该行，
        断言 amount / category / consume_time 与写入值一致。finally 按 marker 清理。
        运行时前置依赖：MCP 常驻服务 sql_bill（或 BILLAGENT_MCP_STDIO=1 时走 stdio 子进程）
        与真实 bill.db；不依赖 LLM。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    marker = _marker()
    today = date.today().isoformat()
    try:
        res = await mcp_client.call_bill_sql(
            TEST_AGENT,
            "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
            [66.6, "餐饮", today, marker],
        )
        assert "成功" in str(res), f"写入应成功，实际: {res}"

        rows = _as_rows(
            await mcp_client.call_bill_sql(
                TEST_AGENT,
                "SELECT amount, category, consume_time FROM bill WHERE remark LIKE ? ORDER BY id DESC LIMIT 1",
                [f"%{marker}%"],
            )
        )
        assert rows, "写入后应能查到记录"
        row = rows[0]
        assert abs(float(row["amount"]) - 66.6) < 0.01, f"金额应一致: {row}"
        assert row["category"] == "餐饮", f"类别应一致: {row}"
        assert row["consume_time"] == today, f"日期应一致: {row}"
    finally:
        await _clean(marker)


@pytest.mark.asyncio
async def test_stat_sum_and_group_accuracy():
    """
    函数功能与逻辑描述：
        验证账单库「统计准确性」：经真实 MCP 写入 (10.0 餐饮) / (20.0 餐饮) / (30.0 交通)
        三条 seed，再按 marker 做 SUM/COUNT 聚合，断言总额为 60.0、笔数为 3；
        并按 category 分组断言聚合结果恰为 {"交通": 30.0, "餐饮": 30.0}。
        finally 按 marker 清理。运行时前置依赖同 test_write_read_roundtrip（MCP sql_bill + bill.db，无 LLM）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    marker = _marker()
    seeds = [(10.0, "餐饮"), (20.0, "餐饮"), (30.0, "交通")]
    try:
        for amount, category in seeds:
            res = await mcp_client.call_bill_sql(
                TEST_AGENT,
                "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
                [amount, category, date.today().isoformat(), marker],
            )
            assert "成功" in str(res), f"seed 写入应成功: {res}"

        rows = _as_rows(
            await mcp_client.call_bill_sql(
                TEST_AGENT,
                "SELECT SUM(amount) AS total, COUNT(*) AS cnt FROM bill WHERE remark LIKE ?",
                [f"%{marker}%"],
            )
        )
        assert rows, "SUM 查询应有结果"
        assert abs(float(rows[0]["total"]) - 60.0) < 0.01, f"SUM 应=60: {rows[0]}"
        assert int(rows[0]["cnt"]) == 3, f"COUNT 应=3: {rows[0]}"

        group = _as_rows(
            await mcp_client.call_bill_sql(
                TEST_AGENT,
                "SELECT category, SUM(amount) AS total FROM bill "
                "WHERE remark LIKE ? GROUP BY category ORDER BY category",
                [f"%{marker}%"],
            )
        )
        agg = {r["category"]: float(r["total"]) for r in group}
        assert agg == {"交通": 30.0, "餐饮": 30.0}, f"分类聚合应精确: {agg}"
    finally:
        await _clean(marker)


@pytest.mark.asyncio
async def test_check_rejects_negative_amount():
    """
    函数功能与逻辑描述：
        验证账单库「数据约束」之负数金额：以 bill_agent 经真实 MCP 尝试写入 amount=-5.0，
        断言返回文本含"失败"（被 CHECK(amount > 0) 约束拒绝），再按 marker 统计
        确认 COUNT 为 0（被拒绝的插入未落库）。finally 按 marker 清理。
        运行时前置依赖：MCP sql_bill 服务与真实 bill.db（含 amount>0 CHECK 约束），不依赖 LLM。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    marker = _marker()
    try:
        res = await mcp_client.call_bill_sql(
            TEST_AGENT,
            "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
            [-5.0, "餐饮", date.today().isoformat(), marker],
        )
        assert "失败" in str(res), f"负数金额应被拒绝，实际: {res}"

        rows = _as_rows(
            await mcp_client.call_bill_sql(
                TEST_AGENT,
                "SELECT COUNT(*) AS cnt FROM bill WHERE remark LIKE ?",
                [f"%{marker}%"],
            )
        )
        assert int(rows[0]["cnt"]) == 0, "被拒绝的插入不应落库"
    finally:
        await _clean(marker)


@pytest.mark.asyncio
async def test_check_rejects_bad_category():
    """
    函数功能与逻辑描述：
        验证账单库「数据约束」之非法类别：以 bill_agent 经真实 MCP 尝试写入
        category="其他"（不在五分类内），断言返回文本含"失败"（被
        CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')) 拒绝），再按 marker 统计
        确认 COUNT 为 0（未落库）。finally 按 marker 清理。
        运行时前置依赖：MCP sql_bill 服务与真实 bill.db（含五分类 CHECK 约束），不依赖 LLM。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    marker = _marker()
    try:
        res = await mcp_client.call_bill_sql(
            TEST_AGENT,
            "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)",
            [10.0, "其他", date.today().isoformat(), marker],
        )
        assert "失败" in str(res), f"非法类别应被拒绝，实际: {res}"

        rows = _as_rows(
            await mcp_client.call_bill_sql(
                TEST_AGENT,
                "SELECT COUNT(*) AS cnt FROM bill WHERE remark LIKE ?",
                [f"%{marker}%"],
            )
        )
        assert int(rows[0]["cnt"]) == 0, "被拒绝的插入不应落库"
    finally:
        await _clean(marker)
