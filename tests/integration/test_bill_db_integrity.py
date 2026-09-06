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

TEST_AGENT = "bill_agent"  # 唯一有账单读写权限的 agent


def _marker() -> str:
    return f"it_db_{uuid.uuid4().hex[:8]}"


def _as_rows(raw) -> list:
    """call_bill_sql 对 SELECT 返回 str(list[dict])，解析为 list[dict]"""
    if isinstance(raw, list):
        return raw
    try:
        data = ast.literal_eval(str(raw))
        return data if isinstance(data, list) else []
    except (ValueError, SyntaxError):
        return []


async def _clean(marker: str):
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
    """写读回环：写入的金额/类别/日期与查回一致"""
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
    """统计准确性：SUM/COUNT/分类聚合与已知 seed 一致"""
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
    """数据约束：负数金额被 DB CHECK(amount>0) 拒绝且不落库"""
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
    """数据约束：非法类别被 DB CHECK(category IN 五类) 拒绝且不落库"""
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
