import pytest
from mcpGateway.client import mcp_client


@pytest.mark.asyncio
async def test_mcp_sql_select():
    """MCP SQL查询：正常读取账单表"""
    res = await mcp_client.call_bill_sql(
        agent_id="stat_agent",
        sql="SELECT COUNT(*) FROM bill",
        params=[]
    )
    assert res is not None
    assert len(str(res)) > 0


@pytest.mark.asyncio
async def test_mcp_sql_write_permission_denied():
    """越权写库：statAgent无写权限，被拦截"""
    with pytest.raises(PermissionError):
        await mcp_client.call_bill_sql(
            agent_id="stat_agent",
            sql="INSERT INTO bill (amount, category, consume_time) VALUES (1, '餐饮', '2026-01-01')",
            params=[]
        )


@pytest.mark.asyncio
async def test_mcp_danger_sql_blocked():
    """危险DDL语句：被SQL网关拦截"""
    res = await mcp_client.call_bill_sql(
        agent_id="bill_agent",
        sql="DROP TABLE bill",
        params=[]
    )
    assert "拒绝" in str(res) or "禁止" in str(res)


@pytest.mark.asyncio
async def test_mcp_llm_base_call():
    """MCP通用模型调用：正常返回"""
    res = await mcp_client.call_llm_base(
        sys_prompt="你是测试助手",
        user_text="请回复OK",
        # M1 起网关强制鉴权：agent_tag 即鉴权主体，须传真实 Agent 标识
        # （默认 "unknown" 无任何权限，会被网关拒绝）
        agent_tag="stat_agent"
    )
    assert len(res.strip()) > 0


@pytest.mark.asyncio
async def test_mcp_finance_llm_permission_denied():
    """越权调用理财模型：被权限拦截"""
    with pytest.raises(PermissionError):
        await mcp_client.call_llm_finance(
            agent_id="stat_agent",
            sys_prompt="",
            user_text="测试"
        )