"""MCP 网关集成测试：经真实 MCP 通道验证 SQL 通道的读写鉴权与危险语句拦截、LLM 通道的权限隔离。

运行时前置依赖：MCP 常驻服务（sql_bill / llm_base / llm_finance；环境变量 BILLAGENT_MCP_STDIO=1
时回退 stdio 子进程模式）与真实 bill.db；LLM 用例还需外部模型可用。不涉及 orchestrator 与子 Agent。
"""
import pytest
from mcpGateway.client import mcp_client


@pytest.mark.asyncio
async def test_mcp_sql_select():
    """
    函数功能与逻辑描述：
        验证 MCP SQL 通道的正常读路径：以 stat_agent（持有 DBPerm.BILL_READ）身份执行
        `SELECT COUNT(*) FROM bill`，断言返回结果非 None 且字符串化后长度大于 0。
        运行时前置依赖：MCP 常驻服务 sql_bill（BILLAGENT_MCP_STDIO=1 时走 stdio 子进程）与真实 bill.db。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    res = await mcp_client.call_bill_sql(
        agent_id="stat_agent",
        sql="SELECT COUNT(*) FROM bill",
        params=[]
    )
    assert res is not None
    assert len(str(res)) > 0


@pytest.mark.asyncio
async def test_mcp_sql_write_permission_denied():
    """
    函数功能与逻辑描述：
        验证 SQL 通道的越权写拦截：stat_agent 只有 BILL_READ、没有 BILL_WRITE，以它执行
        INSERT（走 sql.write / DBPerm.BILL_WRITE）时，兼容层 unwrap 应抛出原生
        PermissionError（旧契约保持）。
        运行时前置依赖：MCP sql_bill 服务与权限表（mcpGateway/rbac_config.py 的 AGENT_PERMISSION_MAP）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with pytest.raises(PermissionError):
        await mcp_client.call_bill_sql(
            agent_id="stat_agent",
            sql="INSERT INTO bill (amount, category, consume_time) VALUES (1, '餐饮', '2026-01-01')",
            params=[]
        )


@pytest.mark.asyncio
async def test_mcp_danger_sql_blocked():
    """
    函数功能与逻辑描述：
        验证危险 DDL 语句在网关侧被拦截：以 bill_agent 身份执行 `DROP TABLE bill`，
        断言返回文本含"拒绝"或"禁止"——DDL 拦截在权限校验之前完成（sql_mcp_base 的五步链），
        故即使调用方持有账单写权限也不会真正执行。
        运行时前置依赖：MCP sql_bill 服务（拦截逻辑在 server 侧 sql_mcp_base.run_sql_logic）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    res = await mcp_client.call_bill_sql(
        agent_id="bill_agent",
        sql="DROP TABLE bill",
        params=[]
    )
    assert "拒绝" in str(res) or "禁止" in str(res)


@pytest.mark.asyncio
async def test_mcp_llm_base_call():
    """
    函数功能与逻辑描述：
        验证 LLM 通道的正常调用路径：以 stat_agent 身份调 call_llm_base（工具 llm.chat，
        required_perm = LLMPerm.LLM_BASE），断言返回文本 strip 后非空。
        agent_tag 传真实 Agent 标识——M1 起网关强制鉴权，默认值 "unknown" 无任何权限会被拒绝。
        运行时前置依赖：MCP llm_base 服务与外部/本地模型可用。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证理财模型通道的权限隔离：stat_agent 只持有 LLMPerm.LLM_BASE、不持有
        LLMPerm.LLM_FINANCE，以它调 call_llm_finance（工具 llm.finance）应抛出原生
        PermissionError，证明两条模型通道各自校验、互不通行。
        运行时前置依赖：MCP llm_finance 服务与权限表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with pytest.raises(PermissionError):
        await mcp_client.call_llm_finance(
            agent_id="stat_agent",
            sys_prompt="",
            user_text="测试"
        )