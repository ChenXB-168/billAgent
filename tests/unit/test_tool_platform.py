# -*- coding: utf-8 -*-
"""M1（D1 工具接入平台）单元测试。

覆盖：平台装配 / 权限过滤 / 鉴权 / 参数校验 / 错误分类 / agent_id 注入 /
      写操作禁用底层重试（D2 联动）/ 兼容层契约。

★全部用例**不拉起真实 MCP 子进程**——凡涉及远端的一律 patch `MCPClient._call_server`，
  鉴权与校验类用例在执行前即被引擎阻断，天然不触达远端。
"""
import pytest
from unittest.mock import AsyncMock, patch

from mcpGateway import get_registry
from mcpGateway.client import mcp_client
from mcpGateway.registry import ToolRegistry
from mcpGateway.tool_model import ToolErrorKind


@pytest.fixture
def reg():
    return get_registry()


# ===================== 装配与发现 =====================
def test_platform_registers_protocol_and_biz_tools(reg):
    """M1 交付 4 协议级工具 + M9（D15）收口 6 业务工具全部注册成功"""
    assert sorted(s.name for s in reg.all()) == [
        "bill.sum_by_category", "bill.sum_by_month", "city_price.get_avg",
        "config.get_latest", "habit.list", "habit.upsert",
        "llm.chat", "llm.finance", "sql.read", "sql.write"]


@pytest.mark.parametrize("agent_id,expected", [
    # M9（D15）：orchestrator 收口后可见 5 只读事实工具（零写，无 sql.write / habit.upsert）
    ("orchestrator_agent", ["bill.sum_by_category", "bill.sum_by_month",
                            "city_price.get_avg", "config.get_latest",
                            "habit.list", "llm.chat"]),
    # M9：bill_agent 承接月度习惯沉淀（habit.upsert）→ 可见面 +1
    ("bill_agent", ["habit.upsert", "llm.chat", "sql.read", "sql.write"]),
    ("stat_agent", ["llm.chat", "sql.read"]),               # 只读 + 通用模型
    ("finance_agent", ["llm.finance", "sql.read"]),         # 只读 + 理财模型（无 llm.chat）
])
def test_list_tools_filters_by_permission(reg, agent_id, expected):
    """★权限过滤即发现过滤"""
    assert sorted(s.name for s in reg.list_tools(agent_id)) == expected


def test_to_openai_schema_never_exposes_agent_id(reg):
    """🔴 安全红线：schema 不得出现 agent_id / agent_tag

    否则 `to_openai_schema` 喂给 LLM 后，模型可伪造调用主体提权（P2 阶段即成真漏洞）。
    """
    for agent_id in ("orchestrator_agent", "bill_agent", "stat_agent", "finance_agent"):
        schema = reg.to_openai_schema(agent_id)
        assert schema, f"{agent_id} 应至少可见一个工具"
        for item in schema:
            props = item["function"]["parameters"].get("properties", {})
            assert "agent_id" not in props
            assert "agent_tag" not in props


# ===================== 执行引擎 =====================
@pytest.mark.asyncio
async def test_invoke_denies_write_for_readonly_agent(reg):
    """越权写：stat_agent 调 sql.write → PERMISSION，且绝不触达远端"""
    with patch.object(mcp_client, "_call_server", new=AsyncMock()) as mock_call:
        res = await reg.invoke("sql.write", {"sql": "INSERT INTO bill (amount) VALUES (1)"},
                               agent_id="stat_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.PERMISSION
    mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_validates_params(reg):
    """参数校验：sql 传非字符串 → VALIDATION"""
    res = await reg.invoke("sql.read", {"sql": 123}, agent_id="stat_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_invoke_unknown_tool(reg):
    res = await reg.invoke("nope.nope", {}, agent_id="stat_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_injects_agent_id_from_context(reg):
    """agent_id 由引擎注入，调用方无需传、也传不了（schema 里没有）"""
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="[]")) as mock_call:
        res = await reg.invoke("sql.read", {"sql": "SELECT 1"}, agent_id="stat_agent")
    assert res.ok is True
    args = mock_call.call_args[0][2]
    assert args["agent_id"] == "stat_agent"
    assert args["sql"] == "SELECT 1"


@pytest.mark.asyncio
async def test_llm_chat_maps_agent_id_to_agent_tag(reg):
    """llm.chat：server 侧参数名为 agent_tag，由 binding 的 inject_agent_id 指定"""
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="ok")) as mock_call:
        res = await reg.invoke("llm.chat", {"system_prompt": "s", "user_input": "u"},
                               agent_id="bill_agent")
    assert res.ok is True
    args = mock_call.call_args[0][2]
    assert args["agent_tag"] == "bill_agent"
    assert "agent_id" not in args


@pytest.mark.asyncio
async def test_write_tool_disables_underlying_retry(reg):
    """🔴 D2 联动：写工具关闭底层连接重试——"已落库但连接断开"时重试 = 重复记账"""
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="ok")) as mock_call:
        await reg.invoke("sql.write", {"sql": "INSERT INTO bill (amount) VALUES (1)"},
                         agent_id="bill_agent")
    assert mock_call.call_args.kwargs["max_retry"] == 0


@pytest.mark.asyncio
async def test_read_tool_keeps_default_retry(reg):
    """读工具沿用底层默认重试，行为与改造前一致（不降级）"""
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="[]")) as mock_call:
        await reg.invoke("sql.read", {"sql": "SELECT 1"}, agent_id="stat_agent")
    assert mock_call.call_args.kwargs["max_retry"] is None


@pytest.mark.asyncio
async def test_unknown_agent_is_denied(reg):
    """治理目标：未标识的调用主体（unknown）无任何权限，一律拒绝

    改造前 call_llm_base 完全不鉴权，网关形同虚设；这是 M1 要补上的缺口。
    """
    res = await reg.invoke("llm.chat", {"system_prompt": "s", "user_input": "u"},
                           agent_id="unknown")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.PERMISSION


@pytest.mark.asyncio
async def test_engine_reports_cost_and_attempts(reg):
    """引擎回填耗时与尝试次数（D3 可观测的天然挂载点）"""
    with patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="[]")):
        res = await reg.invoke("sql.read", {"sql": "SELECT 1"}, agent_id="stat_agent")
    assert res.ok is True
    assert res.cost_ms >= 0
    assert res.attempts == 1


# ===================== 兼容层契约（保护 ~18 处生产调用 + ~40 处测试 patch）=====================
@pytest.mark.asyncio
async def test_compat_call_bill_sql_raises_permission_error():
    """兼容层必须保持「无权限抛 PermissionError」——集成测试有断言依赖此契约"""
    with pytest.raises(PermissionError):
        await mcp_client.call_bill_sql("stat_agent", "INSERT INTO bill (amount) VALUES (1)")


@pytest.mark.asyncio
async def test_compat_call_bill_sql_returns_str():
    with patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="[]")):
        out = await mcp_client.call_bill_sql("stat_agent", "SELECT COUNT(*) FROM bill", [])
    assert out == "[]"


@pytest.mark.asyncio
async def test_compat_call_llm_finance_raises_permission_error():
    with pytest.raises(PermissionError):
        await mcp_client.call_llm_finance("stat_agent", "sys", "user")


@pytest.mark.asyncio
async def test_compat_routes_read_and_write_by_sql_prefix():
    """Q1 决策：兼容层按 SQL 前缀分流到 sql.read / sql.write 两个静态工具"""
    seen = []
    real_invoke = ToolRegistry.invoke

    async def spy(self, name, args, **kw):
        seen.append(name)
        return await real_invoke(self, name, args, **kw)

    with patch.object(ToolRegistry, "invoke", spy), \
            patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="x")):
        await mcp_client.call_bill_sql("bill_agent", "SELECT 1", [])
        await mcp_client.call_bill_sql("bill_agent", "INSERT INTO bill (amount) VALUES (1)", [])
    assert seen == ["sql.read", "sql.write"]
