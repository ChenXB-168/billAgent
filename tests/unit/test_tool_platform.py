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
    """
    函数功能与逻辑描述：
        提供全局单例工具注册表 `get_registry()`：首次访问触发惰性装配（注册 4 个协议级工具
        与 6 个业务事实工具，并绑定 ExecutionEngine），之后复用同一实例。fixture 作用域为
        函数级（默认），纯只读共享、无清理逻辑（共享同一单例，用例之间不重建）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        ToolRegistry：全局单例注册表，供各用例做装配、权限过滤与调用断言。
    """
    return get_registry()


# ===================== 装配与发现 =====================
def test_platform_registers_protocol_and_biz_tools(reg):
    """
    函数功能与逻辑描述：
        验证平台装配的注册清单：M1 交付的 4 个协议级工具（sql.read / sql.write / llm.chat /
        llm.finance）、M9（D15）收口的 6 个业务事实工具（config.get_latest / bill.sum_by_month /
        bill.sum_by_category / city_price.get_avg / habit.list / habit.upsert）与 M15 新增的
        5 个账单改删工具（bill.recent_list / bill.add / bill.update / bill.delete / habit.recalc）
        与 FC 通道工具（llm.chat_fc）全部注册成功。
        断言 reg.all() 的工具名排序后与期望 **16 项**完全一致。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert sorted(s.name for s in reg.all()) == [
        "bill.add", "bill.delete", "bill.recent_list", "bill.sum_by_category",
        "bill.sum_by_month", "bill.update", "city_price.get_avg",
        "config.get_latest", "habit.list", "habit.recalc", "habit.upsert",
        "llm.chat", "llm.chat_fc", "llm.finance", "sql.read", "sql.write"]


@pytest.mark.parametrize("agent_id,expected", [
    # M9（D15）：orchestrator 收口后可见 6 个工具（5 只读事实工具 + llm.chat；零写，无 sql.write / habit.upsert）
    # M15：orchestrator 不可见 bill.recent_list——该工具需 DBPerm.BILL_READ，编排器只有 FACT_READ（不扩权）
    ("orchestrator_agent", ["bill.sum_by_category", "bill.sum_by_month",
                            "city_price.get_avg", "config.get_latest",
                            "habit.list", "llm.chat", "llm.chat_fc"]),
    # M9：bill_agent 承接月度习惯沉淀（habit.upsert）→ 可见面 +1
    # M15：bill_agent 可见 4 个 bill.* 改删工具 + habit.recalc + llm.chat_fc（共 10 个）；
    #       运行时还会按 operate_sub_type 用 op:* tags 二次收窄（见 test_op_tags_* 用例）
    ("bill_agent", ["bill.add", "bill.delete", "bill.recent_list", "bill.update",
                    "habit.recalc", "habit.upsert", "llm.chat", "llm.chat_fc",
                    "sql.read", "sql.write"]),
    # M15：stat/price/finance 本就持有 BILL_READ（读账单），故 bill.recent_list 对它们可见——
    #       这是「权限与可见面一致」而非扩权（它们不参与 FC 循环，实际不会调用）
    ("stat_agent", ["bill.recent_list", "llm.chat", "llm.chat_fc", "sql.read"]),   # 只读 + 通用模型
    # finance_agent 只有 LLM_FINANCE（无 LLM_BASE）→ 看不到 llm.chat / llm.chat_fc
    ("finance_agent", ["bill.recent_list", "llm.finance", "sql.read"]),            # 只读 + 理财模型
])
def test_list_tools_filters_by_permission(reg, agent_id, expected):
    """
    函数功能与逻辑描述：
        验证★权限过滤即发现过滤：list_tools 依 `get_agent_permission(agent_id)` 剔除
        不可见工具——hidden 工具不暴露，required_perm 为空（公开）或 ∈ 该 Agent 权限集合才放行。
        断言按工具名排序后的可见清单与期望完全一致。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
        agent_id (str)：参数化注入的被测调用主体，取值 orchestrator_agent / bill_agent /
            stat_agent / finance_agent，对应四种权限画像。
        expected (list[str])：参数化注入的该 Agent 期望可见工具名清单（已排序），与 agent_id
            一一配对，组合含义即「某权限画像应看到哪些工具」。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert sorted(s.name for s in reg.list_tools(agent_id)) == expected


def test_to_openai_schema_never_exposes_agent_id(reg):
    """
    函数功能与逻辑描述：
        🔴 安全红线：导出给 LLM 的 function-calling schema 中不得出现 agent_id / agent_tag
        参数。校验 4 个业务 Agent 的 to_openai_schema 均非空，且逐项检查其 parameters.properties
        不含这两键——否则模型可伪造调用主体提权（P2 阶段即成真漏洞）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    for agent_id in ("orchestrator_agent", "bill_agent", "stat_agent", "finance_agent"):
        schema = reg.to_openai_schema(agent_id)
        assert schema, f"{agent_id} 应至少可见一个工具"
        for item in schema:
            props = item["function"]["parameters"].get("properties", {})
            assert "agent_id" not in props
            assert "agent_tag" not in props


# ===================== M15：op:* tags 的可见面锁 =====================
@pytest.mark.parametrize("tags,expected", [
    # edit 任务：只能"看清单 + 改"——看不到 bill.delete（操作类型物理锁定，不靠提示词）
    (("op:read", "op:edit"), {"bill.recent_list", "bill.update"}),
    # delete 任务：只能"看清单 + 删"——看不到 bill.update
    (("op:read", "op:delete"), {"bill.recent_list", "bill.delete"}),
    # add 任务：只有新增一个选项（选无可选，发现层在此基本是形式）
    (("op:add",), {"bill.add"}),
])
def test_op_tags_lock_tool_visibility(reg, tags, expected):
    """
    函数功能与逻辑描述：
        验证 ★按操作类型锁定工具可见面（`设计/12` §4.2.2 的 bounded autonomy 落地点）：
        `to_openai_schema(agent_id, tags=...)` 的标签过滤必须把不属于本次操作类型的
        写工具**物理移除**——edit 任务里模型根本看不到 `bill.delete`，因此"改着改着顺手删了"
        在能力层面就不可能（代价是"改→删"切换须交回顶层重规划，见 `设计/12` §4.6）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
        tags：参数化注入的本次任务标签集（对应 operate_sub_type 映射出的 op:* 组合）。
        expected：参数化注入的期望可见工具名集合（与 tags 一一配对）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    names = {item["function"]["name"] for item in reg.to_openai_schema("bill_agent", tags=tags)}
    assert names == expected


def test_habit_recalc_is_invisible_to_every_model_view(reg):
    """
    函数功能与逻辑描述：
        验证 `habit.recalc` **不对模型可见**（`设计/12` §4.2.6）：它是改删后的**内部一致性修复**，
        由代码在写操作成功后自动触发，模型既无必要也无权调用。断言两件事：
        ① 它的 tags 里没有任何 `op:*`（因此不会被任何按操作类型过滤的可见面命中）；
        ② 对所有 op 标签组合取并集，也取不到它（防将来有人随手给它补一个 op 标签）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    spec = reg.get("habit.recalc")
    assert spec is not None
    assert not any(t.startswith("op:") for t in spec.tags)

    seen: set[str] = set()
    for single_tag in ("op:read", "op:add", "op:edit", "op:delete"):
        seen |= {item["function"]["name"]
                 for item in reg.to_openai_schema("bill_agent", tags=(single_tag,))}
    assert "habit.recalc" not in seen


# ===================== 执行引擎 =====================
@pytest.mark.asyncio
async def test_invoke_denies_write_for_readonly_agent(reg):
    """
    函数功能与逻辑描述：
        验证越权写被引擎鉴权阶段拦截：stat_agent 调 sql.write → 返回 ok=False 且错误类型为
        PERMISSION，并断言底层 `_call_server` 绝不被调用（未触达远端）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with patch.object(mcp_client, "_call_server", new=AsyncMock()) as mock_call:
        res = await reg.invoke("sql.write", {"sql": "INSERT INTO bill (amount) VALUES (1)"},
                               agent_id="stat_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.PERMISSION
    mock_call.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_validates_params(reg):
    """
    函数功能与逻辑描述：
        验证参数校验阶段：sql.read 传入非字符串的 sql（123）不符合 params_schema 的
        sql(string) 约束 → 返回 ok=False 且错误类型为 VALIDATION。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    res = await reg.invoke("sql.read", {"sql": 123}, agent_id="stat_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_invoke_unknown_tool(reg):
    """
    函数功能与逻辑描述：
        验证未注册工具的解析分支：调用不存在的工具名 "nope.nope" → 返回 ok=False 且错误
        类型为 NOT_FOUND（引擎第一步按名查契约即失败）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    res = await reg.invoke("nope.nope", {}, agent_id="stat_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.NOT_FOUND


@pytest.mark.asyncio
async def test_injects_agent_id_from_context(reg):
    """
    函数功能与逻辑描述：
        验证 agent_id 由引擎经适配器注入：调用 sql.read（patch 掉 `_call_server`）后，断言
        底层收到的 arguments 中含 agent_id="stat_agent" 与原始 sql；调用方无需传、也传不了
        （schema 里没有 agent_id）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="[]")) as mock_call:
        res = await reg.invoke("sql.read", {"sql": "SELECT 1"}, agent_id="stat_agent")
    assert res.ok is True
    args = mock_call.call_args[0][2]
    assert args["agent_id"] == "stat_agent"
    assert args["sql"] == "SELECT 1"


@pytest.mark.asyncio
async def test_llm_chat_maps_agent_id_to_agent_tag(reg):
    """
    函数功能与逻辑描述：
        验证 llm.chat 的 agent_id 映射：其 binding 的 inject_agent_id 指定远端参数名为
        agent_tag，故调用后底层 arguments 含 agent_tag="bill_agent" 且不含 agent_id。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        🔴 D2 联动：写工具（side_effect=True）关闭底层连接重试——适配器对写工具传
        max_retry=0，"已落库但连接断开"时重试会重复记账。以 bill_agent 调 sql.write
        （patch `_call_server`），断言底层收到的 max_retry 关键字实参为 0。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="ok")) as mock_call:
        await reg.invoke("sql.write", {"sql": "INSERT INTO bill (amount) VALUES (1)"},
                         agent_id="bill_agent")
    assert mock_call.call_args.kwargs["max_retry"] == 0


@pytest.mark.asyncio
async def test_read_tool_keeps_default_retry(reg):
    """
    函数功能与逻辑描述：
        验证读工具（side_effect=False）不改动底层重试：适配器对非写工具传 max_retry=None，
        由 MCPClient 沿用默认重试次数（行为与改造前一致，不降级）。调用 sql.read 后断言
        底层收到的 max_retry 关键字实参为 None。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="[]")) as mock_call:
        await reg.invoke("sql.read", {"sql": "SELECT 1"}, agent_id="stat_agent")
    assert mock_call.call_args.kwargs["max_retry"] is None


@pytest.mark.asyncio
async def test_unknown_agent_is_denied(reg):
    """
    函数功能与逻辑描述：
        验证治理目标：未标识的调用主体（agent_id="unknown"）查权限表得空集合，调 llm.chat
        因缺 LLMPerm.LLM_BASE 被拒绝 → ok=False 且错误类型为 PERMISSION。改造前 call_llm_base
        完全不鉴权、网关形同虚设，这是 M1 要补上的缺口。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    res = await reg.invoke("llm.chat", {"system_prompt": "s", "user_input": "u"},
                           agent_id="unknown")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.PERMISSION


@pytest.mark.asyncio
async def test_engine_reports_cost_and_attempts(reg):
    """
    函数功能与逻辑描述：
        验证引擎的可观测回填（D3 的天然挂载点）：一次成功调用后 ok=True，cost_ms 为非负
        耗时，attempts == 1（首次即成功、无重试）。
    入参说明：
        reg：pytest fixture 注入的全局工具注册表。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="[]")):
        res = await reg.invoke("sql.read", {"sql": "SELECT 1"}, agent_id="stat_agent")
    assert res.ok is True
    assert res.cost_ms >= 0
    assert res.attempts == 1


# ===================== 兼容层契约（保护 ~18 处生产调用 + ~40 处测试 patch）=====================
@pytest.mark.asyncio
async def test_compat_call_bill_sql_raises_permission_error():
    """
    函数功能与逻辑描述：
        验证兼容层保持「无权限抛原生 PermissionError」的旧契约——集成测试有断言依赖此契约。
        stat_agent 调 call_bill_sql 执行 INSERT（走 sql.write，缺 BILL_WRITE）→ unwrap 抛
        PermissionError。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with pytest.raises(PermissionError):
        await mcp_client.call_bill_sql("stat_agent", "INSERT INTO bill (amount) VALUES (1)")


@pytest.mark.asyncio
async def test_compat_call_bill_sql_returns_str():
    """
    函数功能与逻辑描述：
        验证兼容层成功路径返回 str：patch `_call_server` 返回 "[]"，stat_agent 执行 SELECT
        （走 sql.read）→ call_bill_sql 返回 "[]"（签名与返回类型保持不变）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="[]")):
        out = await mcp_client.call_bill_sql("stat_agent", "SELECT COUNT(*) FROM bill", [])
    assert out == "[]"


@pytest.mark.asyncio
async def test_compat_call_llm_finance_raises_permission_error():
    """
    函数功能与逻辑描述：
        验证 call_llm_finance 的无权限契约：stat_agent 缺 LLMPerm.LLM_FINANCE，调用后
        unwrap 抛 PermissionError（权限由执行引擎统一判定）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with pytest.raises(PermissionError):
        await mcp_client.call_llm_finance("stat_agent", "sys", "user")


@pytest.mark.asyncio
async def test_compat_routes_read_and_write_by_sql_prefix():
    """
    函数功能与逻辑描述：
        验证 Q1 决策：兼容层保留按 SQL 前缀分流的旧行为——SELECT 开头走 sql.read、否则走
        sql.write。用 spy 包装 ToolRegistry.invoke 记录被调工具名，依次调用 SELECT 与 INSERT，
        断言记录顺序为 ["sql.read", "sql.write"]。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    seen = []
    real_invoke = ToolRegistry.invoke

    async def spy(self, name, args, **kw):
        """
        函数功能与逻辑描述：
            记录型替身：把本次被调用的工具名追加到 seen 后，原样委托真实 ToolRegistry.invoke，
            从而在保持真实行为的同时观测按 SQL 前缀分流到哪个静态工具。
        入参说明：
            self (ToolRegistry)：patch 后的实例（绑定为 ToolRegistry.invoke）。
            name (str)：工具名。
            args (dict)：工具入参。
            **kw：其余关键字参数（如 agent_id），原样透传给真实 invoke。
        返回值说明：
            ToolResult：真实 ToolRegistry.invoke 的 await 结果。
        """
        seen.append(name)
        return await real_invoke(self, name, args, **kw)

    with patch.object(ToolRegistry, "invoke", spy), \
            patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="x")):
        await mcp_client.call_bill_sql("bill_agent", "SELECT 1", [])
        await mcp_client.call_bill_sql("bill_agent", "INSERT INTO bill (amount) VALUES (1)", [])
    assert seen == ["sql.read", "sql.write"]
