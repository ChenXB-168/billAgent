# -*- coding: utf-8 -*-
"""M15（P3b 自主循环）单元测试：可见面裁剪 / tool loop / 熔断 / 写幂等闸门 / observation 回填。

设计依据：`设计/12` §4.2.1–§4.2.3、§4.6（失败矩阵）。
覆盖（对应 `11` §3 M15 的 P3b 验收方式 ①–③）：
  ① 可见面锁：`edit` 任务取不到 `bill.delete`、`delete` 任务取不到 `bill.update`（真实 registry 断言）；
  ② 循环控制（mock 模型返回固定 tool_calls 序列）：正常闭环 / observation 回填顺序 /
     **追问路径不产生任何 tool_call** / 达 `FC_MAX_TURNS` 熔断转追问 / 写工具重复调用被闸门拦截 /
     工具失败作为 observation 继续 / **无写成功绝不报成功**（红线 7）/ 通道异常转追问；
  ③ 开关等价：`BILL_FC_ENABLED=0` → 走 P3a 确定性路径（由 `test_bill_edit_delete.py` 覆盖；
     本文件只验证默认值下确实走循环）。

★不触网：`mcp_client.call_llm_fc` 被替换为按序列返回的 AsyncMock；
  `get_registry` 打桩为记录型假注册表（工具调用全被捕获）。
"""
import json

import pytest
from unittest.mock import AsyncMock, patch

from agents.bill_agent import nodes as bill_nodes
from agents.bill_agent.nodes import _tool_tags, execute_node
from mcpGateway.tool_model import ToolError, ToolErrorKind, ToolResult

# 模块导入期捕获**真实的** get_registry（此刻 autouse 打桩尚未生效）：
# 可见面锁用例必须用真实注册表验证约束，否则会被假注册表掩盖。
import mcpGateway.registry as _reg_mod

_REAL_GET_REGISTRY = _reg_mod.get_registry

_RECENT = [{"id": 2, "amount": 35.0, "category": "交通",
            "consume_time": "2026-09-13", "remark": "昨天打车35元",
            "create_time": "2026-09-13 20:00:00"}]


class _FakeRegistry:
    """
    函数/类功能与逻辑描述：
        记录型假注册表：`invoke` 记录每次调用并按预设返回结果；`to_openai_schema` 返回预设工具清单
        并记录收到的 tags（用于断言"可见面裁剪"确实按操作类型收窄）。不触网、不连库。
    构造入参说明：
        tools (list | None)：to_openai_schema 的返回（默认空清单）。
        results (dict | None)：{工具名: ToolResult} 的预设返回；未预设的按成功兜底。
    返回值说明：
        构造返回 _FakeRegistry 实例。
    """

    def __init__(self, tools=None, results=None):
        """
        函数功能与逻辑描述：
            保存预设并在实例上初始化调用记录与 tags 记录。
        入参说明：
            tools (list | None)：预设工具清单。
            results (dict | None)：预设工具返回。
        返回值说明：
            无（仅初始化实例属性）。
        """
        self.tools = tools or []
        self.results = results or {}
        self.calls: list[dict] = []
        self.tags_seen = None

    def to_openai_schema(self, agent_id, tags=None):
        """
        函数功能与逻辑描述：
            假导出：记录本次 tags 后返回预设工具清单。
        入参说明：
            agent_id (str)：调用主体。
            tags (tuple | None)：可见面标签。
        返回值说明：
            list：预设工具清单。
        """
        self.tags_seen = tags
        return self.tools

    async def invoke(self, name, args, *, agent_id, trace_id=None,
                     session_id=None, task_id=None):
        """
        函数功能与逻辑描述：
            假 invoke：记录调用并返回预设结果（recent_list 默认返回固定清单）。
        入参说明：
            name (str)：工具名。
            args (dict)：工具入参。
            agent_id (str)：调用主体。
            trace_id / session_id / task_id：引擎上下文（仅记录 task_id）。
        返回值说明：
            ToolResult：预设或兜底的成功结果。
        """
        self.calls.append({"name": name, "args": args, "task_id": task_id})
        if name in self.results:
            return self.results[name]
        if name == "bill.recent_list":
            return ToolResult(ok=True, data=_RECENT)
        if name == "bill.update":
            return ToolResult(ok=True, data={"ok": True, "id": args.get("id"),
                                             "changed": {"amount": [35.0, 50.0]}})
        if name == "bill.delete":
            return ToolResult(ok=True, data={"ok": True, "id": args.get("id"),
                                             "deleted": _RECENT[0]})
        return ToolResult(ok=True, data={"ok": True, "habits": []})

    def names(self) -> list[str]:
        """
        函数功能与逻辑描述：
            返回被调用过的工具名序列（含 habit.recalc），供断言调用链使用。
        入参说明：
            无（隐式 self）。
        返回值说明：
            list[str]：按调用顺序排列的工具名。
        """
        return [c["name"] for c in self.calls]


@pytest.fixture
def fake_reg():
    """
    函数功能与逻辑描述：
        提供一份"模型能看到 2 个工具"的假注册表（edit 任务的典型可见面）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        _FakeRegistry：假注册表实例。
    """
    return _FakeRegistry(tools=[{"type": "function", "function": {"name": "bill.recent_list"}},
                                {"type": "function", "function": {"name": "bill.update"}}])


@pytest.fixture(autouse=True)
def _patch_env(fake_reg):
    """
    函数功能与逻辑描述：
        autouse：把 `get_registry` 打桩为假注册表（节点内为函数级导入，故 patch 定义处生效），
        并确保循环开关处于默认开启态（本文件测的就是循环本身）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（yield 后由 patch 上下文还原）。
    """
    with patch("mcpGateway.registry.get_registry", return_value=fake_reg), \
            patch.object(bill_nodes, "BILL_FC_ENABLED", True):
        yield


def _state(raw: str, op: str) -> dict:
    """
    函数功能与逻辑描述：
        构造 BillState 字典（运行时即 dict），字段与 worker 下发的 init_state 对齐。
    入参说明：
        raw (str)：用户原文。
        op (str)：operate_sub_type（edit / delete）。
    返回值说明：
        dict：可直接喂给 execute_node 的状态字典。
    """
    return {"session_id": "s1", "task_id": "t1", "raw_segments": [raw],
            "operate_sub_type": op, "session_memory": {}, "city": None,
            "month": None, "result": None}


def _payload(cmd) -> dict:
    """
    函数功能与逻辑描述：
        从节点返回的 Command 中取出并解析 result JSON 载荷，供各用例断言。
    入参说明：
        cmd：execute_node 返回的 Command。
    返回值说明：
        dict：解析后的载荷字典。
    """
    return json.loads(cmd.update["result"])


# ===================== ① 可见面锁 =====================
def test_tool_tags_map_by_operate_type():
    """
    函数功能与逻辑描述：
        验证操作类型 → 标签的映射表：`edit` 拿读+改、`delete` 拿读+删、`add` 只拿增，
        未知类型只给只读面（宁可多查一次也不放开写面）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    assert _tool_tags("edit") == ("op:read", "op:edit")
    assert _tool_tags("delete") == ("op:read", "op:delete")
    assert _tool_tags("add") == ("op:add",)
    assert _tool_tags("whatever") == ("op:read",)


@pytest.mark.parametrize("op,forbidden,allowed", [
    ("edit", "bill.delete", "bill.update"),
    ("delete", "bill.update", "bill.delete"),
])
def test_visibility_lock_on_real_registry(op, forbidden, allowed):
    """
    函数功能与逻辑描述：
        ★核心红线：用**真实注册表**验证"操作类型物理锁定可见面"——`edit` 任务里模型根本看不到
        `bill.delete`，`delete` 任务里看不到 `bill.update`。这不是提示词约束，而是能力裁剪：
        模型连工具名都拿不到，就不可能"改着改着顺手删了"。
    入参说明：
        op (str)：参数化注入的操作类型（edit / delete）。
        forbidden (str)：该操作类型下必须**不可见**的工具名。
        allowed (str)：该操作类型下必须**可见**的工具名。
    返回值说明：
        无（断言通过即用例成功）。
    """
    # 直接用模块导入期捕获的**真实**函数引用（绕开本文件的 autouse 打桩，见模块头说明）
    names = {item["function"]["name"]
             for item in _REAL_GET_REGISTRY().to_openai_schema(
                 "bill_agent", tags=_tool_tags(op))}
    assert forbidden not in names
    assert allowed in names
    assert "bill.recent_list" in names


# ===================== ② 循环控制 =====================
@pytest.mark.asyncio
async def test_fc_loop_happy_path_with_observation(fake_reg):
    """
    函数功能与逻辑描述：
        验证标准 tool loop 闭环：第 1 轮查清单 → 第 2 轮改账 → 第 3 轮不再调工具；
        并断言第 2、3 轮的 `extra_messages` 里确实带上了上一轮的 observation
        （`role=tool` 且关联 `tool_call_id`）——这是"模型能看到工具结果再决策"的硬证据。
        最终载荷取自**工具返回值**，模型的一句话总结挂在 `data.reply`。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    responses = [
        {"text": "", "tool_calls": [{"id": "c1", "name": "bill.recent_list",
                                     "arguments": {"limit": 3}}]},
        {"text": "", "tool_calls": [{"id": "c2", "name": "bill.update",
                                     "arguments": {"id": 2, "amount": 50}}]},
        {"text": "已把那笔打车改成 50 元", "tool_calls": None},
    ]
    call = AsyncMock(side_effect=responses)
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call):
        payload = _payload(await execute_node(_state("把昨天那笔打车改成50", "edit")))

    assert payload["success"] is True and payload["msg"] == "账单修改成功"
    assert payload["data"]["id"] == 2
    assert payload["data"]["reply"] == "已把那笔打车改成 50 元"
    # 调用链：查清单 → 改账（+ 重算习惯前的按 id 查月份）
    assert fake_reg.names()[:2] == ["bill.recent_list", "bill.update"]
    assert "habit.recalc" in fake_reg.names()
    # observation 回填：第 2 次调用时 extra_messages 已含 assistant(tool_calls) 与 tool 结果，
    # 且顺序契约是"assistant 在前、tool 紧随其后"（多轮时按轮次成对出现）
    second_kwargs = call.call_args_list[1].kwargs
    msgs = second_kwargs["extra_messages"] or []
    roles = [m["role"] for m in msgs]
    assert roles[:2] == ["assistant", "tool"]
    assert len(roles) % 2 == 0
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1" and "consume_time" in tool_msg["content"]


@pytest.mark.asyncio
async def test_fc_ask_path_produces_no_tool_call(fake_reg):
    """
    函数功能与逻辑描述：
        验证「追问 = 模型不调工具、直接输出话术」：循环的终止条件就是"无 tool_calls"，
        因此追问不需要专门的工具。断言**一次工具调用都没有发生**（"追问路径不产生 tool_call"）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    call = AsyncMock(return_value={"text": "请问您要改哪一笔？", "tool_calls": None})
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call):
        payload = _payload(await execute_node(_state("改一下", "edit")))

    assert payload["success"] is False and payload["error"] == "need_more_info"
    assert payload["data"]["prompt"] == "请问您要改哪一笔？"
    assert fake_reg.names() == []          # 追问路径零工具调用
    assert call.await_count == 1           # 一轮即结束


@pytest.mark.asyncio
async def test_fc_melts_down_at_max_turns(fake_reg):
    """
    函数功能与逻辑描述：
        验证**熔断**：模型每轮都只调读工具（清单）而不给结论时，最多跑 `FC_MAX_TURNS` 轮
        就转追问——自主循环必须有界，否则会无限试探、烧钱且不收敛。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    loop_resp = {"text": "", "tool_calls": [{"id": "c", "name": "bill.recent_list",
                                             "arguments": {"limit": 3}}]}
    call = AsyncMock(return_value=loop_resp)
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call), \
            patch.object(bill_nodes, "FC_MAX_TURNS", 2):
        payload = _payload(await execute_node(_state("改一下那笔", "edit")))

    assert payload["error"] == "need_more_info"
    assert call.await_count == 2                                   # 恰好用满预算
    assert fake_reg.names().count("bill.recent_list") == 2


@pytest.mark.asyncio
async def test_fc_write_idempotency_gate_blocks_second_write(fake_reg):
    """
    函数功能与逻辑描述：
        ★验证**写幂等闸门**：模型第二次调用同一个写工具时被立即拦截并终止循环，
        返回明确的失败载荷（不重复执行、也不继续跑）。这条护栏是为防"模型重复调 `bill.add`
        造成重复记账"而设，直接守住 M5/D2 的成果。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    responses = [
        {"text": "", "tool_calls": [{"id": "c1", "name": "bill.update",
                                     "arguments": {"id": 2, "amount": 50}}]},
        {"text": "", "tool_calls": [{"id": "c2", "name": "bill.update",
                                     "arguments": {"id": 2, "amount": 60}}]},
    ]
    call = AsyncMock(side_effect=responses)
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call):
        payload = _payload(await execute_node(_state("改成50再改成60", "edit")))

    assert payload["success"] is False
    assert "重复调用写工具" in payload["error"]
    assert fake_reg.names().count("bill.update") == 1              # 只执行了一次


@pytest.mark.asyncio
async def test_fc_tool_failure_is_observation_and_continues(fake_reg):
    """
    函数功能与逻辑描述：
        验证「工具失败作为 observation 回填，循环继续」：写工具返回 `ok=False` 时不抛异常、
        不崩循环，而是把 `{ok: false, error}` 交给模型；模型随后不再调工具 → 因**无写成功**
        而落到追问（不是谎报成功）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.results = {"bill.update": ToolResult(
        ok=False, error=ToolError(ToolErrorKind.REMOTE, "未找到该账单，可能已被删除",
                                 tool_name="bill.update"))}
    responses = [
        {"text": "", "tool_calls": [{"id": "c1", "name": "bill.update",
                                     "arguments": {"id": 99, "amount": 50}}]},
        {"text": "那笔账单好像已经不存在了。", "tool_calls": None},
    ]
    call = AsyncMock(side_effect=responses)
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call):
        payload = _payload(await execute_node(_state("把99那笔改成50", "edit")))

    assert payload["success"] is False and payload["error"] == "need_more_info"
    assert payload["data"]["prompt"] == "那笔账单好像已经不存在了。"
    # 失败信息确实以 observation 形式回填给了模型
    second_kwargs = call.call_args_list[1].kwargs
    assert "未找到该账单" in second_kwargs["extra_messages"][1]["content"]


@pytest.mark.asyncio
async def test_fc_never_claims_success_without_write(fake_reg):
    """
    函数功能与逻辑描述：
        ★红线 7：**模型声称"已完成"但没有任何写工具成功**时，绝不能报成功。
        实测模型在没有对应工具的情况下会回答"已删除"——若这里信了文本，用户会以为账改了而实际没改。
        断言：即便模型说了"已删除"，出口仍是追问。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    responses = [
        {"text": "", "tool_calls": [{"id": "c1", "name": "bill.recent_list",
                                     "arguments": {"limit": 3}}]},
        {"text": "已删除那笔账单。", "tool_calls": None},
    ]
    call = AsyncMock(side_effect=responses)
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call):
        payload = _payload(await execute_node(_state("删掉昨天那笔打车", "delete")))

    assert payload["success"] is False
    assert payload["error"] == "need_more_info"
    assert "bill.delete" not in fake_reg.names()


@pytest.mark.asyncio
async def test_fc_channel_error_becomes_ask(fake_reg):
    """
    函数功能与逻辑描述：
        验证通道异常的兜底：`call_llm_fc` 抛 RuntimeError（如 server 返回 `FINANCE_ERROR:` 或
        非法 JSON）时，节点**不把异常抛出图外**，而是转成追问，让用户知道"通道暂时不可用"
        而不是收到一个静默的错误结论。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    call = AsyncMock(side_effect=RuntimeError("FINANCE_ERROR: 模型不可用"))
    with patch.object(bill_nodes.mcp_client, "call_llm_fc", new=call):
        payload = _payload(await execute_node(_state("改成50", "edit")))

    assert payload["success"] is False and payload["error"] == "need_more_info"
    assert "通道" in payload["data"]["prompt"]
