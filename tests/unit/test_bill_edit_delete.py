# -*- coding: utf-8 -*-
"""M15（P3a 确定性驱动）修改 / 删除分支单元测试。

设计依据：`设计/12` §4.2.4 / §4.4 / §4.6 / §4.7 第 3 层（施工序 P3a）。
覆盖（对应 `11` §3 M15 的 P3a 验收方式 ①–④）：
  ① 目标定位规则（纯函数穷举：序号 / 主键 / 金额 / 金额+属性 / 属性 / 指代词 / 空窗口 / 歧义）；
  ② edit 分支五分支（无目标追问 / 目标窗口为空失败 / 部分更新 / 金额≤0 失败 / 无新值追问）+ 跨月重算；
  ③ delete 分支四分支（首次写草稿追问 / 确认后执行 / 未确认不删 / 删 0 行失败并清草稿）；
  ④ add 路径零回归（SQL 形状、参数顺序、task_id 写入逐字不变）。

★不触网、不连库：`mcpGateway.registry.get_registry` 被替换为记录型假注册表（工具调用全被捕获），
  模型侧仅在"确定性路径拿不到新金额"的用例里 patch `parse_json_output`。
"""
import json

import pytest
from unittest.mock import AsyncMock, patch

from datetime import date, timedelta

from agents.bill_agent import bill_target
from agents.bill_agent.nodes import execute_node, BILL_AGENT_ID
from memory.short_memory import SESSION_DRAFT_CACHE, SESSION_MEM, SESSION_ASK_ROUND
from mcpGateway.tool_model import ToolError, ToolErrorKind, ToolResult

# 固定"最近账单清单"（id 倒序，与 bill.recent_list 的返回口径一致）。
# ★日期用**相对今天**动态生成：定位规则里的"昨天/今天"是按当天推导的，
#   若把日期写死，用例会在第二天自动失效（这是 P0-b 那类"跑一次就过期"的坑）。
_TODAY = date.today().isoformat()
_YESTERDAY = (date.today() - timedelta(days=1)).isoformat()
# 跨月用例的目标日期：取「下个月的 5 号」——保证与"昨天所在月"**恒不相同**。
# 反例（本次修复的动因）：原先的目标日期写死为 "2026-10-05"，当"昨天"恰好也落在
#   2026-10 时，源月 == 目标月，代码只重算一次（行为正确），而用例期望两次 ——
#   于是该用例只在 2026-10-02 ~ 10-31 之间必挂。改用「下月」后二者恒不相等。
_NEXT_MONTH_1ST = (date.today().replace(day=1) + timedelta(days=32)).replace(day=1)
_NEXT_MONTH_5TH = _NEXT_MONTH_1ST.replace(day=5).isoformat()
_BILLS = [
    {"id": 3, "amount": 30.0, "category": "购物", "consume_time": _TODAY,
     "remark": "买书", "create_time": f"{_TODAY} 20:00:00"},
    {"id": 2, "amount": 35.0, "category": "交通", "consume_time": _YESTERDAY,
     "remark": "昨天打车35元", "create_time": f"{_YESTERDAY} 20:00:00"},
    {"id": 1, "amount": 35.0, "category": "餐饮", "consume_time": "2026-09-12",
     "remark": "吃饭35元", "create_time": "2026-09-12 20:00:00"},
]


class _FakeRegistry:
    """
    函数/类功能与逻辑描述：
        记录型假工具注册表：把每次 invoke 的（工具名 / 入参 / task_id）记进 calls，
        并按工具名返回预设结果。用于断言"到底调了哪个工具、传了什么参数"——
        这正是 P3a 的核心契约（定位与确认在代码、落库在工具）所必须观测的。
    构造入参说明：
        recent (list | None)：bill.recent_list 的返回清单，默认 None（视为空清单）。
        fail (tuple)：需要伪造失败的工具名（返回 ok=False），默认空。
        changed (dict | None)：bill.update 返回的 changed，默认 None（按入参自动生成）。
    返回值说明：
        构造返回 _FakeRegistry 实例；调用能力见 invoke。
    """

    def __init__(self, recent=None, *, fail=(), changed=None):
        """
        函数功能与逻辑描述：
            保存预设返回值并初始化 calls 记录表。
        入参说明：
            recent (list | None)：最近账单清单。
            fail (tuple)：要伪造失败的工具名集合。
            changed (dict | None)：bill.update 的 changed 返回。
        返回值说明：
            无（仅初始化实例属性）。
        """
        self.recent = recent if recent is not None else []
        self.fail = set(fail)
        self.changed = changed
        self.calls: list[dict] = []

    async def invoke(self, name, args, *, agent_id, trace_id=None,
                     session_id=None, task_id=None):
        """
        函数功能与逻辑描述：
            假 invoke：记录调用后按工具名返回预设 ToolResult（失败集命中则返回 ok=False）。
        入参说明：
            name (str)：工具名。
            args (dict)：工具入参。
            agent_id (str)：调用主体。
            trace_id / session_id / task_id：引擎上下文字段（本桩仅记录 task_id）。
        返回值说明：
            ToolResult：按预设构造的结果对象。
        """
        self.calls.append({"name": name, "args": args,
                           "agent_id": agent_id, "task_id": task_id})
        if name in self.fail:
            return ToolResult(ok=False, error=ToolError(
                ToolErrorKind.REMOTE, "boom", tool_name=name))
        if name == "bill.recent_list":
            return ToolResult(ok=True, data=self.recent)
        if name == "bill.update":
            changed = self.changed
            if changed is None:
                changed = {k: ["旧值", v] for k, v in args.items() if k != "id"}
            return ToolResult(ok=True, data={"ok": True, "id": args.get("id"),
                                             "changed": changed})
        if name == "bill.delete":
            return ToolResult(ok=True, data={
                "ok": True, "id": args.get("id"),
                "deleted": {"id": args.get("id"), "amount": 35.0,
                            "category": "交通", "consume_time": "2026-09-13"}})
        return ToolResult(ok=True, data={"ok": True, "habits": []})

    def names(self) -> list[str]:
        """
        函数功能与逻辑描述：
            返回本次被执行过的工具名序列，供"是否调用某工具/调用了几次"的断言使用。
        入参说明：
            无（隐式 self）。
        返回值说明：
            list[str]：按调用顺序排列的工具名列表。
        """
        return [c["name"] for c in self.calls]


@pytest.fixture
def fake_reg():
    """
    函数功能与逻辑描述：
        提供一份干净的记录型假注册表（默认：最近清单为空），用例可自由改写 recent / fail。
        function 作用域，每个用例拿到全新实例与全新调用记录。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        _FakeRegistry：假注册表实例。
    """
    return _FakeRegistry()


@pytest.fixture(autouse=True)
def _patch_env(fake_reg):
    """
    函数功能与逻辑描述：
        autouse 环境隔离：① 把 `get_registry` 打桩为假注册表（节点内是函数级导入，
        故 patch 目标为定义处 `mcpGateway.registry.get_registry`，与现有 test_bill_agent 同款）；
        ② 清空短期记忆的三个全局字典，避免用例间草稿/计数互相污染。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（yield 结束后由 patch 上下文还原）。
    """
    SESSION_MEM.clear()
    SESSION_DRAFT_CACHE.clear()
    SESSION_ASK_ROUND.clear()
    # ★P3b 之后 edit/delete 默认走自主循环（BILL_FC_ENABLED=1）；本文件测的是 **P3a 确定性路径**，
    #   故显式关掉循环开关，使分派落到规则定位 + 代码确认那一条（对应施工序 P3a）。
    with patch("mcpGateway.registry.get_registry", return_value=fake_reg), \
            patch("agents.bill_agent.nodes.BILL_FC_ENABLED", False):
        yield


def _state(raw: str, op: str, **kw) -> dict:
    """
    函数功能与逻辑描述：
        构造 BillState 字典（TypedDict 运行时即 dict），字段与 worker 下发的 init_state 对齐。
    入参说明：
        raw (str)：用户原文（首段）。
        op (str)：operate_sub_type（add/edit/delete）。
        **kw：覆盖任意字段（如 session_memory）。
    返回值说明：
        dict：可直接喂给 execute_node 的状态字典。
    """
    state = {"session_id": "s1", "task_id": "t1", "raw_segments": [raw],
             "operate_sub_type": op, "session_memory": {}, "city": None,
             "month": None, "result": None}
    state.update(kw)
    return state


# ===================== ① 目标定位规则（纯函数穷举）=====================
def test_locate_explicit_index_and_id():
    """
    函数功能与逻辑描述：
        验证规则①：原文显式指认序号（"第 2 条"）或主键（"id=3"）时直接采纳，
        且序号按清单下标解释（第 1 条 = 最近一笔，与 UI 展示同口径）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    row, reason = bill_target.locate_bill("把第 2 条改成 50", _BILLS)
    assert row["id"] == 2 and reason == bill_target.REASON_OK
    row2, _ = bill_target.locate_bill("id=3 那笔删掉", _BILLS)
    assert row2["id"] == 3


def test_locate_by_amount_unique():
    """
    函数功能与逻辑描述：
        验证规则②：金额唯一命中时采纳（"把 30 那笔改成 50"）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    row, reason = bill_target.locate_bill("把 30 那笔改成 50", _BILLS)
    assert row["id"] == 3 and reason == bill_target.REASON_OK


def test_locate_by_amount_plus_category_narrows():
    """
    函数功能与逻辑描述：
        验证规则③：金额命中多笔（35 元有两笔）时，用**品类线索**收窄到唯一一笔。
        这正是"定位数字 + 属性"协同的场景，也是本规则集存在的主要理由。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    row, reason = bill_target.locate_bill("把 35 那笔打车改成 50", _BILLS, category="交通")
    assert row["id"] == 2 and reason == bill_target.REASON_OK


def test_locate_ambiguous_when_multiple_match():
    """
    函数功能与逻辑描述：
        验证歧义出口：35 元有两笔且无线索可区分时必须返回 AMBIGUOUS（而不是随便挑一笔）——
        定位错=改错账，属不可逆事故，宁可多问一句。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    row, reason = bill_target.locate_bill("把 35 那笔改成 50", _BILLS)
    assert row is None and reason == bill_target.REASON_AMBIGUOUS


def test_locate_by_date_and_category_hint():
    """
    函数功能与逻辑描述：
        验证规则④/⑤：无金额线索时按"日期线索 + 品类"定位（"昨天那笔打车"）。
        日期线索由 `extract_date_hint` 归一为 YYYY-MM-DD 后与 consume_time 前缀比对。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    bills = [dict(b) for b in _BILLS[:1]]
    bills.append({"id": 9, "amount": 12.0, "category": "交通",
                  "consume_time": yesterday, "remark": "昨天打车"})
    row, reason = bill_target.locate_bill("把昨天那笔打车改成 50", bills, category="交通")
    assert row["id"] == 9 and reason == bill_target.REASON_OK


def test_locate_refer_word_only_for_single_candidate():
    """
    函数功能与逻辑描述：
        验证规则⑥（指代词兜底）的**安全边界**：窗口内只有 1 笔时才敢把"那笔"认定为它；
        有多笔时必须返回歧义——多笔场景下用户自己都指不清，规则更不该猜。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    single = [_BILLS[0]]
    row, reason = bill_target.locate_bill("把那笔改成 50", single)
    assert row["id"] == 3 and reason == bill_target.REASON_OK

    row2, reason2 = bill_target.locate_bill("把那笔改成 50", _BILLS)
    assert row2 is None and reason2 == bill_target.REASON_AMBIGUOUS


def test_locate_empty_window_and_no_hit():
    """
    函数功能与逻辑描述：
        验证两种"非歧义"的失败出口：窗口为空（EMPTY，属业务性不可操作）与
        线索完全对不上（NOT_FOUND）。二者对用户的应对方式不同，必须区分。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    assert bill_target.locate_bill("改成 50", []) == (None, bill_target.REASON_EMPTY)
    row, reason = bill_target.locate_bill("把 999 那笔改成 50", _BILLS)
    assert row is None and reason == bill_target.REASON_NOT_FOUND


def test_is_confirm_substring_semantics():
    """
    函数功能与逻辑描述：
        验证确认词判定为**子串命中**（"嗯，删吧"这类带修饰短句要能识别），
        且普通陈述不会被误判为确认。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    assert bill_target.is_confirm("嗯，删吧")
    assert bill_target.is_confirm("确认")
    assert not bill_target.is_confirm("再加一杯咖啡 20 元")
    assert not bill_target.is_confirm("")


# ===================== ② edit 分支 =====================
@pytest.mark.asyncio
async def test_edit_partial_update_amount_only(fake_reg):
    """
    函数功能与逻辑描述：
        验证 edit 的**部分更新**：原文只给了新金额时，工具入参**只含 id + amount**
        （不带 category / consume_date）——这正是"未提到的字段一律不动"的落地保证；
        同时断言 task_id 被透传给工具（崩溃恢复对账依据）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表（用例内设定最近清单）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    cmd = await execute_node(_state("把昨天那笔打车改成50", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True
    assert payload["msg"] == "账单修改成功"
    update_call = [c for c in fake_reg.calls if c["name"] == "bill.update"][0]
    assert update_call["args"] == {"id": 2, "amount": 50.0}
    assert update_call["task_id"] == "t1"
    assert payload["data"]["id"] == 2 and payload["data"]["amount"] == 50.0


@pytest.mark.asyncio
async def test_edit_no_new_value_asks_back(fake_reg):
    """
    函数功能与逻辑描述：
        验证「edit 无任何新值 → 追问」（失败矩阵）：原文既无新金额、也无可识别的品类/日期线索时，
        必须追问而不是"改成原值"或静默成功。此处模型补抽也不给值时仍应追问。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = [_BILLS[0]]                 # 单笔 → 指代词可命中
    with patch("agents.bill_agent.nodes.parse_json_output",
               new=AsyncMock(return_value={"need_more_info": False, "prompt": "",
                                           "valid": True, "amount": None,
                                           "category": "", "consume_date": ""})):
        cmd = await execute_node(_state("把那笔改一下", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["error"] == "need_more_info"
    assert "改成" in payload["data"]["prompt"]
    assert "bill.update" not in fake_reg.names()


@pytest.mark.asyncio
async def test_edit_non_positive_amount_rejected(fake_reg):
    """
    函数功能与逻辑描述：
        验证「新金额 ≤ 0 → 失败（不写库）」：负号在改写动词后仍被识别为非法金额，
        绝不"修正"为正数，且**不产生任何写调用**。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    cmd = await execute_node(_state("把昨天那笔打车改成-30", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["error"] == "消费金额必须大于0"
    assert "bill.update" not in fake_reg.names()


@pytest.mark.asyncio
async def test_edit_target_window_empty_fails(fake_reg):
    """
    函数功能与逻辑描述：
        验证「窗口内没有任何账单 → 业务性失败」：返回 success=False 且**不是**追问
        （追问会让编排侧计轮次，而这里根本没有可操作对象，属另一类出口）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表（最近清单为空）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    cmd = await execute_node(_state("把昨天那笔打车改成50", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["error"] == "最近没有任何账单记录，无法修改"


@pytest.mark.asyncio
async def test_edit_ambiguous_lists_candidates(fake_reg):
    """
    函数功能与逻辑描述：
        验证歧义出口把**候选清单**给到用户（带序号，可直接回"第 2 条"），
        而不是只回一句"听不懂"——这是"追问要给出可执行下一步"的落地。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    cmd = await execute_node(_state("把 35 那笔改成 50", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["error"] == "need_more_info"
    assert "1. " in payload["data"]["prompt"] and "3. " in payload["data"]["prompt"]


@pytest.mark.asyncio
async def test_edit_cross_month_recalcs_both_months(fake_reg):
    """
    函数功能与逻辑描述：
        验证改期跨月时的派生数据修复：`habit.recalc` 必须**同时重算旧月与新月**
        （否则旧月习惯会留着已迁走的金额、新月又缺这笔）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = [_BILLS[1]]                  # id=2，consume_time=昨天，交通
    # ★必须 patch 模型抽取：本句只改日期（"改到"不在金额改写动词表里），确定性路径拿不到金额，
    #   代码会去问模型——若不 patch，会经 `client` 内部**模块级**绑定的 get_registry
    #   走到真实外部模型（测试变慢、且消耗真实额度）。打桩目标与节点内的导入方式必须一致，
    #   这正是 P2 验收②「打桩点同步」要防的那类失配。
    with patch("agents.bill_agent.nodes.parse_json_output",
               new=AsyncMock(return_value={"need_more_info": False, "prompt": "",
                                           "valid": True, "amount": None,
                                           "category": "", "consume_date": ""})):
        cmd = await execute_node(_state(f"把昨天那笔打车改到 {_NEXT_MONTH_5TH}", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True
    recalc_args = [c["args"]["month"] for c in fake_reg.calls if c["name"] == "habit.recalc"]
    # 源月（昨天所在月）与目标月（下月）恒不相同 → 必然重算两次；
    # 两侧都 sorted，消除对 habit.recalc 调用顺序的隐含依赖
    assert sorted(recalc_args) == sorted([_YESTERDAY[:7], _NEXT_MONTH_5TH[:7]])


@pytest.mark.asyncio
async def test_edit_tool_failure_reports_failure(fake_reg):
    """
    函数功能与逻辑描述：
        验证「SQL 写失败 / 更新未生效 → 失败」：工具返回 ok=False 时不得报成功
        （写操作是否成功只以工具返回值为准，`设计/12` §6 红线 7）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表（用例内把 bill.update 设为必失败）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    fake_reg.fail = {"bill.update"}
    cmd = await execute_node(_state("把昨天那笔打车改成50", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert payload["msg"] == "账单更新失败"
    assert "habit.recalc" not in fake_reg.names()


@pytest.mark.asyncio
async def test_delete_skips_fc_when_enabled(fake_reg):
    """
    函数功能与逻辑描述：
        ★缺陷 A 防复发：即使 `BILL_FC_ENABLED=True`，删除也必须走 P3a 确定性路径。
        删除**不可逆**，§4.4 的"先确认后删"不能交给模型裁量（实测 FC 路径会跳过确认直接删库）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    with patch("agents.bill_agent.nodes.BILL_FC_ENABLED", True), \
            patch("agents.bill_agent.nodes._execute_fc", new=AsyncMock()) as fc:
        cmd = await execute_node(_state("删掉昨天那笔打车", "delete"))
    fc.assert_not_awaited()                          # ★没有走 FC
    payload = json.loads(cmd.update["result"])
    assert payload["error"] == "need_more_info"
    assert "确认删除" in payload["data"]["prompt"]
    assert "bill.delete" not in fake_reg.names()     # 只到确认轮，绝不删


@pytest.mark.asyncio
async def test_non_explicit_id_edit_still_uses_fc(fake_reg):
    """
    函数功能与逻辑描述：
        ★分流的**另一半**：自然语言指代（无显式 id）且 FC 开启时，必须走 FC 自主循环
        ——若哪天把"按输入形态分流"的条件写反（把自然语言也塞进 P3a），本用例会失败。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    with patch("agents.bill_agent.nodes.BILL_FC_ENABLED", True), \
            patch("agents.bill_agent.nodes._execute_fc",
                  new=AsyncMock(return_value="FC_PATH")) as fc:
        await execute_node(_state("把昨天那笔打车改成 50", "edit"))
    fc.assert_awaited_once()                         # ★确实走了 FC
    assert "bill.update" not in fake_reg.names()     # 确定性写路径未被使用


@pytest.mark.asyncio
async def test_explicit_id_not_found_falls_back_to_window(fake_reg):
    """
    函数功能与逻辑描述：
        边界：显式 id **查无此行**（主键读数返回 None）→ 必须继续用窗口清单尝试定位
        （用户可能写错 id，但话里还有别的线索），且**绝不因为"没找到"就写库**。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    with patch("mcpGateway.bill_tools.get_bill_by_id", return_value=None):
        cmd = await execute_node(_state("把 id=99999 那笔改成 50", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is not True            # 未命中 → 追问/失败，绝不瞎改
    assert "bill.update" not in fake_reg.names()


@pytest.mark.asyncio
async def test_explicit_id_edit_rejects_non_positive_amount(fake_reg):
    """
    函数功能与逻辑描述：
        显式 id 路径下，"新金额 ≤ 0"的既有红线同样成立：拒绝且**不写库**。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    with patch("mcpGateway.bill_tools.get_bill_by_id",
               return_value={"id": 2, "amount": 35.0, "category": "交通",
                             "consume_time": "2026-09-13", "remark": "昨天打车35元",
                             "create_time": "2026-09-13 20:00:00"}):
        cmd = await execute_node(_state("把 id=2 那笔改成 -5", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False
    assert "bill.update" not in fake_reg.names()


@pytest.mark.asyncio
async def test_explicit_id_delete_not_found_never_deletes(fake_reg):
    """
    函数功能与逻辑描述：
        边界：显式 id 删除但查无此行 → 不写草稿、不删除（防"删了个不存在还报成功"）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    with patch("mcpGateway.bill_tools.get_bill_by_id", return_value=None):
        cmd = await execute_node(_state("删掉 id=99999 那笔", "delete"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is not True
    assert "bill.delete" not in fake_reg.names()
    assert SESSION_DRAFT_CACHE["s1"].get("bill_delete") is None    # 不写草稿


@pytest.mark.asyncio
async def test_fetch_bill_by_id_error_returns_none(fake_reg):
    """
    函数功能与逻辑描述：
        主键读数**异常时返回 None**（由调用方回落窗口规则），绝不把异常抛到图外——
        面板/链路遇到读数故障应降级为"按窗口规则找"，而不是整轮失败。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from agents.bill_agent import nodes as bill_nodes
    with patch("mcpGateway.bill_tools.get_bill_by_id", side_effect=RuntimeError("boom")):
        assert await bill_nodes._fetch_bill_by_id(245) is None


# ============ ③-0 显式 id 契约（2026-09-16 补：防缺陷 D / E 复发）============
# 背景（详见 `设计/11` M15 验收记录第二轮补验收）：
#   缺陷 D：显式 `id=` 只在"最近 20 笔"里查找 → UI 面板（窗口 200）里能选中的 #245
#           被判"没找到符合条件的账单"。修复 = 显式主键**按主键直取该行**，不受窗口约束。
#   缺陷 E：`edit` 走 FC 自主循环时模型把 `bill.update` 的 tool_call 形态当正文输出，
#           工具未执行、数据未改。修复 = **显式 id 恒走 P3a 确定性路径**（用户拍板）。
# 这两条是本轮两个真 bug 的专用防复发用例——缺了它们，同类问题会再次逃过 CI。
_EXPLICIT_ID_ROW = {
    "id": 245, "amount": 30.0, "category": "餐饮", "consume_time": "2026-08-30",
    "remark": "晚饭30元", "create_time": "2026-08-30 20:00:00",
}


@pytest.mark.asyncio
async def test_explicit_id_bypasses_window(fake_reg):
    """
    函数功能与逻辑描述：
        ★显式主键**不受窗口约束**：目标 id 不在 `bill.recent_list` 清单里时，仍必须定位成功
        （而不是回"没找到符合条件的账单"）。防的是缺陷 D——UI 面板窗口比 Agent 定位窗口宽，
        导致"面板里能选中、Agent 说找不到"。
        做法：把假注册表的 recent 清单设为**不含 id=245** 的 _BILLS，并打桩按主键取数命中该行。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS                       # 窗口清单里【没有】245
    with patch("mcpGateway.bill_tools.get_bill_by_id", return_value=_EXPLICIT_ID_ROW):
        cmd = await execute_node(_state("删掉 id=245 那笔", "delete"))
    payload = json.loads(cmd.update["result"])
    assert payload["error"] == "need_more_info"     # 走到确认追问，而不是"没找到"
    assert "确认删除" in payload["data"]["prompt"]
    assert SESSION_DRAFT_CACHE["s1"]["bill_delete"]["id"] == 245
    assert "bill.delete" not in fake_reg.names()    # 定位轮绝不删（安全红线）


@pytest.mark.asyncio
async def test_explicit_id_uses_deterministic_path(fake_reg):
    """
    函数功能与逻辑描述：
        ★显式 id 在 **FC 开启**时也走 P3a 确定性路径：目标由主键唯一确定，不需要也不该
        交给模型自主定位。防的是缺陷 E——FC 路径下模型会把 `bill.update` 的 tool_call
        形态当**正文**输出（回复 `{"name":"bill.update","parameters":{…}}`），工具并未执行、
        数据没改。
        做法：`BILL_FC_ENABLED=True` 时发显式 id 的修改指令，断言 ① FC 入口**未被调用**；
        ② 确定性路径真的把 `bill.update` 发出去了（入参为 id + 新金额）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    with patch("agents.bill_agent.nodes.BILL_FC_ENABLED", True), \
            patch("agents.bill_agent.nodes._execute_fc", new=AsyncMock()) as fc, \
            patch("mcpGateway.bill_tools.get_bill_by_id",
                  return_value=dict(_EXPLICIT_ID_ROW, id=2, amount=35.0)):
        cmd = await execute_node(_state("把 id=2 那笔改成 50", "edit"))
    fc.assert_not_awaited()                        # ★没有走 FC 自主循环
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True
    upd = [c for c in fake_reg.calls if c["name"] == "bill.update"][0]
    assert upd["args"]["id"] == 2 and upd["args"]["amount"] == 50.0


# ===================== ③ delete 分支 =====================
@pytest.mark.asyncio
async def test_delete_first_round_drafts_and_asks(fake_reg):
    """
    函数功能与逻辑描述：
        验证删除的**第一次轮次**：定位到目标后写入 `bill_delete` 草稿槽并追问确认，
        ★本轮**绝不执行删除**（断言 bill.delete 未被调用）——这是"未确认不删"的正面证据。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    fake_reg.recent = _BILLS
    cmd = await execute_node(_state("删掉昨天那笔打车", "delete"))
    payload = json.loads(cmd.update["result"])
    assert payload["error"] == "need_more_info"
    assert "确认删除" in payload["data"]["prompt"]
    assert "bill.delete" not in fake_reg.names()
    assert SESSION_DRAFT_CACHE["s1"]["bill_delete"]["id"] == 2


@pytest.mark.asyncio
async def test_delete_confirm_round_executes(fake_reg):
    """
    函数功能与逻辑描述：
        验证**第二轮的确认执行**：草稿槽存在同一 id 且本轮命中确认词时才调用 `bill.delete`，
        成功后清空草稿并重算该月习惯（派生数据修复）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from memory.short_memory import get_session_memory
    get_session_memory("s1").set_draft("bill_delete", {
        "id": 2, "amount": 35.0, "category": "交通", "consume_time": "2026-09-13"})
    fake_reg.recent = _BILLS
    cmd = await execute_node(_state("确认", "delete"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True and payload["msg"] == "账单删除成功"
    del_call = [c for c in fake_reg.calls if c["name"] == "bill.delete"][0]
    assert del_call["args"] == {"id": 2} and del_call["task_id"] == "t1"
    assert SESSION_DRAFT_CACHE["s1"]["bill_delete"] is None
    assert [c["args"]["month"] for c in fake_reg.calls if c["name"] == "habit.recalc"] == ["2026-09"]


@pytest.mark.asyncio
async def test_delete_without_confirmation_never_deletes(fake_reg):
    """
    函数功能与逻辑描述：
        ★安全红线：草稿存在但本轮**不是**确认词（用户换了话题或改了说法）时，必须不删除
        （走定位轮并覆盖草稿）。这条用例锁住"随口一句'可以'就能删账"的风险。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from memory.short_memory import get_session_memory
    get_session_memory("s1").set_draft("bill_delete", {
        "id": 2, "amount": 35.0, "category": "交通", "consume_time": "2026-09-13"})
    fake_reg.recent = _BILLS
    cmd = await execute_node(_state("算了先别删了", "delete"))
    payload = json.loads(cmd.update["result"])
    assert "bill.delete" not in fake_reg.names()
    assert payload["error"] == "need_more_info"     # 定位不到 → 追问候选


@pytest.mark.asyncio
async def test_delete_zero_row_failure_clears_draft(fake_reg):
    """
    函数功能与逻辑描述：
        验证「目标不存在（删 0 行）→ 失败」且**清空草稿**：否则陈旧草稿会被下一句"确认"复用，
        造成"确认删除的是另一笔"的错觉（工具侧已把删 0 行判失败，此处验证编排侧的一致性）。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表（用例内把 bill.delete 设为必失败）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from memory.short_memory import get_session_memory
    get_session_memory("s1").set_draft("bill_delete", {
        "id": 2, "amount": 35.0, "category": "交通", "consume_time": "2026-09-13"})
    fake_reg.fail = {"bill.delete"}
    cmd = await execute_node(_state("确认删除", "delete"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is False and payload["msg"] == "账单删除失败"
    assert SESSION_DRAFT_CACHE["s1"]["bill_delete"] is None


# ===================== ④ add 路径零回归 =====================
@pytest.mark.asyncio
async def test_add_path_sql_shape_and_task_id_unchanged():
    """
    函数功能与逻辑描述：
        P3a 的回归红线：**add 路径的落库形态逐字不变**——SQL 前缀、参数顺序
        （amount, category, consume_time, remark, task_id）与 task_id 来源都不得改动。
        （P1 只是把同样的 SQL 搬进工具，节点侧 add 分支未被本阶段触碰，此用例即为其护栏。）
    入参说明：
        无（pytest 自动发现并调用；用例内分别 patch parse_json_output 与 call_bill_sql）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    mock_sql = AsyncMock(return_value="成功")
    with patch("agents.bill_agent.nodes.parse_json_output",
               new=AsyncMock(return_value={"need_more_info": False, "prompt": "",
                                           "valid": True, "amount": 35.0,
                                           "category": "餐饮",
                                           "consume_date": "2026-09-13"})), \
            patch("agents.bill_agent.nodes.mcp_client") as mock_client, \
            patch("mcpGateway.registry.get_registry") as mock_reg:
        mock_client.call_bill_sql = mock_sql
        mock_reg.return_value.invoke = AsyncMock(
            return_value=ToolResult(ok=True, data={"ok": True, "habits": []}))
        cmd = await execute_node(_state("昨天打车35元", "add"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True and payload["msg"] == "记账成功"

    kwargs = mock_sql.call_args
    assert kwargs[0][0] == BILL_AGENT_ID                    # 鉴权主体
    sql, params = kwargs[0][1], kwargs[0][2]
    assert sql == ("INSERT INTO bill (amount, category, consume_time, remark, task_id) "
                   "VALUES (?, ?, ?, ?, ?)")
    assert params[0] == 35.0                                 # amount
    assert params[1] == "交通"                                # category（规则兜底：打车→交通）
    assert params[2] == _YESTERDAY                           # consume_time（原文"昨天"优先于模型日期）
    assert "昨天打车35元" in params[3]                        # remark（用户原话）
    assert params[4] == "t1"                                 # task_id（对账依据，代码层写入）


@pytest.mark.asyncio
async def test_explicit_id_locates_beyond_window(fake_reg):
    """
    函数功能与逻辑描述：
        ★UI 点选场景的关键保障：用户用 `id=N` **显式指定**目标时，即便该条不在"最近 3 笔"窗口内，
        也必须能定位到——窗口约束的本意是防"自然语言指代"的歧义，不应用于限制"显式指定"。
        断言两件事：① 取数时 limit 被放宽到 `RECENT_BILL_MAX`；② 目标被正确定位并更新成功。
    入参说明：
        fake_reg：pytest fixture 注入的假注册表。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from config.config import RECENT_BILL_MAX
    fake_reg.recent = [{"id": 42, "amount": 12.0, "category": "餐饮",
                        "consume_time": "2026-01-05", "remark": "很久以前的一笔"}]
    cmd = await execute_node(_state("把 id=42 那笔改成 20", "edit"))
    payload = json.loads(cmd.update["result"])
    assert payload["success"] is True and payload["data"]["id"] == 42
    list_call = [c for c in fake_reg.calls if c["name"] == "bill.recent_list"][0]
    assert list_call["args"]["limit"] == RECENT_BILL_MAX
