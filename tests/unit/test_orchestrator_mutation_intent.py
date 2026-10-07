# -*- coding: utf-8 -*-
"""M15 变更意图（改 / 删）回归用例 —— 防**缺陷 A / B / C** 复发（2026-09-16 补）。

背景（详见 `设计/11` M15 验收记录）：
- **缺陷 C（意图路由）**：`改成 / 改为 / 删掉` 类变更意图**不在编排器词表**里 → 被
  `_is_consume_statement()` 判成"纯消费陈述" → 兜底补 `price` 任务 → 用户说"改"，
  系统回「已为您记账：美妆 60 元（溢价率 …）」这种**假结果**。
- **缺陷 B（结果渲染）**：`build_user_display_text` 用硬下标 `data['category']`，而 delete
  成功载荷只有 `{"id": …}` → 抛 `KeyError`，用户看到「【系统异常】'category'」。
- **缺陷 A（删除确认）**：delete 曾走 FC 自主循环、模型跳过二次确认直接删库（该分派点的
  用例在 `test_bill_edit_delete.py`）。

为什么单独成文件：这些都是**纯函数**（`prune_tasks_by_heuristics` / `_is_consume_statement` /
`build_user_display_text`），不需要真库、真网、真模型，因此用最轻的方式把行为钉死——
避免再次出现"改完只手工验证、没有用例兜底"。
"""
import pytest

from agents.orchestrator.nodes import (
    DELETE_HINT_KEYWORDS,
    EDIT_HINT_KEYWORDS,
    _is_consume_statement,
    build_user_display_text,
    prune_tasks_by_heuristics,
)

# 变更语义的典型输入：显式主键（UI 点选产物）与自然语言指代各两条
_MUTATIONS = [
    ("把 id=515 那笔改成 50", "edit"),
    ("把昨天那笔打车改成 50", "edit"),
    ("删掉 id=245 那笔", "delete"),
    ("删掉昨天那笔打车", "delete"),
]


@pytest.mark.parametrize("text,_sub", _MUTATIONS)
def test_mutation_never_treated_as_consume_statement(text, _sub):
    """
    函数功能与逻辑描述：
        ★缺陷 C 的核心断言：变更指令**绝不能**被判成"纯消费陈述"——一旦被判成陈述，
        兜底逻辑会补 `price` 对标任务，用户说"改"却收到一段比价结论。
    入参说明：
        text (str)：变更指令原文（参数化）。
        _sub (str)：期望的 operate_sub_type（本用例不使用，仅为参数化成对）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    assert _is_consume_statement(text) is False


def test_edit_keywords_share_bill_target_verb_table():
    """
    函数功能与逻辑描述：
        编排层的改动词表与 `bill_target.CHANGE_VERB_RE` **同口径**：后者是定位/新值提取的
        事实源，两处若漂移，会出现"编排认、定位不认"的裂缝。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    from agents.bill_agent.bill_target import CHANGE_VERB_RE

    for kw in ("改成", "改为", "调整为", "修改为", "更新为", "设为"):
        assert kw in EDIT_HINT_KEYWORDS, kw
        assert CHANGE_VERB_RE.search(f"{kw} 50"), kw


@pytest.mark.parametrize("text,sub", _MUTATIONS)
def test_prune_adds_bill_task_for_mutation(text, sub):
    """
    函数功能与逻辑描述：
        ★模型**漏规划**（返回空任务表）时，规则侧必须补出 bill 任务并**带正确的
        operate_sub_type**——这是"改/删"能落到执行层的前提。
    入参说明：
        text (str)：变更指令原文（参数化）。
        sub (str)：期望的 operate_sub_type（edit / delete）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    out = prune_tasks_by_heuristics(text, [])
    assert [t.get("name") for t in out] == ["bill"], out
    assert out[0]["operate_sub_type"] == sub


@pytest.mark.parametrize("text,sub", _MUTATIONS)
def test_prune_drops_misrouted_price_and_keeps_bill(text, sub):
    """
    函数功能与逻辑描述：
        ★缺陷 C 的第二个面：小模型常把变更指令**误规划成 `price`**（实测），规则侧必须
        把它裁掉、换成 bill——否则链路走到 price_agent，用户得到比价结论而非改删结果。
    入参说明：
        text (str)：变更指令原文（参数化）。
        sub (str)：期望的 operate_sub_type。
    返回值说明：
        无（断言通过即用例成功）。
    """
    tasks = [{"name": "price", "raw_segments": [text], "operate_sub_type": "query"}]
    out = prune_tasks_by_heuristics(text, tasks)
    assert [t["name"] for t in out] == ["bill"], out
    assert out[0]["operate_sub_type"] == sub


def test_prune_corrects_model_add_to_edit():
    """
    函数功能与逻辑描述：
        ★"改了却多一笔账"的源头：`TASK_SUB_TYPE_DEFAULTS` 把 bill 兜底成 `add`。当用户
        表达的是变更意图时，规则侧必须**就地纠正**为 `edit`（这一条锁住该纠正不被回退）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    out = prune_tasks_by_heuristics(
        "把 id=515 那笔改成 50",
        [{"name": "bill", "raw_segments": ["把 id=515 那笔改成 50"], "operate_sub_type": "add"}])
    assert out[0]["operate_sub_type"] == "edit"


def test_prune_keeps_normal_add_unchanged():
    """
    函数功能与逻辑描述：
        反向保护：**不含变更意图**的普通记账指令不受本次改动影响（`operate_sub_type` 仍是 add，
        不因新增词表而被误判成 edit）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    out = prune_tasks_by_heuristics(
        "记午饭56元", [{"name": "bill", "raw_segments": ["记午饭56元"], "operate_sub_type": "add"}])
    assert [t["name"] for t in out] == ["bill"]
    assert out[0]["operate_sub_type"] == "add"


def test_delete_result_render_does_not_raise_keyerror():
    """
    函数功能与逻辑描述：
        ★缺陷 B 防复发：delete 成功载荷只有 `{"id": …}`（无 category/amount），
        渲染必须**不抛 KeyError**，并把 id 如实告知用户。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    text = build_user_display_text({
        "success": True, "agent_type": "bill_agent", "msg": "账单删除成功",
        "data": {"id": 245}, "error": None})
    assert "245" in text
    assert "category" not in text


def test_add_edit_render_keeps_fixed_sentence_even_with_reply():
    """
    函数功能与逻辑描述：
        ★防复发（2026-09-17 实际踩到）：add/edit（有 `category`）**必须走固定句式**，
        即使载荷里带了 `reply`。因为 `"记账成功：消费类目 …"` 是**契约句式**，本地确定性渲染
        与下游断言都依赖它；`reply` 是 LLM 生成的随机措辞，优先返回它会让
        `test_multi_round_session_memory` 的 `assert "记账成功" in final_reply` 随机失败。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    text = build_user_display_text({
        "success": True, "agent_type": "bill_agent", "msg": "记账成功",
        "data": {"id": 2, "category": "餐饮", "amount": 35.0, "consume_date": "2026-09-17",
                 "reply": "好的，已经帮您记下了晚饭的支出。"}, "error": None})
    assert "记账成功" in text and "消费类目" in text


def test_delete_result_render_prefers_reply():
    """
    函数功能与逻辑描述：
        FC 路径的成功载荷带 `data.reply`（模型工具循环产出的话术）时**优先采用**，
        不自行拼字段——保证两条路径的话术各由各自的确定性来源决定。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    text = build_user_display_text({
        "success": True, "agent_type": "bill_agent", "msg": "账单删除成功",
        "data": {"id": 245, "reply": "已删除 08-30 那笔餐饮 30 元"}, "error": None})
    assert text == "已删除 08-30 那笔餐饮 30 元"


def test_edit_result_render_keeps_normal_shape():
    """
    函数功能与逻辑描述：
        反向保护：add/edit 的**正常载荷**渲染形态保持不变（防止"为修 delete 而改坏原路径"）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    text = build_user_display_text({
        "success": True, "agent_type": "bill_agent", "msg": "账单修改成功",
        "data": {"id": 2, "category": "交通", "amount": 50.0, "consume_date": "2026-09-13"},
        "error": None})
    assert "交通" in text and "50.0" in text and "2026-09-13" in text


def test_mutation_keywords_do_not_break_other_intents():
    """
    函数功能与逻辑描述：
        反向保护：新增词表不得污染其它意图——"这个月花了多少"（统计）不应被当成变更意图，
        仍应按普通咨询处理（`_is_consume_statement` 不因新增词表而误判）。
    入参说明：
        无。
    返回值说明：
        无（断言通过即用例成功）。
    """
    assert not any(kw in "这个月花了多少" for kw in EDIT_HINT_KEYWORDS)
    assert not any(kw in "这个月花了多少" for kw in DELETE_HINT_KEYWORDS)
    # "打车贵不贵"仍是比价咨询，不该被当成变更
    assert _is_consume_statement("打车贵不贵") is False


def test_mutation_intent_yields_to_income_redline():
    """互斥：含收入词（"工资"）时不得补 bill —— 收入/退款不入账是既有红线。"""
    out = prune_tasks_by_heuristics("把工资那笔删掉", [])
    assert all(t.get("name") != "bill" for t in out), out


def test_mutation_intent_yields_to_refuse_bill():
    """互斥：'别记了，删掉 id=245 那笔' —— 既有'拒绝记账'又有'删除'词时，
    拒绝记账优先，不得补 bill（`has_bill_mutation` 显式与 refuse 互斥）。"""
    out = prune_tasks_by_heuristics("别记了，删掉 id=245 那笔", [])
    assert all(t.get("name") != "bill" for t in out), out


def test_mutation_coexists_with_other_intent():
    """多意图并存：'把昨天那笔改成 50，这个月花了多少' 里 **bill 必须保留**
    （且 sub_type 被纠正为 edit），不得因为还有咨询意图就把变更任务丢掉。"""
    out = prune_tasks_by_heuristics("把昨天那笔改成 50，这个月花了多少", [])
    bill_tasks = [t for t in out if t["name"] == "bill"]
    assert len(bill_tasks) == 1, out
    assert bill_tasks[0]["operate_sub_type"] == "edit"


def test_plain_consume_statement_still_renders_as_consultation():
    """反向保护：纯消费陈述（**无变更词**）仍按陈述处理 → 兜底补 price（原有行为不变）。"""
    assert _is_consume_statement("午饭花了56元") is True
    out = prune_tasks_by_heuristics("午饭花了56元", [])
    assert [t["name"] for t in out] == ["price"]


def test_display_text_legacy_success_none_returns_msg():
    """旧版兜底：`success` 非 True/False（如 None）时原样返回 msg（既有契约不变）。"""
    assert build_user_display_text(
        {"success": None, "agent_type": "bill_agent", "msg": "无返回结果",
         "data": None, "error": None}) == "无返回结果"


def test_display_text_ask_path_returns_prompt():
    """追问路径：success=False 且 error=need_more_info → 回传 data['prompt'] 作为追问语。"""
    text = build_user_display_text(
        {"success": False, "agent_type": "bill_agent", "msg": "", "error": "need_more_info",
         "data": {"prompt": "您想改成什么？"}})
    assert text == "您想改成什么？"
