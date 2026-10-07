# -*- coding: utf-8 -*-
"""M15（P4）账单面板与「定位辅助」单元测试。

设计依据：`设计/12` §4.9、`11` §3 M15 的 P4 验收（① 面板渲染 ② 回调合约 ③ 静态红线 ④ 口径统一）。

★本文件的核心命题是**红线**而不是样式：UI 只做"展示 + 把目标写进输入框"，
  绝不直连写库、绝不触发任何工具调用。样式会变，红线不能变，所以断言写在行为与源码两处：
  - 行为：按钮回调返回 `gr.update(value=...)`，且期间**没有任何 tools 调用发生**；
  - 源码：静态扫描 `app.py` / `bill_panel.py`，禁止出现账单写操作（`bill.update` / `call_bill_sql` 等）。

取数一律打桩 `bill_panel.load_bills`（不触碰生产库）。
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from webUI import bill_panel

# 两个月、共 5 条：前 3 条在可操作窗口内（RECENT_BILL_WINDOW=3），后 2 条超窗
_ROWS = [
    {"id": 9, "amount": 20.0, "category": "餐饮", "consume_time": "2026-09-14",
     "remark": "晚饭20元", "create_time": "2026-09-14 20:00:00"},
    {"id": 8, "amount": 35.0, "category": "交通", "consume_time": "2026-09-13",
     "remark": "打车35元", "create_time": "2026-09-13 20:00:00"},
    {"id": 7, "amount": 120.0, "category": "购物", "consume_time": "2026-09-12",
     "remark": "买鞋", "create_time": "2026-09-12 20:00:00"},
    {"id": 6, "amount": 9.9, "category": "餐饮", "consume_time": "2026-08-30",
     "remark": "早餐", "create_time": "2026-08-30 08:00:00"},
    {"id": 5, "amount": 50.0, "category": "娱乐", "consume_time": "2026-08-29",
     "remark": "电影", "create_time": "2026-08-29 20:00:00"},
]


@pytest.fixture
def rows():
    """
    函数功能与逻辑描述：
        打桩面板取数：把 `bill_panel.load_bills` 替换为固定清单，保证各用例的渲染/选项断言
        不受真实库内容影响，也避免用例写生产库（本模块本就只读，双保险）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        list[dict]：固定账单清单（yield 出，供用例直接查长度）。
    """
    with patch.object(bill_panel, "load_bills", return_value=list(_ROWS)):
        yield _ROWS


# ===================== ① 面板渲染 =====================
def test_render_panel_groups_by_month_and_exposes_ids(rows):
    """
    函数功能与逻辑描述：
        验证面板渲染：按月 `<details>` 折叠（两个月各一组）、每行暴露 `id` 与"第 N 条"序号、
        金额/品类/备注都在——**id 必须出现在面板里**，否则用户无法指认、模型也无从定位
        （`设计/12` §4.2.4「必须返回 id」在 UI 侧的对应要求）。
    入参说明：
        rows：fixture 注入的固定清单。
    返回值说明：
        无（断言通过即用例成功）。
    """
    html = bill_panel.render_bill_panel()
    assert html.count("<details") >= 2                 # 按月分组
    assert "2026-09" in html and "2026-08" in html
    assert "#9" in html and "#5" in html               # 每行带 id
    assert "第 1 条" in html and "第 5 条" in html
    assert "打车35元" in html


def test_render_panel_has_no_unoperable_rows(rows):
    """
    函数功能与逻辑描述：
        ★设计修正（2026-09-15）：面板**不再画"看得见却动不了"的行**。原先超出 `RECENT_BILL_WINDOW`
        的行会标"超出可操作窗口"，但那正是把"自然语言指代"的窗口约束**误用到了 UI 点选**上——
        点选是**显式指定**（id 已确定），不存在指代歧义，故面板展示的每一条都必须可操作。
        本用例守护该结论：渲染结果里不得出现任何"不可操作"标注，且每一条都渲染成可点的行。
    入参说明：
        rows：fixture 注入的固定清单（5 条）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    html = bill_panel.render_bill_panel()
    assert "超出可操作窗口" not in html
    assert html.count("class='bill-row'") == len(_ROWS)


def test_render_panel_empty_placeholder():
    """
    函数功能与逻辑描述：
        验证空账单时的占位：返回提示文案而不是空串或异常（页面不出现"莫名的空白区域"）。
    入参说明：
        无（pytest 自动发现并调用；用例内打桩取数为空）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    with patch.object(bill_panel, "load_bills", return_value=[]):
        html = bill_panel.render_bill_panel()
    assert "暂无账单记录" in html


def test_render_panel_survives_load_exception():
    """
    函数功能与逻辑描述：
        验证取数异常时的降级：`load_bills` 抛异常（如 DB 被锁）时面板返回空态，
        **不把异常抛到 Gradio 回调里**——便利层出问题不该让整个页面崩掉。
    入参说明：
        无（pytest 自动发现并调用；用例内让 load_bills 抛异常）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    with patch.object(bill_panel, "load_bills", side_effect=RuntimeError("db locked")):
        assert "暂无账单记录" in bill_panel.render_bill_panel()


# ===================== ③④ 选择器与口径 =====================
def test_bill_choices_cover_all_rows_and_value_is_id(rows):
    """
    函数功能与逻辑描述：
        ★设计修正（2026-09-15）：① 选择器**列全面板展示的每一条**（不再截断到"最近 3 笔"）；
        ② 选中值是 **id** 而不是序号——序号会因"中间又记了一笔"而整体漂移，进而产生
        "用户点的是第 1 条、Agent 改的却是另一条"这类**改错账**事故；id 是持久标识，不会漂移。
    入参说明：
        rows：fixture 注入的固定清单（5 条）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    choices = bill_panel.bill_choices()
    assert len(choices) == len(_ROWS)                                     # 不再截断到窗口
    assert [value for _, value in choices] == ["9", "8", "7", "6", "5"]   # 选中值 = id
    assert "#9" in choices[0][0]                                          # 标签带 id，便于核对


# ===================== ② 回调合约：只写输入框 =====================
@pytest.mark.parametrize("handler,expect_prefix", [
    (bill_panel.on_edit_selected, "把 id=2 那笔改成"),
    (bill_panel.on_delete_selected, "删掉 id=2 那笔"),
])
def test_selection_handlers_only_write_input_box(handler, expect_prefix):
    """
    函数功能与逻辑描述：
        ★红线验证（回调合约）：两个按钮的回调只返回"写入输入框"的 `gr.update(value=...)`，
        且期间**没有任何工具调用**（把 registry 打桩后断言从未被调用）。
        这正是"UI 按钮只做定位辅助"的落地：降低指认成本，但不越过 RBAC / 审计 / 二次确认。
    入参说明：
        handler：参数化注入的被测回调（修改选中 / 删除选中）。
        expect_prefix (str)：参数化注入的期望文案前缀。
    返回值说明：
        无（断言通过即用例成功）。
    """
    reg = MagicMock()
    reg.invoke = MagicMock(side_effect=AssertionError("UI 不得调用任何工具"))
    with patch("mcpGateway.registry.get_registry", return_value=reg):
        update = handler("2")
    assert isinstance(update, dict)
    assert update.get("value", "").startswith(expect_prefix)


@pytest.mark.parametrize("handler", [bill_panel.on_edit_selected, bill_panel.on_delete_selected])
def test_selection_handlers_noop_without_choice(handler):
    """
    函数功能与逻辑描述：
        验证未选中时的空操作：返回空 `gr.update()`（不含 value），**不改动输入框内容**——
        避免用户已经打了一半的字被按钮清掉。
    入参说明：
        handler：参数化注入的被测回调。
    返回值说明：
        无（断言通过即用例成功）。
    """
    for empty in (None, ""):
        update = handler(empty)
        assert isinstance(update, dict) and "value" not in update


def test_panel_updates_returns_atomic_pair(rows):
    """
    函数功能与逻辑描述：
        验证每轮刷新用的 `panel_updates()`：同时返回面板 HTML 与选择器的 `gr.update(choices=...)`，
        保证"面板"与"可选目标"原子刷新（避免短时间内的不一致）。
    入参说明：
        rows：fixture 注入的固定清单。
    返回值说明：
        无（断言通过即用例成功）。
    """
    html, pick_update = bill_panel.panel_updates()
    assert "#9" in html
    assert isinstance(pick_update, dict) and len(pick_update.get("choices") or []) == len(_ROWS)


# ===================== ③ 静态红线：源码级 =====================
def test_webui_source_contains_no_bill_write_calls():
    """
    函数功能与逻辑描述：
        ★源码级红线：静态扫描 `webUI/app.py` 与 `webUI/bill_panel.py`，禁止出现任何
        账单写通道的字样（工具名 / SQL 片段 / 旧兼容入口）。行为断言只覆盖调用路径，
        源码断言才挡得住"以后有人顺手加一行直连写库"。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    root = Path(__file__).resolve().parents[2]
    banned = ("bill.update", "bill.delete", "bill.add", "call_bill_sql",
              "INSERT INTO bill", "DELETE FROM bill", "UPDATE bill")
    for name in ("webUI/app.py", "webUI/bill_panel.py"):
        text = (root / name).read_text(encoding="utf-8")
        hits = [b for b in banned if b in text]
        assert not hits, f"{name} 出现了账单写通道字样：{hits}"
