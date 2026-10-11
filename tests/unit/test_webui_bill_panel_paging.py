# -*- coding: utf-8 -*-
"""M15（P4）账单面板**按天分页**单测（2026-09-16 补，`设计/11` M15 验收记录第 ⑦ 项）。

为什么单独成文件而不是并进 `test_webui_bill_panel.py`：分页是**本轮新增**的展示行为，
且它必须**完全不依赖真实库**——这里用 monkeypatch 打桩 `load_bills` 构造确定性账单集合，
把"每页 7 天 / 翻页方向 / 边界收敛 / 选择器随页联动 / 页码文案"钉成断言。
（本轮三个真 bug 全部出在"这条组合没人测"：UI 窗口 vs Agent 窗口不一致、
显式 id 越窗、FC 路径写不落地。分页此后有测试兜底。）
"""
import webUI.bill_panel as bp


def _rows(days: int, per_day: int = 1) -> list[dict]:
    """
    构造 `load_bills` 形态的账单行（**id 倒序**，一天 per_day 笔）。

    入参说明：
        days (int)：天数（日期从 2026-09-20 往前递减）。
        per_day (int)：每天笔数。
    返回值说明：
        list[dict]：账单行列表，`consume_time` 形如 "2026-09-20"。
    """
    out, bid = [], 1000
    for i in range(days):
        day = f"2026-09-{20 - i:02d}"
        for j in range(per_day):
            out.append({"id": bid, "amount": 10.0 + j, "category": "餐饮",
                        "consume_time": day, "remark": f"r{bid}", "create_time": day + " 10:00:00"})
            bid -= 1
    return out


def _stub(monkeypatch, rows: list[dict]) -> None:
    """把面板取数打桩为给定行（不触 DB、不触工具层）。"""
    monkeypatch.setattr(bp, "load_bills", lambda limit=bp.UI_BILL_FETCH_LIMIT: rows)


def test_group_by_day_orders_newest_first(monkeypatch):
    _stub(monkeypatch, _rows(3))
    days = bp._group_by_day(bp.load_bills())
    assert [d for d, _ in days] == ["2026-09-20", "2026-09-19", "2026-09-18"]


def test_group_by_day_index_is_global_position(monkeypatch):
    """行内"第 N 条"必须是**全局序号**（与 `bill_target.locate_bill` 同口径）。"""
    _stub(monkeypatch, _rows(2, per_day=2))     # 4 行 → 第 1..4 条
    days = bp._group_by_day(bp.load_bills())
    idxs = [idx for _, rows in days for idx, _ in rows]
    assert idxs == [1, 2, 3, 4]


def test_page_slice_takes_seven_days_per_page():
    days = [(f"d{i}", []) for i in range(10)]
    page0, cur0, pages, total = bp._page_slice(days, 0)
    page1, cur1, _, _ = bp._page_slice(days, 1)
    assert (len(page0), cur0, pages, total) == (7, 0, 2, 10)
    assert (len(page1), cur1) == (3, 1)


def test_page_slice_clamps_out_of_range_and_bad_page():
    days = [(f"d{i}", []) for i in range(3)]
    assert bp._page_slice(days, 99)[1] == 0      # 超出末页 → 收敛到最后一页（仅 1 页时为 0）
    assert bp._page_slice(days, -5)[1] == 0      # 负数 → 0
    assert bp._page_slice(days, None)[1] == 0    # None → 0
    assert bp._page_slice(days, "abc")[1] == 0   # 非数字 → 0
    assert bp._page_slice([], 0)[2] == 1         # 空数据：页数至少 1（不出现 0 页）


def test_page_view_moves_and_never_overshoots(monkeypatch):
    _stub(monkeypatch, _rows(10))                # 10 天 → 2 页（0/1）
    assert bp.page_view(1, 0)[3] == 1            # 更早 → 第 2 页
    assert bp.page_view(1, 1)[3] == 1            # 到底后再点 → 停住（不报错、不越界）
    assert bp.page_view(-1, 1)[3] == 0           # 更新 → 回第 1 页
    assert bp.page_view(-1, 0)[3] == 0           # 到顶后再点 → 停住


def test_page_view_returns_four_tuple_for_atomic_refresh(monkeypatch):
    """返回四元组必须与 `app.py` 的 outputs 顺序一致：面板 / 选择器 / 页码文案 / 页码 state。"""
    _stub(monkeypatch, _rows(10))
    html, pick, label, page = bp.page_view(1, 0)
    assert isinstance(html, str) and isinstance(label, str) and isinstance(page, int)
    assert isinstance(pick, dict) and "choices" in pick and pick.get("value") is None


def test_render_bill_panel_renders_only_current_page(monkeypatch):
    _stub(monkeypatch, _rows(10))
    html0 = bp.render_bill_panel(0)
    html1 = bp.render_bill_panel(1)
    assert html0.count("<details") == bp.UI_BILL_PAGE_DAYS      # 第 1 页恰 7 天
    assert html1.count("<details") == 3                          # 第 2 页剩 3 天
    assert "每页 7 天" in html0
    # ★分组默认折叠：不得出现 open 的 details（首屏观感红线）
    assert "<details open" not in html0 and "<details open" not in html1


def test_bill_choices_lists_only_current_page(monkeypatch):
    _stub(monkeypatch, _rows(10))                                # 每天 1 笔 → 共 10 条
    c0, c1 = bp.bill_choices(0), bp.bill_choices(1)
    assert (len(c0), len(c1)) == (7, 3)
    ids0 = {v for _, v in c0}
    ids1 = {v for _, v in c1}
    assert not (ids0 & ids1)                                    # 两页不重叠
    assert all(v.isdigit() for v in ids0 | ids1)                # 选中值是 **id**（非序号）
    assert "第 1 条" in c0[0][0]                                 # 文案保留全局序号便于复核


def test_page_label_reports_range_and_page(monkeypatch):
    _stub(monkeypatch, _rows(10))
    assert bp.page_label(0) == "第 1-7 天 / 共 10 天（本页 7 笔）· 第 1/2 页"
    assert bp.page_label(1) == "第 8-10 天 / 共 10 天（本页 3 笔）· 第 2/2 页"


def test_page_label_empty_corpus(monkeypatch):
    _stub(monkeypatch, [])
    assert bp.page_label(0) == "暂无账单记录"
    assert "暂无账单记录" in bp.render_bill_panel(0)             # 空态不崩、有占位


def test_panel_updates_resets_to_first_page_and_clears_selection(monkeypatch):
    """每轮对话后刷新：回到最近 7 天并**清空选择**（否则选中项被删后 Radio 持非法值）。"""
    _stub(monkeypatch, _rows(10))
    html, pick = bp.panel_updates()
    assert html.count("<details") == bp.UI_BILL_PAGE_DAYS
    assert pick.get("value") is None
    assert len(pick["choices"]) == 7


def test_group_by_day_missing_date_falls_back(monkeypatch):
    """`consume_time` 缺失/为空 → 归入"未知日期"组（不丢弃、不崩）——代码里有此兜底分支。"""
    rows = [{"id": 7, "amount": 9.0, "category": "餐饮", "consume_time": None,
             "remark": "无日期的账", "create_time": "2026-09-20 09:00:00"}]
    _stub(monkeypatch, rows)
    days = bp._group_by_day(bp.load_bills())
    assert len(days) == 1 and days[0][0] == "未知日期"
    assert len(bp.bill_choices(0)) == 1          # 仍可被选中（不因为缺日期就消失）
    assert "#7" in bp.render_bill_panel(0)


def test_page_view_invalid_delta_is_safe(monkeypatch):
    """非法 `delta`（字符串/None）不得抛异常，页码保持原值（代码里有 try/except 兜底）。"""
    _stub(monkeypatch, _rows(10))
    for bad in ("abc", None, 1.5):
        out = bp.page_view(bad, 1)
        assert isinstance(out[3], int)
        assert 0 <= out[3] <= 1                   # 仍落在合法页范围内


def test_bill_choices_empty_on_empty_corpus(monkeypatch):
    """空库 / 越界页 → 选择器返回空列表（Gradio 组件显示为空，不抛异常）。"""
    _stub(monkeypatch, [])
    assert bp.bill_choices(0) == []
    assert bp.bill_choices(99) == []


def test_paging_supports_more_than_two_pages(monkeypatch):
    """3 页及以上（不止 2 页的情形）：每页 7 天、页码文案与末页余数正确。"""
    _stub(monkeypatch, _rows(20))                 # 20 天 → 3 页（7 / 7 / 6）
    assert bp.page_label(0).endswith("第 1/3 页")
    assert bp.page_label(2).endswith("第 3/3 页")
    assert len(bp._page_slice(bp._group_by_day(bp.load_bills()), 2)[0]) == 6
    assert bp.page_view(1, 2)[3] == 2             # 末页再点"更早" → 停住


def test_panel_updates_accepts_explicit_page(monkeypatch):
    """`panel_updates(page=N)` 渲染指定页：面板天数与选择器条目都随之切换。"""
    _stub(monkeypatch, _rows(20))
    html, pick = bp.panel_updates(2)
    assert html.count("<details") == 6
    assert len(pick["choices"]) == 6


def test_render_bill_panel_escapes_remark_html(monkeypatch):
    """★安全：备注里的 HTML 必须被转义（否则用户备注可注入标签破坏面板结构）。"""
    rows = [{"id": 9, "amount": 1.0, "category": "餐饮", "consume_time": "2026-09-20",
             "remark": "<script>alert(1)</script>", "create_time": "2026-09-20 10:00:00"}]
    _stub(monkeypatch, rows)
    html = bp.render_bill_panel(0)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_load_bills_uses_ui_widened_reader(monkeypatch):
    """★防回退：面板取数必须走 `recent_bills_for_ui`（上限 200），而非模型侧 `recent_bills`。

    若有人把它改回 `recent_bills`，分页会静默退化成"20 笔 / 常常只有 1 页"。
    """
    calls = {}

    def _fake_ui(limit=bp.UI_BILL_FETCH_LIMIT):
        calls["ui"] = limit
        return _rows(10)

    def _fake_agent(limit=None):                   # pragma: no cover - 不应被调用
        calls["agent"] = limit
        return _rows(3)

    monkeypatch.setattr(bp.bill_tools, "recent_bills_for_ui", _fake_ui)
    monkeypatch.setattr(bp.bill_tools, "recent_bills", _fake_agent)
    out = bp.load_bills()
    assert "ui" in calls and "agent" not in calls  # 走 UI 读数
    assert calls["ui"] == bp.UI_BILL_FETCH_LIMIT   # 且用了放宽后的上限
    assert len(out) == 10


def test_load_bills_swallows_reader_exception(monkeypatch):
    """取数异常 → 返回空列表（便利层不阻断页面），渲染层落空态占位。"""
    def _boom(limit=bp.UI_BILL_FETCH_LIMIT):
        raise RuntimeError("db down")

    monkeypatch.setattr(bp.bill_tools, "recent_bills_for_ui", _boom)
    assert bp.load_bills() == []
    assert "暂无账单记录" in bp.render_bill_panel(0)


def test_app_wiring_binds_each_paging_control_exactly_once():
    """★接线静态断言（防本轮踩过的坑复发）：翻页组件**定义各 1 处、绑定各恰 1 次**。

    背景：本轮曾因同文件并行编辑把绑定插到组件定义之前 → `NameError` → WebUI 启动即崩；
    也曾因重复落盘导致一次点击翻两页。这两类事故都能被本断言拦住。
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "webUI" / "app.py").read_text(encoding="utf-8")
    for name in ("page_prev_btn", "page_next_btn", "page_info", "bill_page"):
        assert src.count(f"{name} = gr.") == 1, name          # 定义恰 1 处
    for name in ("page_prev_btn", "page_next_btn"):
        assert src.count(f"{name}.click(") == 1, name          # 绑定恰 1 次
    # 两个翻页事件的 outputs 必须同序（与 page_view 返回的四元组逐一对应）
    assert src.count("outputs=[bill_panel, bill_pick, page_info, bill_page]") == 2
