# -*- coding: utf-8 -*-
"""M15（P4）账单面板：**只读展示 + 只写输入框**的定位辅助组件。

设计依据：`设计/12` §4.9（UI 方案与取舍：按钮只做"定位辅助"）+ `11` §3 M15 的 P4 验收。

★★本模块的**唯一红线**：UI 不做任何写操作。
   它只做两件事——
   ① 读：从 `bill_tools.recent_bills`（**与 Agent 同一实现**，`设计/12` §4.6 口径统一要求）
      取最近账单，渲染成**按天分页**的只读面板（每行带 `id`，供用户指认）；
   ② 翻译：把"用户选中的那条"变成**一句自然语言**写进输入框（`gr.update(value=...)`），
      后续由正常的对话链路（orchestrator → bill_agent）去执行。
   为什么不走"UI 直连写库"：那会绕过 RBAC / 审计 / 参数校验 / **二次确认**四道治理
   （`09` §0.3 原则 2「Agent 只经网关访问数据」），且与"改删必须确认"的交互模型直接冲突
   （`设计/12` §1.2 N1）。UI 面板是**便利层**，不是第二条写通道。

★为什么**按天分页**（每页 7 天）而不是一次铺开全部：
   ① 20 行列表会把首屏和折叠面板一起撑长，观感差（用户明确提出）；
   ② 账单是"按天发生"的，以**天**为翻页粒度符合用户心智（"看上周那些账"），
      比"每页 N 条"更稳定——同一天补记不会让分页整体错位；
   ③ 选择器只列**当前页**的账单，用户在小集合里点选，误点概率低。
   ⚠️ 翻页只在 `recent_bills` 的可见范围内进行（服务端 clamp 上限 `RECENT_BILL_MAX`），
      超出该范围的历史账单不在面板内——这是"与 Agent 同源取数"的必然边界。

★为什么独立成模块而不是写进 `app.py`：① `app.py` import 会拉起整套系统（bootstrap / 编排图），
  本模块保持**可独立单测**；② 回退时删掉 `app.py` 中的接线即回到 M15 之前（`11` 回滚③）。
"""
import html as _html          # 面板文案全部转义后再拼 HTML，防备注里的尖括号破坏结构
import math
from typing import Optional

import gradio as gr

from config.config import RECENT_BILL_WINDOW, UI_BILL_FETCH_LIMIT
from mcpGateway import bill_tools

# 每页展示的**天数**（用户指定：一次 7 天，上下翻页各翻 7 天）。
# 放在本模块而非 config：它是纯 UI 展示参数（与 _PANEL_CSS 同类），不参与 Agent 侧口径。
UI_BILL_PAGE_DAYS = 7

# 面板样式：保持与既有 WebUI 一致的浅色卡片风；只做最小必要内联样式，
# 不引入前端框架/JS（`11` 走查项 3：沿用 Gradio 现有能力，不做 hack）
_PANEL_CSS = (
    "<style>"
    ".bill-panel{font-size:13px;line-height:1.7;}"
    ".bill-panel details{margin:4px 0;}"
    ".bill-panel summary{cursor:pointer;font-weight:600;}"
    ".bill-row{display:flex;gap:8px;padding:2px 0;border-bottom:1px dashed #eee;}"
    ".bill-id{color:#888;min-width:44px;}"
    ".bill-out{color:#c60;font-size:12px;}"
    ".bill-empty{color:#888;}"
    "</style>"
)


def load_bills(limit: int = UI_BILL_FETCH_LIMIT) -> list[dict]:
    """
    函数功能与逻辑描述：
        面板取数：读最近若干笔账单（只读）。★刻意复用 `bill_tools.recent_bills`——
        它是 `bill.recent_list` 工具的**同一个函数**，因此"UI 看到的"与"Agent 认知的"
        永远同源（`设计/12` §4.6：两者口径分叉会让用户指着第 2 条说"这条"而 Agent 理解为别的）。
        读失败（DB 异常等）返回空列表：面板是便利层，任何异常都不该让页面崩。
    入参说明：
        limit (int)：取数条数，默认 `UI_BILL_FETCH_LIMIT`（面板取数上限）。注意该值会被
            工具层 clamp 到 `[RECENT_BILL_WINDOW, RECENT_BILL_MAX]`。
    返回值说明：
        list[dict]：账单列表（id 倒序）；无数据或读取异常时返回 []。
    """
    try:
        return bill_tools.recent_bills_for_ui(limit)
    except Exception:  # noqa: BLE001 —— 便利层不阻断页面
        return []


def _group_by_day(bills: list[dict]) -> list[tuple[str, list[tuple[int, dict]]]]:
    """
    函数功能与逻辑描述：
        把 id 倒序的账单列表按 `consume_time` 的**日期**聚成"天"，返回
        `[(日期, [(全局序号, 行), ...]), ...]`，天顺序与账单顺序一致（最近的天在前）。
        全局序号 = 该行在**整个可见清单**里的下标（从 1 起）——与 `bill_target.locate_bill`
        的"第 N 条"同口径，因此面板上写的"第 N 条"与用户对 Agent 说"第 N 条"指向同一行。
        日期缺失的行归入"未知日期"组（不丢弃、不崩）。
    入参说明：
        bills (list[dict])：`load_bills` 的返回（id 倒序）。
    返回值说明：
        list[tuple[str, list[tuple[int, dict]]]]：按天的分组结果；空入参返回 []。
    """
    groups: dict[str, list[tuple[int, dict]]] = {}
    for idx, row in enumerate(bills, start=1):
        day = str(row.get("consume_time") or "")[:10] or "未知日期"
        groups.setdefault(day, []).append((idx, row))
    return list(groups.items())


def _page_slice(days: list, page: Optional[int]) -> tuple[list, int, int, int]:
    """
    函数功能与逻辑描述：
        分页切片（纯函数）：把"天"列表按 `UI_BILL_PAGE_DAYS` 切成一页，并把页码收敛到
        `[0, 总页数-1]`——页码越界（如翻到边界后又点、或账单被删导致页数变少）时**收敛而非报错**，
        这是面板"永不因数据变化而崩"的一部分。
    入参说明：
        days (list)：`_group_by_day` 的结果。
        page (Optional[int])：期望页码（0 起）；None/非数字按 0 处理。
    返回值说明：
        tuple：`(本页的天, 收敛后的页码, 总页数(至少 1), 总天数)`。
    """
    total = len(days)
    pages = max(1, math.ceil(total / UI_BILL_PAGE_DAYS)) if total else 1
    try:
        cur = int(page) if page is not None else 0
    except (TypeError, ValueError):
        cur = 0
    cur = max(0, min(cur, pages - 1))
    start = cur * UI_BILL_PAGE_DAYS
    return days[start:start + UI_BILL_PAGE_DAYS], cur, pages, total


def page_label(page: int) -> str:
    """
    函数功能与逻辑描述：
        生成翻页状态文案（供按钮旁的 Markdown 显示），形如
        `第 1-7 天 / 共 12 天（20 笔）· 第 1/2 页`。空数据时给出"暂无账单"占位。
    入参说明：
        page (int)：页码（0 起）；内部会自行取数、收敛越界页码。
    返回值说明：
        str：可直接交给 `gr.Markdown` 的单行文案。
    """
    days, cur, pages, total = _page_slice(_group_by_day(load_bills()), page)
    if total == 0:
        return "暂无账单记录"
    cnt = sum(len(rows) for _, rows in days)
    start = cur * UI_BILL_PAGE_DAYS + 1
    end = start + len(days) - 1
    return f"第 {start}-{end} 天 / 共 {total} 天（本页 {cnt} 笔）· 第 {cur + 1}/{pages} 页"


def render_bill_panel(page: int = 0, limit: int = UI_BILL_FETCH_LIMIT) -> str:
    """
    函数功能与逻辑描述：
        把**当前页的天**渲染为只读 HTML 面板：外层为 `gr.HTML` 容器，内部每天一个
        `<details>`（**默认折叠**，只显示"哪天、几笔"），展开后逐行显示
        `id · 第 N 条 · 月-日 · 品类 · 金额 · 备注`。
        为什么默认折叠：面板本身已是"按需展开"的便利层，分组再默认展开会让用户
        在展开折叠面板的一瞬间被整页明细淹没（观感诉求）。
        空数据返回占位文案（不崩、不留空白）。
    入参说明：
        page (int)：页码（0 起，0 = 最近 7 天）；越界由 `_page_slice` 收敛。
        limit (int)：取数条数上限，默认 `UI_BILL_FETCH_LIMIT`。
    返回值说明：
        str：可直接交给 `gr.HTML` 的 HTML 字符串。
    """
    try:
        days, cur, pages, total = _page_slice(_group_by_day(load_bills(limit)), page)
    except Exception:  # noqa: BLE001 —— 便利层自兜底：渲染异常不该让页面崩
        days, cur, pages, total = [], 0, 1, 0
    if not days:
        return (f"{_PANEL_CSS}<div class='bill-panel bill-empty'>"
                "暂无账单记录。直接输入即可记账，例如：记午饭 56 元。</div>")

    parts = [f"{_PANEL_CSS}<div class='bill-panel'>",
             f"<b>第 {cur + 1}/{pages} 页</b>（每页 {UI_BILL_PAGE_DAYS} 天；"
             "在下方选择器里点选任意一条即可定位它——不受「最近 3 笔」限制，"
             "因为点选是<b>显式指定</b>，不存在指代歧义）"]
    for day, rows in days:
        parts.append(f"<details><summary>{_html.escape(day)}（{len(rows)} 笔）</summary>")
        for idx, row in rows:
            amount = row.get("amount")
            amount_text = (f"{float(amount):.0f}" if amount is not None
                           and abs(float(amount) - round(float(amount))) < 0.01
                           else str(amount))
            remark = _html.escape(str(row.get("remark") or "")[:20])
            parts.append(
                f"<div class='bill-row'>"
                f"<span class='bill-id'>#{_html.escape(str(row.get('id')))}</span>"
                f"<span>第 {idx} 条</span>"
                f"<span>{_html.escape(str(row.get('category') or ''))}"
                f" {_html.escape(amount_text)} 元</span>"
                f"<span>{remark}</span></div>")
        parts.append("</details>")
    parts.append(f"<div class='bill-empty'>— 第 {cur + 1}/{pages} 页，共 {total} 天 —</div></div>")
    return "".join(parts)


def bill_choices(page: int = 0, limit: int = UI_BILL_FETCH_LIMIT) -> list[tuple[str, str]]:
    """
    函数功能与逻辑描述：
        生成目标选择器的候选项：**只列当前页的账单**，每项 `(展示文案, 选中值)`。
        选中值用 **`id`**（持久主键）而不是序号——序号会因"中间又记了一笔"而整体漂移，
        把 id 写进输入框后，"用户点的是哪一条"与"Agent 改的是哪一条"永远一致。
        展示文案里保留"第 N 条"（**全局序号**，与 `bill_target.locate_bill` 同口径），
        方便用户口头复核。
    入参说明：
        page (int)：页码（0 起）；越界由 `_page_slice` 收敛。
        limit (int)：取数条数上限，默认 `UI_BILL_FETCH_LIMIT`。
    返回值说明：
        list[tuple[str, str]]：形如 [("第 1 条 · 09-14 交通 35 元（昨天打车35元）· #515", "515"), ...]；
            当前页无账单时返回空列表（组件显示为空）。
    """
    days, _, _, _ = _page_slice(_group_by_day(load_bills(limit)), page)
    choices: list[tuple[str, str]] = []
    for day, rows in days:
        for idx, row in rows:
            label = (f"第 {idx} 条 · {day[5:]} {row.get('category') or ''} "
                     f"{row.get('amount')} 元（{str(row.get('remark') or '')[:12]}）· #{row.get('id')}")
            choices.append((label, str(row.get("id"))))
    return choices


def on_edit_selected(choice: Optional[str]) -> dict:
    """
    函数功能与逻辑描述：
        「修改选中」按钮回调：把选中的 **id** 翻译成一句**自然语言**写进输入框
        （`gr.update(value=...)`），**不执行任何写操作、不调用任何工具**。
        用户随后可补全目标值（如"改成 50"）再发送，走正常对话链路。
    入参说明：
        choice (Optional[str])：目标选择器的选中值（账单 id 字符串）；未选时为 None/空。
    返回值说明：
        dict：`gr.update(value=...)`（写入输入框）；未选中时返回空 `gr.update()`（不改变界面）。
    """
    if not choice:
        return gr.update()
    return gr.update(value=f"把 id={choice} 那笔改成 ")


def on_delete_selected(choice: Optional[str]) -> dict:
    """
    函数功能与逻辑描述：
        「删除选中」按钮回调：同 `on_edit_selected`，只把"删掉 id=X 那笔"写进输入框，
        **不执行删除**——真正的删除必须经对话链路里的**二次确认**（`设计/12` §4.4）。
        这正是"UI 按钮只做定位辅助"的落地：按钮降低指认成本，但不越过治理流程。
    入参说明：
        choice (Optional[str])：目标选择器的选中值（账单 id 字符串）；未选时为 None/空。
    返回值说明：
        dict：`gr.update(value=...)`（写入输入框）；未选中时返回空 `gr.update()`。
    """
    if not choice:
        return gr.update()
    return gr.update(value=f"删掉 id={choice} 那笔")


def panel_updates(page: int = 0, limit: int = UI_BILL_FETCH_LIMIT) -> tuple[str, dict]:
    """
    函数功能与逻辑描述：
        面板的两件套刷新（面板 HTML + 目标选择器选项），供 `app.respond` 在每轮对话结束后调用，
        使"用户刚记的那笔"立刻出现在面板里（`11` P4 验收：每轮刷新）。
        两者一起返回是为了**原子刷新**：面板与选项若分别刷新，会出现"面板有第 3 条、选项只有 2 条"
        的短暂不一致窗口。默认回到第 0 页（最近 7 天）——刚记的账必然落在最近，
        翻页状态因此不需要在 `respond` 里额外传递。
    入参说明：
        page (int)：页码（0 起），默认 0（最近 7 天）。
        limit (int)：取数条数上限，默认 `UI_BILL_FETCH_LIMIT`。
    返回值说明：
        tuple[str, dict]：`(面板 HTML, 目标选择器的 gr.update)`，与 `outputs=[bill_panel, bill_pick]` 对应。
    """
    # ★value=None：每轮刷新同时**清空选择**——否则当"已选中的那条"被删掉或移出清单时，
    #   Radio 会保留一个不在新 choices 里的旧值，Gradio 侧抛校验错误（界面出现红字报错）。
    return (render_bill_panel(page, limit),
            gr.update(choices=bill_choices(page, limit), value=None))


def page_view(delta: int, page: Optional[int]) -> tuple[str, dict, str, int]:
    """
    函数功能与逻辑描述：
        翻页回调（`app.py` 的两个翻页按钮共用）：`delta=+1` 表示"更早 7 天"（页码 +1），
        `delta=-1` 表示"更新 7 天"（页码 -1）。页码在 `[0, 总页数-1]` 内收敛，
        因此反复点到底/到顶都不会报错，只是停住。
        返回四元组以同时刷新：面板 HTML、选择器选项（并清空选择）、页码文案、页码 state。
    入参说明：
        delta (int)：翻页方向步长（+1 = 更早，-1 = 更新）。
        page (Optional[int])：当前页码（来自 `gr.State`）。
    返回值说明：
        tuple：`(面板 HTML, 选择器 gr.update, 页码文案, 新页码)`，
            与 `outputs=[bill_panel, bill_pick, page_info, bill_page]` 逐一对应。
    """
    days = _group_by_day(load_bills())
    _, cur, pages, _ = _page_slice(days, page)
    try:
        new_page = max(0, min(cur + int(delta), pages - 1))
    except (TypeError, ValueError):
        new_page = cur
    return (render_bill_panel(new_page),
            gr.update(choices=bill_choices(new_page), value=None),
            page_label(new_page),
            new_page)


__all__ = ["load_bills", "render_bill_panel", "bill_choices", "page_label", "page_view",
           "on_edit_selected", "on_delete_selected", "panel_updates", "UI_BILL_PAGE_DAYS",
           "RECENT_BILL_WINDOW"]
