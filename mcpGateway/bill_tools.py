# -*- coding: utf-8 -*-
"""M15（`设计/12` §4.2.4）账单改删工具族：4 个 `bill.*` 工具的本地实现。

背景：M15 之前，`operate_sub_type=edit/delete` 已是提示词与草稿槽里的**既成事实**，
但执行层只有一处硬编码 `INSERT`（`agents/bill_agent/nodes.py:393-400`）——
用户说"那笔改成 50"会被**记成一条新账**（`设计/12` §3.1）。
本模块把改删能力下沉为**工具**，由 bill_agent 经 `registry.invoke` 调用。

★设计红线（`设计/12` §4.2 / `设计_从零系列/09` §4.10 第 7 条）：
- **SQL 全部写死在函数内部**，模型只传结构化参数（N6：绝不让模型生成 SQL 文本）；
- 工具只做「参数化的确定性写」，**不做业务判断**（"改成什么/删哪一条"由模型或节点决定）；
- 落库走正门 `utils.common.db`（同 `fact_tools`：本模块 import 会连带初始化 DB 连接，
  属「DB 工具的职责」，为登记在案的合理依赖，见 `fact_tools.py` 模块头 §8.3 #12 对照）。

★写通道语义（`设计/12` §4.2.4）：`db.execute_sql` **只回 bool**（`utils/common.py:120-121`），
既不回 `lastrowid` 也不回 `rowcount`，故一律采用「**先读存在性 → 写 → 后回读校验**」三段式：
- `add`  → 写后按 `task_id` 反查回读 id；
- `update` → 写后回读，逐字段校验"请求值 == 库内值"，并对与写前不同的字段产出 `changed`；
- `delete` → 写后回读，**行仍在即判失败**。

★与旧语义的**刻意分叉**：`utils/common.delete_bill` 对"删 0 行"返回 True（视为成功），
本模块 `delete_bill_record` **删 0 行判失败**（`ok=False`）——模型依赖 `ok` 判断结果，
"删了个不存在的"必须让它看见失败（`设计/12` §4.2.4）。
"""
from config.config import RECENT_BILL_MAX, RECENT_BILL_WINDOW, UI_BILL_FETCH_LIMIT
from utils.common import db

# 账单表列清单（回读投影）：与 `database/init_db.py` 的 bill 建表逐字对齐，
# 显式列出而非 `SELECT *`，避免将来加列时把无关字段回传给模型（observation 精简，`设计/12` §4.2.3）。
_ROW_COLS = "id, amount, category, consume_time, remark, task_id, create_time"
# `bill.recent_list` 对外的字段投影：不含 task_id（对账内部字段，模型无需也不会用）
_LIST_COLS = "id, amount, category, consume_time, remark, create_time"


def _get_bill_row(bill_id: int) -> dict | None:
    """
    函数功能与逻辑描述：
        按主键读取单条账单完整行（含 `task_id`，供内部对账/回读校验使用），
        是「先读存在性 / 后回读校验」三段式的**共读取数点**——update 与 delete 都靠它
        判断"写前是否存在"与"写后是否生效"，避免两处各写一条 SELECT 造成口径分叉。
        查询异常由 `db.query_sql` 内部吞掉并返回空列表，故此处安全退化为"查无此行"。
    入参说明：
        bill_id (int)：bill 表主键 id。
    返回值说明：
        dict | None：命中时返回该行字典（列名为 key，含 task_id）；无此行或查询异常时返回 None。
    """
    rows = db.query_sql(f"SELECT {_ROW_COLS} FROM bill WHERE id = ?", (bill_id,))
    return dict(rows[0]) if rows else None


def recent_bills(limit: int = RECENT_BILL_WINDOW) -> list[dict]:
    """
    函数功能与逻辑描述：
        查询最近若干笔账单明细，是模型「定位目标账单」的唯一读数入口
        （改删前必须先拿到 id，见 `设计/12` §4.2.4「必须返回 id」）。
        排序固定为 `id DESC`——`id` 是持久自增主键，"最近一笔"以**记账先后**为准
        （而非 `consume_time`，用户补记旧日期时后者会错位）。
        ★limit 由**服务端 clamp** 到 `[RECENT_BILL_WINDOW, RECENT_BILL_MAX]`：
        P0-b 实测外部模型会传 `limit=1`，窗口过小会漏掉真正的目标；非法值/未传一律回落到窗口下限。
        返回字段**不含 task_id**（对账内部字段，不进模型可见面）。
    入参说明：
        limit (int)：期望返回条数，默认 RECENT_BILL_WINDOW；非整数或越界值由服务端收敛。
    返回值说明：
        list[dict]：每行 `{id, amount, category, consume_time, remark, create_time}`，
            按 id 倒序；该表无数据时返回空列表 []。
    """
    try:
        want = int(limit)
    except (TypeError, ValueError):
        want = RECENT_BILL_WINDOW
    want = max(RECENT_BILL_WINDOW, min(want, RECENT_BILL_MAX))
    rows = db.query_sql(
        f"SELECT {_LIST_COLS} FROM bill ORDER BY id DESC LIMIT ?", (want,))
    return [dict(r) for r in rows]


def get_bill_by_id(bill_id: int) -> dict | None:
    """
    函数功能与逻辑描述：
        按**主键**取单行账单（`_LIST_COLS` 投影，不含 task_id），供 Agent 侧的确定性定位使用。
        为什么需要它："id=245 那笔"这种**显式主键指认是无歧义的**，不该受"最近 N 笔"窗口约束——
        实测 bug：UI 面板窗口放宽后能选中 #245，而定位侧只拉 20 笔 → 落入窗口外 →
        回答"没找到符合条件的账单"（用户明明是从面板上点下来的）。
        ★**本函数不注册为工具**（`registry` 显式登记 `bill.*` 工具，不扫描本模块），
        模型看不到它、也无法传参进来；它是节点内部定位用的读数。
    入参说明：
        bill_id (int)：bill 表主键 id。
    返回值说明：
        dict | None：命中时返回 `{id, amount, category, consume_time, remark, create_time}`；
            无此行或查询异常（`db.query_sql` 内部吞掉并返回空列表）时返回 None。
    """
    rows = db.query_sql(f"SELECT {_LIST_COLS} FROM bill WHERE id = ?", (bill_id,))
    return dict(rows[0]) if rows else None


def recent_bills_for_ui(limit: int = UI_BILL_FETCH_LIMIT) -> list[dict]:
    """
    函数功能与逻辑描述：
        **UI 面板专用的最近账单读数**（`webUI/bill_panel.py` 调用），与 `recent_bills`
        共用**同一 SQL、同一排序、同一字段投影**——"UI 看到的"与"Agent 认知的"实现同源
        （`设计/12` §4.6），唯一差别是 limit 上限放宽到 `UI_BILL_FETCH_LIMIT`。
        为什么需要它：面板按天翻页浏览，而 `recent_bills` 的 20 笔上限是为**模型上下文预算 D6**
        设的，实际常常只覆盖 3 天，翻页形同虚设（实测）；条数不同本就是契约允许的
        （`设计/12` §4.6：面板展示条数与可操作窗口是两个数）。
        ★**本函数不注册为工具**（`registry` 显式登记 `bill.*` 工具，不扫描本模块），
        模型看不到它、也无法传参进来；`RECENT_BILL_MAX` 因此保持原值不动。
    入参说明：
        limit (int)：期望返回条数，默认 `UI_BILL_FETCH_LIMIT`；非法值回落默认，且上限受其收敛。
    返回值说明：
        list[dict]：每行 `{id, amount, category, consume_time, remark, create_time}`，
            按 id 倒序；该表无数据时返回空列表 []。
    """
    try:
        want = int(limit)
    except (TypeError, ValueError):
        want = UI_BILL_FETCH_LIMIT
    want = max(RECENT_BILL_WINDOW, min(want, UI_BILL_FETCH_LIMIT))
    rows = db.query_sql(
        f"SELECT {_LIST_COLS} FROM bill ORDER BY id DESC LIMIT ?", (want,))
    return [dict(r) for r in rows]


def add_bill(amount: float, category: str, consume_date: str,
             remark: str = "", task_id: str | None = None) -> dict:
    """
    函数功能与逻辑描述：
        新增一笔账单（**写操作**）。SQL 与迁移前 `agents/bill_agent/nodes.py:398-399`
        **逐字一致**（同表、同列序、同参数序），仅把"拼 SQL 的位置"从节点搬进工具——
        故 add 路径的业务语义零变化（`设计/12` §4.2.4 补充 + §8.2 红线 1）。
        ★`task_id` **由引擎经 `inject_ctx` 注入**（同 agent_id 的处理，不进 params_schema）：
        它是崩溃恢复的对账依据（`04` §3.1 修订版），必须由代码保证写入、绝不由模型生成。
        因写通道不回 `lastrowid`，id 采用**写后回读**：有 task_id 时按 task_id 反查最新一条
        （比 `MAX(id)` 抗并发——同 task 内重复 add 时也能取到本 task 刚写的那条）。
        金额在工具层再做一次 `> 0` 防御性校验（节点层已校验，此处防直调）。
    入参说明：
        amount (float)：消费金额（元），必须大于 0。
        category (str)：消费品类，须落在 bill 表 CHECK 白名单内（餐饮/交通/住宿/购物/娱乐）——
            本函数**不重复声明白名单**，DB CHECK 约束是唯一权威（避免三处维护同一份列表）。
        consume_date (str)：消费日期，格式 YYYY-MM-DD。
        remark (str)：备注（通常为用户原话），默认空串。
        task_id (str | None)：本任务标识，由引擎注入；直调（如测试/UI）时可省略，为 None 时退化为 MAX(id) 回读。
    返回值说明：
        dict：成功 `{ok: True, id: <新行主键>}`（id 回读失败时为 None，但 ok 仍为 True——
            行确已写入，不回滚）；失败 `{ok: False, id: None, error: <原因>}`，不抛异常。
    """
    try:
        amount_val = float(amount)
    except (TypeError, ValueError):
        return {"ok": False, "id": None, "error": "消费金额非法"}
    if amount_val <= 0:
        return {"ok": False, "id": None, "error": "消费金额必须大于0"}

    sql = "INSERT INTO bill (amount, category, consume_time, remark, task_id) VALUES (?, ?, ?, ?, ?)"
    if not db.execute_sql(sql, (amount_val, category, consume_date, remark, task_id)):
        return {"ok": False, "id": None, "error": "账单写入失败"}

    if task_id:
        rows = db.query_sql(
            "SELECT id FROM bill WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,))
    else:
        rows = db.query_sql("SELECT MAX(id) AS id FROM bill")
    new_id = rows[0].get("id") if rows else None
    return {"ok": True, "id": new_id}


def update_bill(id: int, amount: float = None, category: str = None,
                consume_date: str = None) -> dict:
    """
    函数功能与逻辑描述：
        对指定 id 的账单做**部分更新**（未传的字段保持原值，**写操作**）。
        实现要点：① 固定模板 `SET x = COALESCE(?, x)` 逐字段覆盖，未传（None）即不改；
        ② `id` 与 `task_id` **永不修改**（task_id 是崩溃恢复对账依据，改了会破坏 D2 成果）；
        ③ 严格三段式——先读存在性（不存在 → 失败不写库）→ 写 → 后回读校验；
        ④ 回读校验逐字段比对"请求值 == 库内值"，任一不符判失败（防"静默未生效"）；
        ⑤ `changed` 只收录**真实发生变化**的字段（请求值与原值相同的字段不算变更）。
        金额在工具层做 `> 0` 校验（负数是非法金额，`设计/12` §4.6）；三个可改字段全为 None
        时直接失败（无新值 = 无操作，交回上层追问）。
        参数名用 `id`（与工具 schema 的属性名一致，LocalAdapter 以 kwargs 直传）。
    入参说明：
        id (int)：目标账单主键 id（来自 `bill.recent_list`），必填。
        amount (float | None)：新金额（元，须 > 0）；None 表示不修改。
        category (str | None)：新品类；None 表示不修改（合法性由 bill 表 CHECK 约束兜底）。
        consume_date (str | None)：新消费日期 YYYY-MM-DD；None 表示不修改（落库列名仍为 consume_time）。
    返回值说明：
        dict：成功 `{ok: True, id, changed: {字段: [旧值, 新值]}}`（`changed` 可能为空 dict，
            表示"传了值但与原值相同"）；失败 `{ok: False, id, changed: {}, error: <原因>}`，不抛异常。
    """
    bill_id = id
    if amount is None and category is None and consume_date is None:
        return {"ok": False, "id": bill_id, "changed": {},
                "error": "未提供任何待修改字段"}

    before = _get_bill_row(bill_id)
    if before is None:
        return {"ok": False, "id": bill_id, "changed": {},
                "error": "未找到该账单，可能已被删除"}

    if amount is not None:
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return {"ok": False, "id": bill_id, "changed": {}, "error": "消费金额非法"}
        if amount <= 0:
            return {"ok": False, "id": bill_id, "changed": {}, "error": "消费金额必须大于0"}

    sql = ("UPDATE bill SET amount = COALESCE(?, amount), category = COALESCE(?, category), "
           "consume_time = COALESCE(?, consume_time) WHERE id = ?")
    if not db.execute_sql(sql, (amount, category, consume_date, bill_id)):
        return {"ok": False, "id": bill_id, "changed": {}, "error": "账单更新失败"}

    after = _get_bill_row(bill_id)
    if after is None:
        return {"ok": False, "id": bill_id, "changed": {}, "error": "账单更新未生效"}

    # 入参名 → 表列名（仅 consume_date ↔ consume_time 一处不同，集中映射避免散落）
    expects = (("amount", amount), ("category", category), ("consume_date", consume_date))
    col_of = {"amount": "amount", "category": "category", "consume_date": "consume_time"}
    changed: dict = {}
    for field, want in expects:
        if want is None:
            continue
        col = col_of[field]
        got = after.get(col)
        if got != want:
            return {"ok": False, "id": bill_id, "changed": changed,
                    "error": f"账单更新未生效：{field}"}
        if before.get(col) != got:
            changed[field] = [before.get(col), got]
    return {"ok": True, "id": bill_id, "changed": changed}


def delete_bill_record(id: int) -> dict:
    """
    函数功能与逻辑描述：
        按 id 删除单条账单（**写操作**），并把被删行快照一并返回——上层据此组织
        "已删除：09-13 交通 35 元"这类回执，无需为回执再查一次库。
        严格三段式：先读存在性（不存在 → 失败，不执行 DELETE）→ 删 → 后回读，
        **行仍在即判失败**（"删 0 行算成功"的旧语义在此刻意不复用，见模块头说明）。
        ★二次确认**不在这里做**：本工具是"确认后的执行件"，确认状态由上层草稿槽管理
        （`设计/12` §4.4，模型不得拥有"跳过确认直接删"的能力）。
        参数名用 `id`（与工具 schema 属性名一致）。
    入参说明：
        id (int)：目标账单主键 id（来自 `bill.recent_list`），必填。
    返回值说明：
        dict：成功 `{ok: True, id, deleted: {原行快照}}`；失败
            `{ok: False, id, deleted: None, error: <原因>}`（含"目标不存在"，均不抛异常）。
    """
    bill_id = id
    before = _get_bill_row(bill_id)
    if before is None:
        return {"ok": False, "id": bill_id, "deleted": None,
                "error": "未找到该账单，可能已被删除"}

    if not db.execute_sql("DELETE FROM bill WHERE id = ?", (bill_id,)):
        return {"ok": False, "id": bill_id, "deleted": None, "error": "账单删除失败"}
    if _get_bill_row(bill_id) is not None:
        return {"ok": False, "id": bill_id, "deleted": None, "error": "账单删除未生效"}
    return {"ok": True, "id": bill_id, "deleted": before}


__all__ = ["recent_bills", "add_bill", "update_bill", "delete_bill_record"]
