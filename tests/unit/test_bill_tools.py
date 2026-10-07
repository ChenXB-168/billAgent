# -*- coding: utf-8 -*-
"""M15（`设计/12` §4.2.4）账单改删工具族单元测试。

★测试策略：**用真实 SQLite 临时库**验证行为，而不是断言"写了某段 SQL 字符串"。
  理由：本里程碑的核心风险是**行为**（排序口径、limit clamp、部分更新语义、
  "删 0 行判失败"），字符串断言只能证明代码里存在某段文本，证明不了行为正确。
  实现：`DatabaseCRUD(db_path=...)` 支持注入库路径（M5 引入），故临时库随 `tmp_path` 销毁，
  **绝不触碰生产 `database/bill.db`**；建表 DDL 与 `database/init_db.py` 的
  bill / monthly_habit 两表逐字一致（含 CHECK 约束——"品类白名单由 DB 兜底"这条
  设计取舍正是靠它验证的）。

覆盖（对应 `11` §3 M15 的 P1 验收方式 ①）：
  排序与字段投影 / limit 双端 clamp 与非法值回落 / add 写 task_id 与回读 id /
  金额与品类约束（不写库）/ update 部分更新与 changed / 目标不存在 / 无字段 /
  同值空 changed / 跨月改期 / delete 快照与 0 行失败 / 不误伤他行 /
  habit.recalc 重算·幂等·空月 / 改删后重算闭环 / 引擎 schema 契约。
"""
import asyncio

import pytest
from unittest.mock import patch

from config.config import RECENT_BILL_MAX, RECENT_BILL_WINDOW
from mcpGateway import bill_tools, fact_tools
from mcpGateway.registry import get_registry
from mcpGateway.tool_model import ToolErrorKind
from utils.common import DatabaseCRUD

# 与 database/init_db.py 的 bill 建表逐字一致（列名 / 类型 / CHECK / 默认值）
_BILL_DDL = """
CREATE TABLE IF NOT EXISTS bill (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    amount REAL CHECK(amount > 0),
    category TEXT CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
    consume_time TEXT,
    remark TEXT,
    task_id TEXT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# 与 database/init_db.py 的 monthly_habit 建表逐字一致（主键 (month, category)）
_HABIT_DDL = """
CREATE TABLE IF NOT EXISTS monthly_habit (
    month TEXT NOT NULL,
    category TEXT NOT NULL CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
    amount_sum REAL NOT NULL DEFAULT 0 CHECK(amount_sum >= 0),
    count INTEGER NOT NULL DEFAULT 0 CHECK(count >= 0),
    avg_amount REAL NOT NULL DEFAULT 0,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (month, category)
)
"""


@pytest.fixture
def temp_db(tmp_path):
    """
    函数功能与逻辑描述：
        提供隔离的临时账单库：在 pytest 的 tmp_path 下新建 SQLite 文件并建好 bill /
        monthly_habit 两张表，然后把 `bill_tools.db`、`fact_tools.db`、`fact_tools.db_lock`
        三个模块级引用临时替换为指向该库的实例（与新建的协程锁），使被测工具函数
        读写临时库而非生产库。function 作用域：每个用例一份全新库，用例之间零污染。
    入参说明：
        tmp_path：pytest 内置 fixture，提供本用例独占的临时目录。
    返回值说明：
        DatabaseCRUD：指向临时库的访问实例，供用例直接查库做断言。
    """
    db = DatabaseCRUD(tmp_path / "bill_test.db")
    db.execute_sql(_BILL_DDL)
    db.execute_sql(_HABIT_DDL)
    # db_lock 每用例新建：跨用例复用同一 asyncio.Lock 会因绑定不同事件循环而报错
    with patch.object(bill_tools, "db", db), \
            patch.object(fact_tools, "db", db), \
            patch.object(fact_tools, "db_lock", asyncio.Lock()):
        yield db
    db.close()


def _seed(db, rows: list[tuple]) -> None:
    """
    函数功能与逻辑描述：
        向临时库直接插入账单种子数据（绕过被测的 add_bill，保证"读/改/删"类用例的前置数据
        不受 add 行为影响，做到失败定位单一）。`create_time` 由表默认值生成，无需显式给。
    入参说明：
        db (DatabaseCRUD)：临时库实例。
        rows (list[tuple])：每项为 (amount, category, consume_time, remark, task_id) 五元组。
    返回值说明：
        无（副作用：向 bill 表插入对应的若干行）。
    """
    for row in rows:
        db.execute_sql(
            "INSERT INTO bill (amount, category, consume_time, remark, task_id) "
            "VALUES (?, ?, ?, ?, ?)", row)


def _count(db) -> int:
    """
    函数功能与逻辑描述：
        读取临时库 bill 表当前行数，供各用例做"是否写库"的断言（失败路径必须一行不写）。
    入参说明：
        db (DatabaseCRUD)：临时库实例。
    返回值说明：
        int：bill 表行数。
    """
    return db.query_sql("SELECT COUNT(*) AS c FROM bill")[0]["c"]


def _count_habit(db) -> int:
    """
    函数功能与逻辑描述：
        读取临时库 monthly_habit 表行数，供幂等用例断言"重算不会把行数越插越多"。
    入参说明：
        db (DatabaseCRUD)：临时库实例。
    返回值说明：
        int：monthly_habit 表行数。
    """
    return db.query_sql("SELECT COUNT(*) AS c FROM monthly_habit")[0]["c"]


# ========== UI / 节点内部读数：主键直取与放宽窗口（2026-09-16 补，防缺陷 D 复发）==========
def test_get_bill_by_id_hit_and_miss(temp_db):
    """
    函数功能与逻辑描述：
        按主键取单行：命中返回该行（**不含 task_id** 的展示投影），未命中返回 None。
        这是"显式 id 定位"的读数基础（节点侧 `_fetch_bill_by_id` 依赖它）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(30.0, "餐饮", "2026-08-30", "晚饭30元", "t1")])
    bid = temp_db.query_sql("SELECT id FROM bill")[0]["id"]
    row = bill_tools.get_bill_by_id(bid)
    assert row is not None and row["id"] == bid and row["remark"] == "晚饭30元"
    assert "task_id" not in row                     # 对账内部字段不进展示投影
    assert bill_tools.get_bill_by_id(bid + 999) is None


def test_get_bill_by_id_beyond_window_is_reachable(temp_db):
    """
    函数功能与逻辑功能描述（★缺陷 D 防复发）：
        主键读数**不受窗口约束**：即使条数远超 `RECENT_BILL_MAX`，仍能取到最早那条。
        缺陷 D 的现象正是"清单里没有 → 报没找到"，而面板窗口比清单宽。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(10.0 + i, "餐饮", "2026-08-01", f"r{i}", f"t{i}") for i in range(25)])
    oldest = temp_db.query_sql("SELECT min(id) AS m FROM bill")[0]["m"]
    # 清单侧：默认只给窗口下限（3 笔），即使显式要 999 也被 clamp 到上限 20 笔
    assert len(bill_tools.recent_bills()) == bill_tools.RECENT_BILL_WINDOW
    assert len(bill_tools.recent_bills(999)) == bill_tools.RECENT_BILL_MAX
    # ★主键直取不受窗口约束：第 25 笔（最老）依然可命中
    assert bill_tools.get_bill_by_id(oldest) is not None


def test_recent_bills_for_ui_widens_limit_and_matches_projection(temp_db):
    """
    函数功能与逻辑描述：
        UI 读数：上限为 `UI_BILL_FETCH_LIMIT`（远宽于模型可见面），**字段投影与
        `recent_bills` 完全一致**（同源同 SQL 的证据），且**不影响模型侧**的 clamp。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(10.0 + i, "餐饮", "2026-08-01", f"r{i}", f"t{i}") for i in range(25)])
    ui_rows = bill_tools.recent_bills_for_ui()
    assert len(ui_rows) == 25                                            # 25 < 200，全给
    assert len(bill_tools.recent_bills(999)) == bill_tools.RECENT_BILL_MAX  # 模型侧仍 20
    assert set(ui_rows[0]) == {"id", "amount", "category", "consume_time",
                               "remark", "create_time"}
    assert bill_tools.UI_BILL_FETCH_LIMIT > bill_tools.RECENT_BILL_MAX    # 两个数不是一个数
    assert ui_rows[0]["id"] > ui_rows[-1]["id"]                           # 仍按 id 倒序


def test_recent_bills_for_ui_invalid_limit_falls_back(temp_db):
    """
    函数功能与逻辑描述：
        非法 limit（非数字）回落默认值，不抛异常（面板取数的健壮性要求）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(10.0, "餐饮", "2026-08-01", "r0", "t0")])
    assert len(bill_tools.recent_bills_for_ui("bad")) == 1
    assert len(bill_tools.recent_bills_for_ui(None)) == 1


# ===================== bill.recent_list =====================
def test_recent_bills_orders_by_id_desc_and_projects_fields(temp_db):
    """
    函数功能与逻辑描述：
        验证清单的排序口径与字段投影：三笔账单按 **id 倒序**返回（"最近一笔"以记账先后为准，
        而非 consume_time），每行只含约定的 6 个字段，且**不含 task_id**（对账内部字段，
        不进模型可见面）。同时确认 remark 全文可见（模型据此辨认"那笔打车的"）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [
        (10.0, "餐饮", "2026-09-01", "早饭", "t1"),
        (20.0, "交通", "2026-09-02", "打车", "t2"),
        (30.0, "购物", "2026-09-03", "买书", "t3"),
    ])
    rows = bill_tools.recent_bills(3)
    assert [r["id"] for r in rows] == [3, 2, 1]
    assert set(rows[0]) == {"id", "amount", "category", "consume_time", "remark", "create_time"}
    assert rows[0]["remark"] == "买书"
    assert rows[0]["amount"] == 30.0


def test_recent_bills_clamps_low_limit_to_window(temp_db):
    """
    函数功能与逻辑描述：
        验证 ★limit 下界 clamp：模型传 `limit=1`（P0-b 实测真实行为）时必须放回
        `RECENT_BILL_WINDOW` 条，否则窗口过小会漏掉用户真正指代的目标（`设计/12` §5.1 修正 1）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(float(i), "餐饮", "2026-09-01", f"第{i}笔", None) for i in range(1, 6)])
    assert len(bill_tools.recent_bills(1)) == RECENT_BILL_WINDOW
    assert len(bill_tools.recent_bills(0)) == RECENT_BILL_WINDOW


def test_recent_bills_clamps_high_limit_to_max(temp_db):
    """
    函数功能与逻辑描述：
        验证 ★limit 上界 clamp：模型传 `limit=999` 时最多返回 `RECENT_BILL_MAX` 条，
        防止一次拉全表撑爆上下文预算（D6）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(float(i), "餐饮", "2026-09-01", f"第{i}笔", None)
                    for i in range(1, RECENT_BILL_MAX + 6)])
    assert len(bill_tools.recent_bills(999)) == RECENT_BILL_MAX


@pytest.mark.parametrize("bad_limit", [None, "abc", "3"])
def test_recent_bills_invalid_limit_falls_back_to_window(temp_db, bad_limit):
    """
    函数功能与逻辑描述：
        验证直调时的非法 limit 容错：None / 非数字字符串等无法转 int 的入参一律回落窗口下限，
        不抛异常（★注意：经网关调用时，非整数 limit 会先被引擎的 JSON Schema 拦成 VALIDATION，
        本容错只对代码直调/UI 取数生效，属第二层防御）。
    入参说明：
        temp_db：临时库 fixture。
        bad_limit：参数化注入的非法 limit 取值（None / "abc" / "3"）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(float(i), "餐饮", "2026-09-01", f"第{i}笔", None) for i in range(1, 6)])
    assert len(bill_tools.recent_bills(bad_limit)) == RECENT_BILL_WINDOW


# ===================== bill.add =====================
def test_add_bill_writes_task_id_and_returns_readback_id(temp_db):
    """
    函数功能与逻辑描述：
        验证 add 的关键契约：① `task_id` 由调用方（引擎注入）写入——这是崩溃恢复的对账依据
        （`04` §3.1），必须由代码保证；② 因写通道不回 `lastrowid`，返回的 id 靠**写后回读**得到，
        且确实指向刚写入的那一行（金额/日期/task_id 全部对得上）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    res = bill_tools.add_bill(35.5, "餐饮", "2026-09-13", "昨天打车35元", task_id="task-1")
    assert res["ok"] is True
    assert isinstance(res["id"], int) and res["id"] > 0
    row = temp_db.query_sql("SELECT * FROM bill WHERE id = ?", (res["id"],))[0]
    assert row["task_id"] == "task-1"
    assert row["amount"] == 35.5
    assert row["consume_time"] == "2026-09-13"
    assert row["remark"] == "昨天打车35元"


def test_add_bill_without_task_id_falls_back_to_max_id(temp_db):
    """
    函数功能与逻辑描述：
        验证直调（无 task_id，如测试或未来的非编排调用）时的回读退化路径：改用 MAX(id) 取回
        新行主键，且库内 task_id 为 NULL——说明注入缺失不会污染对账字段。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(10.0, "餐饮", "2026-09-01", "旧账", None)])
    res = bill_tools.add_bill(20.0, "交通", "2026-09-02")
    assert res["ok"] is True and res["id"] == 2
    assert temp_db.query_sql("SELECT task_id FROM bill WHERE id = 2")[0]["task_id"] is None


@pytest.mark.parametrize("bad_amount", [0, -5, "abc"])
def test_add_bill_rejects_bad_amount_without_write(temp_db, bad_amount):
    """
    函数功能与逻辑描述：
        验证金额防御性校验：0 / 负数 / 不可转数字的入参一律 `ok=False`，且**一行都不写库**
        （失败必须无副作用——否则会留下"金额非法但已落库"的脏数据）。
    入参说明：
        temp_db：临时库 fixture。
        bad_amount：参数化注入的非法金额（0 / -5 / "abc"）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    res = bill_tools.add_bill(bad_amount, "餐饮", "2026-09-01")
    assert res["ok"] is False and res["id"] is None
    assert _count(temp_db) == 0


def test_add_bill_illegal_category_rejected_by_db_check(temp_db):
    """
    函数功能与逻辑描述：
        验证「品类白名单**不在工具层重复声明**」的设计取舍：非法品类由 bill 表的 CHECK 约束
        拒绝，工具把写失败转成 `ok=False` 而不抛异常。这条用例锁住"DB 是白名单唯一权威"，
        防止后续有人在工具层再抄一份列表造成两处漂移。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    res = bill_tools.add_bill(10.0, "饮食", "2026-09-01")
    assert res["ok"] is False
    assert _count(temp_db) == 0


# ===================== bill.update =====================
def test_update_bill_partial_update_keeps_other_columns(temp_db):
    """
    函数功能与逻辑描述：
        验证部分更新语义：只传 `amount` 时，`category` / `consume_time` 必须保持原值
        （`SET x = COALESCE(?, x)` 模板的实际效果），且 `changed` 精确记录
        `{字段: [旧值, 新值]}` 供上层组织回执。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", "t1")])
    res = bill_tools.update_bill(1, amount=50.0)
    assert res["ok"] is True
    assert res["changed"] == {"amount": [35.0, 50.0]}
    row = temp_db.query_sql("SELECT * FROM bill WHERE id = 1")[0]
    assert row["amount"] == 50.0
    assert row["category"] == "交通"
    assert row["consume_time"] == "2026-09-13"
    assert row["task_id"] == "t1"


def test_update_bill_missing_target_fails_without_write(temp_db):
    """
    函数功能与逻辑描述：
        验证"目标不存在"分支：id 查无时**先读存在性即失败**，不执行任何 UPDATE——
        即"温和的错"不会写坏数据；错误文案含"未找到该账单"，供模型转成追问话术。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", None)])
    res = bill_tools.update_bill(9999, amount=1.0)
    assert res["ok"] is False and res["changed"] == {}
    assert "未找到该账单" in res["error"]
    assert temp_db.query_sql("SELECT amount FROM bill WHERE id = 1")[0]["amount"] == 35.0


def test_update_bill_rejects_non_positive_amount(temp_db):
    """
    函数功能与逻辑描述：
        验证 `设计/12` §4.6「新金额 ≤ 0 → 失败（不写库）」：负数金额被工具层拦下，
        库内原值不变（bill 表的 CHECK 约束是第一道，工具层校验是第二道，二者都要有）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", None)])
    res = bill_tools.update_bill(1, amount=-1)
    assert res["ok"] is False
    assert "消费金额必须大于0" in res["error"]
    assert temp_db.query_sql("SELECT amount FROM bill WHERE id = 1")[0]["amount"] == 35.0


def test_update_bill_without_any_field_fails(temp_db):
    """
    函数功能与逻辑描述：
        验证"无新值即无操作"：三个可改字段全为 None 时直接失败并给出可转追问的错误文案
        （对应失败矩阵「edit 无任何新值 → 追问」），不产生一次空 UPDATE。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", None)])
    res = bill_tools.update_bill(1)
    assert res["ok"] is False
    assert res["changed"] == {}
    assert "未提供任何待修改字段" in res["error"]


def test_update_bill_same_value_yields_empty_changed(temp_db):
    """
    函数功能与逻辑描述：
        验证 `changed` 只收录**真实发生变化**的字段：把金额改成与库内相同的值时，
        写操作成功（幂等无害）但 `changed` 为空 dict——上层据此可回答"金额没有变化"，
        而不是谎报"已改为 35"。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", None)])
    res = bill_tools.update_bill(1, amount=35.0)
    assert res["ok"] is True and res["changed"] == {}


def test_update_bill_date_change_maps_to_consume_time(temp_db):
    """
    函数功能与逻辑描述：
        验证入参名与表列名的映射：`consume_date` 落库到 `consume_time` 列，
        且 `changed` 用**入参名**（`consume_date`）而非列名回报——保证模型看到的字段口径
        与它传入的口径一致（跨月改期是"改账"的常见形态，直接关系月度习惯重算）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-08-31", "打车", None)])
    res = bill_tools.update_bill(1, consume_date="2026-09-01")
    assert res["ok"] is True
    assert res["changed"] == {"consume_date": ["2026-08-31", "2026-09-01"]}
    assert temp_db.query_sql(
        "SELECT consume_time FROM bill WHERE id = 1")[0]["consume_time"] == "2026-09-01"


# ===================== bill.delete =====================
def test_delete_bill_returns_snapshot_and_removes_row(temp_db):
    """
    函数功能与逻辑描述：
        验证删除成功路径：返回被删行**快照**（上层无需二次查库即可组织"已删除：09-13 交通 35 元"
        这类回执），并做写后回读确认该行确已消失。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", "t1")])
    res = bill_tools.delete_bill_record(1)
    assert res["ok"] is True
    assert res["deleted"]["amount"] == 35.0
    assert res["deleted"]["remark"] == "打车"
    assert _count(temp_db) == 0


def test_delete_bill_zero_row_is_failure(temp_db):
    """
    函数功能与逻辑描述：
        ★验证与旧语义的刻意分叉：目标 id 不存在时**判失败**（`ok=False`、`deleted=None`），
        而不是沿用 `utils/common.delete_bill` 的"删 0 行也算成功"。模型只认 `ok`，
        必须让"删了个不存在的"显式失败，否则会谎报"已删除"（`设计/12` §4.2.4）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    res = bill_tools.delete_bill_record(9999)
    assert res["ok"] is False and res["deleted"] is None
    assert "未找到该账单" in res["error"]


def test_delete_bill_leaves_other_rows_intact(temp_db):
    """
    函数功能与逻辑描述：
        验证删除的**作用范围精确**：只删指定 id，其它行（含相邻 id）逐字不变——
        防"按条件误删"这类不可逆事故。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(10.0, "餐饮", "2026-09-01", "早饭", None),
                    (20.0, "交通", "2026-09-02", "打车", None),
                    (30.0, "购物", "2026-09-03", "买书", None)])
    assert bill_tools.delete_bill_record(2)["ok"] is True
    left = temp_db.query_sql("SELECT id, amount FROM bill ORDER BY id")
    assert [(r["id"], r["amount"]) for r in left] == [(1, 10.0), (3, 30.0)]


# ===================== habit.recalc =====================
@pytest.mark.asyncio
async def test_recalc_habit_rewrites_month_after_delete(temp_db):
    """
    函数功能与逻辑描述：
        验证习惯整月重算的**修复能力**（这正是 `habit.upsert` 增量语义做不到的）：
        先按 bill 表重算出该月两品类的金额与笔数；随后删除其中一笔交通，再次重算，
        交通品类的金额必须**减少**（而不是像增量 upsert 那样只增不减）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", None),
                    (15.0, "交通", "2026-09-14", "地铁", None),
                    (20.0, "餐饮", "2026-09-14", "外卖", None)])

    res = await fact_tools.recalc_habit("2026-09")
    habits = {h["category"]: h for h in res["habits"]}
    assert habits["交通"]["amount_sum"] == 50.0
    assert habits["交通"]["count"] == 2
    assert habits["交通"]["avg_amount"] == 25.0
    assert habits["餐饮"]["amount_sum"] == 20.0

    assert bill_tools.delete_bill_record(1)["ok"] is True
    res2 = await fact_tools.recalc_habit("2026-09")
    habits2 = {h["category"]: h for h in res2["habits"]}
    assert habits2["交通"]["amount_sum"] == 15.0
    assert habits2["交通"]["count"] == 1


@pytest.mark.asyncio
async def test_recalc_habit_is_idempotent(temp_db):
    """
    函数功能与逻辑描述：
        验证重算幂等：同样数据连跑两次，读回的该月聚合逐字一致（实现是"先删该月、再按
        bill 分组插入"，天然可重复执行），因此改删后重复触发不会累积误差。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    _seed(temp_db, [(35.0, "交通", "2026-09-13", "打车", None)])
    first = await fact_tools.recalc_habit("2026-09")
    second = await fact_tools.recalc_habit("2026-09")
    assert first["habits"] == second["habits"]
    assert _count_habit(temp_db) == 1


@pytest.mark.asyncio
async def test_recalc_habit_empty_month_is_not_error(temp_db):
    """
    函数功能与逻辑描述：
        验证"该月已无账单"是**正常结果**而非错误：返回 `ok=True` 且 habits 为空列表
        （语义 = 该月习惯归零），不抛异常、不阻断上层的改删回执。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    res = await fact_tools.recalc_habit("2026-01")
    assert res["ok"] is True and res["habits"] == []


# ===================== 改删闭环 =====================
@pytest.mark.asyncio
async def test_add_update_delete_roundtrip(temp_db):
    """
    函数功能与逻辑描述：
        验证改删链路的**最小闭环**：add（拿到 id）→ update（同 id 改金额）→ delete（同 id 删除），
        三步均 ok，且最终该行确不存在。这条用例模拟 P3a/P3b 里模型"记一笔 → 改成 50 → 删掉"
        的真实调用序列（工具层的顺序契约）。
    入参说明：
        temp_db：临时库 fixture。
    返回值说明：
        无（断言通过即用例成功）。
    """
    added = bill_tools.add_bill(35.0, "交通", "2026-09-13", "打车", task_id="t-1")
    assert added["ok"] is True
    bill_id = added["id"]

    updated = bill_tools.update_bill(bill_id, amount=50.0)
    assert updated["ok"] is True and updated["changed"]["amount"] == [35.0, 50.0]

    deleted = bill_tools.delete_bill_record(bill_id)
    assert deleted["ok"] is True and deleted["deleted"]["amount"] == 50.0
    assert _count(temp_db) == 0


# ===================== 引擎层契约（schema 即"温和的错"的拦截线）=====================
@pytest.mark.asyncio
@pytest.mark.parametrize("tool,args", [
    ("bill.update", {"id": "1"}),        # id 传字符串 → 引擎判 VALIDATION
    ("bill.delete", {"id": "1"}),
    ("bill.recent_list", {"limit": "3"}),  # limit 非整数 → 同样被拦
])
async def test_engine_rejects_wrong_argument_types(tool, args):
    """
    函数功能与逻辑描述：
        验证 `设计/12` §4.2.4 的"温和的错进不到工具里"：`id` / `limit` 在 schema 中声明为
        `integer`，模型传字符串会被执行引擎的参数校验阶段挡下（返回 VALIDATION），
        **不会触达工具、更不会写库**——参数不合规在网关层即被拦，而非靠工具内部容错。
    入参说明：
        tool (str)：参数化注入的被测工具名（bill.update / bill.delete / bill.recent_list）。
        args (dict)：参数化注入的非法入参（把整型字段传成字符串）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    res = await get_registry().invoke(tool, args, agent_id="bill_agent")
    assert res.ok is False
    assert res.error.kind is ToolErrorKind.VALIDATION


@pytest.mark.asyncio
async def test_add_tool_schema_never_exposes_task_id():
    """
    函数功能与逻辑描述：
        🔴 安全红线：`bill.add` 的 schema **不得出现 `task_id`**——它由引擎经 inject_ctx 注入，
        一旦写进 schema 就会被 `to_openai_schema` 暴露给模型，模型可伪造对账依据（同 agent_id
        的处理，`设计/12` §4.2.4）。同时断言它确实声明了 inject_ctx（注入通道没被误删）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    spec = get_registry().get("bill.add")
    props = spec.params_schema["properties"]
    assert "task_id" not in props
    assert spec.params_schema["additionalProperties"] is False
    assert spec.binding.get("inject_ctx") == ("task_id",)
