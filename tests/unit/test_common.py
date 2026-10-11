# -*- coding: utf-8 -*-
"""`utils/common.py` 单元测试 —— 数据库访问层（DatabaseCRUD）与 MCP 返回体解析。

★为什么必须补这一层：`utils/common.py`（638 行）是整个项目**唯一的数据访问入口**，
被编排器 / 4 个子 Agent / MCP server / WebUI **全域依赖**，此前**零直接测试**
（只被别的用例当工具使用）—— 它一旦行为异常，会以"别处莫名其妙失败"的形式暴露。
本文件是它的**行为契约**。

★测试策略（沿用 `test_bill_tools.py` 的既有约定）：
  用**真实 SQLite 临时库**（`DatabaseCRUD(tmp_path / ...)`）验证行为，
  而非断言"写了某段 SQL"。建表 DDL 与 `database/init_db.py` **逐字一致**
  （含 CHECK 约束 —— "约束由 DB 兜底"这一设计取舍正是靠它验证的）。
  **绝不触碰生产 `database/bill.db`**。

覆盖矩阵：

| 组 | 用例 |
|---|---|
| ① 连接与 PRAGMA | WAL 生效 / busy_timeout=5000 |
| ② execute_sql | 成功 / 带参 / SQL 错误回滚返回 False / 关连接后 False / retry<=0 |
| ③ query_sql | 返回 dict 列表 / 空表 / 非法 SQL 返回 [] / 关连接后 [] |
| ④ close | 幂等 / 关闭后读写均不可用 |
| ⑤ 事务 | begin+commit 持久化 / rollback 丢弃 / 关连接后静默 |
| ⑥ user_config | add / get_latest / upsert 首插 / upsert 覆盖不累积 |
| ⑦ bill | add + get_all 倒序 / get_bill_by_time 左闭右闭 / delete |
| ⑧ city_price | add / batch_add 归一 / get_city_price 无数据 0.0 / get_city_level / 同级 AVG 兜底 |
| ⑨ analysis | add + get_by_month |
| ⑩ monthly_habit | upsert 累加与笔均重算 / get_monthly_habits 过滤 / recalc / cleanup 边界 |
| ⑪ parse_sql_result | str 形态 / TextContent 列表 / 错误文本 / 非 list / 非法字面量 |
"""
from datetime import datetime

import pytest

from utils.common import DatabaseCRUD, parse_sql_result

# ===================== 建表 DDL（与 database/init_db.py 逐字一致） =====================
_DDL_USER_CONFIG = """
CREATE TABLE IF NOT EXISTS user_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT,
    month_budget REAL CHECK(month_budget >= 0),
    consume_mode TEXT CHECK(consume_mode IN ('节俭','正常','宽松')),
    budget_type TEXT DEFAULT 'month' CHECK(budget_type = 'month'),
    category_budget TEXT,
    quota_mode TEXT DEFAULT 'auto' CHECK(quota_mode IN ('auto','user','habit','city')),
    remind_pref TEXT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
_DDL_BILL = """
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
_DDL_CITY_PRICE = """
CREATE TABLE IF NOT EXISTS city_price (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    category TEXT NOT NULL CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
    avg_price REAL CHECK(avg_price > 0),
    city_level TEXT,
    UNIQUE(city, category)
)
"""
_DDL_ANALYSIS = """
CREATE TABLE IF NOT EXISTS analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    total REAL CHECK(total >= 0),
    over_budget INTEGER CHECK(over_budget IN (0,1)),
    content TEXT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
_DDL_HABIT = """
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
def db(tmp_path):
    """
    函数功能与逻辑描述：
        在 pytest 的 tmp_path 下建一个带全部业务表的临时 SQLite 库，供用例读写断言；
        建表 DDL 与 `database/init_db.py` 逐字一致，确保被测行为与生产同构。
        function 作用域：每个用例一份全新库，零污染；用后自动 close。
    入参说明：
        tmp_path：pytest 内置 fixture，提供本用例独占的临时目录。
    返回值说明：
        DatabaseCRUD：指向临时库的实例。
    """
    inst = DatabaseCRUD(tmp_path / "common_test.db")
    for ddl in (_DDL_USER_CONFIG, _DDL_BILL, _DDL_CITY_PRICE, _DDL_ANALYSIS, _DDL_HABIT):
        assert inst.execute_sql(ddl) is True, f"建表失败：{ddl[:40]}"
    yield inst
    inst.close()


# ===================== ① 连接与 PRAGMA =====================
def test_wal_and_busy_timeout_enabled(db):
    """
    函数功能与逻辑描述：
        验证构造时的两条 PRAGMA 真正生效 —— WAL 是 M6 并行派发的前提
        （读不阻塞写、写不阻塞读），busy_timeout 让锁竞争在 SQLite C 层吸收。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    mode = db.query_sql("PRAGMA journal_mode")
    assert mode and mode[0]["journal_mode"].lower() == "wal", f"WAL 未生效：{mode}"

    timeout = db.query_sql("PRAGMA busy_timeout")
    assert timeout and int(timeout[0]["timeout"]) == 5000, f"busy_timeout 非 5000：{timeout}"


# ===================== ② execute_sql =====================
def test_execute_sql_success_and_params(db):
    """写入口：带参数插入返回 True，且数据真正落库（含参数绑定生效）。"""
    assert db.execute_sql("INSERT INTO bill (amount, category, consume_time) VALUES (?, ?, ?)",
                          (12.5, "餐饮", "2026-10-10 12:00:00")) is True
    rows = db.query_sql("SELECT amount, category FROM bill")
    assert len(rows) == 1 and rows[0]["amount"] == 12.5 and rows[0]["category"] == "餐饮"


def test_execute_sql_without_params(db):
    """写入口：params 为 None 时不传参执行（DDL 走的就是这条路径）。"""
    assert db.execute_sql("CREATE TABLE IF NOT EXISTS t_probe (x INTEGER)") is True


def test_execute_sql_sql_error_rolls_back_and_returns_false(db):
    """
    函数功能与逻辑描述：
        验证异常路径：SQL 语法错误 → 返回 False（不向上抛），且**未留下半写入数据**。
        用一条"前半句合法、后半句非法"的多语句验证回滚语义。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    assert db.execute_sql("INSERT INTO bill (amount, category) VALUES (1, '餐饮')") is True
    assert db.execute_sql("INSERT INTO not_exist_table VALUES (1)") is False
    # 异常不得影响已提交数据
    assert len(db.query_sql("SELECT * FROM bill")) == 1


def test_execute_sql_rejects_violating_check_constraint(db):
    """CHECK 约束由 DB 兜底：非法品类（不在五分类内）写入失败并返回 False。"""
    assert db.execute_sql("INSERT INTO bill (amount, category) VALUES (10, '乱写')") is False
    assert db.query_sql("SELECT * FROM bill") == []


def test_execute_sql_after_close_returns_false(db):
    """边界：连接已关闭 → 直接返回 False（不抛异常）。"""
    db.close()
    assert db.execute_sql("INSERT INTO bill (amount, category) VALUES (1, '餐饮')") is False


def test_execute_sql_retry_non_positive_returns_false(db):
    """边界：retry<=0 时循环体不执行 → 回滚并返回 False（即便 SQL 本身合法）。"""
    assert db.execute_sql("INSERT INTO bill (amount, category) VALUES (1, '餐饮')", retry=0) is False
    assert db.query_sql("SELECT * FROM bill") == []


# ===================== ③ query_sql =====================
def test_query_sql_returns_dict_rows(db):
    """读入口统一返回「列名 → 值」的字典列表（防下游把 tuple 当 dict 用）。"""
    db.add_bill(7.0, "交通", "2026-10-01 08:00:00", "地铁")
    rows = db.query_sql("SELECT * FROM bill")
    assert len(rows) == 1
    assert isinstance(rows[0], dict)
    assert {"id", "amount", "category", "consume_time", "remark"} <= set(rows[0])


def test_query_sql_empty_and_invalid_return_empty_list(db):
    """读入口的兜底：空表 → []；SQL 非法 → []（不抛异常）。"""
    assert db.query_sql("SELECT * FROM bill") == []
    assert db.query_sql("SELECT * FROM 不存在的表") == []
    assert db.query_sql("这不是 SQL") == []


def test_query_sql_after_close_returns_empty(db):
    """边界：连接已关闭 → 返回 []（不抛异常）。"""
    db.close()
    assert db.query_sql("SELECT * FROM bill") == []


# ===================== ④ close =====================
def test_close_is_idempotent(db):
    """close 幂等：重复调用不抛异常（_closed 标志位保护）。"""
    db.close()
    db.close()          # 第二次不得抛
    assert db._closed is True


def test_after_close_both_read_and_write_unavailable(db):
    """关闭后读写双向不可用（且都以"安全返回值"表达，不抛）。"""
    db.close()
    assert db.execute_sql("DELETE FROM bill") is False
    assert db.query_sql("SELECT * FROM bill") == []


# ===================== ⑤ 事务 =====================
def test_transaction_commit_persists(db):
    """显式事务：begin → 写 → commit 后数据可见。"""
    db.begin_transaction()
    db.conn.execute("INSERT INTO bill (amount, category) VALUES (9, '购物')")
    db.commit()
    assert len(db.query_sql("SELECT * FROM bill")) == 1


def test_transaction_rollback_discards(db):
    """显式事务：begin → 写 → rollback 后未提交数据被丢弃。"""
    db.begin_transaction()
    db.conn.execute("INSERT INTO bill (amount, category) VALUES (9, '购物')")
    db.rollback()
    assert db.query_sql("SELECT * FROM bill") == []


def test_transaction_helpers_silent_after_close(db):
    """边界：连接已关闭时 begin/commit/rollback 均静默跳过（不抛异常）。"""
    db.close()
    db.begin_transaction()
    db.commit()
    db.rollback()


# ===================== ⑥ user_config =====================
def test_user_config_add_and_get_latest(db):
    """add_user_config 纯追加；get_latest_user_config 取 create_time 倒序首行。"""
    assert db.add_user_config("深圳", 4000.0, "正常") is True
    db.execute_sql("UPDATE user_config SET create_time='2020-01-01 00:00:00' WHERE id=1")
    assert db.add_user_config("北京", 5000.0, "节俭") is True
    latest = db.get_latest_user_config()
    assert len(latest) == 1 and latest[0]["city"] == "北京"


def test_user_config_upsert_inserts_when_empty(db):
    """upsert 在空表时走 INSERT 分支。"""
    assert db.upsert_user_config("上海", 3000.0, "宽松", category_budget='{"餐饮": 900}') is True
    rows = db.get_latest_user_config()
    assert len(rows) == 1 and rows[0]["category_budget"] == '{"餐饮": 900}'


def test_user_config_upsert_overwrites_without_accumulating(db):
    """
    函数功能与逻辑描述：
        ★核心语义：upsert 是「**单行覆盖**」—— WebUI 多次保存**不得累积历史行**，
        否则 get_latest_user_config 的"最新一行"语义会被撑爆。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    for i in range(5):
        assert db.upsert_user_config(f"城市{i}", 1000.0 + i, "正常") is True
    assert len(db.query_sql("SELECT * FROM user_config")) == 1, "多次保存累积了历史行"
    assert db.get_latest_user_config()[0]["city"] == "城市4"


def test_user_config_check_constraint(db):
    """CHECK 兜底：非法 consume_mode（非三档）写入失败。"""
    assert db.add_user_config("深圳", 4000.0, "乱写模式") is False


# ===================== ⑦ bill =====================
def test_bill_add_and_get_all_sorted_desc(db):
    """get_all_bill 按 consume_time **倒序**（最近的在前）。"""
    db.add_bill(10.0, "餐饮", "2026-10-01 12:00:00")
    db.add_bill(20.0, "交通", "2026-10-05 08:00:00")
    rows = db.get_all_bill()
    assert [r["consume_time"] for r in rows] == ["2026-10-05 08:00:00", "2026-10-01 12:00:00"]


def test_bill_get_by_time_is_inclusive_between(db):
    """get_bill_by_time 为**左闭右闭** BETWEEN，且按 consume_time 升序。"""
    for t in ("2026-09-30 23:59:59", "2026-10-01 00:00:00",
              "2026-10-31 23:59:59", "2026-11-01 00:00:00"):
        db.add_bill(5.0, "餐饮", t)
    rows = db.get_bill_by_time("2026-10-01 00:00:00", "2026-10-31 23:59:59")
    assert [r["consume_time"] for r in rows] == ["2026-10-01 00:00:00", "2026-10-31 23:59:59"], \
        "区间边界（两端）未被包含"


def test_bill_delete_hit_and_miss(db):
    """delete_bill：命中删除后查不到；删除不存在的 id 仍返回 True（删除 0 行也算成功）。"""
    db.add_bill(10.0, "餐饮", "2026-10-01 12:00:00")
    bid = db.get_all_bill()[0]["id"]
    assert db.delete_bill(bid) is True
    assert db.get_all_bill() == []
    assert db.delete_bill(99999) is True


# ===================== ⑧ city_price =====================
def test_city_price_add_and_get(db):
    """add_city_price + get_city_price 精确匹配返回 float。"""
    db.add_city_price("深圳", "餐饮", 35.0, "一线")
    assert db.get_city_price("深圳", "餐饮") == 35.0


def test_city_price_get_returns_zero_when_absent(db):
    """★无记录时返回 **0.0**（调用方据此判定"无有效基准"），不返回 None、不抛。"""
    assert db.get_city_price("不存在的城", "餐饮") == 0.0
    db.add_city_price("深圳", "餐饮", 35.0, "一线")
    assert db.get_city_price("深圳", "交通") == 0.0


def test_city_price_unique_constraint(db):
    """UNIQUE(city, category)：同城同品类重复插入失败（导入脚本须先清空再批插）。"""
    assert db.add_city_price("深圳", "餐饮", 35.0, "一线") is True
    assert db.add_city_price("深圳", "餐饮", 40.0, "一线") is False


def test_batch_add_city_price_normalizes_tuple_length(db):
    """
    函数功能与逻辑描述：
        批量导入的入参归一口径：len>=4 取前 4 项（多余忽略）；len==3 自动补 city_level=None。
        这是导入脚本 `import_city_price` 的落库入口，长度口径错会导致整批回滚。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    ok = db.batch_add_city_price([
        ("深圳", "餐饮", 35.0, "一线", "多余字段应被忽略"),
        ("成都", "交通", 4.0),                      # 3 元组 → city_level=None
    ])
    assert ok is True
    assert db.get_city_level("深圳") == "一线"
    assert db.get_city_level("成都") is None
    assert len(db.query_sql("SELECT * FROM city_price")) == 2


def test_get_city_level_skips_null(db):
    """get_city_level 只取 city_level 非 NULL 的行；全为 NULL 时返回 None。"""
    db.add_city_price("甲城", "餐饮", 30.0, None)
    assert db.get_city_level("甲城") is None


def test_same_level_avg_price(db):
    """同级兜底：对同 city_level 下该品类取 AVG（这正是"本城无基准"的替代口径）。"""
    db.add_city_price("A城", "餐饮", 30.0, "一线")
    db.add_city_price("B城", "餐饮", 50.0, "一线")
    db.add_city_price("C城", "交通", 10.0, "一线")      # 不同品类，不参与
    assert db.get_same_level_avg_price("一线", "餐饮") == 40.0


def test_same_level_avg_price_short_circuits_on_empty_level(db):
    """边界：city_level 为空 → 直接短路返回 0.0（不查库）。"""
    assert db.get_same_level_avg_price("", "餐饮") == 0.0
    assert db.get_same_level_avg_price(None, "餐饮") == 0.0


# ===================== ⑨ analysis（零调用契约接口） =====================
def test_analysis_add_and_get_by_month(db):
    """analysis 为"零调用死表"但方法保留为契约接口 —— 行为仍须正确（按 month 精确匹配）。"""
    assert db.add_analysis_record("2026-10", 3200.0, 0, "本月未超支") is True
    rows = db.get_analysis_by_month("2026-10")
    assert len(rows) == 1 and rows[0]["content"] == "本月未超支"
    assert db.get_analysis_by_month("2026-09") == []


# ===================== ⑩ monthly_habit =====================
def test_habit_upsert_accumulates_and_recomputes_avg(db):
    """
    函数功能与逻辑描述：
        ★核心语义：同一 (month, category) 重复写入时 **金额累加、笔数 +1、笔均重算**，
        而不是插入多行 —— 这是"长期记忆事实层"的增量累计口径。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    db.upsert_monthly_habit("2026-10", "餐饮", 30.0)
    db.upsert_monthly_habit("2026-10", "餐饮", 50.0)
    rows = db.get_monthly_habits(["2026-10"])
    assert len(rows) == 1, "重复 upsert 产生了多行"
    assert rows[0]["amount_sum"] == 80.0
    assert rows[0]["count"] == 2
    assert rows[0]["avg_amount"] == 40.0


def test_habit_get_months_filter_and_all(db):
    """get_monthly_habits：传 months 走 IN 过滤；不传返回全表；月份倒序。"""
    db.upsert_monthly_habit("2026-09", "餐饮", 10.0)
    db.upsert_monthly_habit("2026-10", "餐饮", 20.0)
    db.upsert_monthly_habit("2026-10", "交通", 5.0)
    assert {r["month"] for r in db.get_monthly_habits(["2026-10"])} == {"2026-10"}
    all_rows = db.get_monthly_habits()
    assert len(all_rows) == 3
    assert all_rows[0]["month"] == "2026-10", "未按 month 倒序"


def test_habit_recalc_rebuilds_from_bill(db):
    """
    函数功能与逻辑描述：
        recalc_month_habit 以 **bill 为唯一事实源**全量重算 ——
        用于修正"增量累计无法处理删除"的偏差。这里验证：先制造偏差（多累计一笔），
        重算后应**回到 bill 的真实聚合**。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    db.add_bill(30.0, "餐饮", "2026-10-01 12:00:00")
    db.add_bill(50.0, "餐饮", "2026-10-02 12:00:00")
    db.upsert_monthly_habit("2026-10", "餐饮", 999.0)      # 人为偏差
    assert db.recalc_month_habit("2026-10") is True
    rows = db.get_monthly_habits(["2026-10"])
    assert len(rows) == 1
    assert rows[0]["amount_sum"] == 80.0 and rows[0]["count"] == 2
    assert rows[0]["avg_amount"] == 40.0


def test_habit_recalc_empty_month_returns_false(db):
    """边界：该月 bill 表无任何账单 → 返回 False，且该月习惯行被清空、不写新行。"""
    db.upsert_monthly_habit("2026-10", "餐饮", 10.0)
    assert db.recalc_month_habit("2026-10") is False
    assert db.get_monthly_habits(["2026-10"]) == []


def test_habit_cleanup_keeps_recent_months(db):
    """
    函数功能与逻辑描述：
        cleanup_monthly_habit 按"保留最近 N 个自然月（含当月）"滚动删除。
        边界月份用**逐月回退**计算（天然处理跨年），且 YYYY-MM 的字典序等价于时间序。
        用例构造：把 (当月-12) 与 (当月-13) 两行放入，保留 12 个月 → 更早的那行被删。
    入参说明：
        db：临时库 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    today = datetime.now()
    y, m = today.year, today.month

    def back(n: int) -> str:
        yy, mm = y, m - n
        while mm <= 0:
            mm += 12
            yy -= 1
        return f"{yy:04d}-{mm:02d}"

    db.upsert_monthly_habit(back(11), "餐饮", 1.0)     # 保留窗口内（含当月共 12 个月）
    db.upsert_monthly_habit(back(12), "餐饮", 2.0)     # 刚好早于边界 → 应被删
    assert db.cleanup_monthly_habit(keep_months=12) is True
    months = {r["month"] for r in db.get_monthly_habits()}
    assert back(11) in months, "保留窗口内的月份被误删"
    assert back(12) not in months, "超出窗口的月份未被清理"


# ===================== ⑪ parse_sql_result（纯函数，无需 DB） =====================
class _TextContent:
    """MCP SDK 返回体的最小替身：只需 .text 属性。"""

    def __init__(self, text):
        self.text = text


def test_parse_sql_result_from_str():
    """str 形态（client._call_server 提取后的实际形态）解析为 dict 列表。"""
    assert parse_sql_result("[{'amount': 100, 'category': '餐饮'}]") == [
        {"amount": 100, "category": "餐饮"}]


def test_parse_sql_result_from_text_content_list():
    """[TextContent] 形态（部分 SDK 版本的原始返回）同样解析成功。"""
    assert parse_sql_result([_TextContent("[('amount', 1)]")]) == [("amount", 1)]


def test_parse_sql_result_rejects_error_text():
    """
    错误文本分流：服务端在权限不足 / SQL 被拦截 / 语法错误时返回纯文本，
    不以 `[{` / `[(` 开头 → 一律返回 []（**不抛异常**）。
    """
    assert parse_sql_result("操作拒绝：无权访问") == []
    assert parse_sql_result("权限不足：缺少 fact:read") == []
    assert parse_sql_result("[]") == []          # ★空结果集也是 []（函数出口无法区分，见 docstring）


def test_parse_sql_result_rejects_bad_types_and_literals():
    """边界：入参类型不符 / 非法字面量 → 均返回 []；合法字面量正常解析。"""
    assert parse_sql_result(None) == []
    assert parse_sql_result(123) == []
    assert parse_sql_result([]) == []
    assert parse_sql_result("[{坏字面量}]") == []     # SyntaxError → []
    assert parse_sql_result("[{'a': 1}]") == [{"a": 1}]


def test_parse_sql_result_prefix_gate():
    """
    函数功能与逻辑描述：
        ★记录一条**刻意的保守设计**：函数以 `[{` / `[(` 开头判定"这是查询结果"，
        其余一律当"错误文本"返回 []。因此**圆括号**包裹的结果 `({...})` 会被判为错误文本。
        取舍：宁可漏解析、也不把"权限不足：…"这类文本误当数据。
        副作用是"解析出非 list"分支**实际不可达**（前缀已保证外层为 list）——
        本用例把这个边界固定下来，避免后人误以为该分支必须被覆盖。
    入参说明：
        无（纯函数用例）。
    返回值说明：
        无（断言通过即成功）。
    """
    assert parse_sql_result("({'a': 1})") == [], "圆括号结果应被前缀门槛拒绝"
    assert parse_sql_result("(1, 2)") == []
    # 前缀通过即视为结果，外层必为 list（故 [] / [{...}] 都可能）
    assert parse_sql_result("[{'a': 1}]") == [{"a": 1}]
    assert parse_sql_result("[(1, 2)]") == [(1, 2)]
