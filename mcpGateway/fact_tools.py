# -*- coding: utf-8 -*-
"""M9（D15）业务事实工具：orchestrator 9 处 `db.` 直连的**工具化收口落点**。

背景：`agents/orchestrator/nodes.py` 裸连 `utils.common.db` ×9（读×8 写×1）——
绕过网关 = 无鉴权 / 无审计 / 无超时 / 无打点（`02` D15 / `03` §9.9.2）。
本模块把 9 处收口为 **6 个细粒度工具**（protocol=local），经 ToolRegistry + ExecutionEngine
统一鉴权 / 审计 / 超时 / 可观测后，由 orchestrator / bill_agent 经 `registry.invoke` 调用。

★方案差异登记（`11` M9 走查 #4）：`03` §9.9.2 映射表原标 transport=mcp；本实现取 **local**——
单机 SQLite 同进程复用现网 DatabaseCRUD，避免 sql_bill server 端 6 处新增（跨进程成对契约面
扩大）；「走正门」= Registry 门面，与底层协议无关（`08` §1.8「MCP 只是众多协议之一」）。

★§8.3 #12 教训对照：本模块 import `utils.common` 会连带初始化 DB——这是**DB 工具的职责**
（读写在本地 SQLite），且仅在 registry 装配（惰性 `get_registry()`）时被 import，属登记在案的
合理依赖，非"纯净模块被意外拉起 DB"。
"""
from utils.common import db, db_lock


def get_latest_user_config() -> list[dict]:
    """
    函数功能与逻辑描述：
        读取 `user_config` 表中最新一行的用户配置（按 create_time 倒序取第一条），
        是编排器获取城市/月预算/消费模式等事实的统一入口。
        对空结果做了 None 兜底，保证调用方拿到的始终是列表而非 None。
    入参说明：
        无。
    返回值说明：
        list[dict]：字典列表，元素结构与 db.get_latest_user_config 同构（列名为 key）；
            表中无数据时返回空列表 []。
    """
    return db.get_latest_user_config() or []


def sum_month_bills(month: str) -> float:
    """
    函数功能与逻辑描述：
        统计指定自然月的账单金额总和，SQL 用 COALESCE(SUM(amount), 0) 保证无记录时返回 0
        而非 NULL；月份匹配用 substr(consume_time, 1, 7) 截取 YYYY-MM 前缀，
        因此 consume_time 必须是 "YYYY-MM-DD HH:MM:SS" 或 "YYYY-MM-DD" 形态。
        语义与迁移前的 orchestrator.nodes._sum_month_bills 完全一致。
    入参说明：
        month (str)：自然月，格式 "YYYY-MM"，如 "2026-09"。
    返回值说明：
        float：该月账单金额合计；无记录时返回 0.0（由 float() 转换，恒为浮点）。
    """
    sql = "SELECT COALESCE(SUM(amount), 0) AS total FROM bill WHERE substr(consume_time, 1, 7) = ?"
    rows = db.query_sql(sql, (month,))
    return float(rows[0]["total"]) if rows else 0.0


def sum_month_bills_by_category(month: str) -> dict:
    """
    函数功能与逻辑描述：
        统计指定自然月各品类的累计金额，用于预算预警链按品类比对额度。
        SQL 用 COALESCE 保证个别品类无金额时不为 NULL，并 GROUP BY category。
        语义与迁移前的 orchestrator.nodes._sum_month_bills_by_category 完全一致。
    入参说明：
        month (str)：自然月，格式 "YYYY-MM"。
    返回值说明：
        dict：{品类名: 累计金额(float)} 映射，如 {"餐饮": 1234.5, "交通": 300.0}；
            该月无账单时返回空字典 {}（不含 0 值条目，调用方需自行按缺失处理）。
    """
    sql = ("SELECT category, COALESCE(SUM(amount), 0) AS total FROM bill "
           "WHERE substr(consume_time, 1, 7) = ? GROUP BY category")
    rows = db.query_sql(sql, (month,))
    return {r["category"]: float(r["total"]) for r in rows} if rows else {}


def get_city_avg_price(city: str, category: str) -> float:
    """
    函数功能与逻辑描述：
        查询城市同品类基准均价，采用三级兜底策略：
        ① 本城本品类均价（city_price 精确命中）；② 本城无数据时取该城市分级下
        所有城市同品类的平均值（get_same_level_avg_price）；③ 仍无则返回 0.0。
        原 orchestrator._query_city_avg_price 的三段式兜底逻辑整体迁入本工具，
        不保留原实现。入参任一为空即短路返回 0.0，避免无意义的全表扫描。
    入参说明：
        city (str)：城市名，如 "深圳"；空串或 None 直接返回 0.0。
        category (str)：消费品类（限定在 餐饮/交通/住宿/购物/娱乐 白名单内）；空串或 None 直接返回 0.0。
    返回值说明：
        float：可用的基准均价；三段兜底全部失败时返回 0.0（调用方须按「无基准」处理，
            不可当作真实均价 0 使用）。
    """
    if not city or not category:
        return 0.0
    avg = db.get_city_price(city, category)
    if avg and avg > 0:
        return float(avg)
    level = db.get_city_level(city)
    if level:
        avg = db.get_same_level_avg_price(level, category)
        if avg and avg > 0:
            return float(avg)
    return 0.0


def list_habits(months: list[str]) -> list[dict]:
    """
    函数功能与逻辑描述：
        读取指定若干自然月的消费习惯聚合数据（monthly_habit 表，按月+品类汇总频次与金额），
        供编排器构造记账预警与统计上下文使用。对空结果做了 None 兜底。
        语义与迁移前的 orchestrator.nodes._load_recent_habits 完全一致。
    入参说明：
        months (list[str])：月份列表，格式 ["2026-08", "2026-07"]；
            传入空列表时 get_monthly_habits 会走「返回全部」分支，调用方需注意。
    返回值说明：
        list[dict]：习惯记录字典列表（含 month / category / amount_sum / count / avg_amount 等列）；
            无数据时返回空列表 []。
    """
    return db.get_monthly_habits(months) or []


async def upsert_habit(month: str, category: str, amount: float) -> dict:
    """
    函数功能与逻辑描述：
        月度消费习惯沉淀（**写操作**），供 bill_agent 记账成功后回调，把「月+品类」的金额求和、
        笔数 +1、均额重算写入 monthly_habit。写与读回在同一把 db_lock 内完成，
        依据 M3.5「写后读可见」语义保证读回的是刚写入的结果。
        ★必须保持 async：LocalAdapter 对同步函数默认走 to_thread（丢进线程池），
        那样协程级的 db_lock 会随即失效；async 函数被引擎直接 await，
        `async with db_lock` 在事件循环内持锁，与 M3.5「写持锁」语义一致
        （跨进程写仍由 WAL + busy_timeout 兜底）。
        返回值里带 habits 是为了让 bill_agent 更新 pkl 语义层时无需为读而额外申请 fact:read 权限
        （最小权限原则）。
    入参说明：
        month (str)：自然月，格式 "YYYY-MM"。
        category (str)：消费品类（5 类白名单之一），与 month 共同构成 monthly_habit 的唯一键。
        amount (float)：本次消费金额；函数内再做一次 float() 转换以防传入字符串或 Decimal。
    返回值说明：
        dict：固定含 ok / month / category / habits 四个键
            - ok (bool)：恒为 True（失败会由底层抛异常而非返回 False）。
            - month (str) / category (str)：原样回显入参。
            - habits (list[dict])：写后读回的该月习惯聚合记录列表（已转为普通 dict）。
    """
    async with db_lock:  # 写 + 读回同锁（同连接、无并发插入，M3.5「写后读可见」语义）
        db.upsert_monthly_habit(month, category, float(amount))
        rows = db.get_monthly_habits([month]) or []
    return {"ok": True, "month": month, "category": category,
            "habits": [dict(r) for r in rows]}


async def recalc_habit(month: str) -> dict:
    """
    函数功能与逻辑描述：
        月度消费习惯**整月重算**（**写操作**），供账单改删成功后修复派生数据一致性。
        为什么必须重算而不是继续 `habit.upsert` 增量：`upsert_monthly_habit` 是
        「金额累加 + 笔数 +1」语义（`utils/common.py:498-508`），**天生无法表达"改小/删除"**——
        删一笔后再 upsert 只会让该月金额越来越大（`设计/12` §4.5）。
        实现直接复用 `db.recalc_month_habit`（以 `bill` 表为唯一事实源全量重算，幂等）：
        先删该月习惯行，再按品类 GROUP BY 重算插入。
        ★必须保持 async：LocalAdapter 对同步函数默认走 to_thread，协程级 `db_lock` 随即失效；
        async 函数被引擎直接 await，`async with db_lock` 在事件循环内持锁，与 M3.5「写持锁」
        语义一致（与 `upsert_habit` 同族同理由）。写与读回同锁完成，保证「写后读可见」。
    入参说明：
        month (str)：自然月，格式 "YYYY-MM"；匹配 bill 时按 substr(consume_time, 1, 7)。
    返回值说明：
        dict：固定含 ok / month / habits 三个键
            - ok (bool)：恒为 True（重算失败由底层抛异常，不作为业务失败返回）。
            - month (str)：原样回显入参。
            - habits (list[dict])：重算后读回的该月习惯聚合；该月已无账单时为空列表
              （属正常结果，不是错误——"改删后该月归零"是预期状态）。
    """
    async with db_lock:  # 写 + 读回同锁（同连接，M3.5「写后读可见」语义）
        db.recalc_month_habit(month)
        rows = db.get_monthly_habits([month]) or []
    return {"ok": True, "month": month, "habits": [dict(r) for r in rows]}


__all__ = [
    "get_latest_user_config", "sum_month_bills", "sum_month_bills_by_category",
    "get_city_avg_price", "list_habits", "upsert_habit", "recalc_habit",
]
