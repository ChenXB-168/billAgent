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
    """config.get_latest：读 `user_config` 最新一行配置（返回结构与 `db.get_latest_user_config` 同构）。"""
    return db.get_latest_user_config() or []


def sum_month_bills(month: str) -> float:
    """bill.sum_by_month：指定自然月（YYYY-MM）账单金额总和，无数据返回 0（语义同原 nodes._sum_month_bills）。"""
    sql = "SELECT COALESCE(SUM(amount), 0) AS total FROM bill WHERE substr(consume_time, 1, 7) = ?"
    rows = db.query_sql(sql, (month,))
    return float(rows[0]["total"]) if rows else 0.0


def sum_month_bills_by_category(month: str) -> dict:
    """bill.sum_by_category：当月各品类累计 {category: total}（语义同原 nodes._sum_month_bills_by_category）。"""
    sql = ("SELECT category, COALESCE(SUM(amount), 0) AS total FROM bill "
           "WHERE substr(consume_time, 1, 7) = ? GROUP BY category")
    rows = db.query_sql(sql, (month,))
    return {r["category"]: float(r["total"]) for r in rows} if rows else {}


def get_city_avg_price(city: str, category: str) -> float:
    """city_price.get_avg：本城同品类均价；本城无 → 同级城市均价兜底；仍无 → 0。

    原 `orchestrator._query_city_avg_price`（L677-695）三段式兜底逻辑**整体迁入**本工具。
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
    """habit.list：近 N 个自然月消费习惯聚合（`get_monthly_habits`，语义同原 nodes._load_recent_habits）。"""
    return db.get_monthly_habits(months) or []


async def upsert_habit(month: str, category: str, amount: float) -> dict:
    """habit.upsert：月度习惯沉淀（**写**，供 bill_agent 记账成功回调）。

    ★必须 async：LocalAdapter 对同步 fn 默认 `to_thread`（丢进线程池，协程级 `db_lock`
    随即失效）；async fn 被引擎直接 await，`async with db_lock` 在事件循环内持锁，
    与 M3.5「写持锁」语义一致（跨进程写仍由 WAL + busy_timeout 兜底）。

    返回含 `habits`（写后读回该月聚合结果）：供 bill_agent 更新 pkl 语义层，
    避免其为沉淀读而额外持有 `fact:read` 权限（最小权限）。
    """
    async with db_lock:  # 写 + 读回同锁（同连接、无并发插入，M3.5「写后读可见」语义）
        db.upsert_monthly_habit(month, category, float(amount))
        rows = db.get_monthly_habits([month]) or []
    return {"ok": True, "month": month, "category": category,
            "habits": [dict(r) for r in rows]}


__all__ = [
    "get_latest_user_config", "sum_month_bills", "sum_month_bills_by_category",
    "get_city_avg_price", "list_habits", "upsert_habit",
]
