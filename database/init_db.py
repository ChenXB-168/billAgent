# database/init_db.py
# ==============================================
# 数据库初始化脚本 
# 建表一次性操作，无SQL注入风险，无需使用SQLAlchemy
# ==============================================
import sys
from pathlib import Path

# 自动修复路径，否则无法找到其他文件夹
sys.path.append(str(Path(__file__).parent.parent))

from config.config import TASK_STATE_DB
from utils.common import db, logger, DatabaseCRUD

def migrate_bill_table():
    """修复历史 bill 表缺失 CHECK 约束的迁移（IF NOT EXISTS 不会升级旧表）

    流程：建带约束新表 → 清理违规数据 → 复制合规数据 → 换名
    幂等：检测到 DDL 已含 CHECK 约束则直接跳过。
    """
    rows = db.query_sql("SELECT sql FROM sqlite_master WHERE name='bill'")
    if not rows:
        return  # 表不存在，由 init_all_tables 新建
    ddl = rows[0].get("sql", "")
    if "CHECK(amount > 0)" in ddl:
        return  # 已带约束，无需迁移
    logger.warning("bill 表缺少 CHECK 约束，开始迁移修复...")
    # 1. 清理历史违规数据（负数金额/非法类别）
    db.execute_sql(
        "DELETE FROM bill WHERE amount <= 0 OR category NOT IN ('餐饮','交通','住宿','购物','娱乐')"
    )
    # 2. 建带约束新表
    db.execute_sql('''
    CREATE TABLE IF NOT EXISTS bill_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL CHECK(amount > 0),
        category TEXT CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
        consume_time TEXT,
        remark TEXT,
        task_id TEXT,          -- ★M5（D2）：重建表必须带上该列，否则迁移后丢失
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # 3. 复制数据
    db.execute_sql(
        "INSERT INTO bill_new (id, amount, category, consume_time, remark, create_time) "
        "SELECT id, amount, category, consume_time, remark, create_time FROM bill"
    )
    # 4. 换名
    db.execute_sql("DROP TABLE bill")
    db.execute_sql("ALTER TABLE bill_new RENAME TO bill")
    logger.info("bill 表迁移完成，CHECK(amount>0)/CHECK(五分类) 约束已生效")


def init_all_tables(close_after: bool = True):
    """统一初始化全部数据表，IF NOT EXISTS 重复运行不会丢数据

    :param close_after: 结束后是否关闭连接（默认 True，保持独立脚本运行行为）。
        测试/进程内调用需传 False——`db` 是全局单例且 `close()` 单向不可重开，
        关闭后同进程内后续 SQL 会全部失败（见 utils/common.py DatabaseCRUD.close）。
    """
    # 1. 用户配置表
    sql1 = '''
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
    '''

    # 2. 账单表（匹配bill_agent五分类约束，仅存支出正数）
    #    task_id：M5（D2）新增——派发任务的幂等键，崩溃恢复时按此反查账单对账
    #    （`04` §3.1 修订：有记录→completed，无记录→按 retriable 重投/failed）。
    #    可空以兼容历史数据（老账单无 task_id，不参与对账）。
    sql2 = '''
    CREATE TABLE IF NOT EXISTS bill (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL CHECK(amount > 0),
        category TEXT CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
        consume_time TEXT,
        remark TEXT,
        task_id TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''

    # 3. 城市物价对标表（price_agent专用，分类与账单完全统一）
    #    city_level：城市分级（一线/新一线/二线/三线等），用于"本城无均价 → 同级城市均价兜底"
    sql3 = '''
    CREATE TABLE IF NOT EXISTS city_price (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT NOT NULL,
        category TEXT NOT NULL CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
        avg_price REAL CHECK(avg_price > 0),
        city_level TEXT,
        UNIQUE(city, category)
    )
    '''

    # 4. 月度消费分析结果表（finance/stat输出存储）
    sql4 = '''
    CREATE TABLE IF NOT EXISTS analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT NOT NULL,
        total REAL CHECK(total >= 0),
        over_budget INTEGER CHECK(over_budget IN (0,1)),
        content TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''

    # 5. 月度消费习惯表（长期记忆事实层：按 月+品类 聚合频次与金额）
    sql5 = '''
    CREATE TABLE IF NOT EXISTS monthly_habit (
        month TEXT NOT NULL,
        category TEXT NOT NULL CHECK(category IN ('餐饮','交通','住宿','购物','娱乐')),
        amount_sum REAL NOT NULL DEFAULT 0 CHECK(amount_sum >= 0),
        count INTEGER NOT NULL DEFAULT 0 CHECK(count >= 0),
        avg_amount REAL NOT NULL DEFAULT 0,
        update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (month, category)
    )
    '''

    # 6. LLM 调用统计表（D3 可观测：token/耗时归因）
    #    设计依据 `03_逻辑架构.md` §9.3 Tracer：run_id 贯穿的 span 树 + token/耗时统计。
    #    channel：local（Ollama，记 eval_count）/ external（OpenAI 兼容，记 usage）
    #    reasoning_tokens：推理模型的思考消耗——M1 实测 num_predict 预算被 reasoning
    #    占满会导致 content 为空，该字段是定位此类问题的关键指标。
    sql6 = '''
    CREATE TABLE IF NOT EXISTS llm_call_stat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        span_id TEXT,
        parent_span_id TEXT,
        session_id TEXT,
        agent_tag TEXT,
        model TEXT,
        channel TEXT CHECK(channel IN ('local','external')),
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        reasoning_tokens INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        latency_ms INTEGER NOT NULL DEFAULT 0,
        success INTEGER NOT NULL DEFAULT 1 CHECK(success IN (0,1)),
        error_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
    sql6_idx = 'CREATE INDEX IF NOT EXISTS idx_llm_stat_run ON llm_call_stat(run_id)'
    sql6_idx2 = 'CREATE INDEX IF NOT EXISTS idx_llm_stat_agent ON llm_call_stat(agent_tag)'

    # 批量执行建表
    db.execute_sql(sql1)
    db.execute_sql(sql2)
    db.execute_sql(sql3)
    db.execute_sql(sql4)
    db.execute_sql(sql5)
    db.execute_sql(sql6)
    db.execute_sql(sql6_idx)
    db.execute_sql(sql6_idx2)

    # 修复历史库缺失的 CHECK 约束
    migrate_bill_table()
    migrate_user_config_table()
    # 历史 user_config 消费模式英文值迁移为中文
    migrate_user_config_mode()
    # 历史 user_config 表补充新列（category_budget / quota_mode，幂等）
    ensure_user_config_columns()
    # 历史 city_price 表补充 city_level 列（同级城市均价兜底，幂等）
    ensure_city_price_columns()
    # M5（D2）：任务状态独立库建表 + bill.task_id 列（崩溃对账依据，04 §3.1）
    init_task_state_db()
    migrate_bill_add_task_id()
    # M5（D14）：budget_type 收敛为仅 month
    migrate_budget_type_to_month()
    # M7（D3）：span 树持久化表（bill.db）
    ensure_trace_tables(db)

    logger.info("所有数据表校验/创建完成！")
    if close_after:
        db.close()


def migrate_user_config_table():
    """修复历史 user_config 表缺失 CHECK 约束的迁移（IF NOT EXISTS 不会升级旧表）

    旧表结构：consume_mode TEXT（无中文约束）/ month_budget REAL / budget_type TEXT
    流程：建带约束新表 → 规范化数据（英文模式转中文、非法值兜底）→ 复制 → 换名
    幂等：检测到 DDL 已含中文 CHECK 约束则直接跳过。
    """
    rows = db.query_sql("SELECT sql FROM sqlite_master WHERE name='user_config'")
    if not rows:
        return  # 表不存在，由 init_all_tables 新建
    ddl = rows[0].get("sql", "")
    if "CHECK(consume_mode" in ddl:
        return  # 已带约束，无需迁移
    logger.warning("user_config 表缺少 CHECK 约束，开始迁移修复...")
    # 1. 先迁移旧英文消费模式为中文，再建约束表拷贝
    migrate_user_config_mode()
    # 2. 建带约束新表
    db.execute_sql('''
    CREATE TABLE IF NOT EXISTS user_config_new (
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
    ''')
    # 3. 规范化拷贝：负预算置空、非法模式/类型兜底为默认值，不丢行
    db.execute_sql(
        "INSERT INTO user_config_new (id, city, month_budget, consume_mode, budget_type, "
        "category_budget, quota_mode, remind_pref, create_time) "
        "SELECT id, city, "
        "CASE WHEN month_budget < 0 THEN NULL ELSE month_budget END, "
        "CASE WHEN consume_mode IN ('节俭','正常','宽松') THEN consume_mode ELSE '正常' END, "
        # ★M5（D14）：budget_type 一律收敛为 month（year 从未被下游实现）
        "'month', "
        "category_budget, "
        "CASE WHEN quota_mode IN ('auto','user','habit','city') THEN quota_mode ELSE 'auto' END, "
        "remind_pref, "
        "create_time FROM user_config"
    )
    # 4. 换名
    db.execute_sql("DROP TABLE user_config")
    db.execute_sql("ALTER TABLE user_config_new RENAME TO user_config")
    logger.info("user_config 表迁移完成，中文 consume_mode/预算 约束已生效")


def migrate_user_config_mode():
    """历史 user_config 表 consume_mode 英文值迁移为中文（save->节俭 / normal->正常 / ease->宽松）

    与表 CHECK 约束('节俭','正常','宽松')及 CONSUMPTION_MODES 全链路统一为中文。
    幂等：仅更新仍为英文的旧值，重复运行无副作用。
    """
    db.execute_sql(
        """
        UPDATE user_config SET consume_mode = CASE consume_mode
            WHEN 'save' THEN '节俭'
            WHEN 'normal' THEN '正常'
            WHEN 'ease' THEN '宽松'
            ELSE consume_mode END
        WHERE consume_mode IN ('save', 'normal', 'ease')
        """
    )


def ensure_user_config_columns():
    """历史 user_config 表补充新列（category_budget / quota_mode），幂等：
    CREATE TABLE IF NOT EXISTS 不会升级旧表，需检测列存在性后 ALTER TABLE ADD COLUMN。
    """
    cols = db.query_sql("PRAGMA table_info(user_config)")
    if not cols:
        return  # 表不存在，由 init_all_tables 新建
    names = {c.get("name") for c in cols}
    if "category_budget" not in names:
        db.execute_sql("ALTER TABLE user_config ADD COLUMN category_budget TEXT")
        logger.info("user_config 表新增列 category_budget")
    if "quota_mode" not in names:
        db.execute_sql(
            "ALTER TABLE user_config ADD COLUMN quota_mode TEXT DEFAULT 'auto' "
            "CHECK(quota_mode IN ('auto','user','habit','city'))"
        )
        logger.info("user_config 表新增列 quota_mode")
    if "remind_pref" not in names:
        db.execute_sql("ALTER TABLE user_config ADD COLUMN remind_pref TEXT")
        logger.info("user_config 表新增列 remind_pref")


def ensure_city_price_columns():
    """历史 city_price 表补充 city_level 列（同级城市均价兜底用），幂等"""
    cols = db.query_sql("PRAGMA table_info(city_price)")
    if not cols:
        return  # 表不存在，由 init_all_tables 新建
    names = {c.get("name") for c in cols}
    if "city_level" not in names:
        db.execute_sql("ALTER TABLE city_price ADD COLUMN city_level TEXT")
        logger.info("city_price 表新增列 city_level")


# M5（D2）任务状态表 DDL——**单一来源**：`init_task_state_db()` 与 `TaskManager` 共用，
# 避免"建表语句两处维护导致漂移"（`04` §2.4 / `08` §2.3 ①）
TASK_STATE_DDL = '''
    CREATE TABLE IF NOT EXISTS task_state (
        task_id      TEXT PRIMARY KEY,      -- 自然幂等键（orchestrator 生成）
        session_id   TEXT NOT NULL,
        agent_id     TEXT NOT NULL,
        status       TEXT NOT NULL,         -- submitted|running|completed|failed|interrupted
        payload      TEXT,                  -- JSON：派发参数
        result       TEXT,                  -- JSON：执行结果
        retry_cnt    INTEGER DEFAULT 0,
        retriable    INTEGER DEFAULT 1,     -- 有副作用任务置 0（04 §3.3）
        created_at   REAL,
        updated_at   REAL
    )
    '''
TASK_STATE_INDEX_DDLS = (
    'CREATE INDEX IF NOT EXISTS idx_status ON task_state(status, updated_at)',
    'CREATE INDEX IF NOT EXISTS idx_session ON task_state(session_id)',
)


def ensure_task_state_tables(db_inst) -> None:
    """在给定连接上建 `task_state` 表与索引（幂等）。

    ★为什么 `TaskManager` 也要调：状态机不应依赖"启动时先跑过 init_all_tables"的调用顺序——
    否则任何新建库 / 独立测试环境都会 `no such table`。表结构与索引由本函数唯一定义。
    """
    db_inst.execute_sql(TASK_STATE_DDL)
    for ddl in TASK_STATE_INDEX_DDLS:
        db_inst.execute_sql(ddl)


# M7（D3 阶段2）span 树持久化表 DDL——**单一来源**：`init_all_tables` 与 `utils/tracer.py`
# 共用，避免"建表语句两处维护导致漂移"（同 `04` §2.6 llm_call_stat 先例）。
#   kind：orchestrator（编排入口 root）/ agent（worker 子图执行）/ tool（工具调用）
#   status：ok / error；duration_ms：自动计时；error：失败归因文本（截断 500 字符）
TRACE_SPAN_DDL = '''
    CREATE TABLE IF NOT EXISTS trace_span (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id          TEXT,               -- 一次端到端请求 ID（与 llm_call_stat.run_id 关联）
        span_id         TEXT,               -- 当前 span
        parent_span_id  TEXT,               -- 父 span（NULL = root span）
        name            TEXT NOT NULL,      -- orchestrator.run / bill_agent.run / tool.llm.chat
        kind            TEXT,               -- orchestrator / agent / tool
        agent_id        TEXT,               -- 归因主体（orchestrator_agent / bill_agent ...）
        status          TEXT NOT NULL DEFAULT 'ok',   -- ok / error
        duration_ms     INTEGER NOT NULL DEFAULT 0,
        error           TEXT,               -- 失败归因（类型: 消息，截断 500）
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''
TRACE_SPAN_INDEX_DDLS = (
    'CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_span(run_id)',
    'CREATE INDEX IF NOT EXISTS idx_trace_span_id ON trace_span(span_id)',
)


def ensure_trace_tables(db_inst) -> None:
    """在给定连接上建 `trace_span` 表与索引（幂等）。

    ★为什么 `utils/tracer.py` 也要调：可观测埋点不应依赖"启动时先跑过 init_all_tables"的
    调用顺序——否则未初始化库 / 独立测试环境会在打点时报 `no such table`（虽被吞掉但统计缺失）。
    """
    db_inst.execute_sql(TRACE_SPAN_DDL)
    for ddl in TRACE_SPAN_INDEX_DDLS:
        db_inst.execute_sql(ddl)


def init_task_state_db(close_after: bool = True):
    """M5（D2）：初始化任务状态独立库 `database/task_state.db`（`04` §2.4 / `08` §2.3 ①）。

    ★为什么独立库：Durable 每次状态流转一次写（submitted→running→completed），与 bill.db
    同库会争抢 SQLite 写锁；独立库也便于清理与备份隔离（`08` §6 Q4 推荐）。
    """
    ts_db = DatabaseCRUD(TASK_STATE_DB)
    ensure_task_state_tables(ts_db)
    if close_after:
        ts_db.close()


def migrate_bill_add_task_id():
    """M5（D2）：bill 表补 `task_id` 列（可空，兼容历史数据）——崩溃对账依据（`04` §3.1）。

    幂等：`PRAGMA table_info` 已含该列则跳过。历史行 task_id 为 NULL → 恢复时无对账依据，
    按 `retriable` 决定（bill 写操作 retriable=0 → 标记 failed，宁可告警也不重复记账）。
    """
    cols = db.query_sql("PRAGMA table_info(bill)")
    if not cols:
        return  # 表不存在，由 init_all_tables 新建（新表已含 task_id 列）
    names = {c.get("name") for c in cols}
    if "task_id" not in names:
        db.execute_sql("ALTER TABLE bill ADD COLUMN task_id TEXT")
        logger.info("bill 表新增列 task_id（D2 崩溃对账依据）")


def migrate_budget_type_to_month():
    """M5（D14）：`user_config.budget_type` 收敛为仅 `month`（`04` §6.3 变更 2 / `02` D14）。

    背景——三处口径互相矛盾，属"声称支持但未实现"：
      - DB CHECK：`month` / `year`（旧 DDL）
      - LEGACY 校验层：`month` / `week`（`coreModules/target_control.py`）
      - 下游计算：按 `week` 分支（`coreModules/finance_advice.py`）
    → `year` 存得进去但下游不认，`week` 存不进去但校验层放行。

    收敛：数据统一 `month` + 表约束收紧为 `CHECK(budget_type = 'month')`。
    幂等：DDL 已无旧约束（`month','year`）且数据无非 month 值 → 跳过。
    """
    rows = db.query_sql("SELECT sql FROM sqlite_master WHERE name='user_config'")
    if not rows:
        return  # 表不存在，由 init_all_tables 新建（新表已是新约束）
    ddl = rows[0].get("sql", "") or ""
    dirty = db.query_sql(
        "SELECT COUNT(1) AS c FROM user_config WHERE budget_type IS NULL OR budget_type <> 'month'")
    dirty_cnt = (dirty[0].get("c") or 0) if dirty else 0
    if "'month','year'" not in ddl and dirty_cnt == 0:
        return  # 已是新约束且无脏数据
    logger.warning(f"user_config.budget_type 收敛为仅 month（脏数据 {dirty_cnt} 行），开始迁移...")
    if dirty_cnt:
        db.execute_sql("UPDATE user_config SET budget_type='month' "
                       "WHERE budget_type IS NULL OR budget_type <> 'month'")
    # SQLite 无法 ALTER CHECK → 建新约束表 → 复制 → 换名（与 migrate_user_config_table 同模式）
    db.execute_sql('''
    CREATE TABLE IF NOT EXISTS user_config_new (
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
    ''')
    db.execute_sql(
        "INSERT INTO user_config_new (id, city, month_budget, consume_mode, budget_type, "
        "category_budget, quota_mode, remind_pref, create_time) "
        "SELECT id, city, "
        "CASE WHEN month_budget < 0 THEN NULL ELSE month_budget END, "
        "CASE WHEN consume_mode IN ('节俭','正常','宽松') THEN consume_mode ELSE '正常' END, "
        "'month', "
        "category_budget, "
        "CASE WHEN quota_mode IN ('auto','user','habit','city') THEN quota_mode ELSE 'auto' END, "
        "remind_pref, "
        "create_time FROM user_config"
    )
    db.execute_sql("DROP TABLE user_config")
    db.execute_sql("ALTER TABLE user_config_new RENAME TO user_config")
    logger.info("user_config.budget_type 已收敛为仅 month（D14）")


if __name__ == "__main__":
    init_all_tables()