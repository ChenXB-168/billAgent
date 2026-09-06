# ==============================================
# 通用工具类 - CRUD + 日志 + 单库单连接+专属协程锁
# ==============================================
from loguru import logger
from pathlib import Path
from config.config import LOG_PATH, AUDIT_LOG_PATH, DB_PATH
import pysqlite3 as sqlite3
from datetime import datetime
import asyncio
import ast

# 运行日志
logger.add(
    LOG_PATH,
    rotation="1 day",
    retention="7 days",
    format="{time} | {level} | {message}",
    encoding="utf-8"
)

# 独立审计日志（MCP/权限操作专用）
audit_logger = logger.bind(name="audit")
audit_logger.add(
    AUDIT_LOG_PATH,
    rotation="1 day",
    retention="7 days",
    format="{time} | {level} | {extra} | {message}",
    encoding="utf-8"
)

class DatabaseCRUD:
    def __init__(self, db_path: Path | str = DB_PATH):
        """通用 SQLite 访问封装。

        :param db_path: 库路径，默认主业务库 `DB_PATH`。
            ★M5（D2）：新增可选参数以支持 `task_state.db` 独立库——Durable 状态写入频繁，
            与业务库同库会争抢写锁（`04` §2.4 / `08` §6 Q4）。
            默认参数保证既有 `db = DatabaseCRUD()` 单例与全部调用方零改动。
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # M3.5（11 开工清单 / 04 §4.2-4.3）：WAL + busy_timeout。
        # db_lock 是【进程内】锁（04 §4.1 勘误），跨进程（主进程 vs MCP server 子进程）写互斥
        # 只能靠 SQLite 层兜底；WAL 同时是 M6 并行派发的前提（读不阻塞写、写不阻塞读）。
        # 注意：WAL 会生成 bill.db-wal / bill.db-shm（备份须覆盖，07 §2.4 已注明）。
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._closed = False
        logger.info("数据库连接初始化成功（journal_mode=WAL / busy_timeout=5000）")

    def begin_transaction(self):
        """开启事务，批量操作前调用"""
        if not self._closed:
            self.conn.execute("BEGIN TRANSACTION;")

    def commit(self):
        """提交事务"""
        if not self._closed:
            self.conn.commit()

    def rollback(self):
        """回滚事务"""
        if not self._closed:
            self.conn.rollback()

    def execute_sql(self, sql, params=None, retry=3):
        """执行写SQL，自带database locked重试，异常自动回滚"""
        if self._closed:
            logger.error("数据库连接已关闭，无法执行SQL")
            return False

        cur = self.conn.cursor()
        try:
            for i in range(retry):
                try:
                    if params:
                        cur.execute(sql, params)
                    else:
                        cur.execute(sql)
                    self.conn.commit()
                    return True
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and i < retry - 1:
                        # M3.5：busy_timeout=5000 已由 SQLite 在 C 层吸收锁竞争（等待而非立即失败），
                        # 不再需要 time.sleep(0.1)（同步阻塞会冻结事件循环，04 §4.3）；直接重试。
                        continue
                    self.conn.rollback()
                    raise
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"SQL执行失败: {str(e)} | SQL: {sql}")
            self.conn.rollback()
            return False
        finally:
            cur.close()

    def query_sql(self, sql, params=None):
        """执行查询SQL，异常返回空列表并打日志"""
        if self._closed:
            logger.error("数据库连接已关闭，无法执行查询")
            return []

        cur = self.conn.cursor()
        try:
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            rows = cur.fetchall()
            if not rows or not cur.description:
                return []
            # 统一返回字典列表（列名做key），避免下游把tuple当dict用（联调P0坑）
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in rows]
        except Exception as e:
            logger.error(f"查询失败: {str(e)} | SQL: {sql}")
            return []
        finally:
            cur.close()

    def close(self):
        """安全关闭连接，重复调用不报错"""
        if not self._closed:
            try:
                self.conn.close()
                self._closed = True
                logger.info("数据库连接已关闭")
            except Exception as e:
                logger.error(f"关闭数据库连接失败: {str(e)}")

    # ---------- 用户配置表 ----------
    def add_user_config(self, city: str, month_budget: float, consume_mode: str, budget_type: str = "month",
                        category_budget: str = None, quota_mode: str = "auto", remind_pref: str = None):
        """新增用户配置；category_budget 为品类预算 JSON 字符串（如 '{"餐饮": 900, "交通": 300}'），
        quota_mode 为品类额度评估方式（auto/user/habit/city，默认 auto 自动选择），
        remind_pref 为用户个性化提醒偏好（语气/语言/格式，随汇总提示词注入）。"""
        sql = ("INSERT INTO user_config (city, month_budget, consume_mode, budget_type, category_budget, quota_mode, remind_pref) "
               "VALUES (?, ?, ?, ?, ?, ?, ?)")
        return self.execute_sql(sql, (city, month_budget, consume_mode, budget_type, category_budget, quota_mode, remind_pref))

    def get_latest_user_config(self):
        sql = "SELECT * FROM user_config ORDER BY create_time DESC LIMIT 1"
        return self.query_sql(sql)

    def upsert_user_config(self, city: str, month_budget: float, consume_mode: str, budget_type: str = "month",
                           category_budget: str = None, quota_mode: str = "auto", remind_pref: str = None):
        """WebUI 表单写入口：单行覆盖语义（读侧 get_latest_user_config 取"最新一行"= 当前配置）。

        幂等策略：最新行已存在 → 原地 UPDATE 并刷新 create_time（保证永远是最新行）；
        表为空 → 走 add_user_config 插入首行。运行期只应存在一行"当前配置"，
        WebUI 多次保存不会累积历史行（避免 create_time 秒级相同导致 ORDER BY 歧义）。
        形参语义与 add_user_config 完全一致（共用一套列/约束）。
        """
        rows = self.query_sql(
            "SELECT id FROM user_config ORDER BY create_time DESC, id DESC LIMIT 1")
        if rows:
            return self.execute_sql(
                "UPDATE user_config SET city=?, month_budget=?, consume_mode=?, budget_type=?, "
                "category_budget=?, quota_mode=?, remind_pref=?, create_time=CURRENT_TIMESTAMP "
                "WHERE id=?",
                (city, month_budget, consume_mode, budget_type, category_budget, quota_mode,
                 remind_pref, rows[0]["id"]),
            )
        return self.add_user_config(city, month_budget, consume_mode, budget_type,
                                    category_budget, quota_mode, remind_pref)

    # ---------- 账单表（核心） ----------
    def add_bill(self, amount: float, category: str, consume_time: str, remark: str = ""):
        sql = "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)"
        return self.execute_sql(sql, (amount, category, consume_time, remark))

    def get_all_bill(self):
        return self.query_sql("SELECT * FROM bill ORDER BY consume_time DESC")

    def get_bill_by_time(self, start_time: str, end_time):
        sql = "SELECT * FROM bill WHERE consume_time BETWEEN ? AND ? ORDER BY consume_time"
        return self.query_sql(sql, (start_time, end_time))

    def delete_bill(self, bill_id: int):
        sql = "DELETE FROM bill WHERE id = ?"
        return self.execute_sql(sql, (bill_id,))

    # ---------- 城市物价表 ----------
    def add_city_price(self, city: str, category: str, avg_price: float, city_level: str = None):
        """新增城市物价记录；city_level 为城市分级（一线/新一线/二线/三线），用于同级城市均价兜底"""
        sql = "INSERT INTO city_price (city, category, avg_price, city_level) VALUES (?, ?, ?, ?)"
        return self.execute_sql(sql, (city, category, avg_price, city_level))

    def batch_add_city_price(self, data_list: list, retry=3):
        """批量导入物价数据，自带锁重试、异常回滚。
        data_list 元素支持 (city, category, avg_price) 三元组（city_level 兜底 None）
        或 (city, category, avg_price, city_level) 四元组。
        """
        if self._closed:
            logger.error("数据库连接已关闭，无法批量导入")
            return False

        cur = self.conn.cursor()
        try:
            for i in range(retry):
                try:
                    norm = []
                    for item in data_list:
                        if len(item) >= 4:
                            norm.append(tuple(item[:4]))
                        else:
                            norm.append(tuple(item) + (None,))
                    cur.executemany(
                        "INSERT INTO city_price (city, category, avg_price, city_level) VALUES (?, ?, ?, ?)",
                        norm
                    )
                    self.conn.commit()
                    logger.info(f"批量导入物价数据 {len(data_list)} 条成功")
                    return True
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and i < retry - 1:
                        # M3.5：busy_timeout 已在 C 层吸收锁竞争，去掉同步 sleep（防冻结事件循环）
                        continue
                    self.conn.rollback()
                    logger.error(f"批量导入数据库锁死，重试耗尽: {e}")
                    return False
            self.conn.rollback()
            return False
        except Exception as e:
            logger.error(f"批量导入失败: {e}")
            self.conn.rollback()
            return False
        finally:
            cur.close()

    def get_city_price(self, city: str, category: str):
        res = self.query_sql(
            "SELECT avg_price FROM city_price WHERE city = ? AND category = ? LIMIT 1",
            (city, category)
        )
        return float(res[0]["avg_price"]) if res and res[0] and res[0]["avg_price"] is not None else 0.0

    def get_city_level(self, city: str):
        """查询城市分级（一线/新一线/二线/三线）；无数据返回 None"""
        res = self.query_sql(
            "SELECT city_level FROM city_price WHERE city = ? AND city_level IS NOT NULL LIMIT 1",
            (city,)
        )
        return res[0]["city_level"] if res and res[0] and res[0]["city_level"] else None

    def get_same_level_avg_price(self, city_level: str, category: str):
        """同级城市该品类均价兜底：同分级下所有城市同品类的平均值；无数据返回 0.0"""
        if not city_level:
            return 0.0
        res = self.query_sql(
            "SELECT AVG(avg_price) AS avg FROM city_price "
            "WHERE city_level = ? AND category = ? AND avg_price > 0",
            (city_level, category)
        )
        return float(res[0]["avg"]) if res and res[0] and res[0]["avg"] is not None else 0.0

    # ---------- 消费分析表 ----------
    def add_analysis_record(self, month: str, total: float, over_budget: int, content: str):
        sql = "INSERT INTO analysis (month, total, over_budget, content) VALUES (?, ?, ?, ?)"
        return self.execute_sql(sql, (month, total, over_budget, content))

    def get_analysis_by_month(self, month: str):
        return self.query_sql("SELECT * FROM analysis WHERE month = ?", (month,))

    # ---------- 月度消费习惯表（长期记忆事实层：月+品类 聚合频次与金额） ----------
    def upsert_monthly_habit(self, month: str, category: str, amount: float):
        """记账成功后增量累计当月某品类消费习惯：金额求和、笔数+1、均额重算"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = """
        INSERT INTO monthly_habit (month, category, amount_sum, count, avg_amount, update_time)
        VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(month, category) DO UPDATE SET
            amount_sum = monthly_habit.amount_sum + excluded.amount_sum,
            count = monthly_habit.count + 1,
            avg_amount = (monthly_habit.amount_sum + excluded.amount_sum) / (monthly_habit.count + 1),
            update_time = excluded.update_time
        """
        return self.execute_sql(sql, (month, category, amount, amount, now))

    def get_monthly_habits(self, months: list = None) -> list:
        """读取月度消费习惯；months 指定月份列表（如 ['2026-08','2026-07']），None 返回全部"""
        if months:
            placeholders = ",".join("?" * len(months))
            sql = f"SELECT * FROM monthly_habit WHERE month IN ({placeholders}) ORDER BY month DESC, category"
            return self.query_sql(sql, months)
        return self.query_sql("SELECT * FROM monthly_habit ORDER BY month DESC, category")

    def recalc_month_habit(self, month: str):
        """从 bill 表全量重算某月消费习惯（删除/编辑账单后的兜底一致性）"""
        self.execute_sql("DELETE FROM monthly_habit WHERE month = ?", (month,))
        rows = self.query_sql(
            "SELECT category, SUM(amount) AS amount_sum, COUNT(*) AS cnt "
            "FROM bill WHERE substr(consume_time, 1, 7) = ? GROUP BY category",
            (month,)
        )
        if not rows:
            return False
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            amount_sum = float(r["amount_sum"] or 0)
            cnt = int(r["cnt"] or 0)
            avg = round(amount_sum / cnt, 2) if cnt else 0.0
            self.execute_sql(
                "INSERT INTO monthly_habit (month, category, amount_sum, count, avg_amount, update_time) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (month, r["category"], amount_sum, cnt, avg, now)
            )
        return True

    def cleanup_monthly_habit(self, keep_months: int = 12):
        """滚动清理：仅保留最近 keep_months 个自然月（含当月）的习惯数据"""
        today = datetime.now()
        year, month = today.year, today.month
        for _ in range(keep_months - 1):
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        boundary = f"{year:04d}-{month:02d}"
        return self.execute_sql("DELETE FROM monthly_habit WHERE month < ?", (boundary,))

# 单数据库唯一实例 + 该库专属异步锁（一库一锁）
# M3.5 语义（04 §4.1 勘误 / §4.3）：db_lock 是【进程内】锁——跨进程写互斥由 SQLite
# WAL + busy_timeout 兜底（主进程 vs MCP server 子进程）。
# 使用规则：写操作（有副作用）用 async with db_lock 包裹；读操作不加锁
# （同连接同步执行、事件循环单线程天然串行，04 §4.3）。server 子进程（sql_mcp_base）
# 经 to_thread 线程池访问单连接，其 db_lock 是线程安全必需，保留。
db = DatabaseCRUD()
db_lock = asyncio.Lock()


def parse_sql_result(raw_resp) -> list:
    """
    统一解析MCP SQL查询返回，兼容两种形态：
    1. 字符串（client._call_server 提取 resp.content[0].text 后的实际形态）
    2. [TextContent] 对象列表（部分版本MCP SDK原始返回）
    服务端返回的 text 内是单引号 Python 字面量（如 [{'amount': 100}]），
    必须用 ast.literal_eval 安全解析，禁止 eval。
    解析失败/无数据一律返回空列表，通信异常由上层抛出。
    """
    if isinstance(raw_resp, str):
        text_body = raw_resp.strip()
    elif isinstance(raw_resp, list) and raw_resp and hasattr(raw_resp[0], "text"):
        text_body = raw_resp[0].text.strip()
    else:
        return []

    # 以 [{ 或 [( 开头代表查询结果，否则是错误提示文本（权限不足/语法错误）
    if not (text_body.startswith("[{") or text_body.startswith("[(")):
        return []

    try:
        result_data = ast.literal_eval(text_body)
        return result_data if isinstance(result_data, list) else []
    except (ValueError, SyntaxError):
        return []