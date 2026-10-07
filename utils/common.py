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
    """
    函数/类功能与逻辑描述：
        通用 SQLite 访问封装：单条连接 + WAL 日志模式 + busy_timeout，集中承载
        用户配置 / 账单 / 城市物价 / 消费分析 / 月度习惯五类表的读写方法；
        统一约定「写走 execute_sql（锁冲突重试 + 异常自动回滚）、读走 query_sql（返回字典列表）」。
        库路径可参数化：默认主业务库 `DB_PATH`，亦可指向 `task_state.db` 独立库（M5 D2）。
    构造入参说明：
        db_path (Path | str)：SQLite 库文件路径，默认主业务库 `config.DB_PATH`；
            传 `config.TASK_STATE_DB` 即得任务状态独立库实例（见 `04` §2.4 / `08` §6 Q4）。
    返回值说明：
        构造返回 DatabaseCRUD 实例；业务方法多返回 bool（写成功与否）或 list[dict]（查询结果）。
    """

    def __init__(self, db_path: Path | str = DB_PATH):
        """
        函数功能与逻辑描述：
            建立单条 SQLite 连接（check_same_thread=False，供跨线程复用）并开启
            `PRAGMA journal_mode=WAL` 与 `PRAGMA busy_timeout=5000`：前者让读不阻塞写、
            写不阻塞读（M6 并行派发的前提），后者由 SQLite 在 C 层吸收写锁竞争。
            ★M5（D2）：新增可选参数 db_path 以支持 `task_state.db` 独立库——Durable 状态写入频繁，
            与业务库同库会争抢写锁（`04` §2.4 / `08` §6 Q4）；
            默认参数保证既有 `db = DatabaseCRUD()` 单例与全部调用方零改动。
        入参说明：
            db_path (Path | str)：SQLite 库文件路径，默认主业务库 `DB_PATH`；
                传 `TASK_STATE_DB` 时新建/复用任务状态独立库连接。
        返回值说明：
            无（副作用：初始化 self.db_path / self.conn / self._closed，并写一条 info 启动日志）。
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        # M3.5（11 开工清单 / 04 §4.2-4.3）：WAL + busy_timeout。
        # db_lock 是【进程内】锁（04 §4.1 勘误），跨进程（主进程 vs MCP server 子进程）写互斥
        # 只能靠 SQLite 层兜底；WAL 同时是 M6 并行派发的前提（读不阻塞写、写不阻塞读）。
        # 注意：WAL 会生成 bill.db-wal / bill.db-shm（备份须覆盖，07 §5.1 已注明）。
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._closed = False
        logger.info("数据库连接初始化成功（journal_mode=WAL / busy_timeout=5000）")

    def begin_transaction(self):
        """
        函数功能与逻辑描述：
            显式开启事务（`BEGIN TRANSACTION`），供 `import_city_price` 等批量操作
            把「清空 + 批量插入」等步骤绑定为原子单元；连接已关闭时静默跳过，不抛异常。
        入参说明：
            无。
        返回值说明：
            无（副作用：在 self.conn 上发出 BEGIN TRANSACTION；连接已关闭时不产生任何动作）。
        """
        if not self._closed:
            self.conn.execute("BEGIN TRANSACTION;")

    def commit(self):
        """
        函数功能与逻辑描述：
            提交当前事务，把 begin_transaction 之后的写操作持久化；连接已关闭时静默跳过。
        入参说明：
            无。
        返回值说明：
            无（副作用：提交 self.conn 上的未完成事务；连接已关闭时不产生任何动作）。
        """
        if not self._closed:
            self.conn.commit()

    def rollback(self):
        """
        函数功能与逻辑描述：
            回滚当前事务，丢弃 begin_transaction 之后的未提交写操作；连接已关闭时静默跳过。
        入参说明：
            无。
        返回值说明：
            无（副作用：回滚 self.conn 上的未完成事务；连接已关闭时不产生任何动作）。
        """
        if not self._closed:
            self.conn.rollback()

    def execute_sql(self, sql, params=None, retry=3):
        """
        函数功能与逻辑描述：
            统一的写 SQL 入口：逐次尝试执行并提交，仅在捕获到 `database is locked` 且
            仍有剩余次数（i < retry-1）时重试；其余异常与重试耗尽一律回滚并返回 False，
            同时打 error 日志（含 SQL 原文）。M3.5 后不再做同步 sleep——锁竞争已由
            `busy_timeout=5000` 在 SQLite C 层吸收等待，sleep 只会冻结事件循环。
            边界：连接已关闭时直接返回 False；retry<=0 时循环体不执行，直接回滚并返回 False。
        入参说明：
            sql (str)：待执行的写 SQL（INSERT / UPDATE / DELETE 等），SQL 原文不做改写。
            params (tuple | list | None)：绑定参数；为 None 或空时不传参执行。
            retry (int)：锁冲突时的最大尝试次数（含首次），默认 3。
        返回值说明：
            bool：True 表示执行并提交成功；False 表示连接已关闭、锁重试耗尽或抛异常（均已回滚）。
        """
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
        """
        函数功能与逻辑描述：
            统一的查询 SQL 入口：执行后把游标结果统一转成「列名 → 值」的字典列表
            （避免下游把 tuple 当 dict 用），无结果或无可读列时返回空列表；
            连接已关闭或执行抛异常时同样返回空列表并打 error 日志，不向上抛。
        入参说明：
            sql (str)：待执行的查询 SQL（以 SELECT 为主），SQL 原文不做改写。
            params (tuple | list | None)：绑定参数；为 None 或空时不传参执行。
        返回值说明：
            list[dict]：每行一个字典，key 为列名、value 为该行取值；
                查询无结果 / 连接已关闭 / 执行异常时返回空列表 `[]`。
        """
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
        """
        函数功能与逻辑描述：
            安全关闭数据库连接：以 self._closed 作幂等标志，重复调用不会二次 close；
            关闭过程自身抛异常时仅记 error 日志、不向外传播（且不置位 _closed）。
        入参说明：
            无。
        返回值说明：
            无（副作用：关闭 self.conn 并置 self._closed=True，写一条 info 日志）。
        """
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
        """
        函数功能与逻辑描述：
            向 user_config 表追加一行用户配置（纯 INSERT，不判重）；写失败由 execute_sql 回滚并返回 False。
            category_budget 以 JSON 字符串整列落库，供额度评估侧自行反序列化。
        入参说明：
            city (str)：用户所在城市名（如 "深圳"），用于物价对标时的基准城市匹配。
            month_budget (float)：月度总预算，单位元。
            consume_mode (str)：消费模式标识。
            budget_type (str)：预算周期类型，默认 "month"（按月）。
            category_budget (str)：品类预算 JSON 字符串（如 '{"餐饮": 900, "交通": 300}'），可为 None。
            quota_mode (str)：品类额度评估方式，取值 auto/user/habit/city，默认 "auto" 自动选择。
            remind_pref (str)：用户个性化提醒偏好（语气/语言/格式，随汇总提示词注入），可为 None。
        返回值说明：
            bool：True 表示插入并提交成功；False 表示连接已关闭或执行异常（已回滚）。
        """
        sql = ("INSERT INTO user_config (city, month_budget, consume_mode, budget_type, category_budget, quota_mode, remind_pref) "
               "VALUES (?, ?, ?, ?, ?, ?, ?)")
        return self.execute_sql(sql, (city, month_budget, consume_mode, budget_type, category_budget, quota_mode, remind_pref))

    def get_latest_user_config(self):
        """
        函数功能与逻辑描述：
            读取 user_config 表按 create_time 倒序的第一行，作为「当前生效配置」（读侧入口，
            与 upsert_user_config 的单行覆盖写语义配对）。
        入参说明：
            无。
        返回值说明：
            list[dict]：长度 0 或 1 的结果列表；有数据时元素为该行的列名字典型；表为空时返回空列表。
        """
        sql = "SELECT * FROM user_config ORDER BY create_time DESC LIMIT 1"
        return self.query_sql(sql)

    def upsert_user_config(self, city: str, month_budget: float, consume_mode: str, budget_type: str = "month",
                           category_budget: str = None, quota_mode: str = "auto", remind_pref: str = None):
        """
        函数功能与逻辑描述：
            WebUI 表单写入口，实现「单行覆盖」语义（读侧 get_latest_user_config 取"最新一行"= 当前配置）。
            幂等策略：先用 `create_time DESC, id DESC` 双键排序取当前行 id——存在则原地 UPDATE
            七个配置列并刷新 create_time（保证永远是最新行）；表为空则委托 add_user_config 插入首行。
            运行期只应存在一行"当前配置"，WebUI 多次保存不会累积历史行
            （双键排序同时消除 create_time 秒级相同导致的行序歧义）。
        入参说明：
            city (str)：用户所在城市名。
            month_budget (float)：月度总预算，单位元。
            consume_mode (str)：消费模式标识。
            budget_type (str)：预算周期类型，默认 "month"。
            category_budget (str)：品类预算 JSON 字符串，可为 None。
            quota_mode (str)：品类额度评估方式，取值 auto/user/habit/city，默认 "auto"。
            remind_pref (str)：提醒偏好（语气/语言/格式），可为 None；形参语义与 add_user_config 完全一致。
        返回值说明：
            bool：True 表示 UPDATE 或 INSERT 并提交成功；False 表示连接已关闭或执行异常（已回滚）。
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
        """
        函数功能与逻辑描述：
            向 bill 表插入一条账单记录（记一笔账的落库入口），纯 INSERT、不做去重与校验，
            失败由 execute_sql 回滚并返回 False。
        入参说明：
            amount (float)：消费金额，单位元。
            category (str)：消费品类（如 餐饮/交通/住宿/购物/娱乐）。
            consume_time (str)：消费时间字符串（入库原样保存，查询侧按 substr 取年月）。
            remark (str)：备注，默认空串。
        返回值说明：
            bool：True 表示插入并提交成功；False 表示连接已关闭或执行异常（已回滚）。
        """
        sql = "INSERT INTO bill (amount, category, consume_time, remark) VALUES (?, ?, ?, ?)"
        return self.execute_sql(sql, (amount, category, consume_time, remark))

    def get_all_bill(self):
        """
        函数功能与逻辑描述：
            读取 bill 表全部账单，按 consume_time 倒序返回（最近的在前），供账单列表展示。
        入参说明：
            无。
        返回值说明：
            list[dict]：每行一个字典（含 id / amount / category / consume_time / remark）；
                表为空或查询异常时返回空列表。
        """
        return self.query_sql("SELECT * FROM bill ORDER BY consume_time DESC")

    def get_bill_by_time(self, start_time: str, end_time):
        """
        函数功能与逻辑描述：
            按消费时间区间（左闭右闭 BETWEEN）筛选账单，并按 consume_time 升序返回，
            供月度/区间统计聚合使用。
        入参说明：
            start_time (str)：区间起始时间（含），与 consume_time 同格式以便字符串比较。
            end_time (str)：区间结束时间（含）。
        返回值说明：
            list[dict]：命中区间的账单字典列表（按 consume_time 升序）；无命中或异常时为空列表。
        """
        sql = "SELECT * FROM bill WHERE consume_time BETWEEN ? AND ? ORDER BY consume_time"
        return self.query_sql(sql, (start_time, end_time))

    def delete_bill(self, bill_id: int):
        """
        函数功能与逻辑描述：
            按主键 id 删除单条账单；不校验存在性，删除 0 行也视为执行成功。
        入参说明：
            bill_id (int)：bill 表主键 id。
        返回值说明：
            bool：True 表示 DELETE 执行并提交成功（含未命中任何行）；False 表示连接已关闭或执行异常。
        """
        sql = "DELETE FROM bill WHERE id = ?"
        return self.execute_sql(sql, (bill_id,))

    # ---------- 城市物价表 ----------
    def add_city_price(self, city: str, category: str, avg_price: float, city_level: str = None):
        """
        函数功能与逻辑描述：
            向 city_price 表追加一条城市物价基准记录（纯 INSERT，不做去重）；写失败由
            execute_sql 回滚并返回 False。
        入参说明：
            city (str)：城市名（如 "深圳"）。
            category (str)：消费品类，需与记账品类口径一致。
            avg_price (float)：该城市该品类的基准均价，单位元。
            city_level (str)：城市分级（一线/新一线/二线/三线），用于本城无均价时的同级城市均价兜底；可为 None。
        返回值说明：
            bool：True 表示插入并提交成功；False 表示连接已关闭或执行异常（已回滚）。
        """
        sql = "INSERT INTO city_price (city, category, avg_price, city_level) VALUES (?, ?, ?, ?)"
        return self.execute_sql(sql, (city, category, avg_price, city_level))

    def batch_add_city_price(self, data_list: list, retry=3):
        """
        函数功能与逻辑描述：
            用 `executemany` 批量导入城市物价数据（导入脚本 `import_city_price` 的落库入口）：
            入库前先把每条记录规整为统一的 (city, category, avg_price, city_level) 四元组，
            再一次性批插后提交。锁冲突（`database is locked`）且仍有剩余次数时重试
            （busy_timeout 已在 C 层吸收竞争，不再同步 sleep），重试耗尽或其它异常一律回滚。
            边界：连接已关闭时直接返回 False；retry<=0 时循环体不执行，直接回滚并返回 False。
        入参说明：
            data_list (list)：元素为按 (city, category, avg_price[, city_level]) 顺序排列的序列（元组/列表）；
                len >= 4 时取前 4 个（多余元素被忽略），len == 3 时自动补 city_level=None；
                元素少于 3 个会在 executemany 绑定参数时失败并触发整体回滚。
            retry (int)：锁冲突时的最大尝试次数（含首次），默认 3。
        返回值说明：
            bool：True 表示批量插入并提交成功；False 表示连接已关闭、锁重试耗尽或抛异常（均已回滚）。
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
        """
        函数功能与逻辑描述：
            查询指定城市 + 品类的基准均价并转 float 返回，只取首行（精确匹配，不含同级兜底）；
            无记录、avg_price 为 NULL 或查询异常时统一返回 0.0，调用方据此判定「无有效基准」。
        入参说明：
            city (str)：城市名，需与 city_price 表存值精确匹配。
            category (str)：消费品类。
        返回值说明：
            float：命中记录的 avg_price；无数据/为 NULL/异常时返回 0.0。
        """
        res = self.query_sql(
            "SELECT avg_price FROM city_price WHERE city = ? AND category = ? LIMIT 1",
            (city, category)
        )
        return float(res[0]["avg_price"]) if res and res[0] and res[0]["avg_price"] is not None else 0.0

    def get_city_level(self, city: str):
        """
        函数功能与逻辑描述：
            查询城市分级（一线/新一线/二线/三线），用于本城无均价时转走同级城市均价兜底；
            只取 city_level 非 NULL 的首行。
        入参说明：
            city (str)：城市名。
        返回值说明：
            str | None：命中记录的 city_level；无数据、值为 NULL 或异常时返回 None。
        """
        res = self.query_sql(
            "SELECT city_level FROM city_price WHERE city = ? AND city_level IS NOT NULL LIMIT 1",
            (city,)
        )
        return res[0]["city_level"] if res and res[0] and res[0]["city_level"] else None

    def get_same_level_avg_price(self, city_level: str, category: str):
        """
        函数功能与逻辑描述：
            同级城市均价兜底：对同 city_level 下所有城市该品类 `avg_price > 0` 的记录取 AVG，
            作为本城无基准时的替代口径；city_level 为空时直接短路，不查库。
        入参说明：
            city_level (str)：城市分级（一线/新一线/二线/三线）；空值/None 时直接返回 0.0。
            category (str)：消费品类。
        返回值说明：
            float：同级城市该品类均价；无有效样本（或 AVG 为 NULL）时返回 0.0。
        """
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
        """
        函数功能与逻辑描述：
            向 analysis 表写入一条月度分析结果（纯 INSERT，不做去重）。
            ⚠️ 口径说明：analysis 表当前为零调用死表（`04` §2.2 / `10` §5.6.8），
            本方法与 get_analysis_by_month 一并保留为契约接口，暂无调用方。
        入参说明：
            month (str)：月份标识（如 "2026-08"）。
            total (float)：该月消费总额，单位元。
            over_budget (int)：超预算标志/计数（表定义为整数列）。
            content (str)：分析结论正文（通常为面向用户的中文文案）。
        返回值说明：
            bool：True 表示插入并提交成功；False 表示连接已关闭或执行异常（已回滚）。
        """
        sql = "INSERT INTO analysis (month, total, over_budget, content) VALUES (?, ?, ?, ?)"
        return self.execute_sql(sql, (month, total, over_budget, content))

    def get_analysis_by_month(self, month: str):
        """
        函数功能与逻辑描述：
            按月份精确读取 analysis 表中的分析记录（历史遗留月报读取接口）。
            ⚠️ 口径说明：analysis 表当前为零调用死表（`04` §2.2），本方法暂无调用方。
        入参说明：
            month (str)：月份标识（如 "2026-08"），需与入库值精确匹配。
        返回值说明：
            list[dict]：命中月份的分析记录字典列表；无记录或异常时返回空列表。
        """
        return self.query_sql("SELECT * FROM analysis WHERE month = ?", (month,))

    # ---------- 月度消费习惯表（长期记忆事实层：月+品类 聚合频次与金额） ----------
    def upsert_monthly_habit(self, month: str, category: str, amount: float):
        """
        函数功能与逻辑描述：
            记账成功后以 `INSERT ... ON CONFLICT(month, category) DO UPDATE` 增量累计当月该品类消费习惯：
            首行插入时 amount_sum=avg_amount=amount、count=1；(month, category) 已存在时金额累加、
            count+1 并按「(旧 amount_sum + 新增) / (旧 count + 1)」重算 avg_amount，
            update_time 统一写入当前本地时间（INSERT 与 UPDATE 分支都走 excluded.update_time）。
            注意：本方法本身不加锁，主进程由调用方（`fact_tools` 的 habit.upsert）在 db_lock 内调用。
        入参说明：
            month (str)：月份，格式 "YYYY-MM"，与 monthly_habit.month 存值口径一致。
            category (str)：消费品类。
            amount (float)：本次消费金额，单位元；作为增量累加到 amount_sum。
        返回值说明：
            bool：True 表示 upsert 并提交成功；False 表示连接已关闭或执行异常（已回滚）。
        """
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
        """
        函数功能与逻辑描述：
            读取月度消费习惯聚合数据（monthly_habit 表，按月+品类累计金额与笔数）。
            传入 months 时用 IN 占位符按月份列表过滤，未传时返回全表。
            排序固定为 month DESC, category（月份倒序便于取最近月份、同月品类稳定有序）。
            months 为**空列表**时走 `if months` 的假分支，等价于返回全部——调用方若想
            「指定月份但无匹配」得到空结果，应避免传空列表。
        入参说明：
            months (list)：月份列表，元素格式 "YYYY-MM"（如 ['2026-08','2026-07']）；
                默认 None 表示不加月份条件、返回全部记录。（注：SQL 用 f-string 拼占位符，
                占位符个数由列表长度决定，月份值本身始终走参数绑定，不拼接进 SQL。）
        返回值说明：
            list：字典列表，每行为 monthly_habit 的完整列（month / category / amount_sum /
                count / avg_amount / update_time）；无数据时返回 []。
        """
        if months:
            placeholders = ",".join("?" * len(months))
            sql = f"SELECT * FROM monthly_habit WHERE month IN ({placeholders}) ORDER BY month DESC, category"
            return self.query_sql(sql, months)
        return self.query_sql("SELECT * FROM monthly_habit ORDER BY month DESC, category")

    def recalc_month_habit(self, month: str):
        """
        函数功能与逻辑描述：
            以 bill 表为唯一事实源全量重算某月消费习惯，用于账单删除/编辑后修正
            monthly_habit 增量累计可能产生的偏差（增量累计 upsert_monthly_habit 无法处理删除）。
            实现为「先删该月全部习惯行，再按 category 分组重算并逐条插入」，
            因此重算期间该月数据会短暂为空；avg_amount 保留 2 位小数，cnt 为 0 时按 0.0 兜底。
            先删后插不是单事务包装，若插入中途失败会留下不完整数据（调用方可再次重算修复）。
        入参说明：
            month (str)：自然月，格式 "YYYY-MM"；匹配 bill 表时用 substr(consume_time, 1, 7)。
        返回值说明：
            bool：True 表示重算并写入成功；False 表示该月 bill 表中没有任何账单
                （此时该月习惯行已被删除，且不会写入新行）。
        """
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
        """
        函数功能与逻辑描述：
            月度习惯表的滚动清理：删除早于「最近 keep_months 个自然月（含当月）」的月份数据，
            防止该表随使用时长无限增长。边界月份用逐月回退计算而非日期加减，天然处理跨年；
            月份以 YYYY-MM 字符串存储，因此 `month < boundary` 的字典序比较等价于时间先后比较。
            与 memory.long_memory._keep_recent_months_boundary 采用同一套边界算法，
            两边默认值均为 12，修改时需同步，避免长期记忆文本与习惯表保留窗口不一致。
        入参说明：
            keep_months (int)：保留的月份数量（含当月），默认 12；传 1 表示只保留当月。
        返回值说明：
            bool：透传 execute_sql 的执行结果——True 表示删除语句执行成功（不等于有行被删），
                False 表示连接已关闭或执行失败。
        """
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
    函数功能与逻辑描述：
        统一解析 MCP SQL 查询的返回体，兼容两种入参形态：
        ① 字符串（client._call_server 提取 resp.content[0].text 后的实际形态）；
        ② [TextContent] 对象列表（部分 MCP SDK 版本的原始返回）。
        服务端返回的 text 内是单引号 Python 字面量（如 [{'amount': 100}]），
        因此必须用 ast.literal_eval 安全解析，**禁止 eval**（防止服务端内容被利用执行代码）。
        解引用顺序为：先按类型取出 text_body（其它类型直接返回 []），
        再以 `[{` 或 `[(` 前缀做结果/错误文本分流——服务端在权限不足、SQL 被拦截、
        语法错误等场景返回的是「操作拒绝：...」「权限不足：...」这类纯文本，不以前缀开头，
        一律返回 []。解析结果若非 list 同样返回 []。
        注意一个固有局限：空结果集服务端返回的是 "[]"，不满足前缀条件，
        因此「无数据」与「被拒绝」在本函数出口无法区分，均为 []；
        需要区分时调用方应绕过本函数、直接检查原始文本。
        本函数只做解析，不捕获通信异常（由上层负责）。
    入参说明：
        raw_resp：MCP 返回体，可为 str、[TextContent] 列表，或其它类型（其它类型返回 []）。
    返回值说明：
        list：解析成功时返回元素为 dict 的列表；以下情况均返回 []——
            入参类型不符、文本不以 `[{` / `[(` 开头、字面量解析抛 ValueError/SyntaxError、
            或解析出的对象不是 list。
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