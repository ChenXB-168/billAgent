# ==============================================
# 城市物价基准数据集 批量导入脚本
# 执行后自动写入 city_price 表
# ==============================================
import sys
from pathlib import Path

# 修复路径，让脚本能识别项目根目录
sys.path.append(str(Path(__file__).parent.parent))
from utils.common import db, logger

# ======================
# 1. 模拟全国城市物价基准数据（测试用，可直接运行）
# 格式：(城市, 消费品类, 平均单价，单位元)；共 7 城 × 5 品类 = 35 条
# 品类与记账分类一一对应（bill.category 五分类：餐饮 / 交通 / 住宿 / 购物 / 娱乐）
# ======================
PRICE_DATA = [
    # 一线城市
    ("深圳", "餐饮", 36.0),
    ("深圳", "交通", 4.5),
    ("深圳", "住宿", 320.0),
    ("深圳", "购物", 140.0),
    ("深圳", "娱乐", 90.0),

    ("北京", "餐饮", 35.0),
    ("北京", "交通", 4.0),
    ("北京", "住宿", 280.0),
    ("北京", "购物", 120.0),
    ("北京", "娱乐", 80.0),

    ("上海", "餐饮", 38.0),
    ("上海", "交通", 4.5),
    ("上海", "住宿", 300.0),
    ("上海", "购物", 130.0),
    ("上海", "娱乐", 85.0),

    # 新一线/二线城市
    ("杭州", "餐饮", 28.0),
    ("杭州", "交通", 3.0),
    ("杭州", "住宿", 220.0),
    ("杭州", "购物", 90.0),
    ("杭州", "娱乐", 65.0),

    ("成都", "餐饮", 22.0),
    ("成都", "交通", 2.5),
    ("成都", "住宿", 180.0),
    ("成都", "购物", 70.0),
    ("成都", "娱乐", 50.0),

    ("武汉", "餐饮", 24.0),
    ("武汉", "交通", 2.8),
    ("武汉", "住宿", 190.0),
    ("武汉", "购物", 75.0),
    ("武汉", "娱乐", 55.0),

    # 三线及以下城市
    ("洛阳", "餐饮", 18.0),
    ("洛阳", "交通", 1.5),
    ("洛阳", "住宿", 120.0),
    ("洛阳", "购物", 50.0),
    ("洛阳", "娱乐", 35.0)
]

# 城市分级（用于"本城无均价 → 同级城市均价兜底"）
CITY_LEVEL_MAP = {
    "深圳": "一线",
    "北京": "一线",
    "上海": "一线",
    "杭州": "新一线",
    "成都": "新一线",
    "武汉": "新一线",
    "洛阳": "三线",
}


def _with_level(data_list: list) -> list:
    """
    函数功能与逻辑描述：
        把 `PRICE_DATA` 的 (城市, 品类, 均价) 三元组逐条扩成
        (city, category, avg_price, city_level) 四元组，供 `batch_add_city_price` 直接批插
        （该方法要求元素按 city/category/avg_price/city_level 顺序排列）。纯内存转换，
        不回写原列表。
        边界：city 不在 `CITY_LEVEL_MAP` 中时兜底为 "其他"，保证四元组长度恒定、不因缺分级报错。
    入参说明：
        data_list (list)：元素为 (city, category, avg_price) 三元组的序列，通常即 PRICE_DATA。
    返回值说明：
        list：与入参等长的新列表，元素为 (city, category, avg_price, city_level) 四元组，
            顺序与入参一致；data_list 为空时返回空列表。
    """
    out = []
    for city, category, avg_price in data_list:
        out.append((city, category, avg_price, CITY_LEVEL_MAP.get(city, "其他")))
    return out


# ======================
# 2. 执行批量导入（事务保护，失败自动回滚不丢旧数据）
# ======================
def main():
    """
    函数功能与逻辑描述：
        城市物价基准数据集的一次性导入入口，按「begin_transaction → 清空旧数据 → 批量插入
        （附 city_level）→ 校验导入条数 → commit」顺序执行；DELETE / INSERT 返回 False，或
        实际条数与 `len(PRICE_DATA)` 不符时抛 RuntimeError，经 except 分支 rollback 保证旧数据
        不被破坏，finally 无条件 `db.close()`。
        边界：条数校验以 PRICE_DATA 长度为期望值，且 city_price 带 UNIQUE(city, category)，
        故必须先清空再插入；本函数为脚本式入口，调用后全局 db 连接会被关闭。
    入参说明：
        无。
    返回值说明：
        无（仅副作用：读写 city_price 表、提交或回滚事务、关闭 db 连接，并写 logger.info / logger.error）。
    """
    logger.info("开始导入城市物价基准数据集...")
    try:
        # 开启事务：把「清空旧数据 + 批量插入 + 条数校验」绑定为原子单元，
        # 任一步失败整体回滚，旧数据不受影响
        db.begin_transaction()

        # 先清空历史数据
        del_ok = db.execute_sql("DELETE FROM city_price")
        if not del_ok:
            raise RuntimeError("清空旧数据失败")

        # 批量插入新基准（附城市分级，供同级城市均价兜底）
        success = db.batch_add_city_price(_with_level(PRICE_DATA))
        if not success:
            raise RuntimeError("批量插入失败")

        # 校验导入条数
        count_res = db.query_sql("SELECT COUNT(*) FROM city_price")
        count = count_res[0]["COUNT(*)"] if count_res else 0
        if count != len(PRICE_DATA):
            raise RuntimeError(f"导入条数不符，预期{len(PRICE_DATA)}条，实际{count}条")

        db.commit()
        logger.info(f"城市物价基准数据集导入完成！共 {count} 条数据")

    except Exception as e:
        db.rollback()
        logger.error(f"导入失败，已回滚所有变更：{str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main()