# 物价对标规则模块：纯计算、纯规则，无数据库依赖
from utils.common import logger

# 溢价分段阈值（与 price_agent / orchestrator 预警链统一口径）
PREMIUM_THRESHOLD_LOW = -15
PREMIUM_THRESHOLD_NORMAL = 15
PREMIUM_THRESHOLD_HIGH = 40


def calc_premium_evaluate(real_amt: float, base_amt: float) -> dict:
    """
    根据单笔金额与基准均价计算溢价率、档次、评价文案（纯计算，可单测）。
    供 price_agent（物价对标）与 orchestrator（记账预警链）统一复用，保证阈值口径一致。
    Args:
        real_amt: 用户本次消费金额
        base_amt: 城市同品类基准均价
    Returns:
        结构化评价数据：premium_rate、level、conclusion
    """
    if base_amt is None or base_amt <= 0:
        return {"premium_rate": 0.0, "level": "正常", "conclusion": "暂无该城市该品类均价基准，跳过溢价评估"}
    premium_rate = ((real_amt - base_amt) / base_amt) * 100
    premium_rate = round(premium_rate, 2)

    if premium_rate < PREMIUM_THRESHOLD_LOW:
        level = "偏低"
        conclusion = "远低于城市基准均价，性价比极高"
    elif premium_rate <= PREMIUM_THRESHOLD_NORMAL:
        level = "正常"
        conclusion = "与城市基准均价持平，属于常规合理开销"
    elif premium_rate <= PREMIUM_THRESHOLD_HIGH:
        level = "偏高"
        conclusion = "高于城市基准均价，存在小幅消费溢价"
    else:
        level = "过高"
        conclusion = "远高于城市基准均价，本次消费偏贵"

    return {
        "premium_rate": premium_rate,
        "level": level,
        "conclusion": conclusion
    }


class PriceCompare:
    def __init__(self):
        pass

    def calc_premium(self, avg_price: float, pay_amount: float, threshold: float) -> dict:
        """
        纯内存计算溢价率、判断是否超标
        :param avg_price: 本地基准均价
        :param pay_amount: 实际消费金额
        :param threshold: 溢价阈值
        """
        if avg_price <= 0:
            return {"success": False, "msg": "基准均价无效", "data": {}}

        premium_rate = (pay_amount - avg_price) / avg_price if avg_price != 0 else 0.0
        is_over_limit = premium_rate > threshold

        return {
            "success": True,
            "msg": "计算完成",
            "data": {
                "avg_price": round(avg_price, 2),
                "pay_amount": round(pay_amount, 2),
                "premium_rate": round(premium_rate, 4),
                "threshold": threshold,
                "is_over_limit": is_over_limit
            }
        }


# 模块单例
price_compare = PriceCompare()
__all__ = ["price_compare", "calc_premium_evaluate", "PREMIUM_THRESHOLD_LOW",
           "PREMIUM_THRESHOLD_NORMAL", "PREMIUM_THRESHOLD_HIGH"]