# ==============================================
# 物价对标规则模块 - 纯计算 + 纯规则，无数据库依赖
# 说明：溢价阈值在此统一定义，price_agent（物价对标）与 orchestrator（记账预警链）
#       必须复用本模块，避免两处各自维护阈值造成判定口径漂移。
# ==============================================
from utils.common import logger

# 溢价分段阈值（与 price_agent / orchestrator 预警链统一口径，单位为百分比）
PREMIUM_THRESHOLD_LOW = -15
PREMIUM_THRESHOLD_NORMAL = 15
PREMIUM_THRESHOLD_HIGH = 40


def calc_premium_evaluate(real_amt: float, base_amt: float) -> dict:
    """
    函数功能与逻辑描述：
        根据「用户本次实际消费金额」与「城市同品类基准均价」计算溢价率，并映射到
        偏低 / 正常 / 偏高 / 过高 四档，同时生成面向用户的中文评价文案。
        纯计算、无副作用、不访问数据库，便于单测；供 price_agent（物价对标）与
        orchestrator（记账预警链）统一复用，保证阈值口径一致。
        边界处理：当基准均价缺失（None）或非正数时不做溢价计算，直接返回「正常 + 跳过评估」的兜底结果。
    入参说明：
        real_amt (float)：用户本次实际消费金额，单位元。
        base_amt (float)：城市同品类基准均价，单位元；None 或 <=0 视为无有效基准。
    返回值说明：
        dict：三字段结构
            - premium_rate (float)：溢价率百分数，保留 2 位小数；无有效基准时为 0.0。
            - level (str)：溢价档次，取值 偏低 / 正常 / 偏高 / 过高。
            - conclusion (str)：面向用户的中文评价文案。
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
    """
    函数/类功能与逻辑描述：
        物价对标计算器，提供面向对象风格的溢价计算入口；与模块级
        `calc_premium_evaluate` 的区别在于本类返回统一的 {success, msg, data} 响应包，
        便于工具层（MCP 工具返回值契约）直接透传。
    构造入参说明：
        无（无状态类，直接实例化即可）。
    返回值说明：
        构造返回 PriceCompare 实例；业务方法见 `calc_premium`。
    """

    def __init__(self):
        """
        函数功能与逻辑描述：
            空构造，本类为无状态纯计算类，不持有连接、缓存等任何实例资源。
        入参说明：
            无。
        返回值说明：
            无（仅初始化实例，无副作用）。
        """
        pass

    def calc_premium(self, avg_price: float, pay_amount: float, threshold: float) -> dict:
        """
        函数功能与逻辑描述：
            计算单笔消费相对基准均价的溢价率（以小数表示，非百分数），并与传入阈值比较得出
            是否超标；纯内存计算，不做任何持久化。当基准均价非正数时判定为「基准无效」，
            短路返回失败包，不参与后续比较。
        入参说明：
            avg_price (float)：本地基准均价，必须 > 0，否则视为无效基准。
            pay_amount (float)：实际消费金额，单位与 avg_price 保持一致。
            threshold (float)：溢价阈值，与 premium_rate 同量纲（小数）比较。
        返回值说明：
            dict：统一响应包 {success, msg, data}
                - 成功：success=True，msg="计算完成"，data 含 avg_price(2 位小数)、
                  pay_amount(2 位小数)、premium_rate(4 位小数)、threshold、is_over_limit(bool)。
                - 失败：success=False，msg="基准均价无效"，data={}。
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


# 模块单例：全项目共用同一实例（无状态，可安全共享）
price_compare = PriceCompare()
__all__ = ["price_compare", "calc_premium_evaluate", "PREMIUM_THRESHOLD_LOW",
           "PREMIUM_THRESHOLD_NORMAL", "PREMIUM_THRESHOLD_HIGH"]
