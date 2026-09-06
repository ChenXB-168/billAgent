# -*- coding: utf-8 -*-
"""
需求1/4 自动预警链 单元测试（纯逻辑，monkeypatch 隔离依赖，不碰真实 DB）。

覆盖：
- _calc_month_budget_status 四档判定（over / warn / reward / ok）+ 无预算不打扰 + 异常兜底
- _calc_total_budget_fact 总量结构化事实（over/warn/reward 有值；ok 返回 None；单笔不再与总预算对比）
- _calc_alert_facts 三层结构化事实（L1 均价溢价 / L2 品类进度 / L3 总量+归因 / 无基准标记）
"""
import pytest

from agents.orchestrator import nodes
from mcpGateway import fact_tools as ft


def _mk_status(level, month_spent, month_budget, used_ratio, days_left=17, consume_mode="正常"):
    """构造 _calc_month_budget_status 的返回结构"""
    return {
        "level": level,
        "month": "2026-08",
        "month_spent": month_spent,
        "month_budget": month_budget,
        "used_ratio": used_ratio,
        "days_left": days_left,
        "consume_mode": consume_mode,
    }


def _fake_cfg(month_budget=None, category_budget=None, quota_mode="auto", remind_pref=""):
    """async 桩：替代 get_user_target_config（await 普通函数会抛 TypeError）"""
    async def _f(_session):
        if month_budget is None:
            return {}
        return {
            "month_budget": month_budget,
            "consume_mode": "正常",
            "category_budget": category_budget or {},
            "quota_mode": quota_mode,
            "remind_pref": remind_pref,
        }
    return _f


def _fake_sum(spent: float):
    """async 桩：替代 _sum_month_bills"""
    async def _f(_month):
        return spent
    return _f


def _fake_avg_price(avg: float):
    """async 桩：替代 _query_city_avg_price（当地同品类均价）"""
    async def _f(_city, _category):
        return avg
    return _f


def _fake_cat_sum(cat_map: dict):
    """async 桩：替代 _sum_month_bills_by_category（当月各品类累计）"""
    async def _f(_month):
        return cat_map
    return _f


def _fake_quota(quota_info):
    """async 桩：替代 _calc_category_quota（品类合理额度），None=无基准"""
    async def _f(_city, _category):
        return quota_info
    return _f


# ============================================================
# 一、_calc_month_budget_status 档位判定
# ============================================================

@pytest.mark.asyncio
async def test_status_none_without_budget(monkeypatch):
    """无预算配置 → 返回 None（不打扰）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg())
    assert await nodes._calc_month_budget_status() is None


@pytest.mark.asyncio
async def test_status_over(monkeypatch):
    """used_ratio >= 1.0 → over 超支"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(1200.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "over"
    assert st["used_ratio"] == pytest.approx(1.2)
    assert st["month_spent"] == 1200.0
    assert st["month_budget"] == 1000.0


@pytest.mark.asyncio
async def test_status_warn(monkeypatch):
    """0.8 <= used_ratio < 1.0 → warn 接近红线"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(900.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "warn"
    assert st["used_ratio"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_status_reward(monkeypatch):
    """0 < used_ratio <= 0.5 → reward 预算充裕（犒劳）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(300.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "reward"
    assert st["used_ratio"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_status_ok_no_consume(monkeypatch):
    """当月无消费（used_ratio == 0）→ ok 不打扰（月首不犒劳）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(0.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "ok"


@pytest.mark.asyncio
async def test_status_ok_mid_range(monkeypatch):
    """0.5 < used_ratio < 0.8 → ok 正常区间不打扰"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(700.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "ok"
    assert st["used_ratio"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_status_exception_returns_none(monkeypatch):
    """读取配置抛异常 → 返回 None（静默兜底，不影响记账主流程）"""
    def boom(_):
        raise RuntimeError("db broken")
    monkeypatch.setattr(nodes, "get_user_target_config", boom)
    assert await nodes._calc_month_budget_status() is None


@pytest.mark.asyncio
async def test_status_days_left_positive(monkeypatch):
    """days_left 必须为当月剩余自然日（含今天）> 0"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(500.0))
    st = await nodes._calc_month_budget_status()
    assert st["days_left"] > 0


# ============================================================
# 二、_calc_total_budget_fact 总量结构化事实
# ============================================================

def test_total_fact_over():
    """over 档：产出结构化事实（超支），over_categories 初始为空列表"""
    fact = nodes._calc_total_budget_fact(_mk_status("over", 1200, 1000, 1.2))
    assert fact is not None
    assert fact["level"] == "over"
    assert fact["month_spent"] == 1200.0
    assert fact["month_budget"] == 1000.0
    assert fact["over_categories"] == []


def test_total_fact_warn_daily_advice():
    """warn 档：daily_limit = 剩余预算/剩余天数"""
    # 预算 1000，已花 900，剩 100，剩余 10 天 → 每日 10 元
    fact = nodes._calc_total_budget_fact(_mk_status("warn", 900, 1000, 0.9, days_left=10))
    assert fact["level"] == "warn"
    assert fact["remain"] == 100.0
    assert fact["daily_limit"] == 10.0


def test_total_fact_reward():
    """reward 档：预算充裕事实（犒劳由 LLM 汇总提示词生成）"""
    fact = nodes._calc_total_budget_fact(_mk_status("reward", 300, 1000, 0.3))
    assert fact["level"] == "reward"
    assert fact["remain"] == 700.0
    # 非 warn 档不产出每日建议
    assert fact["daily_limit"] is None


def test_total_fact_ok_none():
    """ok 档：不打扰，返回 None"""
    assert nodes._calc_total_budget_fact(_mk_status("ok", 700, 1000, 0.7)) is None


def test_total_fact_no_big_single_vs_total_budget():
    """需求1修正：单笔金额不再与总预算对比（无意义），ok 档大额单笔也不产出事实"""
    assert nodes._calc_total_budget_fact(_mk_status("ok", 700, 1000, 0.7)) is None


def test_total_fact_zero_amount_warn():
    """last_amount 不参与总量计算（异常兜底），warn 档事实不受影响"""
    fact = nodes._calc_total_budget_fact(_mk_status("warn", 900, 1000, 0.9))
    assert fact["level"] == "warn"


# ============================================================
# 三、_calc_alert_facts 三层结构化事实（需求1修正核心）
# ============================================================

@pytest.mark.asyncio
async def test_alert_facts_l1_premium_high(monkeypatch):
    """L1：单笔远超当地品类均价 → 结构化事实（level=过高，溢价率）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(38.0))
    monkeypatch.setattr(nodes, "_calc_category_quota", _fake_quota(None))
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category", _fake_cat_sum({}))
    facts = await nodes._calc_alert_facts(
        _mk_status("ok", 700, 1000, 0.7), 180.0, "餐饮", "上海"
    )
    l1 = facts["l1"]
    assert l1 is not None
    assert l1["category"] == "餐饮"
    assert l1["amount"] == 180.0
    assert l1["city"] == "上海"
    assert l1["avg_price"] == 38.0
    assert l1["premium_rate"] == pytest.approx(373.68, abs=0.1)
    assert l1["level"] == "过高"
    # 无额度基准且当月该品类未消费 → l2 为 None
    assert facts["l2"] is None
    # ok 档 → l3 为 None（不打扰）
    assert facts["l3"] is None


@pytest.mark.asyncio
async def test_alert_facts_l1_premium_normal(monkeypatch):
    """L1：单笔与均价持平 → 正常档"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(38.0))
    monkeypatch.setattr(nodes, "_calc_category_quota", _fake_quota(None))
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category", _fake_cat_sum({}))
    facts = await nodes._calc_alert_facts(
        _mk_status("ok", 700, 1000, 0.7), 36.0, "餐饮", "上海"
    )
    l1 = facts["l1"]
    assert l1["level"] == "正常"
    assert l1["premium_rate"] == pytest.approx(-5.26, abs=0.1)


@pytest.mark.asyncio
async def test_alert_facts_l2_category_progress(monkeypatch):
    """L2：品类进度 = 当月累计 vs 用户设定额度，ratio>=0.8 供 LLM 预警"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(0.0))
    monkeypatch.setattr(nodes, "_calc_category_quota", _fake_quota({"quota": 900.0, "source": "user"}))
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category", _fake_cat_sum({"餐饮": 812.0}))
    facts = await nodes._calc_alert_facts(
        _mk_status("ok", 700, 1000, 0.7), 50.0, "餐饮", "上海"
    )
    l2 = facts["l2"]
    assert l2 is not None
    assert l2["category"] == "餐饮"
    assert l2["spent"] == 812.0
    assert l2["quota"] == 900.0
    assert l2["ratio"] == pytest.approx(0.9022, abs=0.001)
    assert l2["source"] == "user"
    assert l2["need_quota"] is False


@pytest.mark.asyncio
async def test_alert_facts_l2_no_quota_marks_need_quota(monkeypatch):
    """L2：无任何额度基准且当月已消费 → need_quota=True（供 LLM 引导设置）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(0.0))
    monkeypatch.setattr(nodes, "_calc_category_quota", _fake_quota(None))
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category", _fake_cat_sum({"餐饮": 180.0}))
    facts = await nodes._calc_alert_facts(
        _mk_status("ok", 700, 1000, 0.7), 50.0, "餐饮", "上海"
    )
    l2 = facts["l2"]
    assert l2 is not None
    assert l2["need_quota"] is True
    assert l2["quota"] is None
    assert l2["source"] == "none"


@pytest.mark.asyncio
async def test_alert_facts_l3_over_with_attribution(monkeypatch):
    """L3：总量超支 + 点名超支主因品类（over_categories 排序）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(0.0))
    async def _quota(_city, cat):
        return {"quota": {"餐饮": 900.0, "购物": 400.0}.get(cat, 500.0), "source": "habit"}
    monkeypatch.setattr(nodes, "_calc_category_quota", _quota)
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category",
                        _fake_cat_sum({"餐饮": 812.0, "购物": 2000.0}))
    facts = await nodes._calc_alert_facts(
        _mk_status("over", 2812, 1000, 2.8), 50.0, "餐饮", "上海"
    )
    l3 = facts["l3"]
    assert l3 is not None
    assert l3["level"] == "over"
    assert l3["month_spent"] == 2812.0
    assert l3["month_budget"] == 1000.0
    # 超支归因：购物 2000/400 = +400% 排第一，餐饮 812/900 = -9.8% 不超支
    over_cats = l3["over_categories"]
    assert len(over_cats) == 1
    assert over_cats[0]["category"] == "购物"
    assert over_cats[0]["exceed_ratio"] == pytest.approx(4.0, abs=0.01)


@pytest.mark.asyncio
async def test_alert_facts_all_none_when_no_data(monkeypatch):
    """无均价、无额度、无总量状态 → 三层均为 None（LLM 收到空事实，不打扰）"""
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(0.0))
    monkeypatch.setattr(nodes, "_calc_category_quota", _fake_quota(None))
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category", _fake_cat_sum({}))
    facts = await nodes._calc_alert_facts(
        _mk_status("ok", 700, 1000, 0.7), 50.0, "", ""
    )
    assert facts["l1"] is None
    assert facts["l2"] is None
    assert facts["l3"] is None


# ============================================================
# 四、_query_city_avg_price 同级城市均价兜底
# ============================================================

class _FakePriceDB:
    """mock nodes.db 的物价查询接口：own=本城均价 / level=本城分级 / same=同级城市均价"""

    def __init__(self, own=0.0, level=None, same=0.0):
        self.own = own
        self.level = level
        self.same = same

    def get_city_price(self, city, category):
        return self.own

    def get_city_level(self, city):
        return self.level

    def get_same_level_avg_price(self, city_level, category):
        return self.same


# M9（D15）：城市均价兜底逻辑已**整体迁入 `city_price.get_avg` 工具**（`mcpGateway/fact_tools.py`），
# 单测随之指向工具侧（mock `fact_tools.db`）；orchestrator `_query_city_avg_price` 仅转发 + 异常兜底（下两用例）。
def test_city_avg_price_local_hit(monkeypatch):
    """本城有该品类均价 → 直接返回，不走同级兜底（工具语义）"""
    monkeypatch.setattr(ft, "db", _FakePriceDB(own=30.0, level="一线", same=99.9))
    assert ft.get_city_avg_price("广州", "餐饮") == 30.0


def test_city_avg_price_same_level_fallback(monkeypatch):
    """本城无该品类均价 → 同级城市均价兜底（工具语义）"""
    monkeypatch.setattr(ft, "db", _FakePriceDB(own=0.0, level="一线", same=36.5))
    assert ft.get_city_avg_price("广州", "餐饮") == 36.5


def test_city_avg_price_no_fallback(monkeypatch):
    """本城无数据且无分级/同级也无数据 → 返回 0（跳过溢价评估）"""
    monkeypatch.setattr(ft, "db", _FakePriceDB(own=0.0, level=None, same=0.0))
    assert ft.get_city_avg_price("广州", "餐饮") == 0.0


@pytest.mark.asyncio
async def test_query_city_avg_price_forwards_to_tool(monkeypatch):
    """M9：orchestrator 读城市均价 = 经 city_price.get_avg 工具转发（FACT_READ）"""
    async def _fake_invoke(name, args):
        assert name == "city_price.get_avg"
        assert args == {"city": "广州", "category": "餐饮"}
        return 36.5
    monkeypatch.setattr(nodes, "_invoke_fact", _fake_invoke)
    assert await nodes._query_city_avg_price("广州", "餐饮") == 36.5


@pytest.mark.asyncio
async def test_query_city_avg_price_tool_error_falls_back_zero(monkeypatch):
    """M9：工具调用失败 → orchestrator 兜底 0（跳过溢价评估，不阻断预警链）"""
    async def _boom(name, args):
        raise RuntimeError("tool down")
    monkeypatch.setattr(nodes, "_invoke_fact", _boom)
    assert await nodes._query_city_avg_price("广州", "餐饮") == 0.0
