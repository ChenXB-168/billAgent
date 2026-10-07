# -*- coding: utf-8 -*-
"""
需求1/4 自动预警链 单元测试（纯逻辑，monkeypatch 隔离依赖，不碰真实 DB）。

覆盖：
- _calc_month_budget_status 四档判定（over / warn / reward / ok）+ 无预算不打扰 + 异常兜底
- _calc_total_budget_fact 总量结构化事实（over/warn/reward 有值；ok 返回 None；单笔不再与总预算对比）
- _calc_alert_facts 三层结构化事实（L1 均价溢价 / L2 品类进度 / L3 总量+归因 / 无基准标记）
- city_price.get_avg / _query_city_avg_price：本城命中 → 同级城市兜底 → 0 的三级取价，
  以及 orchestrator 侧「工具转发 + 异常兜底 0」的收口
- _calc_category_quota / _load_recent_habits：品类额度「仅已完成月份月均」口径与
  include_current 当月开关（修复月初误报超支）
"""
from datetime import date

import pytest

from agents.orchestrator import nodes
from mcpGateway import fact_tools as ft


def _mk_status(level, month_spent, month_budget, used_ratio, days_left=17, consume_mode="正常"):
    """
    函数功能与逻辑描述：
        构造 `_calc_month_budget_status` 的返回结构 dict，供总量事实/预警事实用例直接注入；
        字段固定为 level/month/month_spent/month_budget/used_ratio/days_left/consume_mode。
    入参说明：
        level：预算档位（over/warn/reward/ok）。
        month_spent：当月已消费金额。
        month_budget：当月预算总额。
        used_ratio：已用比例（0 起，可超 1）。
        days_left：当月剩余自然日，默认 17。
        consume_mode：消费模式文案，默认 "正常"。
    返回值说明：
        dict：`_calc_month_budget_status` 形态的预算状态字典。
    """
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
    """
    函数功能与逻辑描述：
        构造替代 `get_user_target_config` 的 async 桩（await 普通函数会抛 TypeError）：
        month_budget 为 None 时返回空配置（模拟"未设预算"），否则返回完整目标配置。
    入参说明：
        month_budget：月度预算；None 表示未配置。
        category_budget：品类预算字典，默认空。
        quota_mode：额度模式，默认 "auto"。
        remind_pref：提醒偏好，默认空串。
    返回值说明：
        返回 async 桩函数 `_f(session)`，调用后返回用户目标配置 dict。
    """
    async def _f(_session):
        """
        函数功能与逻辑描述：
            替代 get_user_target_config 的 async 桩：month_budget 为 None 时返回空配置
            （模拟"未设预算"），否则返回含月预算/消费模式/品类预算/额度模式/提醒偏好的完整配置。
        入参说明：
            _session：会话标识（本桩忽略，仅为对齐真实 get_user_target_config 的调用签名）。
        返回值说明：
            dict：month_budget 为 None 时返回 {}，否则返回完整用户目标配置字典。
        """
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
    """
    函数功能与逻辑描述：
        构造替代 `_sum_month_bills` 的 async 桩，忽略月份入参直接返回固定月消费额。
    入参说明：
        spent (float)：桩要返回的月度消费额。
    返回值说明：
        返回 async 桩函数 `_f(month)`，调用后返回 spent。
    """
    async def _f(_month):
        """
        函数功能与逻辑描述：
            替代 _sum_month_bills 的 async 桩：忽略月份入参，恒定返回构造时给定的月度消费额，
            用于稳定驱动预算档位判定（over / warn / reward / ok）。
        入参说明：
            _month：自然月字符串（本桩忽略，仅为对齐真实 _sum_month_bills 的调用签名）。
        返回值说明：
            float：构造 _fake_sum 时传入的月消费额 spent。
        """
        return spent
    return _f


def _fake_avg_price(avg: float):
    """
    函数功能与逻辑描述：
        构造替代 `_query_city_avg_price`（当地同品类均价）的 async 桩。
    入参说明：
        avg (float)：桩要返回的平均价（0.0 表示无基准）。
    返回值说明：
        返回 async 桩函数 `_f(city, category)`，调用后返回 avg。
    """
    async def _f(_city, _category):
        """
        函数功能与逻辑描述：
            替代 _query_city_avg_price（当地同品类均价）的 async 桩：忽略城市与品类入参，
            恒定返回构造时给定的均价，用于稳定驱动 L1「单笔 vs 均价」溢价分支。
        入参说明：
            _city：城市名（本桩忽略，仅为对齐真实 _query_city_avg_price 的调用签名）。
            _category：消费品类（本桩忽略）。
        返回值说明：
            float：构造 _fake_avg_price 时传入的均价 avg（0.0 表示无基准）。
        """
        return avg
    return _f


def _fake_cat_sum(cat_map: dict):
    """
    函数功能与逻辑描述：
        构造替代 `_sum_month_bills_by_category`（当月各品类累计）的 async 桩。
    入参说明：
        cat_map (dict)：桩要返回的品类累计映射 {category: amount}。
    返回值说明：
        返回 async 桩函数 `_f(month)`，调用后返回 cat_map。
    """
    async def _f(_month):
        """
        函数功能与逻辑描述：
            替代 _sum_month_bills_by_category 的 async 桩：忽略月份入参，恒定返回构造时给定的
            品类累计映射，用于驱动 L2 品类进度与 L3 超支归因分支。
        入参说明：
            _month：自然月字符串（本桩忽略，仅为对齐真实取数的调用签名）。
        返回值说明：
            dict：构造 _fake_cat_sum 时传入的品类累计映射 {品类: 金额}。
        """
        return cat_map
    return _f


def _fake_quota(quota_info):
    """
    函数功能与逻辑描述：
        构造替代 `_calc_category_quota`（品类合理额度）的 async 桩。
    入参说明：
        quota_info：桩要返回的额度信息 dict；None 表示无基准。
    返回值说明：
        返回 async 桩函数 `_f(city, category)`，调用后返回 quota_info。
    """
    async def _f(_city, _category):
        """
        函数功能与逻辑描述：
            替代 _calc_category_quota（品类合理额度）的 async 桩：忽略城市与品类入参，
            恒定返回构造时给定的额度信息，用于驱动「有额度 → 计算 ratio」与
            「无额度且已消费 → need_quota=True」两条 L2 分支。
        入参说明：
            _city：城市名（本桩忽略，仅为对齐真实 _calc_category_quota 的调用签名）。
            _category：消费品类（本桩忽略）。
        返回值说明：
            dict | None：构造 _fake_quota 时传入的额度信息；None 表示无任何额度基准。
        """
        return quota_info
    return _f


def _month_offset(n: int) -> str:
    """
    函数功能与逻辑描述：
        计算 n 个月前的自然月字符串（n=0 为当月），跨年自动回退年份。
    入参说明：
        n (int)：向前回溯的月数（0 为当月）。
    返回值说明：
        str：形如 "YYYY-MM" 的月份字符串。
    """
    y, m = date.today().year, date.today().month
    m -= n
    while m <= 0:
        m += 12
        y -= 1
    return f"{y:04d}-{m:02d}"


def _fake_habits(rows: list):
    """
    函数功能与逻辑描述：
        构造替代 `_load_recent_habits`（近 N 月习惯聚合）的 async 桩，并把调用参数记录到 `_f.called_with`。
        ★复刻真实取数语义：只返回「查询月份列表内」的行——include_current=False 时月份列表不含当月，
        故当月数据不会被返回（这正是修复的关键）。
    入参说明：
        rows (list)：候选习惯行列表，每行含 month/category/amount_sum/count 等字段。
    返回值说明：
        返回 async 桩函数 `_f(months_back, include_current)`，调用后返回过滤后的行列表。
    """
    async def _f(months_back=3, include_current=True):
        """
        函数功能与逻辑描述：
            替代 _load_recent_habits 的 async 桩：★复刻真实取数语义——先把调用参数记录到
            `_f.called_with` 供断言，再以 date.today() 为基准生成查询月份列表
            （include_current=False 时不含当月），仅返回 month 落在该列表内的行，
            故当月数据不会被返回（这正是修复的关键）。
        入参说明：
            months_back：回溯的自然月数量，默认 3。
            include_current：是否包含当月，默认 True；False 时从上一月起始。
        返回值说明：
            list：rows 中 month 命中查询月份列表的行（复刻真实「只返回查询月份内数据」的语义）。
        """
        _f.called_with = {"months_back": months_back,
                          "include_current": include_current}
        y, m = date.today().year, date.today().month
        if not include_current:
            m -= 1
            if m == 0:
                m, y = 12, y - 1
        months = []
        for _ in range(months_back):
            months.append(f"{y:04d}-{m:02d}")
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        return [r for r in rows if r.get("month") in months]
    _f.called_with = None
    return _f


# ============================================================
# 一、_calc_month_budget_status 档位判定
# ============================================================

@pytest.mark.asyncio
async def test_status_none_without_budget(monkeypatch):
    """
    函数功能与逻辑描述：
        无预算配置时不打扰：get_user_target_config 返回空配置，断言预算状态为 None。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `get_user_target_config` 为无预算 async 桩。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg())
    assert await nodes._calc_month_budget_status() is None


@pytest.mark.asyncio
async def test_status_over(monkeypatch):
    """
    函数功能与逻辑描述：
        used_ratio >= 1.0 判为 over 超支：预算 1000、已花 1200，断言 level=over、ratio 约 1.2、
        month_spent=1200、month_budget=1000。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置桩与 `_sum_month_bills`。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(1200.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "over"
    assert st["used_ratio"] == pytest.approx(1.2)
    assert st["month_spent"] == 1200.0
    assert st["month_budget"] == 1000.0


@pytest.mark.asyncio
async def test_status_warn(monkeypatch):
    """
    函数功能与逻辑描述：
        0.8 <= used_ratio < 1.0 判为 warn 接近红线：预算 1000、已花 900，断言 level=warn、ratio 约 0.9。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置桩与 `_sum_month_bills`。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(900.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "warn"
    assert st["used_ratio"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_status_reward(monkeypatch):
    """
    函数功能与逻辑描述：
        0 < used_ratio <= 0.5 判为 reward 预算充裕（犒劳）：预算 1000、已花 300，断言 level=reward、ratio 约 0.3。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置桩与 `_sum_month_bills`。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(300.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "reward"
    assert st["used_ratio"] == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_status_ok_no_consume(monkeypatch):
    """
    函数功能与逻辑描述：
        当月无消费（used_ratio == 0）判为 ok 不打扰（月首不犒劳）：断言 level=ok。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置桩与 `_sum_month_bills`。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(0.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "ok"


@pytest.mark.asyncio
async def test_status_ok_mid_range(monkeypatch):
    """
    函数功能与逻辑描述：
        0.5 < used_ratio < 0.8 判为 ok 正常区间不打扰：预算 1000、已花 700，断言 level=ok、ratio 约 0.7。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置桩与 `_sum_month_bills`。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(700.0))
    st = await nodes._calc_month_budget_status()
    assert st["level"] == "ok"
    assert st["used_ratio"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_status_exception_returns_none(monkeypatch):
    """
    函数功能与逻辑描述：
        读取配置抛异常时静默兜底返回 None，不影响记账主流程。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `get_user_target_config` 为抛异常函数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    def boom(_):
        """
        函数功能与逻辑描述：
            模拟读取配置抛异常的替身：接收会话参数后抛出 RuntimeError，
            用于验证 _calc_month_budget_status 在配置读取失败时静默兜底返回 None。
        入参说明：
            _：会话标识（本桩忽略，仅为对齐 get_user_target_config 的调用签名）。
        返回值说明：
            无（恒定抛出 RuntimeError("db broken")，不返回）。
        """
        raise RuntimeError("db broken")
    monkeypatch.setattr(nodes, "get_user_target_config", boom)
    assert await nodes._calc_month_budget_status() is None


@pytest.mark.asyncio
async def test_status_days_left_positive(monkeypatch):
    """
    函数功能与逻辑描述：
        days_left 必须为当月剩余自然日（含今天）且 > 0。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置桩与 `_sum_month_bills`。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_sum_month_bills", _fake_sum(500.0))
    st = await nodes._calc_month_budget_status()
    assert st["days_left"] > 0


# ============================================================
# 二、_calc_total_budget_fact 总量结构化事实
# ============================================================

def test_total_fact_over():
    """
    函数功能与逻辑描述：
        over 档产出结构化事实（超支）：断言 level=over、month_spent=1200、month_budget=1000，
        over_categories 初始为空列表。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    fact = nodes._calc_total_budget_fact(_mk_status("over", 1200, 1000, 1.2))
    assert fact is not None
    assert fact["level"] == "over"
    assert fact["month_spent"] == 1200.0
    assert fact["month_budget"] == 1000.0
    assert fact["over_categories"] == []


def test_total_fact_warn_daily_advice():
    """
    函数功能与逻辑描述：
        warn 档产出每日建议：daily_limit = 剩余预算 / 剩余天数（预算 1000、已花 900、剩 10 天 → 每日 10 元）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    # 预算 1000，已花 900，剩 100，剩余 10 天 → 每日 10 元
    fact = nodes._calc_total_budget_fact(_mk_status("warn", 900, 1000, 0.9, days_left=10))
    assert fact["level"] == "warn"
    assert fact["remain"] == 100.0
    assert fact["daily_limit"] == 10.0


def test_total_fact_reward():
    """
    函数功能与逻辑描述：
        reward 档产出预算充裕事实（犒劳文案由 LLM 汇总提示词生成）；断言 remain=700，
        且非 warn 档不产出每日建议（daily_limit 为 None）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    fact = nodes._calc_total_budget_fact(_mk_status("reward", 300, 1000, 0.3))
    assert fact["level"] == "reward"
    assert fact["remain"] == 700.0
    # 非 warn 档不产出每日建议
    assert fact["daily_limit"] is None


def test_total_fact_ok_none():
    """
    函数功能与逻辑描述：
        ok 档不打扰：返回 None。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._calc_total_budget_fact(_mk_status("ok", 700, 1000, 0.7)) is None


def test_total_fact_no_big_single_vs_total_budget():
    """
    函数功能与逻辑描述：
        需求1修正：`_calc_total_budget_fact` 只接收预算状态、已无单笔金额入参，单笔不再与总预算对比；
        ok 档一律返回 None。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert nodes._calc_total_budget_fact(_mk_status("ok", 700, 1000, 0.7)) is None


def test_total_fact_zero_amount_warn():
    """
    函数功能与逻辑描述：
        `_calc_total_budget_fact` 不接收单笔金额（用例名 zero_amount 为历史遗留），
        warn 档仅凭预算状态即产出事实，断言 level=warn 不受影响。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    fact = nodes._calc_total_budget_fact(_mk_status("warn", 900, 1000, 0.9))
    assert fact["level"] == "warn"


# ============================================================
# 三、_calc_alert_facts 三层结构化事实（需求1修正核心）
# ============================================================

@pytest.mark.asyncio
async def test_alert_facts_l1_removed_top_level_does_no_price(monkeypatch):
    """
    函数功能与逻辑描述：
        ★2026-09-17 契约变更（用户指令）：**顶层（orchestrator）不做比价**——
        "品类归类 + 溢价率计算"统一由 `price_agent` 承担，本层 `alert_facts.l1` **恒为 None**。
        本用例由原先的 `test_alert_facts_l1_premium_high` 与 `..._premium_normal` **合并改写**：
        那两条原本断言"顶层用均价算出溢价率与档位"，而**那正是被移除的行为**；现改为
        **反向断言**，锁住"比价不进顶层"这一契约，防止后续把比价逻辑重新搬回顶层。
        同时保留"ok 档 + 无额度基准不打扰"（l2 / l3 均为 None）的断言。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置/均价/额度/品类累计等取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    # 刻意给出"可查到的基准价 38 元 + 金额 180 元（按旧逻辑必然是'过高'）"，
    # 以此证明 l1 为 None 是**行为移除**所致，而不是"取不到数据"。
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(38.0))
    monkeypatch.setattr(nodes, "_calc_category_quota", _fake_quota(None))
    monkeypatch.setattr(nodes, "_sum_month_bills_by_category", _fake_cat_sum({}))
    facts = await nodes._calc_alert_facts(
        _mk_status("ok", 700, 1000, 0.7), 180.0, "餐饮", "上海"
    )
    assert facts["l1"] is None                    # ★顶层不产出任何比价事实
    assert set(facts) == {"l1", "l2", "l3"}       # 结构保持三键（下游兼容）
    assert facts["l2"] is None                    # 无额度基准且当月该品类未消费
    assert facts["l3"] is None                    # ok 档不打扰


@pytest.mark.asyncio
async def test_alert_facts_l2_category_progress(monkeypatch):
    """
    函数功能与逻辑描述：
        L2：品类进度 = 当月累计 vs 用户设定额度（812/900 ≈ 0.9022，ratio>=0.8 供 LLM 预警）；
        断言 spent/quota/ratio/source，且 need_quota=False。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置/均价/额度/品类累计等取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        L2：无任何额度基准且当月已消费 → need_quota=True（供 LLM 引导设置额度）；
        断言 quota 为 None、source="none"。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置/均价/额度/品类累计等取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        L3：总量超支（2812/1000）+ 点名超支主因品类（over_categories 按超出比例排序）；
        断言仅购物超支（2000/400=+400%，餐饮 812/900 未超支）。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置/均价/额度/品类累计等取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(1000))
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(0.0))
    async def _quota(_city, cat):
        """
        函数功能与逻辑描述：
            按品类返回不同额度的 async 替身：餐饮 900 / 购物 400 / 其余默认 500，
            source 固定 "habit"，用于构造「购物超支、餐饮未超支」的 L3 超支归因场景。
        入参说明：
            _city：城市名（本桩忽略，仅为对齐 _calc_category_quota 的调用签名）。
            cat：品类名，决定返回的额度值。
        返回值说明：
            dict：{quota: 该品类额度, source: "habit"}。
        """
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
    """
    函数功能与逻辑描述：
        无均价、无额度、无总量状态（ok 档）且品类/城市为空 → 三层事实均为 None（LLM 收到空事实，不打扰）。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置/均价/额度/品类累计等取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数/类功能与逻辑描述：
        mock nodes.db 的物价查询接口替身：own=本城均价 / level=本城分级 / same=同级城市均价；
        隔离真实数据库，用于验证城市均价兜底逻辑。
    构造入参说明：
        own：本城该品类均价，默认 0.0。
        level：本城城市分级，默认 None（无分级）。
        same：同级城市均价，默认 0.0。
    返回值说明：
        构造返回 _FakePriceDB 实例。
    """

    def __init__(self, own=0.0, level=None, same=0.0):
        """
        函数功能与逻辑描述：
            记录本城均价、城市分级、同级均价三个查询结果，供后续接口方法返回。
        入参说明：
            own：本城该品类均价，默认 0.0。
            level：本城城市分级，默认 None。
            same：同级城市均价，默认 0.0。
        返回值说明：
            无（仅初始化实例，无副作用）。
        """
        self.own = own
        self.level = level
        self.same = same

    def get_city_price(self, city, category):
        """
        函数功能与逻辑描述：
            模拟本城物价查询接口，固定返回构造时传入的本城均价 own。
        入参说明：
            city：城市名（本桩忽略）。
            category：品类名（本桩忽略）。
        返回值说明：
            float：本城该品类均价 own。
        """
        return self.own

    def get_city_level(self, city):
        """
        函数功能与逻辑描述：
            模拟城市分级查询接口，固定返回构造时传入的城市分级 level。
        入参说明：
            city：城市名（本桩忽略）。
        返回值说明：
            str | None：本城分级 level（None 表示无分级）。
        """
        return self.level

    def get_same_level_avg_price(self, city_level, category):
        """
        函数功能与逻辑描述：
            模拟同级城市均价查询接口，固定返回构造时传入的 same。
        入参说明：
            city_level：城市分级（本桩忽略）。
            category：品类名（本桩忽略）。
        返回值说明：
            float：同级城市该品类均价 same。
        """
        return self.same


# M9（D15）：城市均价兜底逻辑已**整体迁入 `city_price.get_avg` 工具**（`mcpGateway/fact_tools.py`），
# 单测随之指向工具侧（mock `fact_tools.db`）；orchestrator `_query_city_avg_price` 仅转发 + 异常兜底（下两用例）。
def test_city_avg_price_local_hit(monkeypatch):
    """
    函数功能与逻辑描述：
        本城有该品类均价 → 直接返回，不走同级兜底（工具语义）：断言返回 30.0。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `fact_tools.db` 为 _FakePriceDB。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(ft, "db", _FakePriceDB(own=30.0, level="一线", same=99.9))
    assert ft.get_city_avg_price("广州", "餐饮") == 30.0


def test_city_avg_price_same_level_fallback(monkeypatch):
    """
    函数功能与逻辑描述：
        本城无该品类均价 → 回退同级城市均价（工具语义）：断言返回 36.5。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `fact_tools.db` 为 _FakePriceDB。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(ft, "db", _FakePriceDB(own=0.0, level="一线", same=36.5))
    assert ft.get_city_avg_price("广州", "餐饮") == 36.5


def test_city_avg_price_no_fallback(monkeypatch):
    """
    函数功能与逻辑描述：
        本城无数据且无分级、同级也无数据 → 返回 0（跳过溢价评估）。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `fact_tools.db` 为 _FakePriceDB。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(ft, "db", _FakePriceDB(own=0.0, level=None, same=0.0))
    assert ft.get_city_avg_price("广州", "餐饮") == 0.0


@pytest.mark.asyncio
async def test_query_city_avg_price_forwards_to_tool(monkeypatch):
    """
    函数功能与逻辑描述：
        M9：orchestrator 读城市均价 = 经 `city_price.get_avg` 工具转发（FACT_READ）；
        断言调用工具名与入参正确、返回值透传 36.5。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `_invoke_fact` 为校验型 async 桩。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    async def _fake_invoke(name, args):
        """
        函数功能与逻辑描述：
            校验型 async 替身：断言经 _invoke_fact 下发的工具名与入参正确，并返回固定均价，
            用于验证 _query_city_avg_price 经 city_price.get_avg 工具（FACT_READ）正确转发。
        入参说明：
            name：被调用的工具名（须为 "city_price.get_avg"）。
            args：工具入参（须为 {"city": "广州", "category": "餐饮"}）。
        返回值说明：
            float：固定返回 36.5，供上层断言透传。
        """
        assert name == "city_price.get_avg"
        assert args == {"city": "广州", "category": "餐饮"}
        return 36.5
    monkeypatch.setattr(nodes, "_invoke_fact", _fake_invoke)
    assert await nodes._query_city_avg_price("广州", "餐饮") == 36.5


@pytest.mark.asyncio
async def test_query_city_avg_price_tool_error_falls_back_zero(monkeypatch):
    """
    函数功能与逻辑描述：
        M9：工具调用失败 → orchestrator 兜底 0（跳过溢价评估，不阻断预警链）。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `_invoke_fact` 为抛异常 async 桩。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    async def _boom(name, args):
        """
        函数功能与逻辑描述：
            模拟工具调用失败的 async 替身：接收工具名与入参后抛出 RuntimeError，
            用于验证 _query_city_avg_price 兜底返回 0.0（跳过溢价评估、不阻断预警链）。
        入参说明：
            name：被调用的工具名（本桩忽略）。
            args：工具入参（本桩忽略）。
        返回值说明：
            无（恒定抛出 RuntimeError("tool down")，不返回）。
        """
        raise RuntimeError("tool down")
    monkeypatch.setattr(nodes, "_invoke_fact", _boom)
    assert await nodes._query_city_avg_price("广州", "餐饮") == 0.0


# ============================================================
# 五、_calc_category_quota 品类额度「月均」口径（2026-09-08 修复）
# ============================================================

@pytest.mark.asyncio
async def test_quota_current_month_excluded(monkeypatch):
    """
    函数功能与逻辑描述：
        修复验证：当月「累计至今」未完成，不得当作整月额度（否则月初误报超支）；
        断言以 include_current=False 取数，且无已完成月份时绝不返回 habit 额度（不能是 38 元）。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置/习惯取数/均价取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    cur = _month_offset(0)
    fake = _fake_habits([{"month": cur, "category": "餐饮",
                          "amount_sum": 38.0, "count": 1}])
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(3000))
    monkeypatch.setattr(nodes, "_load_recent_habits", fake)
    monkeypatch.setattr(nodes, "_query_city_avg_price", _fake_avg_price(0.0))

    q = await nodes._calc_category_quota("广州", "餐饮")

    # ① 必须以 include_current=False 取数（排除当月）
    assert fake.called_with["include_current"] is False
    # ② 无已完成月份 → 绝不返回 habit 额度（不能是 38 元）
    assert q is None or q["source"] != "habit"


@pytest.mark.asyncio
async def test_quota_averages_completed_months(monkeypatch):
    """
    函数功能与逻辑描述：
        已完成月份才计入月均：前 1/2/3 月分别 100/200/300 → 额度 200；断言 source=habit、quota=200。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换配置与习惯取数。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    fake = _fake_habits([
        {"month": _month_offset(1), "category": "餐饮", "amount_sum": 100.0, "count": 2},
        {"month": _month_offset(2), "category": "餐饮", "amount_sum": 200.0, "count": 4},
        {"month": _month_offset(3), "category": "餐饮", "amount_sum": 300.0, "count": 6},
    ])
    monkeypatch.setattr(nodes, "get_user_target_config", _fake_cfg(3000))
    monkeypatch.setattr(nodes, "_load_recent_habits", fake)

    q = await nodes._calc_category_quota("广州", "餐饮")

    assert fake.called_with["include_current"] is False
    assert q["source"] == "habit"
    assert q["quota"] == 200.0        # (100+200+300)/3


@pytest.mark.asyncio
async def test_load_recent_habits_include_current_switch(monkeypatch):
    """
    函数功能与逻辑描述：
        验证 `_load_recent_habits` 的当月开关：默认含当月（finance 注入依赖，向后兼容），
        include_current=False 时跳过当月；断言请求的月份列表构成。
    入参说明：
        monkeypatch：pytest 注入的补丁器，替换 `_invoke_fact` 为记录型 async 桩。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    seen = {}

    async def _fake_invoke(name, args):
        """
        函数功能与逻辑描述：
            记录型 async 替身：把每次工具调用的入参按工具名存入闭包 `seen` 并返回空列表，
            用于验证 _load_recent_habits 请求的月份列表是否含当月（include_current 开关语义）。
        入参说明：
            name：被调用的工具名（本用例为 "habit.list"）。
            args：工具入参（含 months 月份列表，供断言其构成）。
        返回值说明：
            list：固定返回空列表（避免触达真实工具）。
        """
        seen[name] = args
        return []

    monkeypatch.setattr(nodes, "_invoke_fact", _fake_invoke)
    cur = _month_offset(0)

    await nodes._load_recent_habits(3)                         # 默认：含当月
    assert cur in seen["habit.list"]["months"]

    seen.clear()
    await nodes._load_recent_habits(3, include_current=False)  # 排除当月
    assert cur not in seen["habit.list"]["months"]
    assert len(seen["habit.list"]["months"]) == 3
