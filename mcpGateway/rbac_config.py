from enum import Enum


# 数据库权限枚举
class DBPerm(Enum):
    BILL_READ = "bill:read"    # 账单查询权限
    BILL_WRITE = "bill:write"  # 账单增删改权限


# 大模型调用权限枚举
class LLMPerm(Enum):
    LLM_BASE = "llm_base"        # 通用基座模型
    LLM_FINANCE = "llm_finance"  # 理财专属微调模型


# 只读事实权限（D15 / M9：编排器取数"走正门"替代裸连 DB，`03` §9.9.2）
class FactPerm(Enum):
    FACT_READ = "fact:read"    # 只读确定性事实（user_config / bill 聚合 / city_price / monthly_habit 读）


# 月度消费习惯权限（M9：habit.upsert 写下沉 bill_agent）
class HabitPerm(Enum):
    HABIT_READ = "habit:read"    # 预留：习惯读（stat / finance 走工具化时分配，随 M13 落地）
    HABIT_WRITE = "habit:write"  # 习惯写（bill_agent 记账成功顺手沉淀）


# 各Agent权限分配（严格最小权限原则）
AGENT_PERMISSION_MAP = {
    # 编排器：只能调用通用模型 + 只读事实（D15 收口后零写权限）
    "orchestrator_agent": [
        LLMPerm.LLM_BASE,
        FactPerm.FACT_READ
    ],
    # 记账Agent：账单读写 + 习惯写（M9 承接月度习惯沉淀）+ 通用模型
    "bill_agent": [
        DBPerm.BILL_READ,
        DBPerm.BILL_WRITE,
        HabitPerm.HABIT_WRITE,
        LLMPerm.LLM_BASE
    ],
    # 统计Agent：账单只读 + 通用模型
    "stat_agent": [
        DBPerm.BILL_READ,
        LLMPerm.LLM_BASE
    ],
    # 物价Agent：账单只读 + 通用模型
    "price_agent": [
        DBPerm.BILL_READ,
        LLMPerm.LLM_BASE
    ],
    # 理财Agent：账单只读 + 理财专属模型
    "finance_agent": [
        DBPerm.BILL_READ,
        LLMPerm.LLM_FINANCE
    ]
}


def get_agent_permission(agent_id: str) -> list:
    """根据Agent标识获取全部权限列表"""
    return AGENT_PERMISSION_MAP.get(agent_id, [])


__all__ = ["DBPerm", "FactPerm", "HabitPerm", "LLMPerm", "get_agent_permission"]