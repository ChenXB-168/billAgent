from enum import Enum


# 数据库权限枚举
class DBPerm(Enum):
    """
    函数/类功能与逻辑描述：
        账单库读写权限枚举，供 SQL 通道（sql_common.sql_mcp_base）按 SQL 类型校验。
        MEMBER 值即审计日志中记录的权限标签。
    构造入参说明：
        无（枚举成员固定）。
    返回值说明：
        无（枚举类，成员为 DBPerm 实例）。
    """
    BILL_READ = "bill:read"    # 账单查询权限（SELECT）
    BILL_WRITE = "bill:write"  # 账单增删改权限（INSERT / UPDATE / DELETE）


# 大模型调用权限枚举
class LLMPerm(Enum):
    """
    函数/类功能与逻辑描述：
        大模型调用权限枚举，区分「通用基座」与「理财专属」两条模型通道，
        供 llm_base / llm_finance 两个 MCP 服务各校各的权限，实现通道级最小权限隔离。
    构造入参说明：
        无（枚举成员固定）。
    返回值说明：
        无（枚举类，成员为 LLMPerm 实例）。
    """
    LLM_BASE = "llm_base"        # 通用基座模型
    LLM_FINANCE = "llm_finance"  # 理财专属微调模型


# 只读事实权限（D15 / M9：编排器取数"走正门"替代裸连 DB，`03` §9.9.2）
class FactPerm(Enum):
    """
    函数/类功能与逻辑描述：
        只读确定性事实权限枚举，用于让 orchestrator 经工具层读取事实数据，
        替代其直接裸连数据库（D15 收口，`03` §9.9.2），从而保持编排器零写权限。
        注意：当前代码仅定义了权限枚举，FactPerm 尚未被 AGENT_PERMISSION_MAP 之外的
        校验点消费，属于契约先行定义。
    构造入参说明：
        无（枚举成员固定）。
    返回值说明：
        无（枚举类，成员为 FactPerm 实例）。
    """
    FACT_READ = "fact:read"    # 只读确定性事实（user_config / bill 聚合 / city_price / monthly_habit 读）


# 月度消费习惯权限（M9：habit.upsert 写下沉 bill_agent）
class HabitPerm(Enum):
    """
    函数/类功能与逻辑描述：
        月度消费习惯权限枚举。写权限随 M9 下沉给 bill_agent（记账成功后顺手沉淀习惯），
        读权限当前仅作预留，待 stat / finance 改为工具化取数后分配。
    构造入参说明：
        无（枚举成员固定）。
    返回值说明：
        无（枚举类，成员为 HabitPerm 实例）。
    """
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
    """
    函数功能与逻辑描述：
        按 Agent 标识查表返回其被授予的权限枚举列表，是 MCP 网关做 RBAC 校验的统一数据源
        （SQL 通道校验 DBPerm，LLM 通道校验 LLMPerm）。
        采用「查不到即空列表」的失败安全策略：未登记的 Agent（含拼写错误、伪造标识）
        一律拿到空权限，从而被各校验点拒绝，不会意外获得默认权限。
    入参说明：
        agent_id (str)：Agent 标识，需与 AGENT_PERMISSION_MAP 的键完全一致
            （如 orchestrator_agent / bill_agent / stat_agent / price_agent / finance_agent）。
    返回值说明：
        list：权限枚举成员列表（DBPerm / LLMPerm / FactPerm / HabitPerm 混合）；
            标识未登记时返回 []，且返回的是映射表中原列表的引用，调用方不应就地修改。
    """
    return AGENT_PERMISSION_MAP.get(agent_id, [])


__all__ = ["DBPerm", "FactPerm", "HabitPerm", "LLMPerm", "get_agent_permission"]