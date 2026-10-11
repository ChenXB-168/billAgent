# ==============================================
# RBAC 权限矩阵单元测试：逐 agent 校验数据库/模型权限集合的最小权限边界
# ==============================================
from mcpGateway.rbac_config import get_agent_permission, DBPerm, LLMPerm


def test_orchestrator_permission():
    """
    函数功能与逻辑描述：
        校验编排器（orchestrator_agent）的权限集合——持有通用模型权限 LLM_BASE，
        不含账单读/写权限，也不含理财专属模型权限，用于锁定编排层「不直接触碰业务库」的边界。
        覆盖场景：无入参直接查询权限集合，逐项断言各权限的有无。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    perms = get_agent_permission("orchestrator_agent")
    assert LLMPerm.LLM_BASE in perms
    assert DBPerm.BILL_READ not in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_FINANCE not in perms


def test_bill_agent_permission():
    """
    函数功能与逻辑描述：
        校验记账 Agent（bill_agent）的权限集合——账单读写权限 + 通用模型权限，
        且不含理财专属模型权限（记账不调用理财模型）。
        覆盖场景：无入参直接查询权限集合，逐项断言各权限的有无。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    perms = get_agent_permission("bill_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE in perms
    assert LLMPerm.LLM_BASE in perms
    assert LLMPerm.LLM_FINANCE not in perms


def test_stat_agent_permission():
    """
    函数功能与逻辑描述：
        校验统计 Agent（stat_agent）的权限集合——仅账单读权限（无写权限）+ 通用模型权限，
        验证只读 Agent 不能写库。
        覆盖场景：无入参直接查询权限集合，逐项断言各权限的有无。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    perms = get_agent_permission("stat_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_BASE in perms


def test_price_agent_permission():
    """
    函数功能与逻辑描述：
        校验物价 Agent（price_agent）的权限集合——仅账单读权限（无写权限）+ 通用模型权限。
        覆盖场景：无入参直接查询权限集合，逐项断言各权限的有无。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    perms = get_agent_permission("price_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_BASE in perms


def test_finance_agent_permission():
    """
    函数功能与逻辑描述：
        校验理财 Agent（finance_agent）的权限集合——账单只读（无写权限）+ 理财专属模型权限。
        覆盖场景：无入参直接查询权限集合，逐项断言各权限的有无。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    perms = get_agent_permission("finance_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_FINANCE in perms


def test_unknown_agent_permission():
    """
    函数功能与逻辑描述：
        校验未登记主体（如 hacker_agent）的权限集合为空，验证「未标识主体一律无权限」的默认拒绝策略。
        覆盖场景：无入参直接查询未知 agent 的权限集合，断言集合长度为 0。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    perms = get_agent_permission("hacker_agent")
    assert len(perms) == 0
