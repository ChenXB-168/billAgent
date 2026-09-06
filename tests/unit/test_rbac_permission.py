from mcpGateway.rbac_config import get_agent_permission, DBPerm, LLMPerm


def test_orchestrator_permission():
    """编排器：仅通用模型权限，无数据库权限"""
    perms = get_agent_permission("orchestrator_agent")
    assert LLMPerm.LLM_BASE in perms
    assert DBPerm.BILL_READ not in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_FINANCE not in perms


def test_bill_agent_permission():
    """记账Agent：账单读写 + 通用模型"""
    perms = get_agent_permission("bill_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE in perms
    assert LLMPerm.LLM_BASE in perms
    assert LLMPerm.LLM_FINANCE not in perms


def test_stat_agent_permission():
    """统计Agent：仅账单读权限，无写权限"""
    perms = get_agent_permission("stat_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_BASE in perms


def test_price_agent_permission():
    """物价Agent：仅账单读权限"""
    perms = get_agent_permission("price_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_BASE in perms


def test_finance_agent_permission():
    """理财Agent：账单只读 + 理财专属模型"""
    perms = get_agent_permission("finance_agent")
    assert DBPerm.BILL_READ in perms
    assert DBPerm.BILL_WRITE not in perms
    assert LLMPerm.LLM_FINANCE in perms


def test_unknown_agent_permission():
    """未知Agent：无任何权限"""
    perms = get_agent_permission("hacker_agent")
    assert len(perms) == 0