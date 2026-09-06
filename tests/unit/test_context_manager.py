"""M12（D6）ContextManager 单元测试：分层预算 / 溢出策略 / 红线 / 接入回退。

验收锚点：构造超长上下文用例 → 不溢出（各层估算 ≤ 预算）且关键信息保留
（system 不可截、task_results 受保护块不截、user_docs 最低保留 1 段、history 保留最近）。
"""
import pytest

from agents.context_manager import (
    ContextManager,
    ContextResult,
    estimate_tokens,
    context_manager as _shared_cm,
)


# ---------------- estimate_tokens 近似估算 ----------------
def test_estimate_tokens_empty_and_units():
    assert estimate_tokens("") == 0
    # 纯中文 100 字：1.5 token/字 → 150
    assert estimate_tokens("费" * 100) == 150
    # 纯 ASCII 100 字符：0.3 token/字符 → 30
    assert estimate_tokens("a" * 100) == 30
    # 非空中文文本 token > 0
    assert estimate_tokens("本月支出分析") > 0


# ---------------- 预算内逐字节零改动（回归零风险锚点） ----------------
def test_build_context_no_change_when_within_budget():
    cm = ContextManager()
    history = "用户：上个月花了多少？\n助手：上月支出 3200 元。"
    long_memory = "历史习惯：餐饮均额 45 元。"
    user_docs = "（用户笔记《健康饮食》）少喝奶茶。"
    result = cm.build_context(
        system="系统角色规则（固定）",
        history=history,
        long_memory=long_memory,
        user_docs=user_docs,
        task_results=[("【bill】已记账：餐饮 56 元", False)],
    )
    assert result.truncated is False
    assert result.layer("history") == history
    assert result.layer("long_memory") == long_memory
    assert result.layer("user_docs") == user_docs
    assert result.layer("task_results") == "【bill】已记账：餐饮 56 元"


# ---------------- history 滑窗：保留最近（超预算触发） ----------------
def test_history_keep_tail_when_over_budget():
    cm = ContextManager()
    old_tail = "最早：吃饭花费" + "很" * 3000          # 远古消息，应被滑出
    recent_tail = "最近：今天午饭吃了 28 元" + "哦" * 50  # 最近消息，应保留
    history = old_tail + "\n" + recent_tail
    result = cm.build_context(system="", history=history)
    assert result.truncated is True
    kept = result.layer("history")
    assert estimate_tokens(kept) <= cm.budget_for("history")
    assert "最近：" in kept          # 保留最近内容
    assert "最早：" not in kept      # 最早内容被滑出


# ---------------- long_memory reduce_top_k：保留头部 ----------------
def test_long_memory_keep_head_paragraphs():
    cm = ContextManager()
    paras = "\n".join(f"习惯段{i}：" + "费" * 150 for i in range(8))
    assert estimate_tokens(paras) > cm.budget_for("long_memory")
    result = cm.build_context(system="", long_memory=paras)
    assert result.truncated is True
    kept = result.layer("long_memory")
    assert estimate_tokens(kept) <= cm.budget_for("long_memory")
    assert "习惯段0" in kept   # 检索结果靠前段最相关 → 保留
    assert "习惯段7" not in kept


# ---------------- user_docs：最低保留 1 段（宁少勿空） ----------------
def test_user_docs_min_keep_one_paragraph():
    cm = ContextManager()
    # 多段超预算：至少保留首段，且不放空
    paras = "\n".join(f"笔记段{i}：" + "内" * 120 for i in range(10))
    result = cm.build_context(system="", user_docs=paras)
    assert result.truncated is True
    kept = result.layer("user_docs")
    assert estimate_tokens(kept) <= cm.budget_for("user_docs")
    assert "笔记段0" in kept
    assert kept.strip() != ""      # 最低保留 1 段（截空即违规）

    # 单段超预算：宁截头部保留也不整体丢弃
    one_huge = "单" * 2000
    result2 = cm.build_context(system="", user_docs=one_huge)
    assert result2.truncated is True
    assert result2.layer("user_docs") != ""
    assert estimate_tokens(result2.layer("user_docs")) <= cm.budget_for("user_docs")


# ---------------- task_results：受保护块（确定性底稿）不截（R9） ----------------
def test_task_results_protected_block_never_truncated():
    cm = ContextManager()
    protected = "【确定性事实底稿】本月支出3200元，预算3000元，超支200元（达107%）" + "。" * 3000
    result = cm.build_context(
        system="", task_results=[(protected, True)])
    assert result.truncated is True          # 超预算已登记
    assert result.layer("task_results") == protected  # 受保护块完整保留
    assert estimate_tokens(result.layer("task_results")) == estimate_tokens(protected)


def test_task_results_plain_blocks_truncated_tail():
    cm = ContextManager()
    big = "大段任务结果文本" + "结" * 3000
    result = cm.build_context(system="", task_results=[(big, False)])
    assert result.truncated is True
    kept = result.layer("task_results")
    assert estimate_tokens(kept) <= cm.budget_for("task_results")
    assert kept != ""


# ---------------- truncate_text 直接截断 ----------------
def test_truncate_text_modes():
    cm = ContextManager()
    text = "头部内容" + "中" * 300 + "尾部内容"
    head = cm.truncate_text(text, 200, mode="tail")   # 保留头部
    assert "头部内容" in head
    tail = cm.truncate_text(text, 200, mode="head")   # 保留尾部（滑窗）
    assert "尾部内容" in tail
    assert estimate_tokens(head) <= 200
    assert estimate_tokens(tail) <= 200


# ---------------- system 层不可截（超限 WARN 保留，不混入 layers） ----------------
def test_system_layer_not_truncatable():
    cm = ContextManager()
    huge_system = "角色" * 5000   # 远超 system 预留，但 system 不可截
    result = cm.build_context(system=huge_system, history="短")
    # system 不参与可截分层（调用方分传 sys/user），层内不出现、也不被截断
    assert "system" not in result.layers
    assert result.layer("history") == "短"


# ---------------- 窗口换算 / 常量 ----------------
def test_budget_conversion():
    cm = ContextManager(context_window=4096, output_reserve=1024)
    # 输入预算 = 4096-1024 = 3072；system 预留 = min(400, 3072*0.15=460) = 400
    assert cm.input_budget == 3072
    assert cm.system_budget == 400
    pool = 3072 - 400  # 2672
    assert cm.budget_for("history") == round(pool * 0.25)
    assert cm.budget_for("long_memory") == round(pool * 0.20)
    assert cm.budget_for("user_docs") == round(pool * 0.25)
    assert cm.budget_for("task_results") == round(pool * 0.30)


# ---------------- 接入点异常回退：stat build_stat_prompt 组件抛错 → 原文直发 ----------------
def test_stat_history_fallback_when_component_raises(monkeypatch):
    """走查『失败与恢复』：组件异常 → 接入点回退原拼装，不阻断 stat 链路。"""
    import agents.stat_agent.nodes as stat_nodes

    def _boom(**kwargs):
        raise RuntimeError("组件内部故障")

    # M14（D17）：stat 的接入点实例为带路由窗口的 _cm_stat（共享实例已不承载业务接入）
    monkeypatch.setattr(stat_nodes._cm_stat, "build_context", _boom)
    state = {
        "session_memory": {"history_text": "历史原文内容"},
        "user_input": "上个月餐饮花了多少",
    }
    sys_p, usr_p = stat_nodes.build_stat_prompt(state)
    assert sys_p
    assert "历史原文内容" in usr_p   # 回退：原文完整进入模板
