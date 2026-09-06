from langgraph.graph import StateGraph, START, END
from agents.finance_agent.state import FinanceState
from agents.finance_agent.nodes import (
    build_prompt_node, execute_finance_node, verify_query_node, reply_node
)

builder = StateGraph(FinanceState)
builder.add_node("build_prompt_node", build_prompt_node)
builder.add_node("execute_finance_node", execute_finance_node)
builder.add_node("verify_query_node", verify_query_node)  # D9 Reflection（M11）
builder.add_node("reply_node", reply_node)

# 起点直接进入业务节点，移除recv_task_node链路
builder.add_edge(START, "build_prompt_node")
builder.add_edge("build_prompt_node", "execute_finance_node")
# D9 Reflection（M11）：execute 成功流默认进 verify；verify 按决策回 execute 重生成或发结果
builder.add_edge("execute_finance_node", "verify_query_node")
builder.add_conditional_edges(
    "verify_query_node",
    lambda state: state.get("verify_decision") or "reply",
    {"retry": "execute_finance_node", "reply": "reply_node"},
)
builder.add_edge("reply_node", END)

finance_agent_graph = builder.compile()
__all__ = ["finance_agent_graph"]
