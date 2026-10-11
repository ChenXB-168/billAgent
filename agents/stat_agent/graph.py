from langgraph.graph import StateGraph, START, END
from agents.stat_agent.state import StatState
from agents.stat_agent.nodes import (
    parse_query_node, execute_query_node, verify_query_node, reply_node
)

builder = StateGraph(StatState)
builder.add_node("parse_query_node", parse_query_node)
builder.add_node("execute_query_node", execute_query_node)
builder.add_node("verify_query_node", verify_query_node)  # D9 Reflection（M10）
builder.add_node("reply_node", reply_node)

# 起点直接进入首个业务节点
builder.add_edge(START, "parse_query_node")
# 条件边：parse_query_node 解析失败（raw_ir 非 dict）时跳过 execute_query_node 直达 reply_node
# 之前用静态边，Command(goto="reply_node") 被 langgraph 忽略，导致 parse 失败后仍执行
# execute_query_node 且 raw_ir=None，触发 AttributeError 崩溃。
builder.add_conditional_edges(
    "parse_query_node",
    lambda state: "execute" if isinstance(state.get("raw_ir"), dict) else "reply",
    {"execute": "execute_query_node", "reply": "reply_node"},
)
# D9 Reflection（M10）：execute 成功流默认进 verify；verify 按决策回 parse 重生成或发结果
builder.add_edge("execute_query_node", "verify_query_node")
builder.add_conditional_edges(
    "verify_query_node",
    lambda state: state.get("verify_decision") or "reply",
    {"retry": "parse_query_node", "reply": "reply_node"},
)
builder.add_edge("reply_node", END)

stat_agent_graph = builder.compile()
__all__ = ["stat_agent_graph"]
