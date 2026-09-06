from langgraph.graph import StateGraph, START, END
from agents.orchestrator.state import OrchState
from agents.orchestrator.nodes import (
    plan_node, dispatch_node, wait_result_node, collect_node
)

builder = StateGraph(OrchState)

# 注册节点
builder.add_node("plan_node", plan_node)
builder.add_node("dispatch_node", dispatch_node)
builder.add_node("wait_result_node", wait_result_node)
builder.add_node("collect_node", collect_node)

# 主链路
builder.add_edge(START, "plan_node")
builder.add_edge("plan_node", "dispatch_node")
builder.add_edge("wait_result_node", "dispatch_node")
builder.add_edge("collect_node", END)

# 编译
orchestrator_graph = builder.compile()
__all__ = ["orchestrator_graph"]