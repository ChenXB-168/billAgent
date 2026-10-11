from langgraph.graph import StateGraph, START, END
from agents.bill_agent.state import BillState
from agents.bill_agent.nodes import execute_node, reply_node

builder = StateGraph(BillState)

builder.add_node("execute_node", execute_node)
builder.add_node("reply_node", reply_node)

# 起点直接进业务执行节点
builder.add_edge(START, "execute_node")
builder.add_edge("execute_node", "reply_node")
builder.add_edge("reply_node", END)

bill_agent_graph = builder.compile()
__all__ = ["bill_agent_graph"]