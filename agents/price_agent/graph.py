from langgraph.graph import StateGraph, START, END
from agents.price_agent.state import PriceState
from agents.price_agent.nodes import execute_node, reply_node

builder = StateGraph(PriceState)
builder.add_node("execute_node", execute_node)
builder.add_node("reply_node", reply_node)

# 移除recv_task，外层worker提前接收消息并初始化状态
builder.add_edge(START, "execute_node")
builder.add_edge("execute_node", "reply_node")
builder.add_edge("reply_node", END)

price_agent_graph = builder.compile()
__all__ = ["price_agent_graph"]