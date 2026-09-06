"""MCP网关统一导出入口，上层仅需from mcpGateway import xxx"""
from .client import mcp_client
from .rbac_config import get_agent_permission
from .audit_recorder import write_audit_log
from .a2a_queue import A2AMessageBus, a2a_bus
# M1（D1 工具接入平台）：工具注册中心。新代码统一 `get_registry().invoke(...)`
# ★惰性装配——import 此处不会拉起任何 MCP 子进程，首次 invoke 时才装配
from .registry import get_registry

__all__ = [
    "mcp_client",
    "get_agent_permission",
    "write_audit_log",
    "A2AMessageBus",
    "a2a_bus",
    "get_registry"
]