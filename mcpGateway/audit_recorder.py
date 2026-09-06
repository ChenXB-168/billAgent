import time
from utils.common import audit_logger

def record_operate(agent_id: str, tool_name: str, content: str, success: bool):
    """
    MCP网关全操作审计日志
    :param agent_id: 操作Agent标识
    :param tool_name: 操作分类标签(sql_select/llm_base)
    :param content: 操作原始内容(SQL/用户提问)
    :param success: 是否执行成功
    """
    audit_logger.info(
        "MCP网关操作",
        extra={
            "timestamp": round(time.time(), 2),
            "agent_id": agent_id,
            "tool": tool_name,
            "content": content,
            "success": success
        }
    )

# 对外统一别名
write_audit_log = record_operate
__all__ = ["write_audit_log"]