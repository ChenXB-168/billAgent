# ==============================================
# MCP 网关操作审计记录器 - 统一审计日志落盘入口
# 日志文件由 utils.common.audit_logger 绑定（AUDIT_LOG_PATH，独立于运行日志）
# ==============================================
import time
from utils.common import audit_logger


def record_operate(agent_id: str, tool_name: str, content: str, success: bool):
    """
    函数功能与逻辑描述：
        向独立审计日志写入一条结构化操作记录，用于事后追溯「哪个 Agent 发起、调用哪类工具、
        原文内容是什么、是否成功」。仅落盘，不做鉴权、限流或异常捕获，也不改变调用方控制流。
        鉴权失败与执行失败路径同样需要调用本函数留痕，因此本函数对 success=False 不做任何过滤。
    入参说明：
        agent_id (str)：发起操作的 Agent 标识，如 orchestrator_agent / bill_agent / finance_agent。
        tool_name (str)：操作分类标签，用于按类聚合检索。实际取值由各调用点自行约定，
            当前代码中出现的取值为：SQL_BILL（账单 SQL 通道，见 sql_server/server_bill.py 与
            client.py）、{db_identifier}_{op_type} 形式（如 bill_select / bill_insert，
            见 sql_common/sql_mcp_base.py）、LLM_FINANCE 与 llm_finance
            （理财模型通道，见 client.py 与 server_llm_finance.py）。
        content (str)：操作原始内容，通常为 SQL 语句或用户提问原文；
            异常路径下由调用方改写为错误摘要（如 "{sql} 异常信息：..."）。
        success (bool)：本次操作是否成功。失败场景必须以 False 调用，以便审计侧可检索。
    返回值说明：
        无（仅写日志，无返回值、无其它副作用）。
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