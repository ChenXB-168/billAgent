import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from mcpGateway.sql_common.sql_mcp_base import run_sql_logic
from utils.common import db
from mcpGateway.audit_recorder import write_audit_log
from config.config import MCP_HOST, MCP_SERVER_PORTS, MCP_HTTP_PATH, MCP_STDIO_FALLBACK

# M3：常驻 HTTP 服务（`03` §9.8.2）——host/port 由 config 统一定义，随 bootstrap 拉起
mcp = FastMCP(
    "mcp-sql-bill",
    host=MCP_HOST,
    port=MCP_SERVER_PORTS["sql_bill"],
    streamable_http_path=MCP_HTTP_PATH,
)

@mcp.tool()
async def exec_sql(agent_id: str, sql: str, params: list | None = None):
    """
    函数功能与逻辑描述：
        账单库的 MCP 工具入口，把请求转交给公共执行体 run_sql_logic（库标识固定为 "bill"，
        连接实例固定为 utils.common.db 单例），从而复用统一的危险语句拦截、RBAC 鉴权、
        db_lock 串行化与审计逻辑，本函数自身不重复实现任何安全判断。
        另在本层额外记一条「发起SQL调用」审计；注意该条审计是无条件 success=True 的入口流水，
        真正的成败结论以 run_sql_logic 内部记录的 success=False 条目为准。
        外层兜底捕获异常并把异常 repr 文本化返回，保证 MCP 边界不向上抛栈。
    入参说明：
        agent_id (str)：调用方 Agent 标识，用于权限校验与审计归属。
        sql (str)：待执行的 SQL 原文。
        params (list | None)：SQL 绑定参数，默认 None 表示不带参执行。
    返回值说明：
        list[TextContent]：长度为 1 的文本结果列表，内容与 run_sql_logic 的返回语义一致
            （查询结果为结果列表的 str 形态，写操作为 "SQL执行成功"/"SQL执行失败"，
            拒绝为「操作拒绝：...」/「权限不足：...」）；本层捕获到异常时返回
            "SQL执行内部异常：{repr(e)}"。
    """
    try:
        res = await run_sql_logic(
            agent_id=agent_id,
            db_identifier="bill",
            sql=sql,
            params=params,
            db_inst=db
        )
        write_audit_log(agent_id, "SQL_BILL", f"发起SQL调用：{sql}", success=True)
        return res
    except Exception as e:
        err_msg = f"SQL执行内部异常：{repr(e)}"
        write_audit_log(agent_id, "SQL_BILL", f"异常:{sql}", success=False)
        return [TextContent(type="text", text=err_msg)]

if __name__ == "__main__":
    # ★M4（D8 沙箱）：进程 self-prison（POSIX RLIMIT_AS/RLIMIT_CORE；Windows no-op，由 bootstrap Job 兜底）
    from mcpGateway.sandbox import apply_process_limits
    apply_process_limits()
    if MCP_STDIO_FALLBACK:
        # ★M3 回滚：BILLAGENT_MCP_STDIO=1 时 server 走 stdio（与客户端回滚开关配对）
        print("[SQL-BILL MCP] 服务启动成功（stdio 回滚模式）", file=sys.stderr)
        mcp.run()
    else:
        print(f"[SQL-BILL MCP] 服务启动成功 http://{MCP_HOST}:{MCP_SERVER_PORTS['sql_bill']}{MCP_HTTP_PATH}", file=sys.stderr)
        mcp.run(transport="streamable-http")