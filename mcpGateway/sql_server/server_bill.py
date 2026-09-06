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
    """账单数据库统一SQL执行入口"""
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