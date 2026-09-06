from mcp.types import TextContent
from mcpGateway.rbac_config import get_agent_permission, DBPerm
from mcpGateway.audit_recorder import write_audit_log
from mcpGateway.sql_common.sql_validator import get_sql_operation
from utils.common import DatabaseCRUD, db_lock
import anyio
from functools import partial

async def run_sql_logic(
    agent_id: str,
    db_identifier: str,
    sql: str,
    params: list | None,
    db_inst: DatabaseCRUD
):
    agent_perms = get_agent_permission(agent_id)
    op_type, danger_tag = get_sql_operation(sql)

    # 拦截DDL危险语句（锁外并发执行）
    if danger_tag is not None:
        write_audit_log(agent_id, f"{db_identifier}_{op_type}", sql, False)
        return [TextContent(type="text", text=f"操作拒绝：禁止执行{danger_tag}等DDL语句")]

    # 无法识别的SQL（锁外并发执行）
    if op_type == "unknown":
        write_audit_log(agent_id, f"{db_identifier}_{op_type}", sql, False)
        return [TextContent(type="text", text="操作拒绝：不支持该类型SQL语句")]

    # 粗粒度读写权限校验（锁外并发执行）
    if op_type == "select":
        required_perm = DBPerm.BILL_READ
        perm_desc = "读取"
    else:
        required_perm = DBPerm.BILL_WRITE
        perm_desc = "写入"

    if required_perm not in agent_perms:
        write_audit_log(agent_id, f"{db_identifier}_{op_type}", sql, False)
        return [TextContent(type="text", text=f"权限不足：当前Agent无{db_identifier}库{perm_desc}操作权限")]

    try:
        # 仅数据库IO加锁，粒度极小；同步SQL放入线程，不阻塞协程循环
        async with db_lock:
            if op_type == "select":
                func = partial(db_inst.query_sql, sql, params)
                data = await anyio.to_thread.run_sync(func)
                write_audit_log(agent_id, f"{db_identifier}_{op_type}", sql, True)
                return [TextContent(type="text", text=str(data))]
            else:
                func = partial(db_inst.execute_sql, sql, params)
                ok = await anyio.to_thread.run_sync(func)
                if ok:
                    write_audit_log(agent_id, f"{db_identifier}_{op_type}", sql, True)
                    return [TextContent(type="text", text="SQL执行成功")]
                else:
                    return [TextContent(type="text", text="SQL执行失败")]
    except Exception as e:
        err_msg = f"{sql} 异常信息：{str(e)}"
        write_audit_log(agent_id, f"{db_identifier}_{op_type}", err_msg, False)
        return [TextContent(type="text", text=f"SQL执行失败：{str(e)}")]