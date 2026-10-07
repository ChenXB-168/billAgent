# ==============================================
# MCP SQL 通道公共逻辑 - 鉴权 + 危险语句拦截 + 加锁执行
# 各 sql_server/*.py 只负责声明 FastMCP 与传库标识，本模块承载全部安全与执行逻辑
# ==============================================
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
    """
    函数功能与逻辑描述：
        所有 SQL MCP 服务的公共执行体，按「分类 → 危险拦截 → 鉴权 → 加锁执行 → 审计」五步串行：
        1) 用 get_sql_operation 判定 SQL 类型；2) 命中 DDL 等危险词直接拒绝；
        3) unknown 类型（如以注释开头的 SQL）一并拒绝；4) 按 op_type 映射到 DBPerm.BILL_READ
        （select）或 DBPerm.BILL_WRITE（insert/update/delete）做权限校验，不足即拒绝；
        5) 通过后仅在数据库 IO 临界区持有 db_lock，并把同步 SQLite 调用丢进线程池执行，
        避免阻塞事件循环（select 走 query_sql，写走 execute_sql）。
        全程所有分支（拒绝、成功、异常）都会写审计日志；不抛异常，失败一律转成 TextContent 文本返回，
        以保证 MCP 边界上调用方拿到的是可读文案而非栈信息。
    入参说明：
        agent_id (str)：调用方 Agent 标识，用于查权限表并写入审计。
        db_identifier (str)：库标识，仅用于拼装审计标签与错误提示文案（当前实际传值 "bill"），
            不参与路由，真正的库对象由 db_inst 给出。
        sql (str)：待执行的 SQL 原文。
        params (list | None)：SQL 绑定参数；为 None 或空列表时不传参执行（由 DatabaseCRUD 内部判空）。
        db_inst (DatabaseCRUD)：目标库连接实例，由各 server 模块注入（如 utils.common.db）。
    返回值说明：
        list[TextContent]：长度为 1 的文本结果列表，全部为字符串化内容
            - 拒绝：文本以「操作拒绝：」或「权限不足：」开头（如 "操作拒绝：禁止执行DROP等DDL语句"）。
            - 查询成功：文本为查询结果列表的 str() 形态（如 "[{'amount': 100}]"），
              需由上层用 utils.common.parse_sql_result 解析。
            - 写成功/失败：文本分别为 "SQL执行成功" / "SQL执行失败"；异常时为 "SQL执行失败：{异常}"。
    """
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