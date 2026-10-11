def get_sql_operation(sql_raw: str) -> tuple[str, str | None]:
    """
    函数功能与逻辑描述：
        SQL 类型分级器，为 MCP SQL 通道提供「先分类、后鉴权」的第一道闸门。
        实现为纯前缀匹配（先大写化再比较）：命中危险操作名 → danger；否则识别四类常规操作；
        其余一律 unknown。判定顺序上危险词优先，确保 DDL 不会因前缀正则宽松而漏网。
        安全侧失败设计：SQL 注释开头（如 `/* hint */ SELECT`）会落到 unknown 而被拒绝，
        宁可误杀不可放过；同理本函数不解析语义，不做语法校验（真正的语法错误由 SQLite 报错）。
    入参说明：
        sql_raw (str)：待判定的 SQL 原文，允许首尾存在空白。
    返回值说明：
        tuple[str, str | None]：二元组 (操作标识, 危险类型)
            - 操作标识 (str)：select / insert / update / delete / danger / unknown 之一。
            - 危险类型 (str | None)：命中危险操作时返回命中的关键字（CREATE/DROP/ALTER/TRUNCATE/
              GRANT/RENAME/ATTACH/DETACH/VACUUM/PRAGMA/REINDEX），否则为 None。
    """
    sql = sql_raw.strip().upper()
    # ★M4（D8 危险操作白名单）：除建表/删表/改表类 DDL，还须拦——
    #   ATTACH/DETACH（可把外部 db 挂进会话）、VACUUM（整库重写）、
    #   PRAGMA（写类 pragma 可改 journal 模式等）、REINDEX（重建索引）。
    # 判定方式为前缀匹配，注释开头（`/* hint */ SELECT`）会落到 unknown → 拒绝（安全侧失败，可接受）。
    danger_ops = ("CREATE", "DROP", "ALTER", "TRUNCATE", "GRANT", "RENAME",
                  "ATTACH", "DETACH", "VACUUM", "PRAGMA", "REINDEX")
    for op in danger_ops:
        if sql.startswith(op):
            return "danger", op
    if sql.startswith("SELECT"):
        return "select", None
    elif sql.startswith("INSERT"):
        return "insert", None
    elif sql.startswith("UPDATE"):
        return "update", None
    elif sql.startswith("DELETE"):
        return "delete", None
    else:
        return "unknown", None