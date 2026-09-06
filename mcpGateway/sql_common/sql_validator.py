def get_sql_operation(sql_raw: str) -> tuple[str, str | None]:
    """
    解析SQL语句类型
    返回 (操作标识, 危险类型None/CREATE/DROP等)
    操作标识：select / insert / update / delete / danger / unknown
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