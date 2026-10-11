from typing import TypedDict, Optional, List, Dict


class BillState(TypedDict):
    """
    函数/类功能与逻辑描述：
        记账 Agent 的私有状态定义（TypedDict），独立生命周期，不与全局状态共享。
        由编排器经 A2A 投递的任务载荷构造，仅在 bill_agent 子图内流转：
        execute_node 读取原始片段并完成抽取/落库，reply_node 据此生成回传结果。
        字段与 A2A 消息路由（task_id）及前端会话（session_id）一一对应。
    构造入参说明：
        无（TypedDict 无语义构造器，实例由 dict 字面量直接构造，字段按需填充）。
    返回值说明：
        无（类型定义，不是可调用对象；运行时表现为普通 dict）。
    """
    session_id: str                  # 会话唯一标识，与 A2A 消息、短期记忆共用一个口径
    task_id: str                     # 单次任务标识，A2A 消息路由匹配使用
    raw_segments: List[str]          # 待记账的原始文本片段（用户原话切分后的列表）
    operate_sub_type: str            # 操作类型 add/edit/delete（M15：orchestrator 判定后下发）。
                                     # 用途有二：① 决定子层工具可见面（op:* tags 过滤）；
                                     # ② 放宽"必须有金额数字"的前置拦截（edit/delete 常无新金额）。
    session_memory: Optional[Dict]   # 会话记忆快照（history_text + draft_store，M15 补齐接线：
                                     # dispatch 早已下发、worker 此前丢弃）。★读面用于删除二次确认
                                     # 的草稿读取；**写/清**仍须经 get_session_memory(session_id)
                                     # 的 set_draft/clear_draft——export_all 是浅拷贝，就地改不会写回。
    city: Optional[str]              # 城市名，供物价对标取基准均价用；未识别到时为 None
    month: Optional[str]             # 记账所属月份（YYYY-MM）；未指定时由执行节点按当天推导
    result: Optional[str]            # 本节点生成的最终回传文本（reply_node 写入）
