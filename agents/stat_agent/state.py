from typing import TypedDict, Optional, Dict, Any


class StatState(TypedDict):
    """
    函数/类功能与逻辑描述：
        统计 Agent 的私有状态定义（TypedDict），独立生命周期，不与全局状态共享。
        流转链路为 parse_query_node（自然语言 → 查询 IR）→ execute_query_node（执行取数）
        → verify_query_node（D9 Reflection 语义自检，可回退重生成）→ reply_node（回传结果）。
        verify_* 三个字段承载自检回路：非空反馈触发重解析，次数受 MAX_VERIFY_ATTEMPTS=2 约束，
        决策字段直接驱动图中 verify 节点的条件边。
    构造入参说明：
        无（TypedDict 无语义构造器，实例由 dict 字面量构造；A2A 投递时仅填 session_id /
        task_id / user_input / session_memory 等必需项，其余字段由各节点按需补齐）。
    返回值说明：
        无（类型定义，不是可调用对象；运行时表现为普通 dict）。
    """
    session_id: str
    task_id: str               # 单次任务标识，A2A消息路由匹配使用
    user_input: str
    session_memory: Dict[str, Any]
    target_control: Optional[Dict]
    raw_ir: Optional[Dict]
    final_struct: Optional[Dict]
    # D9 Reflection（M10）：结果自检相关字段
    verify_feedback: Optional[str]   # 上轮校验反馈（非空时 parse 注入修正提示）
    verify_attempt: int              # 已反馈重生成次数（上限 MAX_VERIFY_ATTEMPTS=2）
    verify_decision: str             # verify 路由决策：retry=回 parse 重生成 / reply=发送结果
