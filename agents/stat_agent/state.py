from typing import TypedDict, Optional, Dict, Any

class StatState(TypedDict):
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