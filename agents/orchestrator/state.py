from typing import TypedDict, Optional, List, Dict, Any

class OrchState(TypedDict):
    """编排调度Agent全局状态"""
    user_input: str                  # 用户原始输入
    session_id: str                  # 会话唯一ID
    session_history: Optional[str]   # 入口兼容占位（历史遗留，运行期不跨节点传递；dispatch 每轮自行读会话内存）

    task_plan: Dict[str, dict]       # 任务计划DAG：任务名→状态、依赖、参数
    current_task: Optional[str]      # 当前执行中的任务
    all_task_results: List[str]      # 所有任务结果【⚠️注意：这里存旧文本，后续逐步废弃】
    final_reply: Optional[str]       # 最终汇总回复

    error_msg: Optional[str]         # 全局异常
    current_agent: str               # 当前执行Agent标识
    dispatch_round: int              # 调度轮次计数（防死循环）

    # ============ 新增下面两行 ============
    current_sub_agent_struct: Optional[Dict[str, Any]]   # 当前子Agent解析后结构化数据
    agent_result_cache: Dict[str, Dict[str, Any]]       # 会话缓存 key=agent_type(bill_agent)