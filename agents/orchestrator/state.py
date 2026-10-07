from typing import TypedDict, Optional, List, Dict, Any


class OrchState(TypedDict):
    """
    函数/类功能与逻辑描述：
        编排调度 Agent 的全局状态定义（TypedDict），是编排图四个节点
        （plan_node → dispatch_node → wait_result_node → dispatch_node 循环 → collect_node）
        之间唯一的数据载体。dispatch_node 每轮挑一个就绪任务投递给子 Agent，
        wait_result_node 领取结果并写回 agent_result_cache，collect_node 汇总生成 final_reply。
        dispatch_round 是防死循环的核心字段，每轮派发自增。
    构造入参说明：
        无（TypedDict 无语义构造器，实例由 dict 字面量构造；调用方至少需提供
        user_input 与 session_id，其余字段可留给各节点补齐）。
    返回值说明：
        无（类型定义，不是可调用对象；运行时表现为普通 dict）。
    """
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
    cancelled: bool                  # M16/M19：本轮已终止（不再派发新波次、且 collect 不产出业务回复）
    # M19（D20）：终止**原因**——`cancelled` 只说"终止了"，本字段说"为什么终止"，供 collect 出对应回执。
    #   取值：None（未终止）| "user_cancel"（M16 用户取消）| "task_timeout"（子任务超时）| "task_error"（子任务异常）
    #   ★为什么与 cancelled 并存而非替换：cancelled 已被三处节点判断复用（dispatch/wait/collect），
    #     新增字段可保证既有判断零改动、只增量补"原因"维度。
    abort_reason: Optional[str]

    current_sub_agent_struct: Optional[Dict[str, Any]]   # 当前子Agent解析后结构化数据
    agent_result_cache: Dict[str, Dict[str, Any]]       # 会话缓存 key=agent_type(bill_agent)
