from typing import TypedDict, Optional, Dict, Any, List


class FinanceState(TypedDict, total=False):
    """理财分析Agent私有状态，独立生命周期，不与全局共享

    字段分类（对齐 docx 规范 + 实际 nodes.py 下标读写方式）：
    1. A2A通信必填标识：session_id / task_id
    2. Orchestrator下发原始任务数据：raw_segments / operate_sub_type / session_memory
    3. 前置依赖数据（主Agent缓存下发）：target_control / bill_result / stat_result / price_result
    4. LLM中间缓存：user_query / llm_input_prompt / llm_raw_answer
    5. 统一输出层：final_struct / error_stack
    """
    # A2A通信必填标识
    session_id: str
    task_id: str

    # Orchestrator 下发原始任务数据
    raw_segments: List[str]
    operate_sub_type: str
    session_memory: Optional[Dict[str, Any]]

    # 前置依赖数据（主Agent缓存下发）
    target_control: Optional[Dict[str, Any]]
    bill_result: Optional[Dict[str, Any]]
    stat_result: Optional[Dict[str, Any]]
    price_result: Optional[Dict[str, Any]]
    # 历史消费习惯（近N月聚合：品类/笔数/金额/均额，来自 monthly_habit 表，主Agent下发）
    habit_data: Optional[List[Dict[str, Any]]]
    # M8：用户资料检索结果（`memory/user_docs.py` search_user_docs，主Agent下发；
    #     list[dict]: {"doc_id","text","score"}，检索失败/无结果为空列表）
    user_docs: Optional[List[Dict[str, Any]]]

    # 内部中间缓存
    user_query: Optional[str]
    llm_input_prompt: Optional[str]
    llm_raw_answer: Optional[str]

    # D16 防幻觉（M11）：确定性规则事实底稿（预算/当月支出/余额/百分比等，白名单唯一来源）
    fact_sheet: Optional[Dict[str, Any]]
    # D9 Reflection（M11 复用 M10 模式）：输出语义自检
    verify_feedback: Optional[str]   # 上轮校验反馈（非空时 execute 注入修正提示）
    verify_attempt: int              # 已反馈重生成次数（上限 MAX_VERIFY_ATTEMPTS=2）
    verify_decision: str             # verify 路由决策：retry=回 execute 重生成 / reply=发送结果

    # 统一输出字段（和price/stat完全对齐）
    final_struct: Optional[Dict[str, Any]]
    error_stack: Optional[str]
