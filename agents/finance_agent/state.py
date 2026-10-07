from typing import TypedDict, Optional, Dict, Any, List


class FinanceState(TypedDict, total=False):
    """
    函数/类功能与逻辑描述：
        理财分析 Agent 的私有状态定义（TypedDict），独立生命周期，不与全局状态共享。
        流转链路为 build_prompt_node（组装提示词与事实底稿）→ execute_finance_node
        （调专用模型 + D16 数字白名单闸门）→ verify_query_node（D9 语义自检，可带 feedback 回
        execute 重生成，上限 MAX_VERIFY_ATTEMPTS=2）→ reply_node（序列化后经 A2A 回传）。
        声明为 total=False，即所有键都可缺省——各节点按需读写，缺键由 state.get 兜底。
        字段分类（对齐 docx 规范 + 实际 nodes.py 下标读写方式）：
        1. A2A 通信必填标识：session_id / task_id
        2. Orchestrator 下发原始任务数据：raw_segments / operate_sub_type / session_memory
        3. 前置依赖数据（主 Agent 缓存下发）：target_control / bill_result / stat_result /
           price_result / habit_data / user_docs
        4. LLM 中间缓存：user_query / llm_input_prompt / llm_raw_answer
        5. 防幻觉与自检（D16/D9 · M11）：fact_sheet / verify_feedback / verify_attempt / verify_decision
        6. 统一输出层：final_struct / error_stack
    构造入参说明：
        无（TypedDict 无语义构造器，实例由 dict 字面量构造；外层 worker 至少注入
        session_id、task_id、raw_segments，其余字段由各节点补齐）。
    返回值说明：
        无（类型定义，不是可调用对象；运行时表现为普通 dict）。
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
