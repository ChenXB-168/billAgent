from typing import TypedDict, List, Optional, Dict, Any

class PriceState(TypedDict):
    """
    函数/类功能与逻辑描述：
        物价对标 Agent 的私有状态定义（TypedDict），独立生命周期，不与全局状态共享。
        流转链路为 execute_node（LLM 抽取消费信息 → 查 city_price 基准均价 → 算溢价率）
        → reply_node（把 final_struct 序列化后经 A2A 回传），两节点间无条件边。
        标识字段由外层 worker 初始化，抽取结果与输出载荷由节点写入。
    构造入参说明：
        无（TypedDict 无语义构造器，实例由 dict 字面量构造；外层 worker 至少注入
        session_id、task_id、raw_segments）。
    返回值说明：
        无（类型定义，不是可调用对象；运行时表现为普通 dict）。
    """
    # A2A路由标识，由外层Worker初始化传入
    session_id: str
    task_id: str
    # 拆分后的消费文本片段
    raw_segments: List[str]
    # LLM提取结果缓存
    extract_info: Optional[Dict[str, Any]]
    # 返回载荷
    final_struct: Optional[Dict[str, Any]]
    # 异常堆栈缓存
    error_stack: Optional[str]