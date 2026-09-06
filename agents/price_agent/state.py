from typing import TypedDict, List, Optional, Dict, Any

class PriceState(TypedDict):
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