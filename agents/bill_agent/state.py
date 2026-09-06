from typing import TypedDict, Optional, List

class BillState(TypedDict):
    """记账Agent私有状态，独立生命周期，不与全局共享"""
    session_id: str
    task_id: str
    raw_segments: List[str]
    city: Optional[str]
    month: Optional[str]
    result: Optional[str]