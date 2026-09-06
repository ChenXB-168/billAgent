import pytest
from memory.short_memory import ShortSessionMemory, get_session_memory, SESSION_MEM


def test_memory_create_and_add():
    """新建会话，添加消息，读取正常"""
    sid = "test_short_001"
    mem = get_session_memory(sid)
    
    mem.add_msg("user", "你好")
    mem.add_msg("agent", "你好呀")
    
    history = mem.get_history()
    assert "user: 你好" in history
    assert "agent: 你好呀" in history


def test_memory_limit_truncation():
    """超过限制条数，只返回最近N轮"""
    sid = "test_short_002"
    mem = get_session_memory(sid)
    
    # 写入10轮对话
    for i in range(10):
        mem.add_msg("user", f"问题{i}")
        mem.add_msg("agent", f"回答{i}")
    
    # 只取最近3轮
    history = mem.get_history(limit=3)
    lines = [l for l in history.strip().split("\n") if l]
    
    # 3轮 = 6行（user+agent）
    assert len(lines) == 6
    # 最新的内容在最后
    assert "问题9" in history
    assert "问题0" not in history


def test_memory_session_isolation():
    """不同会话记忆完全隔离"""
    sid1 = "mem_A"
    sid2 = "mem_B"
    
    mem1 = get_session_memory(sid1)
    mem2 = get_session_memory(sid2)
    
    mem1.add_msg("user", "我是用户A")
    mem2.add_msg("user", "我是用户B")
    
    hist1 = mem1.get_history()
    hist2 = mem2.get_history()
    
    assert "用户A" in hist1
    assert "用户B" not in hist1
    assert "用户B" in hist2
    assert "用户A" not in hist2


def test_memory_clear():
    """清空记忆后无历史记录"""
    sid = "test_short_003"
    mem = get_session_memory(sid)
    mem.add_msg("user", "测试内容")
    
    mem.clear()
    history = mem.get_history()
    assert history.strip() == ""


def test_get_session_memory_singleton():
    """同一会话ID获取到同一个记忆实例"""
    sid = "test_short_004"
    mem1 = get_session_memory(sid)
    mem2 = get_session_memory(sid)
    
    mem1.add_msg("user", "单例测试")
    hist2 = mem2.get_history()
    assert "单例测试" in hist2


def test_session_mem_global_storage():
    """全局存储字典结构正确"""
    sid = "test_short_005"
    get_session_memory(sid)
    assert sid in SESSION_MEM
    assert isinstance(SESSION_MEM[sid], list)