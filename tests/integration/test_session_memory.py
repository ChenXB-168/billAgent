from memory.short_memory import get_session_memory


def test_memory_add_and_get():
    """记忆写入与读取：历史记录正确"""
    sid = "mem_test_001"
    mem = get_session_memory(sid)

    mem.add_msg("user", "你好")
    mem.add_msg("agent", "你好呀")

    history = mem.get_history()
    assert "user: 你好" in history
    assert "agent: 你好呀" in history


def test_memory_session_isolation():
    """会话隔离：两个会话记忆互不影响"""
    sid1 = "mem_A"
    sid2 = "mem_B"

    mem1 = get_session_memory(sid1)
    mem2 = get_session_memory(sid2)

    mem1.add_msg("user", "我是A用户")
    mem2.add_msg("user", "我是B用户")

    hist1 = mem1.get_history()
    hist2 = mem2.get_history()

    assert "A用户" in hist1
    assert "B用户" not in hist1
    assert "B用户" in hist2
    assert "A用户" not in hist2


def test_memory_limit():
    """记忆限制：只返回最近N轮"""
    sid = "mem_test_002"
    mem = get_session_memory(sid)

    for i in range(10):
        mem.add_msg("user", f"第{i}轮")
        mem.add_msg("agent", f"回复{i}")

    history = mem.get_history(limit=6)
    lines = [l for l in history.strip().split("\n") if l]
    # 6轮对应12行（user+agent各一行）
    assert len(lines) <= 12