from memory.short_memory import get_session_memory


def test_memory_add_and_get():
    """
    函数功能与逻辑描述：
        验证短期记忆的写入与读取：取会话 mem_test_001 的门面实例，依次追加
        ("user","你好") 与 ("agent","你好呀") 两条消息，断言 get_history()（默认 3 轮裁剪）
        返回的文本中分别含 "user: 你好" 与 "agent: 你好呀"。消息为纯内存全局字典，
        本用例不做清理（重复运行会继续追加，不影响断言）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = "mem_test_001"
    mem = get_session_memory(sid)

    mem.add_msg("user", "你好")
    mem.add_msg("agent", "你好呀")

    history = mem.get_history()
    assert "user: 你好" in history
    assert "agent: 你好呀" in history


def test_memory_session_isolation():
    """
    函数功能与逻辑描述：
        验证会话隔离：分别在 mem_A 与 mem_B 两个会话中各写入一条用户消息，断言各自
        get_history() 只含本会话内容、不含另一会话内容（A 见 A 不见 B，B 见 B 不见 A），
        证明短期记忆按 session_id 分桶、互不串扰。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证滑动窗口裁剪：向会话 mem_test_002 连续写入 10 轮（每轮 user + agent 共 20 条消息），
        再以 get_history(limit=6) 读取，断言非空行数 ≤ 12——即只返回最近 6 轮
        （msg_max = limit * 2 = 12 条消息，与 add_msg 的写入条数对应）。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = "mem_test_002"
    mem = get_session_memory(sid)

    for i in range(10):
        mem.add_msg("user", f"第{i}轮")
        mem.add_msg("agent", f"回复{i}")

    history = mem.get_history(limit=6)
    lines = [l for l in history.strip().split("\n") if l]
    # 6轮对应12行（user+agent各一行）
    assert len(lines) <= 12