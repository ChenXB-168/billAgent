# ==============================================
# 短期会话记忆单元测试：消息读写、条数截断、会话隔离、清空与单例语义
# ==============================================
import pytest
from memory.short_memory import ShortSessionMemory, get_session_memory, SESSION_MEM


def test_memory_create_and_add():
    """
    函数功能与逻辑描述：
        验证会话记忆的基础写入与读取：新建会话后追加 user/agent 两条消息，
        get_history() 返回的文本按「role: content」格式包含这两条内容。
        覆盖场景：空会话 → 首次写入 → 读取回显。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = "test_short_001"
    mem = get_session_memory(sid)

    mem.add_msg("user", "你好")
    mem.add_msg("agent", "你好呀")

    history = mem.get_history()
    assert "user: 你好" in history
    assert "agent: 你好呀" in history


def test_memory_limit_truncation():
    """
    函数功能与逻辑描述：
        验证超限截断：写入 10 轮（20 条）消息后，get_history(limit=3) 只返回最近 3 轮，
        即 6 行（每轮 user+agent 两条）；最早的「问题0」被滑出，最新的「问题9」保留。
        覆盖场景：历史条数超过 limit 时的滑窗保留最近。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证不同会话的记忆互不串扰：分别向两个会话写入不同内容后，
        各自 get_history() 只含本会话内容，不含对方内容。
        覆盖场景：多会话并存时的读写隔离。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证清空语义：写入一条消息后调用 clear()，get_history() 返回空白文本。
        覆盖场景：会话结束清理后的历史为空。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = "test_short_003"
    mem = get_session_memory(sid)
    mem.add_msg("user", "测试内容")

    mem.clear()
    history = mem.get_history()
    assert history.strip() == ""


def test_get_session_memory_singleton():
    """
    函数功能与逻辑描述：
        验证同一会话 ID 返回同一记忆实例：mem1 写入的消息，通过 mem2 读取可见，
        说明 get_session_memory 复用同一底层存储（会话级单例语义）。
        覆盖场景：同一 sid 多次获取实例后数据共享。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = "test_short_004"
    mem1 = get_session_memory(sid)
    mem2 = get_session_memory(sid)

    mem1.add_msg("user", "单例测试")
    hist2 = mem2.get_history()
    assert "单例测试" in hist2


def test_session_mem_global_storage():
    """
    函数功能与逻辑描述：
        验证全局存储字典结构：get_session_memory(sid) 后 sid 出现在模块级 SESSION_MEM 中，
        且对应值为 list（消息列表）。
        覆盖场景：会话注册即写入全局存储结构。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    sid = "test_short_005"
    get_session_memory(sid)
    assert sid in SESSION_MEM
    assert isinstance(SESSION_MEM[sid], list)
