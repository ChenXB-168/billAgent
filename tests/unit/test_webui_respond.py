# -*- coding: utf-8 -*-
"""`webUI/app.py` 单元测试 —— WebUI 主流程（`respond`）与会话/信号/端口 helper。

★为什么必须补这一层：`webUI/app.py`（980 行）是**用户交互的唯一入口**，
此前只测了它的 `bill_panel` 子模块，**`respond()` 主流程完全未测** ——
而它承载了「空输入短路 / 记忆落库 / 编排图驱动 / 异常兜底 / 面板刷新」整条链路。

★测试策略：**打桩全部外部依赖**（编排图 / 短期记忆 / 面板 / M7 埋点），
只验证 `respond` 自身的**编排与边界行为**，不启动 gradio、不连 MCP、不写真库。
`_loop` 用真实事件循环（`run_until_complete` 是 `respond` 的既有契约）。

覆盖矩阵：

| 用例 | 验证点 |
|---|---|
| `test_empty_input_short_circuits` | 空输入（None / "" / 空白）→ 4 元组短路，**不落库、不调 agent、不刷面板** |
| `test_normal_path_appends_history_and_returns_4tuple` | 正常路径 → 历史 +2 条、返回 4 元组、面板已刷新 |
| `test_user_message_persisted_to_short_memory` | 用户提问先落短期记忆（与 collect_node 成对） |
| `test_init_state_shape_passed_to_graph` | ★传给编排图的 state 字段完整（含 M16/M19 的 `cancelled` / `abort_reason`） |
| `test_agent_exception_becomes_reply_not_raise` | Agent 异常 → 转「【系统异常】…」文案，**不向上抛**（页面不崩） |
| `test_none_final_reply_falls_back` | `final_reply` 为 None → 「无返回结果」 |
| `test_history_none_treated_as_empty` | `history=None` 按空列表处理 |
| `test_new_session_rotates_session_id` | 「新会话」换号（切断短期记忆分区） |
| `test_port_in_use_detects_listener` | 端口探测：有监听 → True，空闲 → False |
| `test_install_signal_handlers_registers_sigterm` | 信号注册：SIGTERM/SIGINT 均被注册，且处理函数抛 KeyboardInterrupt |
"""
import asyncio
from contextlib import contextmanager

import pytest

import webUI.app as app
from utils import tracer


class _FakeGraph:
    """编排图替身：记录收到的 state，可配置返回值或抛异常。"""

    def __init__(self):
        self.calls: list[dict] = []
        self.result: dict = {"final_reply": "好的，已记下"}
        self.exc: Exception | None = None

    async def ainvoke(self, state):
        self.calls.append(state)
        if self.exc is not None:
            raise self.exc
        return self.result


class _FakeMemory:
    """短期记忆替身：记录写入的 (role, content)。"""

    def __init__(self):
        self.msgs: list[tuple] = []

    def add_msg(self, role, content):
        self.msgs.append((role, content))


@pytest.fixture()
def app_env(monkeypatch):
    """
    函数功能与逻辑描述：
        隔离 `webUI/app.py` 的全部外部依赖，使 `respond` 可在**无 gradio 服务、
        无 MCP、无真实编排图**的前提下被直接调用：
        独立事件循环（`respond` 走 `_loop.run_until_complete`）、假编排图、
        假短期记忆、假面板刷新、并把 M7 埋点（`run_scope` / `span`）替换为 no-op 上下文
        —— 后者避免测试向生产 `bill.db` 的 span 表写数据。
    入参说明：
        monkeypatch：pytest 内置 fixture，用例结束自动回滚全部替换。
    返回值说明：
        tuple：(webUI.app 模块, 假编排图, 假短期记忆)。
    """
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(app, "_loop", loop)

    graph = _FakeGraph()
    monkeypatch.setattr(app, "AGENT_GRAPH", {"orchestrator_agent": graph})

    mem = _FakeMemory()
    monkeypatch.setattr(app, "get_session_memory", lambda session_id: mem)
    monkeypatch.setattr(app, "panel_updates", lambda: ("<panel-html>", {"choices": ["a"]}))

    @contextmanager
    def _noop_ctx(*args, **kwargs):
        yield

    monkeypatch.setattr(tracer, "run_scope", _noop_ctx)
    monkeypatch.setattr(tracer, "span", _noop_ctx)

    yield app, graph, mem
    loop.close()


# ===================== ① 空输入短路 =====================
@pytest.mark.parametrize("bad_input", [None, "", "   ", "\n\t "])
def test_empty_input_short_circuits(app_env, bad_input):
    """
    函数功能与逻辑描述：
        验证空输入边界（None / 空串 / 纯空白）：直接短路返回 4 元组，
        **不落库、不调编排图、不刷新面板**（避免无谓的库查询与一次完整 Agent 调用）。
        返回的第 3/4 项须是 gr.update()（本次不刷面板）。
    入参说明：
        app_env：依赖隔离 fixture。
        bad_input：参数化的各类空输入。
    返回值说明：
        无（断言通过即成功）。
    """
    module, graph, mem = app_env
    history, msg, panel, pick = module.respond(bad_input, [])

    assert history == [] and msg == ""
    assert graph.calls == [], "空输入不得触发编排图"
    assert mem.msgs == [], "空输入不得写入短期记忆"


# ===================== ② 正常路径 =====================
def test_normal_path_appends_history_and_returns_4tuple(app_env):
    """正常路径：历史追加 user/assistant 两条；返回 4 元组；面板与选择器被刷新。"""
    module, graph, _mem = app_env
    history = [{"role": "user", "content": "之前的话"}]

    out = module.respond("记午饭30元", history)

    assert isinstance(out, tuple) and len(out) == 4
    assert out[1] == "", "第 2 项应为空串（清空输入框）"
    assert out[2] == "<panel-html>", "第 3 项应为面板 HTML（每轮刷新）"
    assert out[3] == {"choices": ["a"]}, "第 4 项应为目标选择器 gr.update"
    assert len(out[0]) == 3, "应在原有 1 条基础上追加 user/assistant 两条"
    assert out[0][-2] == {"role": "user", "content": "记午饭30元"}
    assert out[0][-1] == {"role": "assistant", "content": "好的，已记下"}


def test_user_message_persisted_to_short_memory(app_env):
    """用户提问须先落短期记忆（与 collect_node 写入的 Agent 回复成对，构成完整上下文）。"""
    module, _graph, mem = app_env
    module.respond("记午饭30元", [])
    assert mem.msgs == [("user", "记午饭30元")], f"落库消息不符：{mem.msgs}"


def test_init_state_shape_passed_to_graph(app_env):
    """
    函数功能与逻辑描述：
        ★契约：传给编排图的 state 必须是**完整字段集** —— 尤其 M16 的 `cancelled`
        与 M19 的 `abort_reason`：入口统一初始化，不依赖节点侧兜底默认值
        （漏一个字段会导致终止语义在节点里读到 None 而行为漂移）。
    入参说明：
        app_env：依赖隔离 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    module, graph, _mem = app_env
    module.respond("记午饭30元", [])

    assert len(graph.calls) == 1
    state = graph.calls[0]
    assert state["user_input"] == "记午饭30元"
    assert state["session_id"] == module.SESSION_ID
    assert state["cancelled"] is False
    assert state["abort_reason"] is None
    assert state["current_agent"] == "orchestrator_agent"
    assert {"task_plan", "current_task", "all_task_results", "final_reply",
            "error_msg", "session_history"} <= set(state)


# ===================== ③ 异常兜底 =====================
def test_agent_exception_becomes_reply_not_raise(app_env):
    """
    函数功能与逻辑描述：
        ★可靠性契约：编排图抛异常时，`respond` 必须**捕获并转成文案回复**，
        绝不向上抛 —— 否则 gradio 事件回调报错、页面崩溃。
    入参说明：
        app_env：依赖隔离 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    module, graph, _mem = app_env
    graph.exc = RuntimeError("MCP 通道不可用")

    history, _msg, _panel, _pick = module.respond("记午饭30元", [])

    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "【系统异常】执行失败：MCP 通道不可用"
    assert len(history) == 2, "异常时仍应补齐 user + assistant 成对消息"


def test_none_final_reply_falls_back(app_env):
    """边界：编排图返回的 final_reply 为 None（或空）→ 回退「无返回结果」文案。"""
    module, graph, _mem = app_env
    graph.result = {"final_reply": None}
    history, _msg, _p, _k = module.respond("你好", [])
    assert history[-1]["content"] == "无返回结果"


def test_history_none_treated_as_empty(app_env):
    """边界：history 传 None → 按空列表处理（不抛 AttributeError）。"""
    module, _graph, _mem = app_env
    history, _msg, _p, _k = module.respond("你好", None)
    assert isinstance(history, list) and len(history) == 2


# ===================== ④ 会话与 helper =====================
def test_new_session_rotates_session_id(app_env):
    """
    函数功能与逻辑描述：
        「新会话」按钮只做一件事：**换 SESSION_ID** ——
        短期记忆（对话历史 / 三类草稿 / 追问计数）在 `memory/short_memory.py` 中按 session_id
        分区存放，换号即等价于全新会话，无需重启任何后台组件。
    入参说明：
        app_env：依赖隔离 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    module, _graph, _mem = app_env
    before = module.SESSION_ID
    out = module.new_session()
    assert module.SESSION_ID != before, "新会话未切换 SESSION_ID"
    assert out == [], "新会话应返回空历史以清空 chatbot"


def test_port_in_use_detects_listener(tmp_path):
    """`_port_in_use`：有监听返回 True，空闲端口返回 False（单实例保护依赖它）。"""
    import socket

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))          # 由 OS 分配空闲端口
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert app._port_in_use(port) is True
    finally:
        srv.close()
    assert app._port_in_use(port) is False, "监听关闭后应判定为空闲"


def test_install_signal_handlers_registers_sigterm(monkeypatch):
    """
    函数功能与逻辑描述：
        ★M23（D26）L2 的行为契约：`_install_signal_handlers` 必须为 **SIGTERM 与 SIGINT
        都注册处理函数**，且该处理函数的行为是**抛 KeyboardInterrupt**（复用既有冒泡路径，
        从而触发 `main()` 的 `finally: shutdown()`）。
        实现方式：打桩 `signal.signal` 捕获注册行为（**不真注册**，避免污染 pytest 进程）。
    入参说明：
        monkeypatch：pytest 内置 fixture。
    返回值说明：
        无（断言通过即成功）。
    """
    import signal as _signal

    registered: dict = {}

    def _fake_signal(sig, handler):
        registered[sig] = handler

    monkeypatch.setattr(_signal, "signal", _fake_signal)
    app._install_signal_handlers()

    assert _signal.SIGTERM in registered, "未注册 SIGTERM → kill -15 / docker stop 下 finally 不执行"
    assert _signal.SIGINT in registered, "未注册 SIGINT"
    with pytest.raises(KeyboardInterrupt):
        registered[_signal.SIGTERM](_signal.SIGTERM, None)


def test_install_signal_handlers_degrades_on_failure(monkeypatch):
    """边界：`signal.signal` 抛 ValueError/OSError（非主线程等）→ 只打告警，**不向上抛**。"""
    import signal as _signal

    def _boom(sig, handler):
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(_signal, "signal", _boom)
    app._install_signal_handlers()          # 不得抛异常
