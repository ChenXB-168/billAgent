# -*- coding: utf-8 -*-
"""M15 FC（function calling）通道单元测试：模型接入层 + 网关工具 + 兼容层封装。

★全部用例**不触网、不拉子进程**：provider 层用 monkeypatch 替换请求入口（外部走
  `requests.Session.post`、本地走 `requests.post`，与 `test_model_service.py` 同一套打桩约定），
  网关层 patch `MCPClient._call_server`。

覆盖（对应 `11` §3 M15 的 P2 验收方式 ①②④）：
  ① provider 单测（全 mock）：tools 透传进请求体 / extra_messages 组装顺序 / tool_calls 归一 /
     **有 tool_calls 时空正文不算 EMPTY_CONTENT** / arguments 非法 JSON 容错；
  ② 打桩点同步：本文件同时断言"改动请求入口会立刻击穿打桩"这一点可见（用真实 provider 打桩）；
  ④ `llm.chat_fc` 的 schema 契约：`tools` / `extra_messages` 为可选、无 agent_id / agent_tag；
  ⑤ 兼容层 `call_llm_fc`：JSON 还原为 dict、`FINANCE_ERROR:` 前缀转 RuntimeError、非法 JSON 转 RuntimeError；
  ⑥ Router 原样透传：tools / extra_messages 不得 merge 进 options。
"""
import json

import pytest
import requests
from unittest.mock import AsyncMock, patch

from mcpGateway.client import mcp_client
from mcpGateway.registry import get_registry
from modelService.chat_model import build_messages, normalize_tool_calls
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import OpenAICompatProvider
from modelService.router import ModelRouter

_TOOLS = [{"type": "function",
           "function": {"name": "bill.recent_list",
                        "description": "查最近账单",
                        "parameters": {"type": "object", "properties": {}}}}]


class _FcResp:
    """
    函数/类功能与逻辑描述：
        假外部响应：200 + 指定的 message 体（content / tool_calls 由构造参数决定），
        用于验证 FC 通道的解析与"空正文 + 有工具调用"的判定修正。不触网。
    构造入参说明：
        message (dict)：要放进 choices[0].message 的字典（content / tool_calls）。
    返回值说明：
        无（仅作为 Session.post 的替身对象）。
    """

    status_code = 200

    def __init__(self, message: dict):
        """
        函数功能与逻辑描述：
            保存待返回的 message 体，供 json() 组装成 OpenAI 兼容响应。
        入参说明：
            message (dict)：choices[0].message 的内容。
        返回值说明：
            无（仅赋值实例属性）。
        """
        self._message = message

    def raise_for_status(self):
        """
        函数功能与逻辑描述：
            模拟成功响应的状态检查：固定不抛异常（status_code=200）。
        入参说明：
            无（隐式 self）。
        返回值说明：
            无（直接返回）。
        """
        pass

    def json(self):
        """
        函数功能与逻辑描述：
            组装 OpenAI 兼容响应体：choices[0].message 取构造时给的 message，
            并附加最小 usage 计数，避免埋点解析分支干扰被测逻辑。
        入参说明：
            无（隐式 self）。
        返回值说明：
            dict：含 choices 与 usage 的假响应体。
        """
        return {"choices": [{"message": self._message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class _CapturingLocalResp:
    """
    函数/类功能与逻辑描述：
        假本地响应 + 请求体捕获器：模拟 Ollama /api/chat 返回（message.content + 计数），
        同时把收到的 payload 记到类属性 last_payload，供"tools 是否透传"与"消息顺序"断言。
    构造入参说明：
        无（无构造参数）。
    返回值说明：
        无（仅作为 requests.post 的替身对象）。
    """

    status_code = 200
    last_payload = None

    def raise_for_status(self):
        """
        函数功能与逻辑描述：
            模拟成功响应：固定不抛异常。
        入参说明：
            无（隐式 self）。
        返回值说明：
            无（直接返回）。
        """
        pass

    def json(self):
        """
        函数功能与逻辑描述：
            返回 Ollama 风格的固定响应体（正文 "本地返回" + 两个计数字段）。
        入参说明：
            无（隐式 self）。
        返回值说明：
            dict：含 message 与计数字段的假响应体。
        """
        return {"message": {"content": "本地返回"}, "prompt_eval_count": 1, "eval_count": 1}


# ── 1. 纯函数契约：消息顺序与 tool_calls 归一 ─────────────────────────────
def test_build_messages_order_and_skip_blank_system():
    """
    函数功能与逻辑描述：
        验证**消息顺序契约**（M15 抽公共的目的就是让它只有一处实现）：system → user → extra，
        system 为空白时不写入，extra_messages 中的非法元素被跳过而非抛出。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    msgs = build_messages("sys", "usr", [{"role": "assistant", "content": "a"},
                                         {"role": "tool", "content": "obs"},
                                         "bad-element"])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert [m["role"] for m in build_messages("   ", "usr")] == ["user"]


def test_normalize_tool_calls_parses_json_arguments_and_fills_id():
    """
    函数功能与逻辑描述：
        验证 tool_calls 归一的两种后端形态：OpenAI（arguments 为 JSON 字符串、带 id）与
        Ollama（arguments 已是 dict、无 id，按位置补 `call_{idx}`）。归一后上层无需感知后端差异。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    openai_style = [{"id": "call_1", "type": "function",
                     "function": {"name": "bill.update", "arguments": '{"id": 3, "amount": 50}'}}]
    assert normalize_tool_calls(openai_style) == [
        {"id": "call_1", "name": "bill.update", "arguments": {"id": 3, "amount": 50}}]

    ollama_style = [{"function": {"name": "bill.delete", "arguments": {"id": 7}}}]
    assert normalize_tool_calls(ollama_style) == [
        {"id": "call_0", "name": "bill.delete", "arguments": {"id": 7}}]


def test_normalize_tool_calls_tolerates_bad_arguments_and_entries():
    """
    函数功能与逻辑描述：
        验证归一的容错口径（宁可达上限也不硬失败）：arguments 非法 JSON → 置空 dict（下游会被
        引擎判参数不合规并作为 observation 回填）；无函数名 / 非 dict 元素直接丢弃；None 输入返回空列表。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    assert normalize_tool_calls(None) == []
    assert normalize_tool_calls([{"function": {"arguments": "{not json"}}]) == []
    assert normalize_tool_calls([{"id": "x", "function": {"name": "f", "arguments": "{bad"}}]) == [
        {"id": "x", "name": "f", "arguments": {}}]


# ── 2. Provider 层：tools 透传 + tool_calls 解析 + 空正文判定修正 ───────────
def test_external_provider_passes_tools_and_returns_tool_calls(monkeypatch):
    """
    函数功能与逻辑描述：
        验证外部通道的 FC 关键行为：① `tools` 被写进**请求体顶层**（不是 options 内，
        绝不被 filter_options 过滤）；② 返回的 tool_calls 被归一为 dict 形态；
        ③ ★**正文为空但有 tool_calls 时不得抛 EMPTY_CONTENT**——FC 场景模型本就用工具调用
        代替自然语言回复，若沿用 M14 的"空正文即可重试"判定，会把正常 FC 响应误判为失败并重试。
    入参说明：
        monkeypatch：pytest fixture 注入，把 `requests.Session.post` 换成捕获型假响应。
    返回值说明：
        无（断言通过即用例成功）。
    """
    seen = {}

    def _fake_post(*args, **kwargs):
        """
        函数功能与逻辑描述：
            假请求入口：记录本次 payload 后返回一个"空正文 + 单个 tool_call"的响应。
        入参说明：
            *args：位置参数（忽略）。
            **kwargs：含 json=payload 的关键字参数。
        返回值说明：
            _FcResp：构造好的假响应对象。
        """
        seen.update(kwargs.get("json") or {})
        return _FcResp({"content": "",
                        "tool_calls": [{"id": "call_9", "type": "function",
                                        "function": {"name": "bill.recent_list",
                                                     "arguments": "{}"}}]})

    monkeypatch.setattr(requests.Session, "post", _fake_post)
    prov = OpenAICompatProvider("ut-model")
    res = prov.generate(system="s", user="u", options={"temperature": 0.1},
                        agent_tag="bill_agent", tools=_TOOLS)
    assert seen["tools"] == _TOOLS              # 协议结构在顶层（与 messages 同级）
    assert "options" not in seen                # 外部通道无 options 子层：参数平铺，tools 不可能被塞进白名单
    assert res.text == ""
    assert res.tool_calls == [{"id": "call_9", "name": "bill.recent_list", "arguments": {}}]


def test_external_provider_extra_messages_appended_in_order(monkeypatch):
    """
    函数功能与逻辑描述：
        验证多轮 tool 消息的回填顺序：system → user → extra（assistant tool_calls 在前、
        tool 结果在后）。顺序错乱会让模型看到"先有观察结果、后有调用"的倒序对话，直接破坏循环。
    入参说明：
        monkeypatch：pytest fixture 注入，把 `requests.Session.post` 换成捕获型假响应。
    返回值说明：
        无（断言通过即用例成功）。
    """
    seen = {}

    def _fake_post(*args, **kwargs):
        """
        函数功能与逻辑描述：
            假请求入口：记录 messages 后返回普通正文响应。
        入参说明：
            *args：位置参数（忽略）。
            **kwargs：含 json=payload 的关键字参数。
        返回值说明：
            _FcResp：正文为 "done" 的假响应。
        """
        seen["messages"] = (kwargs.get("json") or {}).get("messages")
        return _FcResp({"content": "done"})

    monkeypatch.setattr(requests.Session, "post", _fake_post)
    extra = [{"role": "assistant", "content": "", "tool_calls": []},
             {"role": "tool", "tool_call_id": "call_9", "content": "[]"}]
    OpenAICompatProvider("ut-model").generate(
        system="sys", user="usr", options={}, agent_tag="bill_agent",
        extra_messages=extra)
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "tool"]


def test_local_provider_passes_tools_into_payload(monkeypatch):
    """
    函数功能与逻辑描述：
        验证本地通道的 tools 透传：`tools` 必须出现在 payload 顶层，而**不在** `options` 内
        （否则会被 filter_options 按契约边界丢弃并告警，通道静默失效）。同时确认不传 tools 时
        请求体里不出现该键——保证既有调用零影响。
    入参说明：
        monkeypatch：pytest fixture 注入，把 `requests.post` 换成捕获型假响应。
    返回值说明：
        无（断言通过即用例成功）。
    """
    captured = {}

    def _fake_post(*args, **kwargs):
        """
        函数功能与逻辑描述：
            假请求入口：记录 payload 后返回固定本地响应。
        入参说明：
            *args：位置参数（忽略）。
            **kwargs：含 json=payload 的关键字参数。
        返回值说明：
            _CapturingLocalResp：假本地响应对象。
        """
        captured.clear()
        captured.update(kwargs.get("json") or {})
        return _CapturingLocalResp()

    monkeypatch.setattr(requests, "post", _fake_post)
    prov = OllamaProvider("ut-local")
    prov.generate(system="s", user="u", options={"temperature": 0.2},
                  agent_tag="bill_agent", tools=_TOOLS)
    assert captured["tools"] == _TOOLS
    assert "tools" not in captured.get("options", {})     # 不得混进 options

    prov.generate(system="s", user="u", options={"temperature": 0.2}, agent_tag="bill_agent")
    assert "tools" not in captured                        # 不传则完全等价于 M14 形态


def test_router_passes_tools_through_without_merging_into_options(monkeypatch):
    """
    函数功能与逻辑描述：
        验证 Router 的**原样透传**契约：tools / extra_messages 必须既不被 merge 进 options、
        也不被丢弃（不 merge、不转译、不白名单）。用一个记录型假 provider 捕获 Router 实际下发的参数。
    入参说明：
        monkeypatch：pytest fixture 注入，开外部、不封本地，使 resolve 命中外部假 provider。
    返回值说明：
        无（断言通过即用例成功）。
    """
    import modelService.router as router_mod
    from modelService.chat_model import Capabilities, ChatModel, ChatResult

    class _Recorder(ChatModel):
        """
        函数/类功能与逻辑描述：
            记录型假后端：把最近一次 generate 收到的 tools / extra_messages / options 记下来，
            用于断言 Router 的透传行为；不触网。
        构造入参说明：
            无。
        返回值说明：
            无（实例自带 last 字典）。
        """

        def __init__(self):
            """
            函数功能与逻辑描述：
                初始化记录容器。
            入参说明：
                无。
            返回值说明：
                无（仅初始化 self.last）。
            """
            self.last = {}

        @property
        def capabilities(self) -> Capabilities:
            """
            函数功能与逻辑描述：
                声明为 openai-compat 后端（使 Router 走外部预算分支），窗口 8192。
            入参说明：
                无（隐式 self）。
            返回值说明：
                Capabilities：外部能力契约。
            """
            return Capabilities(backend="openai-compat", supported_params=frozenset(),
                                reasoning=True, context_window=8192)

        def generate(self, *, system, user, options, agent_tag, timeout=None,
                     tools=None, extra_messages=None):
            """
            函数功能与逻辑描述：
                记录本次收到的参数并返回固定成功结果。
            入参说明：
                system/user/options/agent_tag/timeout：标准参数（本桩忽略值）。
                tools (list|None)：function-calling 清单。
                extra_messages (list|None)：追加消息。
            返回值说明：
                ChatResult：text="ok"、channel="external"。
            """
            self.last = {"options": dict(options), "tools": tools,
                         "extra_messages": extra_messages}
            return ChatResult(text="ok", model="rec", channel="external")

    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", True)
    rec = _Recorder()
    ModelRouter(local=rec, external=rec).generate(
        system="s", user="u", agent_tag="bill_agent",
        tools=_TOOLS, extra_messages=[{"role": "tool", "content": "x"}])
    assert rec.last["tools"] == _TOOLS
    assert rec.last["extra_messages"] == [{"role": "tool", "content": "x"}]
    assert "tools" not in rec.last["options"] and "extra_messages" not in rec.last["options"]


# ── 3. 网关工具与兼容层 ────────────────────────────────────────────────────
def test_llm_chat_fc_schema_contract():
    """
    函数功能与逻辑描述：
        验证 `llm.chat_fc` 的注册契约（`设计/12` §4.3）：① 必填仅 system_prompt / user_input，
        tools 与 extra_messages 为**可选**；② 四个属性齐备；③ 与既有红线一致——
        `agent_id` / `agent_tag` **不得出现**在 schema 里（由适配器注入，防 LLM 伪造提权）。
    入参说明：
        无（pytest 自动发现并调用）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    spec = get_registry().get("llm.chat_fc")
    assert spec is not None
    props = spec.params_schema["properties"]
    assert set(props) == {"system_prompt", "user_input", "tools", "extra_messages"}
    assert spec.params_schema["required"] == ["system_prompt", "user_input"]
    assert "agent_id" not in props and "agent_tag" not in props
    assert spec.binding["tool"] == "llm_chat_fc"


@pytest.mark.asyncio
async def test_call_llm_fc_parses_json_payload():
    """
    函数功能与逻辑描述：
        验证兼容层 `call_llm_fc` 把 server 返回的 JSON 字符串还原为 dict（text + tool_calls），
        并确认 tools / extra_messages 被透传到工具入参、`agent_tag` 由适配器注入（调用方不传）。
    入参说明：
        无（pytest 自动发现并调用；用例内 patch `_call_server` 与 `MCPAdapter.invoke` 的入参可见性）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    payload = json.dumps({"text": "好的", "tool_calls": [
        {"id": "call_1", "name": "bill.recent_list", "arguments": {}}]}, ensure_ascii=False)
    with patch.object(mcp_client, "_call_server", new=AsyncMock(return_value=payload)) as mock_call:
        out = await mcp_client.call_llm_fc(
            "bill_agent", "sys", "usr", tools=_TOOLS,
            extra_messages=[{"role": "tool", "content": "[]"}])
    assert out["text"] == "好的"
    assert out["tool_calls"][0]["name"] == "bill.recent_list"
    args = mock_call.call_args[0][2]                 # 远端 payload
    assert args["tools"] == _TOOLS
    assert args["extra_messages"] == [{"role": "tool", "content": "[]"}]
    assert args["agent_tag"] == "bill_agent"         # 适配器注入，非调用方传入


@pytest.mark.asyncio
async def test_call_llm_fc_error_prefix_raises_runtime_error():
    """
    函数功能与逻辑描述：
        验证错误协议：server 侧业务失败以 `FINANCE_ERROR:` 前缀字符串返回，兼容层必须转成
        RuntimeError——**不得**静默返回空 tool_calls，否则循环会把"通道坏了"误当成"模型不想调工具"
        而给出错误结论（`设计/12` §6 红线 7：写操作是否成功只以工具返回值 ok 为准）。
    入参说明：
        无（pytest 自动发现并调用；用例内 patch `_call_server` 返回错误前缀串）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    with patch.object(mcp_client, "_call_server",
                      new=AsyncMock(return_value="FINANCE_ERROR: 模型不可用")):
        with pytest.raises(RuntimeError):
            await mcp_client.call_llm_fc("bill_agent", "sys", "usr", tools=_TOOLS)


@pytest.mark.asyncio
async def test_call_llm_fc_invalid_json_raises_runtime_error():
    """
    函数功能与逻辑描述：
        验证返回体非法 JSON 时的兜底：抛 RuntimeError 而非静默给空结果——
        通道异常必须显式暴露（上层据此转追问或降级），不能被当成"没有工具调用"。
    入参说明：
        无（pytest 自动发现并调用；用例内 patch `_call_server` 返回非 JSON 文本）。
    返回值说明：
        无（断言通过即用例成功）。
    """
    with patch.object(mcp_client, "_call_server", new=AsyncMock(return_value="not-a-json")):
        with pytest.raises(RuntimeError):
            await mcp_client.call_llm_fc("bill_agent", "sys", "usr")
