# -*- coding: utf-8 -*-
"""
M14（D17）模型接入层 harness 单元测试。

★全部用例**不依赖外部 API / 不连 Ollama**（离线可跑，外部额度耗尽也能验证）。
覆盖目标（`08` §8.11 验收判据，对应 `02` D17 验收）：
  1. 路由集中：resolve 决策表 4 组合（DISABLE / FORCE / AGENTS / 本地兜底）+ 配置矛盾抛错（验收#3 路由集中）
  2. 参数语义隔离：越界参数过滤 + warning（R10 不静默），绝不转译（验收#2 参数语义隔离）
  3. reasoning 预算告警：reasoning 模型 max_tokens < 下限 → warning（验收#1 换模型不崩：reasoning 不再挤空 content）
  4. 重试分类：可重试（429/空 content）进预算池；不可重试（401/400/Timeout）短路（`08` §8.7 重试设计）
  5. fallback：finance 场景外部失败回退本地；DISABLE_LOCAL_LLM=1 禁回退（`08` §8.7）
  6. 兼容层语义：ollama_base_call / external_llm_call 返回 str（FINANCE_ERROR 前缀协议不变）
  7. usage 解析边界：parse_usage 对缺失 usage / 无 reasoning details 兜底 0，total 优先服务端原始值
  8. 窗口（D6 接管依据）：context_window_for 按实际所选后端窗口返回（本地 4k / 外部 8k）
"""
import logging

import pytest

import modelService.router as router_mod
from modelService.chat_model import (
    Capabilities, ChatError, ChatModel, ChatResult, filter_options,
)
from modelService.providers.ollama import OllamaProvider
from modelService.providers.openai_compat import (
    OpenAICompatProvider, parse_usage, REASONING_MIN_OUTPUT_BUDGET,
)
from modelService.llm_loader import ollama_base_call, external_llm_call
import modelService.llm_loader as llm_loader_mod
from modelService.router import ModelRouter

# 避免真实网络（★2026-09-15 更新）：打桩目标分两处，改请求入口必须同步本文件，否则打桩失效、用例真连网——
#   外部通道（OpenAICompatProvider / external_llm_call）走模块级 `_SESSION.post`（**显式直连**，`trust_env=False`，M15）
#     → 统一 patch `requests.Session.post`（一处覆盖该 Session 实例，无需 import 项目模块）；
#   本地通道（ollama_base_call）仍走模块函数 `requests.post`。
import requests


# ── 工具：Fake Provider（不碰网络，测 Router 逻辑）────────────────────────
class _FakeLocal(ChatModel):
    """
    函数/类功能与逻辑描述：
        本地后端假实现（继承 ChatModel），全程不发起网络请求，仅用于验证 ModelRouter 的
        路由 / 重试 / fallback 逻辑；通过构造时的 fail_kind 注入固定失败模式，
        并用 calls 计数器断言 Router 实际发起的调用次数。
    构造入参说明：
        fail_kind (str)：关键字参数，None=正常返回 | "retryable"=抛可重试 ChatError(HTTP_429) |
            "fatal"=抛不可重试 ChatError(HTTP_401)。
    返回值说明：
        构造返回 _FakeLocal 实例；generate 成功时返回 ChatResult(text="local-ok")。
    """
    backend = "ollama"

    def __init__(self, *, fail_kind=None):
        """
        函数功能与逻辑描述：
            按 fail_kind 预置失败模式与成功标记（_ok），并把调用计数器 calls 归零；
            纯内存赋值，不连接 Ollama、不发任何请求。
        入参说明：
            fail_kind (str)：关键字参数，None=成功 | "retryable" | "fatal"，默认 None。
        返回值说明：
            无（仅初始化 self._fail_kind / self._ok / self.calls）。
        """
        self._fail_kind = fail_kind  # None=成功 | "retryable" | "fatal"
        self._ok = fail_kind is None
        self.calls = 0

    @property
    def capabilities(self) -> Capabilities:
        """
        函数功能与逻辑描述：
            返回本地后端能力契约：后端标识 ollama、不额外声称支持参数、reasoning=False、
            上下文窗口 4096，使 Router 的路由判定与窗口查询落入「本地」分支。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Capabilities：backend="ollama"、supported_params=空 frozenset、reasoning=False、context_window=4096。
        """
        return Capabilities(backend="ollama", supported_params=frozenset(),
                            reasoning=False, context_window=4096)

    def generate(self, *, system, user, options, agent_tag, timeout=None,
                 tools=None, extra_messages=None):
        """
        函数功能与逻辑描述：
            模拟本地后端调用：每次调用先自增 calls，再按 fail_kind 决定抛异常或返回成功结果，
            不触网。retryable 分支抛 HTTP_429（进重试池），fatal 分支抛 HTTP_401（立即短路），
            成功分支返回 text="local-ok"、channel="local" 的 ChatResult。
        入参说明：
            system (str)：关键字参数，系统提示词（本假实现不参与推理）。
            user (str)：关键字参数，用户输入（本假实现不参与推理）。
            options (dict)：关键字参数，生成参数（本假实现忽略）。
            agent_tag (str)：关键字参数，Agent 标识（本假实现忽略）。
            timeout (int)：关键字参数，超时秒数（本假实现忽略），默认 None。
            tools (list)：关键字参数，function-calling 工具清单（M15；本假实现忽略），默认 None。
            extra_messages (list)：关键字参数，tool 循环追加消息（M15；本假实现忽略），默认 None。
        返回值说明：
            ChatResult：成功时 text="local-ok"、model="fake-local"、channel="local"；
                失败时抛出 ChatError，不返回。
        """
        self.calls += 1
        if not self._ok:
            if self._fail_kind == "retryable":
                raise ChatError("fake retryable", retryable=True, error_type="HTTP_429")
            raise ChatError("fake fatal", retryable=False, error_type="HTTP_401")
        return ChatResult(text="local-ok", model="fake-local", channel="local")


class _FakeExternal(ChatModel):
    """
    函数/类功能与逻辑描述：
        外部后端假实现（继承 ChatModel），全程不发起网络请求，用于验证 Router 的外部路由 /
        重试预算池 / fallback 回退；与 _FakeLocal 的差异是 backend="openai-compat"、
        reasoning=True、窗口 8192，从而稳定命中「外部」分支。
    构造入参说明：
        fail_kind (str)：关键字参数，None=正常返回 | "retryable"=HTTP_429 | "fatal"=HTTP_401。
    返回值说明：
        构造返回 _FakeExternal 实例；generate 成功时返回 ChatResult(text="external-ok")。
    """

    def __init__(self, *, fail_kind=None):
        """
        函数功能与逻辑描述：
            按 fail_kind 预置失败模式与成功标记（_ok），并把调用计数器 calls 归零；
            纯内存赋值，不连接任何外部 API。
        入参说明：
            fail_kind (str)：关键字参数，None=成功 | "retryable" | "fatal"，默认 None。
        返回值说明：
            无（仅初始化 self._fail_kind / self._ok / self.calls）。
        """
        self._fail_kind = fail_kind
        self._ok = fail_kind is None
        self.calls = 0

    @property
    def capabilities(self) -> Capabilities:
        """
        函数功能与逻辑描述：
            返回外部后端能力契约：后端标识 openai-compat、reasoning=True、
            上下文窗口 8192，使 Router 的路由判定与窗口查询落入「外部」分支。
        入参说明：
            无（隐式 self）。
        返回值说明：
            Capabilities：backend="openai-compat"、supported_params=空 frozenset、reasoning=True、context_window=8192。
        """
        return Capabilities(backend="openai-compat", supported_params=frozenset(),
                            reasoning=True, context_window=8192)

    def generate(self, *, system, user, options, agent_tag, timeout=None,
                 tools=None, extra_messages=None):
        """
        函数功能与逻辑描述：
            模拟外部后端调用：每次调用先自增 calls，再按 fail_kind 决定抛异常或返回成功结果，
            不触网。retryable 分支抛 HTTP_429（进重试池），fatal 分支抛 HTTP_401（立即短路），
            成功分支返回 text="external-ok"、channel="external" 的 ChatResult。
        入参说明：
            system (str)：关键字参数，系统提示词（本假实现忽略）。
            user (str)：关键字参数，用户输入（本假实现忽略）。
            options (dict)：关键字参数，生成参数（本假实现忽略）。
            agent_tag (str)：关键字参数，Agent 标识（本假实现忽略）。
            timeout (int)：关键字参数，超时秒数（本假实现忽略），默认 None。
            tools (list)：关键字参数，function-calling 工具清单（M15；本假实现忽略），默认 None。
            extra_messages (list)：关键字参数，tool 循环追加消息（M15；本假实现忽略），默认 None。
        返回值说明：
            ChatResult：成功时 text="external-ok"、model="fake-ext"、channel="external"；
                失败时抛出 ChatError，不返回。
        """
        self.calls += 1
        if not self._ok:
            if self._fail_kind == "retryable":
                raise ChatError("fake retryable", retryable=True, error_type="HTTP_429")
            raise ChatError("fake fatal", retryable=False, error_type="HTTP_401")
        return ChatResult(text="external-ok", model="fake-ext", channel="external")


@pytest.fixture()
def fake_providers():
    """
    函数功能与逻辑描述：
        提供一对「本地 + 外部」假后端（_FakeLocal, _FakeExternal），均不触网，
        供需要同时持有两端实例的 Router 用例注入使用；函数级作用域，
        每个用例拿到全新实例（calls 计数从 0 开始），无资源需要清理。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        tuple：(_FakeLocal, _FakeExternal) 二元组，两者均为成功模式（fail_kind=None）。
    """
    return _FakeLocal(), _FakeExternal()


# ── 1. 路由集中：resolve 决策表 ────────────────────────────────────────────
def test_resolve_local_default(monkeypatch):
    """
    函数功能与逻辑描述：
        验证路由决策表第 5 档兜底：外部未启用（EXTERNAL_LLM_ENABLED=False，等价于未配置 API Key）、
        未封禁本地、未强制外部时，stat_agent 必须命中本地后端（1.8B 免费基线）。
        构造方式：monkeypatch 改写 router_mod 的三个路由开关后构造 ModelRouter 并调用 resolve。
        关键断言：resolve 返回的 provider.capabilities.backend == "ollama"。
    入参说明：
        monkeypatch：pytest fixture 注入，临时改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED=False / FORCE_EXTERNAL_LLM=False，用例结束后自动还原。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", False)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", False)
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    provider, _ = router.resolve("stat_agent")
    assert provider.capabilities.backend == "ollama"


def test_resolve_external_by_agent(monkeypatch):
    """
    函数功能与逻辑描述：
        验证决策表第 4 档「按 agent 白名单走外部」：外部已启用且 agent ∈ EXTERNAL_LLM_AGENTS 时命中外部，
        白名单之外的 bill_agent 则回落本地，确保白名单未覆盖的 agent 不受影响。
        构造方式：monkeypatch 把 EXTERNAL_LLM_AGENTS 显式设为 frozenset({"orchestrator_agent"})
        （与 config 默认值一致），并开启外部、保持本地可用、不强制外部。
        关键断言：orchestrator_agent → backend=="openai-compat"；bill_agent → backend=="ollama"。
        覆盖边界：白名单内 / 白名单外两条分支。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED=True / FORCE_EXTERNAL_LLM=False / EXTERNAL_LLM_AGENTS。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_AGENTS", frozenset({"orchestrator_agent"}))
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    provider, _ = router.resolve("orchestrator_agent")
    assert provider.capabilities.backend == "openai-compat"
    provider2, _ = router.resolve("bill_agent")   # 不在集合 → 本地
    assert provider2.capabilities.backend == "ollama"


def test_resolve_external_force(monkeypatch):
    """
    函数功能与逻辑描述：
        验证决策表第 3 档强开关：FORCE_EXTERNAL_LLM=True（测试/联调期全量走强模型）时，
        即使是白名单之外的 bill_agent 也一律命中外部后端。
        构造方式：monkeypatch 开启 FORCE_EXTERNAL_LLM 与 EXTERNAL_LLM_ENABLED，保持本地可用。
        关键断言：bill_agent 的 provider.capabilities.backend == "openai-compat"。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED=True / FORCE_EXTERNAL_LLM=True。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", True)
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    provider, _ = router.resolve("bill_agent")
    assert provider.capabilities.backend == "openai-compat"


def test_resolve_disable_local_goes_external(monkeypatch):
    """
    函数功能与逻辑描述：
        验证决策表第 1 档硬开关及其配置矛盾分支：DISABLE_LOCAL_LLM=True 且外部已启用时，
        bill_agent 必须走外部；若此时外部也未启用（本地被封 + 外部不可用 = 配置矛盾），
        resolve 必须立即抛 RuntimeError 暴露问题，而不是静默跑本地或静默失败。
        构造方式：先 monkeypatch 封本地 + 开外部断言走外部，再单独把 EXTERNAL_LLM_ENABLED 置 False 触发矛盾。
        关键断言：首次 resolve 得 backend=="openai-compat"；矛盾配置下 pytest.raises(RuntimeError)。
        覆盖边界：正常外部路由 + 配置矛盾抛错两条路径。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=True /
            EXTERNAL_LLM_ENABLED（先 True 后 False）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", True)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    provider, _ = router.resolve("bill_agent")
    assert provider.capabilities.backend == "openai-compat"
    # 本地被封但外部未启用 = 配置矛盾 → 立即抛错（不默默跑/静默失败）
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", False)
    with pytest.raises(RuntimeError):
        router.resolve("bill_agent")


def test_resolve_prefer_external_finance(monkeypatch):
    """
    函数功能与逻辑描述：
        验证 finance 通道差异（prefer_external=True）：外部可用即走外部，且不查 EXTERNAL_LLM_AGENTS
        白名单；外部不可用时回落本地（QLoRA 小模型）。
        构造方式：ModelRouter(..., prefer_external=True)，先开外部断言命中外部，再关外部断言回落本地。
        关键断言：外部启用 → backend=="openai-compat"；外部关闭 → backend=="ollama"。
        覆盖边界：外部可用 / 外部不可用两条分支。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED（先 True 后 False）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    router = ModelRouter(_FakeLocal(), _FakeExternal(), prefer_external=True)
    provider, _ = router.resolve("finance_agent")
    assert provider.capabilities.backend == "openai-compat"
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", False)
    provider2, _ = router.resolve("finance_agent")   # 无 API → 本地 QLoRA
    assert provider2.capabilities.backend == "ollama"


# ── 2. 参数语义隔离：越界过滤 + warning（R10 不静默，绝不转译）─────────────
def test_filter_options_keeps_supported_only(caplog):
    """
    函数功能与逻辑描述：
        验证参数语义隔离（验收#2）：openai-compat 契约只认 temperature / max_tokens，
        传入的 num_predict / num_ctx 必须被直接丢弃并打 WARNING（R10 不静默），绝不转译为 max_tokens。
        构造方式：在 caplog.at_level(WARNING) 下直接调用 chat_model.filter_options，
            传入四个参数与 openai-compat 的支持参数集合。
        关键断言：返回值仅含 {temperature, max_tokens}（值原样保留）；WARNING 记录中出现 "num_predict"。
    入参说明：
        caplog：pytest fixture 注入，捕获 logging 记录，用于断言越界参数确实产生了告警。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    with caplog.at_level(logging.WARNING):
        kept = filter_options({"temperature": 0.3, "max_tokens": 2048,
                               "num_predict": 1000, "num_ctx": 4096},
                              frozenset({"temperature", "max_tokens"}),
                              "openai-compat")
    assert kept == {"temperature": 0.3, "max_tokens": 2048}   # 绝不转译，直接丢弃
    assert any("num_predict" in r.message for r in caplog.records)


def test_reasoning_budget_warning(caplog, monkeypatch):
    """
    函数功能与逻辑描述：
        验证 reasoning 预算护栏：推理模型下 max_tokens 低于 REASONING_MIN_OUTPUT_BUDGET（512）时，
        OpenAICompatProvider.generate 必须打 WARNING（R10 不静默），但仍正常返回 content、不因告警报错。
        构造方式：monkeypatch 把 requests.post 替换为返回 200 + 正常 content 的假响应（不触网），
            以 max_tokens=REASONING_MIN_OUTPUT_BUDGET-1（511）调用 generate。
        关键断言：返回文本 == "ok"；WARNING 消息同时含 "max_tokens" 与 "挤空"。
        覆盖边界：max_tokens 恰好低于下限的告警触发点。
    入参说明：
        caplog：pytest fixture 注入，捕获 WARNING 级日志以断言告警文案。
        monkeypatch：pytest fixture 注入，把 requests.post 替换为假响应 _Resp，避免真实 HTTP 调用。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        """
        函数/类功能与逻辑描述：
            假 HTTP 响应对象：模拟 requests.post 的返回，提供 generate 所需的
            status_code / raise_for_status() / json() 接口，固定返回 200 与正常 content。
        构造入参说明：
            无（无构造参数，字段以类属性形式固定）。
        返回值说明：
            无（仅作为 requests.post 的替身对象）。
        """

        status_code = 200

        def raise_for_status(self):
            """
            函数功能与逻辑描述：
                模拟 requests 成功响应的状态检查：固定不抛异常（status_code=200）。
            入参说明：
                无（隐式 self）。
            返回值说明：
                无（直接返回，无副作用）。
            """
            pass

        def json(self):
            """
            函数功能与逻辑描述：
                返回固定的 OpenAI 兼容响应体：choices[0].message.content="ok"，
                并附带 usage 三项计数字段，供 parse_usage 解析。
            入参说明：
                无（隐式 self）。
            返回值说明：
                dict：含 choices 与 usage 的假响应体。
            """
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
    with caplog.at_level(logging.WARNING):
        r = prov.generate(system="s", user="u",
                          options={"max_tokens": REASONING_MIN_OUTPUT_BUDGET - 1},
                          agent_tag="ut")
    assert r.text == "ok"
    assert any("max_tokens" in m and "挤空" in m for m in caplog.messages)


# ── 3. 重试分类 ────────────────────────────────────────────────────────────
def test_provider_401_not_retryable(monkeypatch):
    """
    函数功能与逻辑描述：
        验证不可重试分类：鉴权失败（HTTP 401）必须抛 retryable=False 的 ChatError，
        使 Router 立即短路、不占用重试预算池（重试只烧钱）。
        构造方式：monkeypatch requests.post 返回 status_code=401 的假响应，
            其 raise_for_status 构造携带 response 的 requests.HTTPError。
        关键断言：pytest.raises(ChatError)；ei.value.retryable 为 False；error_type == "HTTP_401"。
    入参说明：
        monkeypatch：pytest fixture 注入，把 requests.post 替换为 401 假响应 _Resp。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        """
        函数/类功能与逻辑描述：
            假 HTTP 401 响应：提供 status_code / text 与抛 HTTPError 的 raise_for_status，
            用于触发 OpenAICompatProvider 的不可重试分类分支。
        构造入参说明：
            无（字段以类属性形式固定）。
        返回值说明：
            无（仅作为 requests.post 的替身对象）。
        """

        status_code = 401
        text = '{"error":"unauthorized"}'

        def raise_for_status(self):
            """
            函数功能与逻辑描述：
                模拟 requests.Response.raise_for_status 在 401 时抛出 HTTPError，
                并携带一个构造好的 response（status_code=401、_content 为错误 JSON），
                使 Provider 能从 http_err.response 取到状态码与详情文本。
            入参说明：
                无（隐式 self）。
            返回值说明：
                无（固定抛出 requests.exceptions.HTTPError，不返回）。
            """
            import requests as rq
            resp = rq.Response()
            resp.status_code = 401
            resp._content = b'{"error":"unauthorized"}'
            raise rq.exceptions.HTTPError("401 Client Error", response=resp)

    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
    with pytest.raises(ChatError) as ei:
        prov.generate(system="s", user="u", options={}, agent_tag="ut")
    assert not ei.value.retryable
    assert ei.value.error_type == "HTTP_401"


def test_provider_429_retryable(monkeypatch):
    """
    函数功能与逻辑描述：
        验证可重试分类：限流（HTTP 429）必须抛 retryable=True 的 ChatError，
        从而被 Router 纳入统一重试预算池（429 属瞬时错误，重试有收益）。
        构造方式：monkeypatch requests.post 返回 status_code=429 的假响应，
            其 raise_for_status 抛出携带 response 的 HTTPError。
        关键断言：pytest.raises(ChatError)；ei.value.retryable 为 True；error_type == "HTTP_429"。
    入参说明：
        monkeypatch：pytest fixture 注入，把 requests.post 替换为 429 假响应 _Resp。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        """
        函数/类功能与逻辑描述：
            假 HTTP 429 响应：提供 status_code / text 与抛 HTTPError 的 raise_for_status，
            用于触发 OpenAICompatProvider 的可重试分类分支。
        构造入参说明：
            无（字段以类属性形式固定）。
        返回值说明：
            无（仅作为 requests.post 的替身对象）。
        """

        status_code = 429
        text = "rate limited"

        def raise_for_status(self):
            """
            函数功能与逻辑描述：
                模拟 requests.Response.raise_for_status 在 429 时抛出 HTTPError，
                并携带 status_code=429 的 response，使 Provider 判定 retryable=True 且 error_type="HTTP_429"。
            入参说明：
                无（隐式 self）。
            返回值说明：
                无（固定抛出 requests.exceptions.HTTPError，不返回）。
            """
            import requests as rq
            resp = rq.Response()
            resp.status_code = 429
            resp._content = b"rate limited"
            raise rq.exceptions.HTTPError("429 Too Many Requests", response=resp)

    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
    with pytest.raises(ChatError) as ei:
        prov.generate(system="s", user="u", options={}, agent_tag="ut")
    assert ei.value.retryable
    assert ei.value.error_type == "HTTP_429"


def test_provider_empty_content_retryable(monkeypatch):
    """
    函数功能与逻辑描述：
        验证「请求成功但 content 为空」被归类为可重试（error_type=EMPTY_CONTENT）：
        常见成因是 reasoning 占满 max_tokens 预算，落池重试有实际收益。
        构造方式：monkeypatch requests.post 返回 200 + content="" 的假响应，
            以 max_tokens=512（等于下限，不触发预算告警）调用 generate。
        关键断言：pytest.raises(ChatError)；ei.value.retryable 为 True；error_type == "EMPTY_CONTENT"。
    入参说明：
        monkeypatch：pytest fixture 注入，把 requests.post 替换为空 content 假响应 _Resp。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        """
        函数/类功能与逻辑描述：
            假 HTTP 200 且 content 为空的响应：提供 status_code / raise_for_status / json，
            用于驱动 EMPTY_CONTENT 可重试分支。
        构造入参说明：
            无（字段以类属性形式固定）。
        返回值说明：
            无（仅作为 requests.post 的替身对象）。
        """

        status_code = 200

        def raise_for_status(self):
            """
            函数功能与逻辑描述：
                模拟成功响应：固定不抛异常（status_code=200），把错误留给 content 判空逻辑触发。
            入参说明：
                无（隐式 self）。
            返回值说明：
                无（直接返回，无副作用）。
            """
            pass

        def json(self):
            """
            函数功能与逻辑描述：
                返回 choices[0].message.content 为空串的假响应体（usage 计数齐全），
                使 Provider 抛 EMPTY_CONTENT 且判定可重试。
            入参说明：
                无（隐式 self）。
            返回值说明：
                dict：content 为空串的假响应体（含 usage 三项计数）。
            """
            return {"choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 400,
                              "total_tokens": 401}}

    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
    with pytest.raises(ChatError) as ei:
        prov.generate(system="s", user="u", options={"max_tokens": 512},
                      agent_tag="ut")
    assert ei.value.retryable
    assert ei.value.error_type == "EMPTY_CONTENT"


def test_execute_retry_budget_attempts(fake_providers):
    """
    函数功能与逻辑描述：
        验证统一重试预算池语义：可重试错误按 max_attempts 次数耗尽预算后才抛出，
        不可重试错误则在首次失败即短路；以假后端的调用计数区分两种行为。
        构造方式：用 fake_providers 提供的本地实例，分别配 fail_kind="retryable"（HTTP_429）
        与 fail_kind="fatal"（HTTP_401）的外部假后端构造两个 Router，直接调用 execute(max_attempts=3)。
        关键断言：可重试后端 calls==3（占满预算池）；不可重试后端 calls==1（401 短路不重试）。
    入参说明：
        fake_providers：pytest fixture 注入，提供 (_FakeLocal, _FakeExternal) 假后端二元组；
            本用例仅取用其中的本地实例作为 Router 的本地端，外部端另用 fail_kind 重新构造。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    local, external = fake_providers
    ext_flaky = _FakeExternal(fail_kind="retryable")   # 每次都失败
    router = ModelRouter(local, ext_flaky)
    with pytest.raises(ChatError):
        router.execute(ext_flaky, system="s", user="u", options={},
                       agent_tag="ut", max_attempts=3)
    assert ext_flaky.calls == 3          # 429 占满预算池
    ext_fatal = _FakeExternal(fail_kind="fatal")
    router2 = ModelRouter(local, ext_fatal)
    with pytest.raises(ChatError):
        router2.execute(ext_fatal, system="s", user="u", options={},
                        agent_tag="ut", max_attempts=3)
    assert ext_fatal.calls == 1          # 401 短路，不重试


# ── 4. fallback（finance 场景）─────────────────────────────────────────────
def test_generate_fallback_to_local(monkeypatch, fake_providers):
    """
    函数功能与逻辑描述：
        验证 fallback 回退：finance 通道（prefer_external=True + allow_fallback=True）下，
        外部后端致命失败（HTTP_401，不可重试）后应回退本地并返回本地结果。
        构造方式：monkeypatch 开外部、不封本地，构造 prefer_external=True + allow_fallback=True 的
        Router（本地成功、外部 fatal），调用 generate。
        关键断言：result.text == "local-ok"（来自本地）；external.calls == 1 且 local.calls == 1
        （外部失败一次后回退本地一次，本地侧不重试）。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED=True。
        fake_providers：pytest fixture 注入，提供假后端二元组；本用例实际另建 local/external 实例
            （外部 fail_kind="fatal"），fixture 在此仅用于保持用例签名一致。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    local = _FakeLocal()
    external = _FakeExternal(fail_kind="fatal")
    router = ModelRouter(local, external, prefer_external=True, allow_fallback=True)
    result = router.generate(system="s", user="u", agent_tag="finance_agent")
    assert result.text == "local-ok"
    assert external.calls == 1 and local.calls == 1


def test_generate_no_fallback_when_local_disabled(monkeypatch, fake_providers):
    """
    函数功能与逻辑描述：
        验证禁回退护栏：DISABLE_LOCAL_LLM=True 时，即便 allow_fallback=True、外部失败，
        也必须直接抛 ChatError 而绝不回退本地（本地已被封死）。
        构造方式：monkeypatch 封本地并开外部，构造 prefer_external=True + allow_fallback=True 的 Router，
        generate 时外部 fatal 失败。
        关键断言：pytest.raises(ChatError)；local.calls == 0（证明一次都没走到本地）。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=True /
            EXTERNAL_LLM_ENABLED=True。
        fake_providers：pytest fixture 注入，提供假后端二元组；本用例另建 local/external 实例
            （外部 fail_kind="fatal"），fixture 在此仅用于保持用例签名一致。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", True)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    local = _FakeLocal()
    external = _FakeExternal(fail_kind="fatal")
    router = ModelRouter(local, external, prefer_external=True, allow_fallback=True)
    with pytest.raises(ChatError):
        router.generate(system="s", user="u", agent_tag="finance_agent")
    assert local.calls == 0


def test_generate_base_no_fallback(monkeypatch, fake_providers):
    """
    函数功能与逻辑描述：
        验证 base 通道不偷偷回退：allow_fallback 保持默认 False 时，走外部的 orchestrator_agent
        在外部 fatal 失败后必须直接抛 ChatError，不得静默回退本地。
        构造方式：monkeypatch 开外部、不封本地，并把 EXTERNAL_LLM_AGENTS 设为 {"orchestrator_agent"}；
        构造默认参数的 Router（外部 fail_kind="fatal"）后 generate。
        关键断言：pytest.raises(ChatError)；local.calls == 0（无隐式回退）。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED=True / EXTERNAL_LLM_AGENTS。
        fake_providers：pytest fixture 注入，提供假后端二元组；本用例另建 local/external 实例
            （外部 fail_kind="fatal"），fixture 在此仅用于保持用例签名一致。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_AGENTS", frozenset({"orchestrator_agent"}))
    local = _FakeLocal()
    external = _FakeExternal(fail_kind="fatal")
    router = ModelRouter(local, external)
    with pytest.raises(ChatError):
        router.generate(system="s", user="u", agent_tag="orchestrator_agent")
    assert local.calls == 0


# ── 5. 兼容层语义：签名/返回协议不变 ───────────────────────────────────────
class _LocalOkResp:
    """
    函数/类功能与逻辑描述：
        假 Ollama 成功响应：模拟 /api/chat 的返回结构（message.content +
        prompt_eval_count / eval_count），供兼容层 ollama_base_call 用例替换 requests.post。
    构造入参说明：
        无（字段以类属性固定，无构造参数）。
    返回值说明：
        无（仅作为 requests.post 的替身对象）。
    """

    status_code = 200

    def raise_for_status(self):
        """
        函数功能与逻辑描述：
            模拟 requests 成功响应的状态检查：固定不抛异常（status_code=200）。
        入参说明：
            无（隐式 self）。
        返回值说明：
            无（直接返回，无副作用）。
        """
        pass

    def json(self):
        """
        函数功能与逻辑描述：
            返回 Ollama /api/chat 的固定成功响应体：message.content="本地返回"，
            并附带 prompt_eval_count=10、eval_count=5 作为 token 等价计量。
        入参说明：
            无（隐式 self）。
        返回值说明：
            dict：含 message 与 prompt_eval_count / eval_count 的假响应体。
        """
        return {"message": {"content": "本地返回"}, "prompt_eval_count": 10,
                "eval_count": 5}


def test_ollama_base_call_returns_text(monkeypatch):
    """
    函数功能与逻辑描述：
        验证兼容层 ollama_base_call 成功路径：返回纯文本且不带 FINANCE_ERROR 前缀
        （正常文本不会被业务层误判为错误）。

        ★M22 适配：本地通道**默认已弃用**（config.DISABLE_LOCAL_LLM 默认 1，走到即抛
        RuntimeError）。本用例测的是"**回滚开关打开后旧通道仍可用**"这条路径，故须先
        把 llm_loader 模块内的 DISABLE_LOCAL_LLM 置为 False —— 注意不能用 setenv：
        该常量在 config 导入时即已读定（Final），改环境变量不会影响已导入的值。

        构造方式：monkeypatch requests.post 为 _LocalOkResp（200 + message.content="本地返回"），
            调用 ollama_base_call("sys", "user内容")。
        关键断言：返回值 == "本地返回"；且不以 "FINANCE_ERROR" 开头。
        入参说明：
        monkeypatch：pytest fixture 注入，把 requests.post 替换为成功假响应 _LocalOkResp，
            并把 llm_loader.DISABLE_LOCAL_LLM 置 False 以打开本地回滚通道。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(llm_loader_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: _LocalOkResp())
    text = ollama_base_call("sys", "user内容")
    assert text == "本地返回"
    assert not text.startswith("FINANCE_ERROR")


def test_ollama_base_call_error_prefix(monkeypatch):
    """
    函数功能与逻辑描述：
        验证兼容层错误协议：底层异常必须被统一转成 "FINANCE_ERROR: ..." 前缀字符串返回，
        保持业务层 startswith("FINANCE_ERROR") 的旧判断语义不变（不向调用方抛异常）。

        ★M22 适配：同上一条，须先把 llm_loader.DISABLE_LOCAL_LLM 置 False 打开回滚通道，
        否则会在入口处就抛 RuntimeError，走不到异常字符串化这条路径。

        构造方式：monkeypatch requests.post 抛 requests.exceptions.ConnectionError。
        关键断言：返回值以 "FINANCE_ERROR" 开头。
        入参说明：
        monkeypatch：pytest fixture 注入，把 requests.post 替换为必然抛 ConnectionError 的 _boom，
            并把 llm_loader.DISABLE_LOCAL_LLM 置 False 以打开本地回滚通道。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(llm_loader_mod, "DISABLE_LOCAL_LLM", False)

    def _boom(*a, **k):
        """
        函数功能与逻辑描述：
            假 requests.post：无条件抛出 requests.exceptions.ConnectionError("refused")，
            模拟本地 Ollama 服务不可达，用于驱动兼容层的异常字符串化路径。
        入参说明：
            *a：位置参数（被忽略，仅保持与 requests.post 一致的调用签名）。
            **k：关键字参数（被忽略）。
        返回值说明：
            无（必然抛出 requests.exceptions.ConnectionError，不返回）。
        """
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", _boom)
    text = ollama_base_call("sys", "user内容")
    assert text.startswith("FINANCE_ERROR")


def test_ollama_base_call_raises_when_local_disabled():
    """
    函数功能与逻辑描述：
        ★M22 行为守卫：本地通道**默认已弃用**（config.DISABLE_LOCAL_LLM 默认 1）。
        此时 ollama_base_call 必须**立即抛 RuntimeError**，而不是静默用 1.8B 小模型
        跑出一个又慢又不可靠的结果 —— 走到该函数说明模型路由有遗漏，应当**暴露**而非兜底。
        本用例把这条硬开关行为用测试锁死，防止日后误把默认值改回 0 而无人察觉。
    入参说明：
        无（读取 llm_loader 模块内已导入的 DISABLE_LOCAL_LLM 常量）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert llm_loader_mod.DISABLE_LOCAL_LLM is True, \
        "M22 约定本地通道默认弃用；若此处为 False，说明默认值被改回，需同步本用例与 README"
    with pytest.raises(RuntimeError, match="本地模型已被禁用"):
        ollama_base_call("sys", "user内容")


def test_external_llm_call_returns_text(monkeypatch):
    """
    函数功能与逻辑描述：
        验证兼容层 external_llm_call 的两条路径：外部成功时返回纯文本；
        当响应 content 恒为空（可重试错误）时，重试耗尽预算后转为 FINANCE_ERROR 前缀字符串返回。
        构造方式：先 monkeypatch requests.post 返回 200 + content="外部返回"，断言成功文本；
            再换成 content="" 的假响应，触发 EMPTY_CONTENT 重试耗尽。
        关键断言：首次 text == "外部返回"；第二次 text2 以 "FINANCE_ERROR" 开头。
        覆盖边界：正常返回 + 空 content 重试耗尽两种情形。
    入参说明：
        monkeypatch：pytest fixture 注入，先后把 requests.post 替换为 _Resp（正常）与
            _EmptyResp（空 content）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    class _Resp:
        """
        函数/类功能与逻辑描述：
            假外部成功响应：200 + choices[0].message.content="外部返回" + usage 计量，
            用于验证 external_llm_call 的成功返回。
        构造入参说明：
            无（字段以类属性固定）。
        返回值说明：
            无（仅作为 requests.post 的替身对象）。
        """

        status_code = 200

        def raise_for_status(self):
            """
            函数功能与逻辑描述：
                模拟成功响应：固定不抛异常（status_code=200）。
            入参说明：
                无（隐式 self）。
            返回值说明：
                无（直接返回，无副作用）。
            """
            pass

        def json(self):
            """
            函数功能与逻辑描述：
                返回固定的 OpenAI 兼容成功响应体：content="外部返回"，usage 三项计数齐全。
            入参说明：
                无（隐式 self）。
            返回值说明：
                dict：含 choices 与 usage 的假响应体。
            """
            return {"choices": [{"message": {"content": "外部返回"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _Resp())
    text = external_llm_call("sys", "user", "ut-model")
    assert text == "外部返回"

    # 空 content（可重试）×3 占满预算 → FINANCE_ERROR
    class _EmptyResp:
        """
        函数/类功能与逻辑描述：
            假外部「空 content」响应：200 但 choices[0].message.content=""，使 Provider 抛
            EMPTY_CONTENT（可重试），用于验证 external_llm_call 重试耗尽后回退为 FINANCE_ERROR 字符串。
        构造入参说明：
            无（字段以类属性固定）。
        返回值说明：
            无（仅作为 requests.post 的替身对象）。
        """

        status_code = 200

        def raise_for_status(self):
            """
            函数功能与逻辑描述：
                模拟成功响应：固定不抛异常（status_code=200），把错误留给 content 判空逻辑触发。
            入参说明：
                无（隐式 self）。
            返回值说明：
                无（直接返回，无副作用）。
            """
            pass

        def json(self):
            """
            函数功能与逻辑描述：
                返回 choices[0].message.content 为空串的假响应体（usage 计数齐全），
                用于触发 EMPTY_CONTENT 可重试分支并耗尽重试预算。
            入参说明：
                无（隐式 self）。
            返回值说明：
                dict：content 为空串的假响应体。
            """
            return {"choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    monkeypatch.setattr(requests.Session, "post", lambda *a, **k: _EmptyResp())
    text2 = external_llm_call("sys", "user", "ut-model")
    assert text2.startswith("FINANCE_ERROR")


# ── 5.1 显式直连（M15）────────────────────────────────────────────────────
def test_external_provider_disables_system_proxy():
    """
    函数功能与逻辑描述：
        验证外部 Provider 的**显式直连**契约（M15）：模块级 `_SESSION.trust_env` 必须为 False。
        背景：requests 默认会读环境变量**与 Windows 注册表**的系统代理，一旦运行环境留有失效的代理残留
        （如代理软件异常退出留下的 127.0.0.1:7892，端口已无人监听），外部调用会全部以 ProxyError 失败，
        且报错与"模型不可用"难以区分（2026-09-15 实测踩坑）。显式直连把"看系统心情"变成确定行为。
        关键断言：trust_env 恒为 False。
        覆盖边界：防回归——若后续重构把请求入口改回裸 `requests.post`，本用例即失败。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    from modelService.providers import openai_compat

    assert openai_compat._SESSION.trust_env is False


# ── 6. usage 解析边界 ──────────────────────────────────────────────────────
def test_parse_usage_empty_and_no_details():
    """
    函数功能与逻辑描述：
        验证 parse_usage 的边界兜底：入参为 {} / None / 无 reasoning details 时各计数按 0 处理；
        且 total_tokens 优先采用服务端原始值，服务端未给才回退为 prompt+completion+reasoning 之和。
        构造方式：直接以三组输入调用模块级纯函数 parse_usage（无网络、无副作用）。
        关键断言：parse_usage({}) 四键全 0；{"prompt_tokens":7,"completion_tokens":3} →
        reasoning_tokens==0 且 total_tokens==10（自算回退）；parse_usage(None)["total_tokens"]==0。
        覆盖边界：空字典 / None / 缺 usage / 缺 completion_tokens_details。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    assert parse_usage({}) == {"prompt_tokens": 0, "completion_tokens": 0,
                               "reasoning_tokens": 0, "total_tokens": 0}
    u = parse_usage({"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert u == {"prompt_tokens": 7, "completion_tokens": 3,
                 "reasoning_tokens": 0, "total_tokens": 10}
    assert parse_usage(None)["total_tokens"] == 0


# ── 7. 窗口（D6 接管依据）──────────────────────────────────────────────────
def test_context_window_for_by_route(monkeypatch):
    """
    函数功能与逻辑描述：
        验证窗口查询按「实际所选后端」返回（D6 ContextManager 预算换算依据，`08` §8.3 铁律3）：
        默认走外部的 orchestrator_agent 返回外部 8k，走本地的 stat_agent 返回本地 4k，
        证明窗口不是写死的单一常量。
        构造方式：monkeypatch 开外部并把 EXTERNAL_LLM_AGENTS 设为 {"orchestrator_agent"}，
            构造 ModelRouter(_FakeLocal(), _FakeExternal()) 后调用 context_window_for。
        关键断言：stat_agent == 4096（本地）；orchestrator_agent == 8192（外部）。
    入参说明：
        monkeypatch：pytest fixture 注入，改写 router_mod.DISABLE_LOCAL_LLM=False /
            EXTERNAL_LLM_ENABLED=True / FORCE_EXTERNAL_LLM=False / EXTERNAL_LLM_AGENTS。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_AGENTS", frozenset({"orchestrator_agent"}))
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    assert router.context_window_for("stat_agent") == 4096
    assert router.context_window_for("orchestrator_agent") == 8192
