# -*- coding: utf-8 -*-
"""
M14（D17）模型接入层 harness 单元测试。

★全部用例**不依赖外部 API / 不连 Ollama**（离线可跑，外部额度耗尽也能验证）。
覆盖目标（`08` §8.11 验收判据，对应 `02` D17 验收）：
  1. 路由集中：resolve 决策表 4 组合（DISABLE / FORCE / AGENTS / 本地兜底）+ 配置矛盾抛错
  2. 参数语义隔离：越界参数过滤 + warning（R10 不静默），绝不转译（验收#2 口径）
  3. reasoning 预算告警：reasoning 模型 max_tokens < 下限 → warning（验收#2）
  4. 重试分类：可重试（429/空 content）进预算池；不可重试（401/400/Timeout）短路（验收#3）
  5. fallback：finance 场景外部失败回退本地；DISABLE_LOCAL_LLM=1 禁回退
  6. 兼容层语义：ollama_base_call / external_llm_call 返回 str（FINANCE_ERROR 前缀协议不变）
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
from modelService.router import ModelRouter

# 避免 ollama 侧误发请求（兼容层用例全部 patch requests.post）
import requests


# ── 工具：Fake Provider（不碰网络，测 Router 逻辑）────────────────────────
class _FakeLocal(ChatModel):
    backend = "ollama"

    def __init__(self, *, fail_kind=None):
        self._fail_kind = fail_kind  # None=成功 | "retryable" | "fatal"
        self._ok = fail_kind is None
        self.calls = 0

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(backend="ollama", supported_params=frozenset(),
                            reasoning=False, context_window=4096)

    def generate(self, *, system, user, options, agent_tag, timeout=None):
        self.calls += 1
        if not self._ok:
            if self._fail_kind == "retryable":
                raise ChatError("fake retryable", retryable=True, error_type="HTTP_429")
            raise ChatError("fake fatal", retryable=False, error_type="HTTP_401")
        return ChatResult(text="local-ok", model="fake-local", channel="local")


class _FakeExternal(ChatModel):
    def __init__(self, *, fail_kind=None):
        self._fail_kind = fail_kind
        self._ok = fail_kind is None
        self.calls = 0

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(backend="openai-compat", supported_params=frozenset(),
                            reasoning=True, context_window=8192)

    def generate(self, *, system, user, options, agent_tag, timeout=None):
        self.calls += 1
        if not self._ok:
            if self._fail_kind == "retryable":
                raise ChatError("fake retryable", retryable=True, error_type="HTTP_429")
            raise ChatError("fake fatal", retryable=False, error_type="HTTP_401")
        return ChatResult(text="external-ok", model="fake-ext", channel="external")


@pytest.fixture()
def fake_providers():
    return _FakeLocal(), _FakeExternal()


# ── 1. 路由集中：resolve 决策表 ────────────────────────────────────────────
def test_resolve_local_default(monkeypatch):
    """未启用外部（无 API key）→ 本地兜底（1.8B 免费基线）"""
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", False)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", False)
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    provider, _ = router.resolve("stat_agent")
    assert provider.capabilities.backend == "ollama"


def test_resolve_external_by_agent(monkeypatch):
    """外部已启用且 agent ∈ EXTERNAL_LLM_AGENTS（默认 orchestrator_agent）→ 外部"""
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
    """FORCE_EXTERNAL_LLM=1（测试/联调全量走强模型）→ 任意 agent 都外部"""
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", True)
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    provider, _ = router.resolve("bill_agent")
    assert provider.capabilities.backend == "openai-compat"


def test_resolve_disable_local_goes_external(monkeypatch):
    """DISABLE_LOCAL_LLM=1 → 外部（本地被封死；配置矛盾则立即抛错暴露）"""
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
    """finance 通道差异：prefer_external=外部可用即外部（不查 EXTERNAL_LLM_AGENTS）"""
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
    """openai-compat 契约只有 temperature/max_tokens：num_predict/num_ctx 越界 → 过滤+告警"""
    with caplog.at_level(logging.WARNING):
        kept = filter_options({"temperature": 0.3, "max_tokens": 2048,
                               "num_predict": 1000, "num_ctx": 4096},
                              frozenset({"temperature", "max_tokens"}),
                              "openai-compat")
    assert kept == {"temperature": 0.3, "max_tokens": 2048}   # 绝不转译，直接丢弃
    assert any("num_predict" in r.message for r in caplog.records)


def test_reasoning_budget_warning(caplog, monkeypatch):
    """reasoning 模型 max_tokens < 下限 → warning（thinking 会挤空 content 的根源告警）"""
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    with caplog.at_level(logging.WARNING):
        r = prov.generate(system="s", user="u",
                          options={"max_tokens": REASONING_MIN_OUTPUT_BUDGET - 1},
                          agent_tag="ut")
    assert r.text == "ok"
    assert any("max_tokens" in m and "挤空" in m for m in caplog.messages)


# ── 3. 重试分类 ────────────────────────────────────────────────────────────
def test_provider_401_not_retryable(monkeypatch):
    """鉴权失败（401）不可重试——重试只烧钱"""
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        status_code = 401
        text = '{"error":"unauthorized"}'

        def raise_for_status(self):
            import requests as rq
            resp = rq.Response()
            resp.status_code = 401
            resp._content = b'{"error":"unauthorized"}'
            raise rq.exceptions.HTTPError("401 Client Error", response=resp)

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    with pytest.raises(ChatError) as ei:
        prov.generate(system="s", user="u", options={}, agent_tag="ut")
    assert not ei.value.retryable
    assert ei.value.error_type == "HTTP_401"


def test_provider_429_retryable(monkeypatch):
    """限流（429）可重试 → 进 Router 预算池"""
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        status_code = 429
        text = "rate limited"

        def raise_for_status(self):
            import requests as rq
            resp = rq.Response()
            resp.status_code = 429
            resp._content = b"rate limited"
            raise rq.exceptions.HTTPError("429 Too Many Requests", response=resp)

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    with pytest.raises(ChatError) as ei:
        prov.generate(system="s", user="u", options={}, agent_tag="ut")
    assert ei.value.retryable
    assert ei.value.error_type == "HTTP_429"


def test_provider_empty_content_retryable(monkeypatch):
    """请求成功但 content 空 → 可重试（常因 reasoning 吃满预算，落池重试）"""
    prov = OpenAICompatProvider("ut-model")

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 400,
                              "total_tokens": 401}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    with pytest.raises(ChatError) as ei:
        prov.generate(system="s", user="u", options={"max_tokens": 512},
                      agent_tag="ut")
    assert ei.value.retryable
    assert ei.value.error_type == "EMPTY_CONTENT"


def test_execute_retry_budget_attempts(fake_providers):
    """统一重试预算池：可重试错误重试到预算耗尽；不可重试立即短路"""
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
    """finance：外部失败 → 回退本地（allow_fallback + prefer_external）"""
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    local = _FakeLocal()
    external = _FakeExternal(fail_kind="fatal")
    router = ModelRouter(local, external, prefer_external=True, allow_fallback=True)
    result = router.generate(system="s", user="u", agent_tag="finance_agent")
    assert result.text == "local-ok"
    assert external.calls == 1 and local.calls == 1


def test_generate_no_fallback_when_local_disabled(monkeypatch, fake_providers):
    """DISABLE_LOCAL_LLM=1 时外部失败禁止回退本地（本地被封死）"""
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", True)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    local = _FakeLocal()
    external = _FakeExternal(fail_kind="fatal")
    router = ModelRouter(local, external, prefer_external=True, allow_fallback=True)
    with pytest.raises(ChatError):
        router.generate(system="s", user="u", agent_tag="finance_agent")
    assert local.calls == 0


def test_generate_base_no_fallback(monkeypatch, fake_providers):
    """base 通道（allow_fallback=False）：外部失败不偷偷回退本地"""
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
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "本地返回"}, "prompt_eval_count": 10,
                "eval_count": 5}


def test_ollama_base_call_returns_text(monkeypatch):
    """ollama_base_call 返回纯文本（无 FINANCE_ERROR 前缀=成功）"""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _LocalOkResp())
    text = ollama_base_call("sys", "user内容")
    assert text == "本地返回"
    assert not text.startswith("FINANCE_ERROR")


def test_ollama_base_call_error_prefix(monkeypatch):
    """异常统一转 FINANCE_ERROR 前缀字符串（业务层 startswith 语义不变）"""
    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", _boom)
    text = ollama_base_call("sys", "user内容")
    assert text.startswith("FINANCE_ERROR")


def test_external_llm_call_returns_text(monkeypatch):
    """external_llm_call 返回纯文本（成功）；空 content 走重试池后仍失败 → FINANCE_ERROR"""
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "外部返回"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    text = external_llm_call("sys", "user", "ut-model")
    assert text == "外部返回"

    # 空 content（可重试）×3 占满预算 → FINANCE_ERROR
    class _EmptyResp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": ""}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    monkeypatch.setattr(requests, "post", lambda *a, **k: _EmptyResp())
    text2 = external_llm_call("sys", "user", "ut-model")
    assert text2.startswith("FINANCE_ERROR")


# ── 6. usage 解析边界 ──────────────────────────────────────────────────────
def test_parse_usage_empty_and_no_details():
    """usage 缺失 / 无 reasoning details → 安全兜底 0；total 优先原始值"""
    assert parse_usage({}) == {"prompt_tokens": 0, "completion_tokens": 0,
                               "reasoning_tokens": 0, "total_tokens": 0}
    u = parse_usage({"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert u == {"prompt_tokens": 7, "completion_tokens": 3,
                 "reasoning_tokens": 0, "total_tokens": 10}
    assert parse_usage(None)["total_tokens"] == 0


# ── 7. 窗口（D6 接管依据）──────────────────────────────────────────────────
def test_context_window_for_by_route(monkeypatch):
    """orchestrator（默认外部 8k）窗口 > stat（本地 4k）——ContextManager 预算换算依据"""
    monkeypatch.setattr(router_mod, "DISABLE_LOCAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_ENABLED", True)
    monkeypatch.setattr(router_mod, "FORCE_EXTERNAL_LLM", False)
    monkeypatch.setattr(router_mod, "EXTERNAL_LLM_AGENTS", frozenset({"orchestrator_agent"}))
    router = ModelRouter(_FakeLocal(), _FakeExternal())
    assert router.context_window_for("stat_agent") == 4096
    assert router.context_window_for("orchestrator_agent") == 8192
