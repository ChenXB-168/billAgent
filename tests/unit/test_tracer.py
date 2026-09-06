# -*- coding: utf-8 -*-
"""
D3 Tracer 单元测试（阶段1）

★全部用例**不依赖外部 API**，可离线运行（外部额度耗尽时也能验证）。
覆盖目标：
  1. llm_call_stat 表可用、统计记录能落库
  2. OpenAI 兼容 usage 解析正确（含 reasoning_tokens）
  3. run_scope / span 上下文能写入 run_id、span_id、parent_span_id
  4. **铁律验证**：Tracer 自身异常不得阻断主链路（`03_逻辑架构.md:534`）
  5. 聚合查询可用（按 agent 归因）
"""
import pytest

from database.init_db import init_all_tables
from utils.tracer import (
    record_llm_call, run_scope, span, summary_by_agent, summary_by_run,
    current_run_id, current_span_id, new_run_id,
)
# M14（D17）：usage 解析收口于 Provider（llm_loader 双埋点 helper 已消除，见 `11` §3 M14）
from modelService.providers.openai_compat import parse_usage

TEST_AGENT = "tracer_unit_test_agent"
# M7 起 span 结束会落 trace_span 表：run_scope 用例固定 run_id，便于清理
# （历史随机 run_id 的残留由模块级 _ensure_table 一次性清除）
TEST_RUN_ID = "rtest_ut_runscope"


@pytest.fixture(scope="module", autouse=True)
def _ensure_table():
    """确保统计表存在（幂等，重复运行安全）

    ★close_after=False：`db` 是全局单例且 close() 单向不可重开，
    默认 init_all_tables() 会关闭它，导致后续用例的 SQL 全部失败。
    """
    init_all_tables(close_after=False)
    from utils.common import db
    # 阶段1 遗留清理：M7 前 run_scope 用例以随机 run_id 落 span（plan/dispatch 为
    # 本文件专用 span 名，真实链路为 orchestrator.run/{agent}.run/tool.*，可安全删除）
    db.execute_sql("DELETE FROM trace_span WHERE name IN ('plan', 'dispatch')")
    db.execute_sql("DELETE FROM trace_span WHERE run_id = ?", (TEST_RUN_ID,))


@pytest.fixture(autouse=True)
def _clean():
    """每条用例前后清理本用例写入的测试行，避免污染统计"""
    from utils.common import db
    db.execute_sql("DELETE FROM llm_call_stat WHERE agent_tag = ?", (TEST_AGENT,))
    db.execute_sql("DELETE FROM trace_span WHERE run_id = ?", (TEST_RUN_ID,))
    yield
    db.execute_sql("DELETE FROM llm_call_stat WHERE agent_tag = ?", (TEST_AGENT,))
    db.execute_sql("DELETE FROM trace_span WHERE run_id = ?", (TEST_RUN_ID,))


def _rows():
    from utils.common import db
    return db.query_sql(
        "SELECT * FROM llm_call_stat WHERE agent_tag = ? ORDER BY id ASC", (TEST_AGENT,))


def test_table_created():
    """统计表可用"""
    record_llm_call(agent_tag=TEST_AGENT, model="ut-model", channel="external",
                    prompt_tokens=10, completion_tokens=5, latency_ms=100)
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["total_tokens"] == 15
    assert rows[0]["success"] == 1


def test_external_usage_parsing():
    """OpenAI 兼容 usage 解析：含推理模型的 reasoning_tokens"""
    fake_resp = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 460,           # 故意 ≠ 120+80，验证优先取原始 total
            "completion_tokens_details": {"reasoning_tokens": 260},
        },
    }
    u = parse_usage(fake_resp)
    assert u["prompt_tokens"] == 120
    assert u["completion_tokens"] == 80
    assert u["reasoning_tokens"] == 260   # 推理消耗——定位"预算被 reasoning 占满"的关键指标
    assert u["total_tokens"] == 460       # 优先取原始 total（≠ prompt+completion）
    # 解析结果经 record_llm_call 落库（M14 起埋点统一收口于此）
    record_llm_call(agent_tag=TEST_AGENT, model="hy3", channel="external",
                    latency_ms=1234, success=True,
                    prompt_tokens=u["prompt_tokens"],
                    completion_tokens=u["completion_tokens"],
                    reasoning_tokens=u["reasoning_tokens"],
                    total_tokens=u["total_tokens"])
    row = _rows()[0]
    assert row["prompt_tokens"] == 120
    assert row["completion_tokens"] == 80
    assert row["reasoning_tokens"] == 260
    assert row["total_tokens"] == 460
    assert row["latency_ms"] == 1234
    assert row["channel"] == "external"
    assert row["success"] == 1


def test_run_scope_and_span_context():
    """run_id / span_id / parent_span_id 能随上下文写入记录"""
    with run_scope(session_id="sess-ut", run_id=TEST_RUN_ID) as run_id:
        assert run_id and current_run_id() == run_id
        with span("plan") as outer:
            record_llm_call(agent_tag=TEST_AGENT, model="m", channel="external",
                            prompt_tokens=1, completion_tokens=1)
            with span("dispatch") as inner:
                record_llm_call(agent_tag=TEST_AGENT, model="m", channel="external",
                                prompt_tokens=2, completion_tokens=2)
                assert current_span_id() == inner

    rows = _rows()
    assert len(rows) == 2
    assert all(r["run_id"] == run_id for r in rows)
    assert all(r["session_id"] == "sess-ut" for r in rows)
    assert rows[0]["span_id"] == outer
    assert rows[0]["parent_span_id"] is None          # 首层 span 无父
    assert rows[1]["span_id"] == inner
    assert rows[1]["parent_span_id"] == outer         # 嵌套 span 记录父节点


def test_tracer_never_blocks_pipeline(monkeypatch):
    """★铁律（`03`:534）：Tracer 自身故障不得抛异常、不得影响主链路"""
    import utils.common

    def _boom(*a, **kw):
        raise RuntimeError("模拟 DB 故障")

    # tracer 内部为延迟 import（from utils.common import db），拿到的是该单例实例，
    # 故 patch 实例方法即可命中
    monkeypatch.setattr(utils.common.db, "execute_sql", _boom)
    # 不得抛出任何异常
    record_llm_call(agent_tag=TEST_AGENT, model="m", channel="external", prompt_tokens=1)


def test_summary_by_agent():
    """按 agent 聚合：调用次数 / token / 失败数"""
    for i in range(3):
        record_llm_call(agent_tag=TEST_AGENT, model="m", channel="external",
                        prompt_tokens=10, completion_tokens=5, latency_ms=100,
                        success=(i != 2), error_type=None if i != 2 else "EMPTY_CONTENT")

    summary = [r for r in summary_by_agent() if r["agent_tag"] == TEST_AGENT]
    assert len(summary) == 1
    s = summary[0]
    assert s["call_cnt"] == 3
    assert s["prompt_tokens"] == 30
    assert s["completion_tokens"] == 15
    assert s["total_tokens"] == 45
    assert s["fail_cnt"] == 1          # 失败调用被单独计数


def test_summary_by_run_empty_is_safe():
    """查询不存在的 run_id 返回空列表，不抛异常"""
    assert summary_by_run(new_run_id()) == []
