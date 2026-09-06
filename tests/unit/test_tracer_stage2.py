# -*- coding: utf-8 -*-
"""M7（D3 阶段2）Tracer 测试：run_id 规则 / span 树落库 / 工具层打点 / 查看脚本渲染。

对应验收（`08` §5.2 M7 行 / `11` §3 M7）：
- run_id 规则与 task_id 同构（r{ms}{hex3}，08 §3.2）
- run_scope 支持恢复既有 run_id（worker 跨协程）
- span 结束自动落 trace_span（kind/status/duration/error），异常标 error 且不吞异常
- record_tool_span 挂 parent（engine ⑨ 在 {agent}.run span 内执行）
- trace_view.render_span_tree ASCII 渲染
- 无 run 上下文时工具 span 静默跳过（不污染统计）

★表隔离：本文件所有写入均用固定测试 run_id 前缀 `rtest_m7_`，finally 清理，
不污染真实 run 数据（与 llm_call_stat 阶段1 测试同策略）。
"""
import asyncio
import re
import unittest

from database.init_db import ensure_trace_tables
from utils.common import db
from utils import tracer
from utils.trace_view import render_span_tree

# 每次用例的测试 run_id（固定便于清理）
RUN_PREFIX = "rtest_m7_"


def _cleanup():
    try:
        db.execute_sql("DELETE FROM trace_span WHERE run_id LIKE ?", (RUN_PREFIX + "%",))
        db.execute_sql("DELETE FROM llm_call_stat WHERE run_id LIKE ?", (RUN_PREFIX + "%",))
    except Exception:
        pass


def setUpModule():  # noqa: N802 —— unittest 钩子
    ensure_trace_tables(db)


def tearDownModule():  # noqa: N802
    _cleanup()


class TracerStage2Test(unittest.TestCase):
    def setUp(self):
        _cleanup()

    def tearDown(self):
        _cleanup()

    # ── run_id 规则与 run_scope 复用 ──────────────────────────────
    def test_run_id_rule_matches_task_id_style(self):
        rid = tracer.new_run_id()
        # r{ms}{hex3}：08 §3.2，与 task_id t{ms}{hex3} 同构
        self.assertRegex(rid, r"^r\d{13}[0-9a-f]{6}$")

    def test_run_scope_generates_and_restores(self):
        with tracer.run_scope(session_id="s1") as rid:
            self.assertEqual(tracer.current_run_id(), rid)
            self.assertRegex(rid, r"^r\d{13}[0-9a-f]{6}$")
        self.assertIsNone(tracer.current_run_id())

    def test_run_scope_accepts_existing_run_id(self):
        # worker 跨协程恢复场景：编排入口生成 → dispatch 下发 → worker 显式重建
        with tracer.run_scope(session_id="s1", run_id=RUN_PREFIX + "reuse") as rid:
            self.assertEqual(rid, RUN_PREFIX + "reuse")
            self.assertEqual(tracer.current_run_id(), rid)

    def test_bind_run_id_writes_current_process_context(self):
        tracer.bind_run_id(RUN_PREFIX + "bound", session_id="s_bound")
        try:
            self.assertEqual(tracer.current_run_id(), RUN_PREFIX + "bound")
        finally:
            tracer._RUN_ID.set(None)  # 测试后清理（bind 不 reset，见函数文档）

    # ── span 落库 / 树结构 ────────────────────────────────────────
    def test_span_writes_row_and_tree(self):
        with tracer.run_scope(session_id="s2", run_id=RUN_PREFIX + "tree"):
            with tracer.span("orchestrator.run", kind="orchestrator",
                             agent_id="orchestrator_agent"):
                with tracer.span("bill_agent.run", kind="agent",
                                 agent_id="bill_agent"):
                    pass
        rows = tracer.load_spans(RUN_PREFIX + "tree")
        self.assertEqual(len(rows), 2)
        # 落库序 = 退出序（子先父后），用结构定位而非行序
        root = next(r for r in rows if not r["parent_span_id"])
        child = next(r for r in rows if r["parent_span_id"] == root["span_id"])
        self.assertEqual(root["name"], "orchestrator.run")
        self.assertIsNone(root["parent_span_id"])
        self.assertEqual(child["name"], "bill_agent.run")
        self.assertEqual(child["parent_span_id"], root["span_id"])
        self.assertEqual(child["kind"], "agent")
        self.assertEqual(child["agent_id"], "bill_agent")
        self.assertEqual(child["status"], "ok")

    def test_span_marks_error_and_reraises(self):
        with tracer.run_scope(session_id="s3", run_id=RUN_PREFIX + "err"):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent"):
                    raise RuntimeError("boom")
        rows = tracer.load_spans(RUN_PREFIX + "err")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("RuntimeError: boom", rows[0]["error"])

    def test_record_tool_span_parent_and_noise_skip(self):
        # 在 {agent}.run span 内记录工具调用 → parent 自动归属
        with tracer.run_scope(session_id="s4", run_id=RUN_PREFIX + "tool"):
            with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent"):
                tracer.record_tool_span(tool_name="llm.chat", agent_id="bill_agent",
                                        ok=True, duration_ms=123)
                tracer.record_tool_span(tool_name="sql.write", agent_id="bill_agent",
                                        ok=False, duration_ms=45, error="insert failed")
        rows = tracer.load_spans(RUN_PREFIX + "tool")
        self.assertEqual(len(rows), 3)
        agent_span = next(r for r in rows if r["name"] == "bill_agent.run")
        tools = [r for r in rows if r["kind"] == "tool"]
        self.assertEqual(len(tools), 2)
        for t in tools:
            self.assertEqual(t["parent_span_id"], agent_span["span_id"])
            self.assertEqual(t["agent_id"], "bill_agent")
        self.assertEqual(tools[0]["name"], "tool.llm.chat")
        self.assertEqual(tools[0]["duration_ms"], 123)
        self.assertEqual(tools[1]["status"], "error")
        self.assertIn("insert failed", tools[1]["error"])

    def test_tool_span_without_run_context_is_skipped(self):
        # 无 run_scope（registry 独立直调 / 单测）→ 静默不落库，不污染统计
        tracer.record_tool_span(tool_name="llm.chat", agent_id="x", ok=True,
                                duration_ms=10)
        self.assertEqual(tracer.load_spans(RUN_PREFIX + "never"), [])

    # ── 查看脚本渲染 ──────────────────────────────────────────────
    def test_render_span_tree(self):
        with tracer.run_scope(session_id="s5", run_id=RUN_PREFIX + "render"):
            with tracer.span("orchestrator.run", kind="orchestrator",
                             agent_id="orchestrator_agent"):
                with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent"):
                    tracer.record_tool_span(tool_name="llm.chat", agent_id="bill_agent",
                                            ok=True, duration_ms=10)
                with tracer.span("stat_agent.run", kind="agent", agent_id="stat_agent"):
                    pass
        spans = tracer.load_spans(RUN_PREFIX + "render")
        tree = render_span_tree(spans)
        self.assertIn("orchestrator.run", tree)
        self.assertIn("bill_agent.run", tree)
        self.assertIn("stat_agent.run", tree)
        self.assertIn("tool.llm.chat", tree)
        self.assertIn("ok 10ms", tree)  # 工具行含耗时
        # ASCII 树线（CLI 与 Windows GBK 兼容）
        self.assertIn("+-- ", tree)
        self.assertIn("`-- ", tree)

    def test_span_accepts_explicit_parent_cross_coroutine(self):
        # worker 场景：编排协程退出 root span 后，worker 协程凭 task_data 里的
        # run_id + parent_span_id 把 {agent}.run 挂回同一棵根（树合并为一棵）
        with tracer.run_scope(session_id="s7", run_id=RUN_PREFIX + "merge"):
            with tracer.span("orchestrator.run", kind="orchestrator",
                             agent_id="orchestrator_agent") as root_sid:
                pass  # root span 先退出落库
            # 模拟 worker 协程（无外层 span 栈）
            with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent",
                             parent_span_id=root_sid):
                tracer.record_tool_span(tool_name="llm.chat", agent_id="bill_agent",
                                        ok=True, duration_ms=5)
        rows = tracer.load_spans(RUN_PREFIX + "merge")
        self.assertEqual(len(rows), 3)
        bill = next(r for r in rows if r["name"] == "bill_agent.run")
        tool = next(r for r in rows if r["kind"] == "tool")
        self.assertEqual(bill["parent_span_id"], root_sid)
        self.assertEqual(tool["parent_span_id"], bill["span_id"])

    def test_render_empty(self):
        self.assertIn("无 span 记录", render_span_tree([]))


class ExecutionEngineSpanTest(unittest.IsolatedAsyncioTestCase):
    """executor ⑨ 打点接入 trace_span 的端到端（fake adapter，不连 MCP server）"""

    async def asyncSetUp(self):
        _cleanup()
        from mcpGateway.executor import ExecutionEngine
        from mcpGateway.tool_model import ToolSpec
        import time as _time

        spec = ToolSpec(
            name="demo_tool",
            description="d",
            params_schema={},
            protocol="local",
            binding={"func": lambda: None},
            required_perm=None,
            side_effect=False,
            auditable=False,
            idempotent=False,
            timeout=None,
            max_retry=0,
            tags=(),
        )

        class _FakeReg:
            def get(self, name):
                return spec if name == "demo_tool" else None

            def get_adapter(self, protocol):
                class _A:
                    async def invoke(self, s, args, ctx):
                        await asyncio.sleep(0.002)
                        return {"echo": args}
                return _A()

        self._engine = ExecutionEngine(_FakeReg(), audit=None)
        self._t0 = _time

    async def test_engine_records_tool_span_under_agent_span(self):
        with tracer.run_scope(session_id="s6", run_id=RUN_PREFIX + "engine"):
            with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent"):
                res = await self._engine.invoke("demo_tool", {"x": 1},
                                                agent_id="bill_agent")
        self.assertTrue(res.ok)
        rows = tracer.load_spans(RUN_PREFIX + "engine")
        tools = [r for r in rows if r["kind"] == "tool"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "tool.demo_tool")
        self.assertEqual(tools[0]["agent_id"], "bill_agent")
        self.assertGreaterEqual(tools[0]["duration_ms"], 1)
        self.assertEqual(tools[0]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
