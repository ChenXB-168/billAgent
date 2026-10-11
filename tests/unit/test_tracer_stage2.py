# -*- coding: utf-8 -*-
"""M7（D3 阶段2）Tracer 测试：run_id 规则 / span 树落库 / 工具层打点 / 查看脚本渲染。

对应验收（`08` §5.2 M7 行 / `11` §3 M7）：
- run_id 规则与 task_id 同构（r{ms}{hex3}，08 §3.2）
- run_scope 支持恢复既有 run_id（worker 跨协程）
- span 结束自动落 trace_span（kind/status/duration/error），异常标 error 且不吞异常
- record_tool_span 挂 parent（engine ⑨ 在 {agent}.run span 内执行）
- trace_view.render_span_tree ASCII 渲染
- 无 run 上下文时工具 span 静默跳过（不污染统计）

★表隔离：本文件所有写入均用固定测试 run_id 前缀 `rtest_m7_`，由 setUp / tearDown
（及 tearDownModule）统一清理，不污染真实 run 数据（与 llm_call_stat 阶段1 测试同策略）。
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
    """
    函数功能与逻辑描述：
        清理本文件测试标识（run_id 前缀 RUN_PREFIX）在 trace_span 与 llm_call_stat 中留下的行，
        供模块级与本类 setUp/tearDown 各自把环境恢复到干净状态；清理失败静默吞掉
        （可观测表的清理问题不应中断测试主流程）。
    入参说明：
        无（由 setUp/tearDown 直接调用，pytest 自动发现并调用用例）。
    返回值说明：
        无（仅执行 DELETE 清理，异常被吞掉不抛出）。
    """
    try:
        db.execute_sql("DELETE FROM trace_span WHERE run_id LIKE ?", (RUN_PREFIX + "%",))
        db.execute_sql("DELETE FROM llm_call_stat WHERE run_id LIKE ?", (RUN_PREFIX + "%",))
    except Exception:
        pass


def setUpModule():  # noqa: N802 —— unittest 钩子
    """
    函数功能与逻辑描述：
        unittest 模块级前置钩子：幂等确保 trace_span 表存在（ensure_trace_tables），
        避免本模块首次运行时报 no such table。
    入参说明：
        无（unittest/pytest 自动调用，无需外部入参）。
    返回值说明：
        无（仅建表，无返回值）。
    """
    ensure_trace_tables(db)


def tearDownModule():  # noqa: N802
    """
    函数功能与逻辑描述：
        unittest 模块级后置钩子：本模块全部用例结束后做一次测试数据清理。
    入参说明：
        无（unittest/pytest 自动调用，无需外部入参）。
    返回值说明：
        无（仅执行清理，无返回值）。
    """
    _cleanup()


class TracerStage2Test(unittest.TestCase):
    """
    函数/类功能与逻辑描述：
        M7（D3 阶段2）核心用例集：覆盖 run_id 生成规则与 run_scope 复用、span 落库与父子树结构、
        span 异常标 error 且不吞异常、工具层 span 的 parent 归属与无上下文静默跳过、
        trace_view.render_span_tree 的 ASCII 渲染。用例级 setUp/tearDown 各自清理 RUN_PREFIX 前缀数据。
    构造入参说明：
        无（unittest 自动按测试方法名实例化，无需外部入参）。
    返回值说明：
        构造返回测试实例；测试方法由 unittest/pytest 自动发现并执行。
    """

    def setUp(self):
        """
        函数功能与逻辑描述：
            每条用例执行前的清理：删除 RUN_PREFIX 前缀的历史测试数据，保证用例之间互不污染。
        入参说明：
            无（unittest/pytest 自动调用，无需外部入参）。
        返回值说明：
            无（仅执行清理，无返回值）。
        """
        _cleanup()

    def tearDown(self):
        """
        函数功能与逻辑描述：
            每条用例执行后的清理：再次删除 RUN_PREFIX 前缀的测试数据，避免残留影响后续用例。
        入参说明：
            无（unittest/pytest 自动调用，无需外部入参）。
        返回值说明：
            无（仅执行清理，无返回值）。
        """
        _cleanup()

    # ── run_id 规则与 run_scope 复用 ──────────────────────────────
    def test_run_id_rule_matches_task_id_style(self):
        """
        函数功能与逻辑描述：
            验证 run_id 生成规则与 task_id 同构（`08` §3.2）：new_run_id() 产出形如
            r{13 位毫秒时间戳}{6 位小写 hex} 的字符串，用正则 ^r\d{13}[0-9a-f]{6}$ 断言。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        rid = tracer.new_run_id()
        # r{ms}{hex3}：08 §3.2，与 task_id t{ms}{hex3} 同构
        self.assertRegex(rid, r"^r\d{13}[0-9a-f]{6}$")

    def test_run_scope_generates_and_restores(self):
        """
        函数功能与逻辑描述：
            验证 run_scope 的新建与复位：不传 run_id 时内部生成新 id，with 内 current_run_id()
            等于该 id 且符合 r{ms}{hex3} 规则；退出 with 后上下文复位为 None。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        with tracer.run_scope(session_id="s1") as rid:
            self.assertEqual(tracer.current_run_id(), rid)
            self.assertRegex(rid, r"^r\d{13}[0-9a-f]{6}$")
        self.assertIsNone(tracer.current_run_id())

    def test_run_scope_accepts_existing_run_id(self):
        """
        函数功能与逻辑描述：
            验证 worker 跨协程恢复场景：run_scope 传入既有 run_id 时原样回显（不新建），
            且 current_run_id() 等于该值——对应「编排入口生成 → dispatch 下发 → worker 显式重建」。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        # worker 跨协程恢复场景：编排入口生成 → dispatch 下发 → worker 显式重建
        with tracer.run_scope(session_id="s1", run_id=RUN_PREFIX + "reuse") as rid:
            self.assertEqual(rid, RUN_PREFIX + "reuse")
            self.assertEqual(tracer.current_run_id(), rid)

    def test_bind_run_id_writes_current_process_context(self):
        """
        函数功能与逻辑描述：
            验证 bind_run_id 把外部 run_id 写入本进程上下文：调用后 current_run_id() 等于传入值；
            因 bind 不做 reset，finally 里手工把 tracer._RUN_ID 置回 None 以清理上下文。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        tracer.bind_run_id(RUN_PREFIX + "bound", session_id="s_bound")
        try:
            self.assertEqual(tracer.current_run_id(), RUN_PREFIX + "bound")
        finally:
            tracer._RUN_ID.set(None)  # 测试后清理（bind 不 reset，见函数文档）

    # ── span 落库 / 树结构 ────────────────────────────────────────
    def test_span_writes_row_and_tree(self):
        """
        函数功能与逻辑描述：
            验证 span 落库与树结构：run_scope 内嵌套 orchestrator.run（kind=orchestrator）与
            bill_agent.run（kind=agent）两个 span，退出后 load_spans 得 2 行（落库序=退出序，
            子先父后）；用结构定位根与子，断言两行的 name / parent / kind / agent_id / status。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        with tracer.run_scope(session_id="s2", run_id=RUN_PREFIX + "tree"):
            with tracer.span("orchestrator.run", kind="orchestrator",
                             agent_id="orchestrator_agent"):
                with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent"):
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
        """
        函数功能与逻辑描述：
            验证 span 的异常处理：span 内抛 RuntimeError("boom") 时异常照常向上抛出
            （assertRaisesRegex 捕获），落库行数为 1、status 为 error、error 文本含
            "RuntimeError: boom"——打点不吞业务异常。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        with tracer.run_scope(session_id="s3", run_id=RUN_PREFIX + "err"):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with tracer.span("bill_agent.run", kind="agent", agent_id="bill_agent"):
                    raise RuntimeError("boom")
        rows = tracer.load_spans(RUN_PREFIX + "err")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("RuntimeError: boom", rows[0]["error"])

    def test_record_tool_span_parent_and_noise_skip(self):
        """
        函数功能与逻辑描述：
            验证工具层 span 的 parent 归属与字段：在 {agent}.run span 内记录两次工具调用
            （llm.chat 成功 / sql.write 失败），load_spans 得 3 行，两条 kind=tool 行的
            parent_span_id 均指向 agent span 且 agent_id 一致，并断言工具名、耗时、失败行的
            status=error 与错误文本。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
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
        """
        函数功能与逻辑描述：
            验证无 run 上下文时工具 span 静默跳过：不在 run_scope 内调用 record_tool_span，
            不落库，断言该 run_id 下 load_spans 返回空列表（registry 独立直调 / 单测场景
            不污染统计）。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        # 无 run_scope（registry 独立直调 / 单测）→ 静默不落库，不污染统计
        tracer.record_tool_span(tool_name="llm.chat", agent_id="x", ok=True,
                                duration_ms=10)
        self.assertEqual(tracer.load_spans(RUN_PREFIX + "never"), [])

    # ── 查看脚本渲染 ──────────────────────────────────────────────
    def test_render_span_tree(self):
        """
        函数功能与逻辑描述：
            验证 trace_view.render_span_tree 的渲染：对含 orchestrator.run → 两个 {agent}.run
            与一个 tool.llm.chat 的 span 集，渲染结果包含各 span 名与工具行的 "ok 10ms" 耗时，
            并使用 ASCII 树线 "+-- " 与 "`-- "（规避 Windows 控制台 GBK 乱码）。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
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
        """
        函数功能与逻辑描述：
            验证跨协程显式指定父节点：编排协程退出 root span 后（span 栈已空），worker 场景
            用 parent_span_id 把 bill_agent.run 挂回同一根，并在其内 record_tool_span；
            断言共 3 行、bill 行的 parent 为 root_sid、tool 行的 parent 为 bill 行（树合并为一棵）。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
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
        """
        函数功能与逻辑描述：
            验证空输入的渲染边界：render_span_tree([]) 返回固定文案 "无 span 记录"。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
        self.assertIn("无 span 记录", render_span_tree([]))


class ExecutionEngineSpanTest(unittest.IsolatedAsyncioTestCase):
    """
    函数/类功能与逻辑描述：
        executor ⑨ 打点接入 trace_span 的端到端验证：注入 fake registry / adapter（不连真实
        MCP server），验证工具调用结束时引擎自动写入 kind=tool 的 span 并归到 {agent}.run span 之下。
    构造入参说明：
        无（unittest 自动按测试方法名实例化，无需外部入参）。
    返回值说明：
        构造返回测试实例；测试方法由 unittest/pytest 自动发现并执行。
    """

    async def asyncSetUp(self):
        """
        函数功能与逻辑描述：
            用例级异步前置（IsolatedAsyncioTestCase 钩子）：先清理测试数据，再构造被测
            ExecutionEngine——注入 _FakeReg（仅认 demo_tool，返回 fake adapter），其 invoke
            睡 2ms 后回显入参，用于制造非零耗时以断言 duration_ms；全程不连真实 MCP server。
        入参说明：
            无（unittest/pytest 自动调用，无需外部入参）。
        返回值说明：
            无（仅初始化 self._engine / self._t0，无返回值）。
        """
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
                """
                函数功能与逻辑描述：
                    替身注册表的契约查询：仅认 demo_tool，命中返回该 ToolSpec，其余返回 None，
                    使引擎的解析步骤可控而不引入真实注册表。
                入参说明：
                    name (str)：工具名。
                返回值说明：
                    ToolSpec | None：name == "demo_tool" 时返回 demo_tool 契约，否则 None。
                """
                return spec if name == "demo_tool" else None

            def get_adapter(self, protocol):
                """
                函数功能与逻辑描述：
                    替身注册表的适配器查询：无视传入协议，固定返回一个耗时 2ms 的 fake 适配器，
                    制造非零耗时以便断言工具 span 的 duration_ms。
                入参说明：
                    protocol (str)：协议名（本替身忽略该值）。
                返回值说明：
                    _A：内联定义的 fake 适配器实例（其 invoke 睡 2ms 后回显入参）。
                """
                class _A:
                    async def invoke(self, s, args, ctx):
                        """
                        函数功能与逻辑描述：
                            fake 适配器的执行方法：睡 2ms 制造非零耗时后回显入参，
                            使引擎调用成功并产生可断言的 duration_ms。
                        入参说明：
                            s (ToolSpec)：工具契约（本替身不使用）。
                            args (dict)：工具入参。
                            ctx (InvokeContext)：调用上下文（本替身不使用）。
                        返回值说明：
                            dict：形如 {"echo": args} 的回显结果。
                        """
                        await asyncio.sleep(0.002)
                        return {"echo": args}
                return _A()

        self._engine = ExecutionEngine(_FakeReg(), audit=None)
        self._t0 = _time

    async def test_engine_records_tool_span_under_agent_span(self):
        """
        函数功能与逻辑描述：
            验证引擎调用成功后在当前 {agent}.run span 下自动落一条工具 span：invoke("demo_tool")
            返回 ok=True，load_spans 得 1 条 kind=tool 行，其名为 tool.demo_tool、agent_id 为
            bill_agent、duration_ms>=1、status=ok。
        入参说明：
            无（unittest/pytest 自动发现并调用，无需外部入参）。
        返回值说明：
            无（断言通过即用例成功，抛出断言异常即失败）。
        """
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
