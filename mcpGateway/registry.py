# -*- coding: utf-8 -*-
"""ToolRegistry：注册层（fail-fast）+ 发现层（权限过滤）+ 平台装配。

设计依据：`设计/08_架构演进方案与决策记录.md` §1.5 / §1.6。

- **注册层**：工具的唯一真相源。注册时 fail-fast——一切不一致都在装配时报出，
  绝不让坏工具带到运行时。
- **发现层**：让"这个 Agent 能用哪些工具"成为可查询、可导出、可自检的一等公民，
  `to_openai_schema()` 是 D7 与「LLM 自主选工具」的地基。
"""
from typing import TYPE_CHECKING, Any

from mcpGateway.rbac_config import get_agent_permission
from mcpGateway.tool_model import ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from mcpGateway.adapters import ProtocolAdapter
    from mcpGateway.executor import ExecutionEngine


class ToolRegistry:
    """注册层 + 发现层。执行由 `ExecutionEngine` 负责（本类持有其引用）。"""

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._adapters: dict[str, "ProtocolAdapter"] = {}
        self._engine: "ExecutionEngine | None" = None

    # ─────────────── 注册 API ───────────────
    def register(self, spec: ToolSpec) -> None:
        """注册一个工具。★fail-fast：校验不通过 / 重复注册 → 装配即报错"""
        self._validate_spec(spec)
        if spec.name in self._specs:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._specs[spec.name] = spec

    def register_adapter(self, protocol: str, adapter: "ProtocolAdapter") -> None:
        if protocol in self._adapters:
            raise ValueError(f"协议适配器重复注册: {protocol}")
        self._adapters[protocol] = adapter

    def tool(self, *, name: str | None = None, description: str,
             schema: dict, **kw) -> Any:
        """装饰器：本地函数一行注册成工具。

            @registry.tool(name="calc.premium", description="计算保费", schema={...})
            def calc_premium(...): ...
        """
        def deco(fn):
            to_thread = kw.pop("to_thread", True)
            self.register(ToolSpec(
                name=name or fn.__name__, description=description,
                params_schema=schema, protocol="local",
                binding={"func": fn, "to_thread": to_thread}, **kw))
            return fn
        return deco

    def unregister(self, name: str) -> None:
        """热卸载（D7 Skills 动态加载 / 卸载用）"""
        self._specs.pop(name, None)

    def _validate_spec(self, spec: ToolSpec) -> None:
        """★fail-fast：注册时的全部校验集中在这里"""
        if not spec.name or not spec.description or not spec.params_schema:
            raise ValueError(f"工具[{spec.name}] 缺少必填字段")
        if spec.protocol not in self._adapters:
            raise ValueError(f"工具[{spec.name}] 的协议[{spec.protocol}]未注册适配器")
        if spec.protocol == "mcp" and not (
                spec.binding.get("server_key") and spec.binding.get("tool")):
            raise ValueError(f"工具[{spec.name}] mcp 绑定缺 server_key/tool")
        if spec.protocol == "local" and "func" not in spec.binding:
            raise ValueError(f"工具[{spec.name}] local 绑定缺 func")

    def bind_engine(self, engine: "ExecutionEngine") -> None:
        self._engine = engine

    # ─────────────── 发现 API ───────────────
    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def get_adapter(self, protocol: str) -> "ProtocolAdapter":
        adapter = self._adapters.get(protocol)
        if adapter is None:
            raise KeyError(f"协议[{protocol}]未注册适配器")
        return adapter

    def all(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def has(self, name: str) -> bool:
        return name in self._specs

    def list_tools(self, agent_id: str) -> list[ToolSpec]:
        """★权限过滤即发现过滤：
        ① hidden=True 的内部工具不暴露
        ② required_perm=None → 公开工具，放行
        ③ required_perm ∈ get_agent_permission(agent_id) → 放行"""
        perms = set(get_agent_permission(agent_id))
        return [s for s in self._specs.values()
                if not s.hidden and (s.required_perm is None or s.required_perm in perms)]

    def to_openai_schema(self, agent_id: str,
                         tags: tuple[str, ...] | None = None) -> list[dict]:
        """导出为 OpenAI/Ollama function-calling 格式（LLM 自主选工具的输入）"""
        return [{"type": "function", "function": s.to_llm_schema()}
                for s in self.list_tools(agent_id)
                if not tags or set(tags) & set(s.tags)]

    def to_mcp_schema(self, agent_id: str) -> list[dict]:
        """导出为 MCP 工具列表格式（调试 / 网关转发用）"""
        return [{"name": s.name, "description": s.description,
                 "inputSchema": s.params_schema}
                for s in self.list_tools(agent_id)]

    async def verify_schemas(self) -> dict[str, list[str]]:
        """离线自检：对比本地注册的工具契约与 MCP 远端 tools/list 是否漂移。

        ★不进启动路径（MCP 子进程启动约 26~29s），作为 CI / 手动命令执行。
        """
        report: dict[str, list[str]] = {}
        for spec in self._specs.values():
            if spec.protocol != "mcp":
                continue
            adapter = self._adapters["mcp"]
            remote = {t["name"]: t for t in await adapter.list_remote_tools()}
            want = spec.binding["tool"]
            if want not in remote:
                report.setdefault(spec.name, []).append(f"远端缺少工具: {want}")
            elif _schema_diff(spec.params_schema, remote[want].get("inputSchema") or {}):
                report.setdefault(spec.name, []).append(
                    "；".join(_schema_diff(spec.params_schema,
                                           remote[want].get("inputSchema") or {})))
        return report

    # ─────────────── 统一执行入口（门面）───────────────
    async def invoke(self, name: str, args: dict, *, agent_id: str,
                     trace_id: str | None = None,
                     session_id: str | None = None):
        """业务代码只认识这一个入口，内部委托给 ExecutionEngine"""
        if self._engine is None:
            raise RuntimeError("ExecutionEngine 未绑定")
        return await self._engine.invoke(name, args, agent_id=agent_id,
                                         trace_id=trace_id, session_id=session_id)


def _schema_diff(local: dict, remote: dict) -> list[str]:
    """比较本地契约与远端声明，返回差异列表（空列表 = 一致）。

    只比「属性名集合」与「必填集合」：远端 schema 由 FastMCP 依类型注解生成，
    没有 additionalProperties 之类约束，全量深比较只会产生无意义噪声。
    远端多出的 `agent_id` 属预期差异（本地刻意不声明，防 LLM 伪造提权）。
    """
    local_props = set((local or {}).get("properties") or {})
    remote_props = set((remote or {}).get("properties") or {}) - {"agent_id", "run_id"}
    local_req = set((local or {}).get("required") or [])
    remote_req = set((remote or {}).get("required") or []) - {"agent_id", "run_id"}

    diff = []
    if local_props != remote_props:
        diff.append(f"属性不一致 本地={sorted(local_props)} 远端={sorted(remote_props)}")
    if local_req != remote_req:
        diff.append(f"必填不一致 本地={sorted(local_req)} 远端={sorted(remote_req)}")
    return diff


# ══════════════════ 平台装配 ══════════════════
_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """全局唯一工具注册中心（首次访问时惰性装配）。

    ★为什么是惰性 + 函数内延迟导入：
      ① 打破 `client → registry → adapters → client` 的模块级导入环；
      ② 保证"不跑 bootstrap 也能用"——集成测试直接调 `mcp_client.*` 时自动装配完成。
    """
    global _registry
    if _registry is None:
        _registry = _build_platform()
    return _registry


def _build_platform() -> ToolRegistry:
    """装配工具接入平台：适配器 → 4 个协议级工具（P0）→ 绑定执行引擎。

    工具清单与权限映射见 `08` §1.10。
    """
    # 函数内导入：此刻 client / adapters / executor 均已就绪，且不构成模块级循环导入
    from mcpGateway.adapters import LocalAdapter, MCPAdapter
    from mcpGateway.client import mcp_client
    from mcpGateway.executor import ExecutionEngine
    from mcpGateway.rbac_config import DBPerm, FactPerm, HabitPerm, LLMPerm
    # M4：工具级超时收敛到 config（TOOL_TIMEOUT_SQL / TOOL_TIMEOUT_LLM）
    from config.config import TOOL_TIMEOUT_SQL, TOOL_TIMEOUT_LLM

    reg = ToolRegistry()
    reg.register_adapter("mcp", MCPAdapter(mcp_client))
    reg.register_adapter("local", LocalAdapter())

    # ── 第一批：协议级工具（P0）──
    # ★M4 超时收紧（依据 `11` M4 / config TOOL_TIMEOUT_*，环境变量可配、≤0=关闭）：
    #   M1（2026-08-30）设 900s 是 stdio 短连接时代的"防挂死 + 绝不误杀"值——当时每次
    #   调用付 26~29s 冷启动，本地 1.8B 单次 llm.chat 实测 189s；初版 300s 因冷启动叠加被
    #   误触发，wait_for 取消协程会让 stdio_client 抛 "unhandled errors in a TaskGroup"
    #   （回归实录 2026-08-30）。M3 常驻化消除冷启动后按工具类型区分收紧：
    #     SQL=30s（SQLite 本地库 + busy_timeout 5s，M3 实测调用 135ms，30s 极宽裕）
    #     LLM=300s（本地 189s 仍在覆盖范围内，外部 API 更快）。
    # max_retry=0 ：底层 `_call_server` 已自带 2 次连接重试，外层再重试会放大成乘数效应
    #                （2×N 次），故引擎层 M1 阶段不重试，行为与现状一致。
    def _sql_props(desc: str) -> dict:
        return {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": desc},
                "params": {"type": "array", "description": "占位符参数列表", "items": {}},
            },
            "required": ["sql"],
            "additionalProperties": False,
        }

    llm_schema = {
        "type": "object",
        "properties": {
            "system_prompt": {"type": "string", "description": "系统提示词"},
            "user_input": {"type": "string", "description": "用户输入"},
        },
        "required": ["system_prompt", "user_input"],
        "additionalProperties": False,
    }

    reg.register(ToolSpec(
        name="sql.read",
        description="只读查询账单数据库（SELECT），返回查询结果",
        params_schema=_sql_props("SELECT 查询语句"),
        protocol="mcp", binding={"server_key": "sql_bill", "tool": "exec_sql"},
        required_perm=DBPerm.BILL_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("sql", "db")))

    reg.register(ToolSpec(
        name="sql.write",
        description="写入账单数据库（INSERT / UPDATE / DELETE）",
        params_schema=_sql_props("写操作 SQL 语句"),
        protocol="mcp", binding={"server_key": "sql_bill", "tool": "exec_sql"},
        required_perm=DBPerm.BILL_WRITE, side_effect=True, auditable=True,
        idempotent=False, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("sql", "db")))

    reg.register(ToolSpec(
        name="llm.chat",
        description="调用通用基座大模型（结构化提取 / 文案生成 / 意图规划）",
        params_schema=llm_schema, protocol="mcp",
        # ★server 侧参数名为 agent_tag（用于差异化推理参数，见踩坑记录坑 12），
        #   由引擎注入而非写进 schema，避免 LLM 伪造调用主体提权。
        binding={"server_key": "llm_base", "tool": "llm_chat",
                 "inject_agent_id": "agent_tag", "inject_run_id": True},
        required_perm=LLMPerm.LLM_BASE, side_effect=False,
        auditable=False,        # 对齐现状：原 call_llm_base 不写审计，避免高频刷屏
        idempotent=False, timeout=TOOL_TIMEOUT_LLM, max_retry=0, tags=("llm",)))

    reg.register(ToolSpec(
        name="llm.finance",
        description="调用理财专属大模型生成理财建议（仅理财 Agent 可用）",
        params_schema=llm_schema, protocol="mcp",
        binding={"server_key": "llm_finance", "tool": "finance_chat",
                 "inject_run_id": True},
        required_perm=LLMPerm.LLM_FINANCE, side_effect=False, auditable=True,
        idempotent=False, timeout=TOOL_TIMEOUT_LLM, max_retry=0, tags=("llm", "finance")))

    # ── 第二批：业务事实工具（M9 / D15，P0-4+P0-5）──
    # 收口 orchestrator 9 处 `db.` 直连：读走网关（FACT_READ）+ habit 写下沉 bill_agent（HABIT_WRITE）。
    # ★方案差异：transport=local（`11` M9 走查 #4 登记；`03` §9.9.2 表原标 mcp，文档回写时修订）。
    #   local 工具执行：同步 fn → 引擎 to_thread；habit.upsert 为 async（协程内持 db_lock）。
    from mcpGateway import fact_tools

    def _empty_props() -> dict:
        return {"type": "object", "properties": {},
                "required": [], "additionalProperties": False}

    def _month_props() -> dict:
        return {"type": "object",
                "properties": {"month": {"type": "string", "description": "自然月 YYYY-MM"}},
                "required": ["month"], "additionalProperties": False}

    city_price_schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名"},
            "category": {"type": "string", "description": "消费品类"},
        },
        "required": ["city", "category"],
        "additionalProperties": False,
    }

    habit_list_schema = {
        "type": "object",
        "properties": {
            "months": {"type": "array", "description": "自然月列表 YYYY-MM",
                       "items": {"type": "string"}},
        },
        "required": ["months"],
        "additionalProperties": False,
    }

    habit_upsert_schema = {
        "type": "object",
        "properties": {
            "month": {"type": "string", "description": "自然月 YYYY-MM"},
            "category": {"type": "string", "description": "消费品类"},
            "amount": {"type": "number", "description": "本次消费金额（元）"},
        },
        "required": ["month", "category", "amount"],
        "additionalProperties": False,
    }

    reg.register(ToolSpec(
        name="config.get_latest",
        description="读取用户最新预算/城市配置（user_config 最新一行）",
        params_schema=_empty_props(), protocol="local",
        binding={"func": fact_tools.get_latest_user_config, "to_thread": True},
        required_perm=FactPerm.FACT_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("fact", "db")))

    reg.register(ToolSpec(
        name="bill.sum_by_month",
        description="查询指定自然月账单金额总和",
        params_schema=_month_props(), protocol="local",
        binding={"func": fact_tools.sum_month_bills, "to_thread": True},
        required_perm=FactPerm.FACT_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("fact", "db")))

    reg.register(ToolSpec(
        name="bill.sum_by_category",
        description="查询指定自然月各消费品类累计金额 {category: total}",
        params_schema=_month_props(), protocol="local",
        binding={"func": fact_tools.sum_month_bills_by_category, "to_thread": True},
        required_perm=FactPerm.FACT_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("fact", "db")))

    reg.register(ToolSpec(
        name="city_price.get_avg",
        description="查询当地同品类基准均价（本城无则同级城市兜底，仍无返回 0）",
        params_schema=city_price_schema, protocol="local",
        binding={"func": fact_tools.get_city_avg_price, "to_thread": True},
        required_perm=FactPerm.FACT_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("fact", "db")))

    reg.register(ToolSpec(
        name="habit.list",
        description="查询近 N 个自然月消费习惯聚合（monthly_habit）",
        params_schema=habit_list_schema, protocol="local",
        binding={"func": fact_tools.list_habits, "to_thread": True},
        required_perm=FactPerm.FACT_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("fact", "db")))

    reg.register(ToolSpec(
        name="habit.upsert",
        description="月度消费习惯沉淀（写 monthly_habit：金额累加 / 笔数+1 / 均额重算）",
        params_schema=habit_upsert_schema, protocol="local",
        # ★async 函数：协程内持 db_lock，防 LocalAdapter to_thread 丢锁（`11` M9 走查 #2/#5）
        binding={"func": fact_tools.upsert_habit, "to_thread": False},
        required_perm=HabitPerm.HABIT_WRITE, side_effect=True, auditable=True,
        idempotent=False, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("fact", "db", "write")))

    reg.bind_engine(ExecutionEngine(reg))
    return reg


__all__ = ["ToolRegistry", "get_registry"]
