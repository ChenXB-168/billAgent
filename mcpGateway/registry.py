# -*- coding: utf-8 -*-
"""ToolRegistry：注册层（fail-fast）+ 发现层（权限过滤）+ 平台装配。


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
    """
    函数/类功能与逻辑描述：
        工具接入平台的注册层 + 发现层。注册层是工具的唯一真相源，注册时 fail-fast，
        一切不一致都在装配期报出；发现层让"这个 Agent 能用哪些工具"可查询、可导出、可自检。
        执行不在本类内实现，而是持有 `ExecutionEngine` 引用并统一委托，`invoke` 为唯一门面。
    构造入参说明：
        无（构造内部自建三个空容器：工具契约表、协议适配器表、执行引擎引用）。
    返回值说明：
        构造返回 ToolRegistry 实例；业务方法见 `register` / `list_tools` / `invoke` 等。
    """

    def __init__(self):
        """
        函数功能与逻辑描述：
            初始化注册表的三个内部容器，均从空开始，不加载任何内置工具；
            工具契约与协议适配器全部由平台装配函数 `_build_platform` 显式注册。
        入参说明：
            无。
        返回值说明：
            无（仅初始化 _specs / _adapters / _engine 三个实例属性，无副作用）。
        """
        self._specs: dict[str, ToolSpec] = {}
        self._adapters: dict[str, "ProtocolAdapter"] = {}
        self._engine: "ExecutionEngine | None" = None

    # ─────────────── 注册 API ───────────────
    def register(self, spec: ToolSpec) -> None:
        """
        函数功能与逻辑描述：
            注册一个工具契约；★fail-fast：先做全量校验（`_validate_spec`），再检查是否重名，
            任一不通过都在装配期直接抛错，绝不让坏工具带到运行时。
        入参说明：
            spec (ToolSpec)：工具契约对象，spec.name 为全局唯一键，须满足 `_validate_spec` 各项约束。
        返回值说明：
            无（副作用：成功时以 spec.name 为键写入 _specs；校验失败或重名抛 ValueError）。
        """
        self._validate_spec(spec)
        if spec.name in self._specs:
            raise ValueError(f"工具重复注册: {spec.name}")
        self._specs[spec.name] = spec

    def register_adapter(self, protocol: str, adapter: "ProtocolAdapter") -> None:
        """
        函数功能与逻辑描述：
            注册协议适配器，供所有声明该 protocol 的工具复用；同一协议只允许一个适配器，
            重复注册直接抛 ValueError，保证"一个协议一个实现"的唯一性。
        入参说明：
            protocol (str)：协议名，取值与 ToolSpec.protocol 对应（"mcp" / "local" / "http"）。
            adapter (ProtocolAdapter)：该协议的适配器实例，需实现平台统一的调用接口。
        返回值说明：
            无（副作用：成功时以 protocol 为键写入 _adapters；重复注册抛 ValueError）。
        """
        if protocol in self._adapters:
            raise ValueError(f"协议适配器重复注册: {protocol}")
        self._adapters[protocol] = adapter

    def tool(self, *, name: str | None = None, description: str,
             schema: dict, **kw) -> Any:
        """
        函数功能与逻辑描述：
            装饰器工厂：把一个本地函数一行注册成 protocol="local" 的 ToolSpec，并以函数对象
            填充 binding.func，最后原样返回被装饰函数。注册动作发生在返回的 `deco` 真正装饰
            函数的那一刻，而非调用本方法时。
        入参说明：
            name (str | None)：工具名，省略时回退取被装饰函数的 __name__，默认 None。
            description (str)：给 LLM 看的工具描述，直接决定它选不选这个工具。
            schema (dict)：入参 JSON Schema，同时驱动参数校验与 LLM 可见性。
            **kw (Any)：透传给 ToolSpec 的其余可选字段；其中 to_thread 由 deco 单独弹出，
                用于声明该本地函数是否放线程池执行（默认 True）。
        返回值说明：
            Any：返回装饰器 `deco`；deco 接收被装饰函数 fn 并返回同一个 fn。
        """
        def deco(fn):
            """
            函数功能与逻辑描述：
                实际执行注册的内层装饰器：先弹出 to_thread，再构造 ToolSpec 并调用 `register`，
                注册完成后原样返回 fn，不包裹、不改签名、不改变其调用行为。
            入参说明：
                fn (Callable)：被装饰的本地函数对象，作为 binding.func 写入契约。
            返回值说明：
                Callable：原样返回入参 fn 本身。
            """
            to_thread = kw.pop("to_thread", True)
            self.register(ToolSpec(
                name=name or fn.__name__, description=description,
                params_schema=schema, protocol="local",
                binding={"func": fn, "to_thread": to_thread}, **kw))
            return fn
        return deco

    def unregister(self, name: str) -> None:
        """
        函数功能与逻辑描述：
            热卸载指定工具（Skills 动态加载 / 卸载用）；采用 pop(name, None) 语义，
            工具不存在时静默返回、不抛错，保证重复卸载幂等。
        入参说明：
            name (str)：待卸载的工具名。
        返回值说明：
            无（副作用：从 _specs 中移除该工具；不存在时注册表无任何变化）。
        """
        self._specs.pop(name, None)

    def _validate_spec(self, spec: ToolSpec) -> None:
        """
        函数功能与逻辑描述：
            注册前的私有校验入口，★fail-fast：全部一致性检查集中在此，任一不通过即抛 ValueError。
            依次校验：name / description / params_schema 必填字段非空 → protocol 已注册适配器
            → mcp 协议 binding 须同时含 server_key 与 tool → local 协议 binding 须含 func。
        入参说明：
            spec (ToolSpec)：待校验的工具契约。
        返回值说明：
            无（通过则静默返回；不通过抛 ValueError，无其它副作用）。
        """
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
        """
        函数功能与逻辑描述：
            绑定执行引擎实例（依赖注入），使 `invoke` 能把调用委托给 ExecutionEngine；
            未绑定时 `invoke` 将抛 RuntimeError，属于装配缺陷而非运行时业务错误。
        入参说明：
            engine (ExecutionEngine)：执行引擎实例，由平台装配阶段创建后传入。
        返回值说明：
            无（副作用：写入 self._engine）。
        """
        self._engine = engine

    # ─────────────── 发现 API ───────────────
    def get(self, name: str) -> ToolSpec | None:
        """
        函数功能与逻辑描述：
            按工具名查契约，不做任何权限过滤（权限过滤由 `list_tools` 负责）；
            查不到时返回 None 而非抛错，调用方自行判空。
        入参说明：
            name (str)：工具名。
        返回值说明：
            ToolSpec | None：命中返回对应契约对象；未注册返回 None。
        """
        return self._specs.get(name)

    def get_adapter(self, protocol: str) -> "ProtocolAdapter":
        """
        函数功能与逻辑描述：
            按协议名取适配器；与 `get` 的"查不到返回 None"不同，适配器缺失属装配缺陷，
            此处直接抛 KeyError 报错，避免错误被静默吞掉后在运行期才暴露。
        入参说明：
            protocol (str)：协议名（"mcp" / "local" / "http"）。
        返回值说明：
            ProtocolAdapter：命中的适配器实例；未注册则抛 KeyError（不返回 None）。
        """
        adapter = self._adapters.get(protocol)
        if adapter is None:
            raise KeyError(f"协议[{protocol}]未注册适配器")
        return adapter

    def all(self) -> list[ToolSpec]:
        """
        函数功能与逻辑描述：
            返回全部已注册工具的契约快照，不做权限 / hidden 过滤，供自检、统计等内部用途；
            返回的是新建列表，外部增删该列表不会影响注册表本身。
        入参说明：
            无。
        返回值说明：
            list[ToolSpec]：全部契约列表；注册表为空时返回空列表。
        """
        return list(self._specs.values())

    def has(self, name: str) -> bool:
        """
        函数功能与逻辑描述：
            判断工具是否已注册，仅判断键是否存在，不读取契约内容、不校验权限。
        入参说明：
            name (str)：工具名。
        返回值说明：
            bool：已注册返回 True，否则 False。
        """
        return name in self._specs

    def list_tools(self, agent_id: str) -> list[ToolSpec]:
        """
        函数功能与逻辑描述：
            列出某 Agent 可见的工具；★权限过滤即发现过滤，在发现阶段就把不可用工具剔除。
            过滤规则：① hidden=True 的内部工具不暴露；
            ② required_perm=None → 公开工具，放行；
            ③ required_perm ∈ get_agent_permission(agent_id) → 放行。
        入参说明：
            agent_id (str)：调用方 Agent 标识，用于查询其权限集合。
        返回值说明：
            list[ToolSpec]：过滤后的契约列表，顺序与注册顺序一致；无可见工具时返回空列表。
        """
        perms = set(get_agent_permission(agent_id))
        return [s for s in self._specs.values()
                if not s.hidden and (s.required_perm is None or s.required_perm in perms)]

    def to_openai_schema(self, agent_id: str,
                         tags: tuple[str, ...] | None = None) -> list[dict]:
        """
        函数功能与逻辑描述：
            导出为 OpenAI/Ollama function-calling 格式（LLM 自主选工具的输入）；先经
            `list_tools` 做权限过滤，再按 tags 做二次能力过滤（任一标签命中即保留），
            每个元素由 ToolSpec.to_llm_schema() 生成。
        入参说明：
            agent_id (str)：调用方 Agent 标识，决定可见工具范围。
            tags (tuple[str, ...] | None)：能力标签过滤器，默认 None；为 None 或空元组时不做标签过滤。
        返回值说明：
            list[dict]：形如 [{"type": "function", "function": {...}}] 的列表；无匹配工具时返回空列表。
        """
        return [{"type": "function", "function": s.to_llm_schema()}
                for s in self.list_tools(agent_id)
                if not tags or set(tags) & set(s.tags)]

    def to_mcp_schema(self, agent_id: str) -> list[dict]:
        """
        函数功能与逻辑描述：
            导出为 MCP 工具列表格式（调试 / 网关转发用）；同样先经 `list_tools` 做权限过滤，
            再映射为 MCP 的 name / description / inputSchema 三字段结构，不做 tags 过滤。
        入参说明：
            agent_id (str)：调用方 Agent 标识，决定可见工具范围。
        返回值说明：
            list[dict]：形如 [{"name": ..., "description": ..., "inputSchema": ...}] 的列表；无可见工具时返回空列表。
        """
        return [{"name": s.name, "description": s.description,
                 "inputSchema": s.params_schema}
                for s in self.list_tools(agent_id)]

    async def verify_schemas(self) -> dict[str, list[str]]:
        """
        函数功能与逻辑描述：
            离线自检：遍历本地 protocol="mcp" 的工具契约，逐个向对应 MCP 服务拉取 tools/list
            并比对契约是否漂移——远端缺工具记一条，schema 与远端不一致记一条。
            ★不进启动路径（M1 时代 MCP 子进程冷启动约 26~29s；M3 常驻化后冷启动已消除，
            但每次自检仍需向各服务发起一次 tools/list 往返），作为 CI / 手动命令执行。
        入参说明：
            无。
        返回值说明：
            dict[str, list[str]]：{工具名: [差异描述, ...]}；检查项全部一致时返回空 dict {}；
                非 mcp 协议的工具不参与比对。
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
                     session_id: str | None = None,
                     task_id: str | None = None):
        """
        函数功能与逻辑描述：
            统一执行入口（门面）：业务代码只认识这一个方法，内部把调用原样委托给
            ExecutionEngine，参数校验、权限判定、超时与审计等横切逻辑均由引擎负责，
            本类不重复实现，从而保证"本地工具 / MCP 工具"对上层无差别。
        入参说明：
            name (str)：工具名。
            args (dict)：工具入参，须满足该工具的 params_schema。
            agent_id (str)：调用方 Agent 标识（关键字限定），引擎据此做权限判定与审计。
            trace_id (str | None)：链路追踪 ID，默认 None。
            session_id (str | None)：会话 ID，默认 None。
            task_id (str | None)：单次任务标识（M15 新增），默认 None；经引擎装入 InvokeContext，
                供声明了 `binding["inject_ctx"]` 的工具（如 bill.add）作为附加形参取用。
                与 agent_id 同属「调用方身份面」，**不进 params_schema**。
        返回值说明：
            返回 ExecutionEngine.invoke 的 await 结果（ToolResult）；未绑定引擎时抛 RuntimeError。
        """
        if self._engine is None:
            raise RuntimeError("ExecutionEngine 未绑定")
        return await self._engine.invoke(name, args, agent_id=agent_id,
                                         trace_id=trace_id, session_id=session_id,
                                         task_id=task_id)


def _schema_diff(local: dict, remote: dict) -> list[str]:
    """
    函数功能与逻辑描述：
        比较本地工具契约与远端 MCP 声明，返回差异描述列表（空列表 = 一致）。
        只比「属性名集合」与「必填集合」：远端 schema 由 FastMCP 依类型注解生成，
        没有 additionalProperties 之类约束，全量深比较只会产生无意义噪声。
        远端多出的 `agent_id` / `run_id` 属预期差异：二者由引擎注入而非写进 schema
        （本地刻意不声明，防 LLM 伪造提权），故比对前先从远端集合中剔除。
    入参说明：
        local (dict)：本地 ToolSpec.params_schema，可为 None（按空 dict 处理）。
        remote (dict)：远端返回的 inputSchema，可为 None（按空 dict 处理）。
    返回值说明：
        list[str]：逐项差异描述；属性集合与必填集合都一致时返回空列表。
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
    """
    函数功能与逻辑描述：
        返回全局唯一工具注册中心；首次访问时惰性装配，之后复用模块级缓存实例。
        ★为什么是惰性 + 函数内延迟导入：① 打破 `client → registry → adapters → client`
        的模块级导入环；② 保证"不跑 bootstrap 也能用"——集成测试直接调 `mcp_client.*`
        时自动装配完成。
    入参说明：
        无。
    返回值说明：
        ToolRegistry：全局单例注册表；首次调用时触发 `_build_platform` 完成装配。
    """
    global _registry
    if _registry is None:
        _registry = _build_platform()
    return _registry


def _build_platform() -> ToolRegistry:
    """
    函数功能与逻辑描述：
        装配工具接入平台：注册协议适配器 → 注册 4 个协议级工具（P0）、6 个业务事实工具
        （M9 / D15，收口编排器直连）与 5 个账单改删工具（M15：4 个 `bill.*` + `habit.recalc`，
        共 15 个）→ 绑定执行引擎，返回可直接使用的 ToolRegistry。
        工具清单与权限映射见 `08` §1.10；M15 新增部分的契约见 `设计/12` §4.2.4。
    入参说明：
        无。
    返回值说明：
        ToolRegistry：装配完成、已绑定 ExecutionEngine 的注册表实例。
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
    #     LLM=60s（★M22 收紧：原 300s 按"本地 1.8B 单次 189s"的**旧基线**所设；
    #              本地通道已弃用、生产走外部强模型【秒级】，60s 为防挂死兜底，
    #              且为超时链最内环：工具(60) < 任务(120) < 编排器等待(150)）。
    # max_retry=0 ：底层 `_call_server` 已自带 2 次连接重试，外层再重试会放大成乘数效应
    #                （2×N 次），故引擎层 M1 阶段不重试，行为与现状一致。
    #                ★现状提醒：**当前所有工具均为 0**，即引擎层的重试 + 指数退避代码
    #                （`executor._execute`）处于**未启用**的预留状态 —— 对外表述须如实，
    #                不可说成"重试已在平台层生效"（重试实际发生在 `client._call_server`）。
    def _sql_props(desc: str) -> dict:
        """
        函数功能与逻辑描述：
            构造 SQL 类工具的入参 JSON Schema，统一 sql.read / sql.write 的契约形态：
            仅 description 因读写用途不同而参数化，其余结构与约束完全一致，避免两条 SQL
            工具的契约各自漂移。
        入参说明：
            desc (str)：写入 sql 字段的 description 文案，用于区分读 / 写用途。
        返回值说明：
            dict：JSON Schema；properties 含 sql(string) 与 params(array，元素不限类型)，
                仅 sql 必填，并设 additionalProperties=False 禁止多余参数。
        """
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

    # M15：FC 通道专用 schema——比 llm.chat 多出 tools / extra_messages 两个**协议结构**参数
    #   （与 messages 同级，不是调参项；由 bill_agent 代码构造，模型看不到本工具）。
    llm_fc_schema = {
        "type": "object",
        "properties": {
            "system_prompt": {"type": "string", "description": "系统提示词"},
            "user_input": {"type": "string", "description": "用户输入"},
            "tools": {"type": "array", "description": "function-calling 工具清单（OpenAI 格式）",
                      "items": {"type": "object"}},
            "extra_messages": {"type": "array",
                               "description": "多轮 tool 消息（assistant tool_calls / tool 结果回填）",
                               "items": {"type": "object"}},
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

    # ★M15：FC 通道（`设计/12` §4.3）——**新增工具而非改 `llm.chat` 契约**。
    #   为什么不复用 llm.chat：FC 调用的返回是**结构化**的（正文 + tool_calls），而 llm.chat 的
    #   返回契约是**纯文本 str**（18+ 处既有调用与 ~40 处测试 patch 依赖它）。用"传了 tools 就
    #   返回 JSON、否则返回文本"这种按参数分叉的形态表达，会让同一工具的契约产生二义，后续调用方
    #   极易踩坑；新增一个工具则：① 旧契约零风险；② 可单独下线（`unregister("llm.chat_fc")` = 回退，
    #   满足 M15 回滚要求）；③ 符合项目"加能力 = 加工具"的既有范式（`设计/12` §4.2.7）。
    #   ★工具清单从哪进模型：仍走网关（`registry.to_openai_schema(agent_id, tags=...)` → 作为本工具的
    #   tools 参数），不新增直连 provider 的旁路（`09` §0.3 原则 2：Agent 只经网关访问模型/数据）。
    reg.register(ToolSpec(
        name="llm.chat_fc",
        description="带 function-calling 通道的基座模型调用，返回 JSON（正文 text + 工具调用 tool_calls）",
        params_schema=llm_fc_schema, protocol="mcp",
        binding={"server_key": "llm_base", "tool": "llm_chat_fc",
                 "inject_agent_id": "agent_tag", "inject_run_id": True},
        required_perm=LLMPerm.LLM_BASE, side_effect=False,
        auditable=False,        # 同 llm.chat：对齐现状不审计 LLM 调用，避免高频刷屏
        idempotent=False, timeout=TOOL_TIMEOUT_LLM, max_retry=0, tags=("llm", "fc")))

    # ── 第二批：业务事实工具（M9 / D15，P0-4+P0-5）──
    # 收口 orchestrator 9 处 `db.` 直连：读走网关（FACT_READ）+ habit 写下沉 bill_agent（HABIT_WRITE）。
    # ★方案差异：transport=local（`11` M9 走查 #4 登记；`03` §9.9.2 表原标 mcp，文档回写时修订）。
    #   local 工具执行：同步 fn → 引擎 to_thread；habit.upsert 为 async（协程内持 db_lock）。
    from mcpGateway import fact_tools

    def _empty_props() -> dict:
        """
        函数功能与逻辑描述：
            构造「无入参」工具的 JSON Schema，供 config.get_latest 这类零参数工具复用，
            避免为每个零参工具手写一份空 schema。
        入参说明：
            无。
        返回值说明：
            dict：JSON Schema；properties 为空 dict、required 为空列表，
                additionalProperties=False 明确表示不接受任何参数。
        """
        return {"type": "object", "properties": {},
                "required": [], "additionalProperties": False}

    def _month_props() -> dict:
        """
        函数功能与逻辑描述：
            构造「按自然月查询」类工具的入参 JSON Schema，供 bill.sum_by_month /
            bill.sum_by_category 复用，统一月份参数的类型与取值格式约定。
        入参说明：
            无。
        返回值说明：
            dict：JSON Schema；仅含必填的 month(string，格式 YYYY-MM)，
                additionalProperties=False 禁止多余参数。
        """
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

    # ── 第三批：账单改删工具族（M15 / `设计/12` §4.2.4）──
    # 4 个 bill.* 工具（SQL 全部写死在工具内部，模型只传结构化参数，N6 红线）+ 1 个
    # habit.recalc（改删后整月重算派生习惯，**不打 op:* → 不进任何模型可见面**）。
    # ★op:* tags 是「按操作类型锁定工具可见面」的物理开关：bill_agent 依 operate_sub_type 调
    #   `to_openai_schema(agent_id, tags=("op:read", "op:edit"))` 之类，于是 edit 任务里模型
    #   **根本看不到** `bill.delete`（不靠提示词约束，靠可见面裁剪，`设计/12` §4.2.2）。
    #   ★两层过滤的语义边界：第一层 RBAC（list_tools 按 Agent 权限画像）是**授权**，
    #   第二层 tags 只是在既有授权**内**进一步收窄——tags 不授予任何权限。
    from mcpGateway import bill_tools

    bill_recent_schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer",
                      "description": "期望返回条数（服务端按窗口下限/上限收敛，可不传）"},
        },
        "required": [],
        "additionalProperties": False,
    }

    bill_add_schema = {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "消费金额（元，须大于 0）"},
            "category": {"type": "string", "description": "消费品类：餐饮/交通/住宿/购物/娱乐"},
            "consume_date": {"type": "string", "description": "消费日期 YYYY-MM-DD"},
            "remark": {"type": "string", "description": "备注（用户原话）"},
        },
        "required": ["amount", "category", "consume_date"],
        "additionalProperties": False,
    }

    bill_update_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "账单记录 id（须先经 bill.recent_list 取得）"},
            "amount": {"type": "number", "description": "新金额（元，须大于 0）；不修改则不传"},
            "category": {"type": "string", "description": "新品类；不修改则不传"},
            "consume_date": {"type": "string", "description": "新消费日期 YYYY-MM-DD；不修改则不传"},
        },
        "required": ["id"],
        "additionalProperties": False,
    }

    bill_delete_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer", "description": "账单记录 id（须先经 bill.recent_list 取得）"},
        },
        "required": ["id"],
        "additionalProperties": False,
    }

    habit_recalc_schema = {
        "type": "object",
        "properties": {"month": {"type": "string", "description": "自然月 YYYY-MM"}},
        "required": ["month"],
        "additionalProperties": False,
    }

    reg.register(ToolSpec(
        name="bill.recent_list",
        description="查询最近几笔账单明细（含 id / 金额 / 品类 / 消费时间 / 备注），改删账单前先用它定位目标",
        params_schema=bill_recent_schema, protocol="local",
        binding={"func": bill_tools.recent_bills, "to_thread": True},
        # ★权限取 BILL_READ 而非 FACT_READ（`09` §4.10 原标 FACT_READ，落地时修订）：
        #   本工具读的就是账单表，语义与 DBPerm.BILL_READ（"账单查询权限 SELECT"）精确对应；
        #   用 FACT_READ 会迫使给 bill_agent 扩一条它并不需要的 fact:read（该权限覆盖
        #   user_config / city_price / monthly_habit 读，属越权面扩大，违反最小权限）。
        required_perm=DBPerm.BILL_READ, side_effect=False, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("bill", "op:read")))

    reg.register(ToolSpec(
        name="bill.add",
        description="新增一笔账单（金额 / 品类 / 消费日期必填）",
        params_schema=bill_add_schema, protocol="local",
        # ★task_id 经引擎从调用上下文注入（inject_ctx），**不进 params_schema**：
        #   它是崩溃恢复对账依据，必须由代码保证，模型既不该看到也不能伪造（同 agent_id 处理）。
        binding={"func": bill_tools.add_bill, "to_thread": True, "inject_ctx": ("task_id",)},
        required_perm=DBPerm.BILL_WRITE, side_effect=True, auditable=True,
        idempotent=False, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("bill", "op:add")))

    reg.register(ToolSpec(
        name="bill.update",
        description="修改指定账单（部分更新：只传要改的字段，未传字段保持原值）",
        params_schema=bill_update_schema, protocol="local",
        binding={"func": bill_tools.update_bill, "to_thread": True},
        required_perm=DBPerm.BILL_WRITE, side_effect=True, auditable=True,
        idempotent=False, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("bill", "op:edit")))

    reg.register(ToolSpec(
        name="bill.delete",
        description="按 id 删除指定账单（需用户已确认）",
        params_schema=bill_delete_schema, protocol="local",
        binding={"func": bill_tools.delete_bill_record, "to_thread": True},
        required_perm=DBPerm.BILL_WRITE, side_effect=True, auditable=True,
        idempotent=False, timeout=TOOL_TIMEOUT_SQL, max_retry=0, tags=("bill", "op:delete")))

    reg.register(ToolSpec(
        name="habit.recalc",
        description="按账单表整月重算月度消费习惯（账单改删后的一致性修复，幂等）",
        params_schema=habit_recalc_schema, protocol="local",
        # ★async 函数：协程内持 db_lock（同 habit.upsert 的理由，见 fact_tools.recalc_habit）
        binding={"func": fact_tools.recalc_habit, "to_thread": False},
        required_perm=HabitPerm.HABIT_WRITE, side_effect=True, auditable=True,
        idempotent=True, timeout=TOOL_TIMEOUT_SQL, max_retry=0,
        # ★不打 op:* —— 该工具**不对模型可见**（由代码在写操作成功后自动触发，`设计/12` §4.2.6）
        tags=("fact", "db", "write")))

    reg.bind_engine(ExecutionEngine(reg))
    return reg


__all__ = ["ToolRegistry", "get_registry"]
