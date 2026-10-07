# billAgent 学习知识框架（Python 基础 + Agent 领域 + 面试考察）

> 用途：解决"不知道该问什么、不知道问到什么程度"的问题。
> 用法：按章推进；每学完一个知识点，用 §0.2 的模板向我提问，我按"概念 → 项目依据 → 取舍"三层回答。
> 索引约定：文中所有论断均附 `文件:行号` 或 `文件:符号名`。**行号会随代码变动漂移，函数名/类名才是稳定锚点**（依据 `代码阅读路线.md:4`）。

---

## 第 0 章 先解决"怎么问、问到什么程度"

### 0.1 五级深度标准（用来自查"我学到了没有"）

| 级别 | 标准 | 自测问法 |
|---|---|---|
| L1 能定义 | 用自己的话说清概念，不背术语 | "GIL 是什么？" |
| L2 能定位 | 指出它在项目哪个文件、哪个函数 | "GIL 在项目里影响了哪个决策？" |
| L3 能讲取舍 | 说清为什么选 A 不选 B，代价是什么 | "为什么用协程不用进程？" |
| L4 能讲边界 | 说清什么情况下会失效、怎么发现 | "协程方案什么时候会崩？" |
| L5 能改进 | 提出可落地的演进方案 | "要支持多机部署该怎么改？" |

**面试的分水岭在 L3**：只会 L1/L2 的人叫"背书党"，能答 L3 的人叫"做过项目的人"。L4/L5 决定是白菜价还是 SSP。

### 0.2 五类提问模板（直接复制改词）

| 类型 | 模板 | 适用时机 |
|---|---|---|
| 概念澄清 | `XX 是什么？它解决什么问题？没有它会怎样？` | 完全没听过这个名词时 |
| 定位 | `XX 在这个项目里用在哪？给我 1-2 处最典型的代码` | 想知道"学了有什么用" |
| 取舍 | `这里为什么选 A 不选 B？换 B 会付出什么代价？` | 读代码看到奇怪的设计时 |
| 边界 | `这套机制在什么情况下会失效？项目里有兜底吗？` | 准备面试深挖时 |
| 延伸 | `如果要支持 YY，需要改哪些文件？` | 想展示系统思维时 |

### 0.3 三个提问纪律

1. **一次问一个知识点**，不要一次抛 5 个名词——我会逐个压缩成三句话，学不透。
2. **先自己读指定的那 1 处代码再问**，否则你拿到的是"二手结论"，面试时接不住追问。
3. **每个知识点问完，必须能答出 L3**（为什么这么选），否则这个点对面试无用。

---

## 第 1 章 Python 基础框架（按项目真实使用组织）

> 本章每个知识点都标了**项目依据**。这个项目没有用到的语言特性（如 `multiprocessing`、`Pydantic`、`typing.Protocol`），
> 我单独列在 §1.7，避免你在无用知识点上花时间。

### 1.1 模块一：并发与执行模型（你点名的重点）

#### P1.1 并发 vs 并行

- **并发**：单核上交替推进多个任务（宏观看像同时，微观是轮流）。
- **并行**：多核上真正同时执行。
- **项目依据**：`README.md` §7.5 —— 架构图明确标注「主进程：单进程多协程」，即**并发不并行**。

#### P1.2 GIL（全局解释器锁）★核心

**定义**：CPython 解释器的一把全局互斥锁——同一进程内，**同一时刻只有一个线程能执行 Python 字节码**。

**两个推论**（面试必答这两个）：

1. 多线程**不能**加速 CPU 密集型任务（算不动的还是算不动，还多了切换开销）。
2. 多线程**能**加速 IO 密集型任务——线程执行到 `read()`/网络请求等阻塞调用时会**主动释放 GIL**。

**项目里的三处实证**（这是你区别于"背书党"的地方）：

| 现象 | 代码依据 | 说明 |
|---|---|---|
| 重活丢独立进程 | `startup/bootstrap.py:97` `subprocess.Popen` | 3 个 MCP 服务是独立进程，天然绕开 GIL，各自吃一个核 |
| 阻塞调用丢线程池 | `agents/orchestrator/nodes.py:1312` `await asyncio.to_thread(search_history_consume, ...)` | FAISS 向量检索是同步 CPU 计算，直接调用会**冻结整个事件循环**（因为单协程阻塞 = 全进程卡住） |
| 子 Agent 用协程不用进程 | `README.md` §11 "子 Agent 用协程不用进程" | 理由写在文档里：主进程不做重计算（推理在外部 API / 独立进程），协程足以覆盖 IO 等待 |

> **为什么必须要 `asyncio.to_thread`？** 这是理解 GIL + 协程的关键连接点：
> 协程是**单线程**的，一个协程里放同步阻塞代码，事件循环里其他所有任务全部停摆。
> `to_thread` 把阻塞函数扔到线程池，阻塞期间释放 GIL，事件循环继续调度其他协程。

#### P1.3 进程 / 线程 / 协程三选一（选型决策表）

| 维度 | 进程 | 线程 | 协程 |
|---|---|---|---|
| 调度者 | 操作系统 | 操作系统 | **用户态事件循环**（无内核切换） |
| 内存开销 | 大（独立地址空间） | 中（共享内存，栈 MB 级） | 极小（KB 级） |
| 受 GIL 限制 | 否 | 是 | 是（但不受害，因为是单线程交替） |
| 切换成本 | 高 | 中 | 极低 |
| 适合 | CPU 密集 / 隔离故障 | IO 密集（老旧代码） | **高并发 IO（本项目）** |
| 通信 | IPC / 网络（要序列化） | 共享内存（要加锁） | 共享内存（无锁，单线程无竞争） |
| 项目出处 | `bootstrap.py:97` <br>`subprocess.Popen` | `memory/user_docs.py:163` <br>`threading.RLock` | `startup/bootstrap.py:399` <br>`asyncio.create_task` |

**项目三种全用上了**——这就是最好的学习素材：

- **协程为主**：1 个 orchestrator + 4 个常驻 worker（`bootstrap.py:399` 挂到 `WORKER_TASKS`）
- **进程为辅**：3 个 MCP 常驻服务（`bootstrap.py:97`），以及 stdio 回滚模式下的子进程（`mcpGateway/client.py:172`）
- **线程少量**：`asyncio.to_thread`（`mcpGateway/adapters.py:234` 本地工具执行）、`threading.RLock`（索引重建锁）、`threading.Timer`（`webUI/app.py:503` 延迟退出）

#### P1.4 协程原理与 asyncio 核心 API

**必须能讲清的四件事**：

1. `async def` 定义的函数被调用时**不执行**，只返回协程对象；真正跑起来要靠事件循环 `await` 或 `create_task`。
2. `await` 的语义是"**把控制权交还事件循环**，等这个操作完成再回来"——协程并发的本质就在这里。
3. 事件循环是**单线程**的，所以协程之间没有数据竞争（但一个协程阻塞就全阻塞，见 P1.2）。
4. 三大同步原语：`Queue`（传数据）、`Event`（通知/等待）、`Lock`（互斥）。

**项目 API 索引（按学习优先级排序）**：

| API | 项目位置 | 用途 |
|---|---|---|
| `asyncio.Queue` | `mcpGateway/a2a_queue.py:39` `defaultdict(asyncio.Queue)` | A2A 总线：每个 Agent 一条队列 |
| `asyncio.Event` | `a2a_queue.py:40,215,236` | 结果就绪通知（`set()` / `event.wait()`） |
| `asyncio.wait_for` | `a2a_queue.py:146,239`；`mcpGateway/executor.py:217` | 超时熔断（总线 240s、工具按 spec.timeout） |
| `asyncio.gather` | `agents/orchestrator/nodes.py:1765` | 整波并发收集结果（耗时 = max(T) 而非 ΣT） |
| `asyncio.create_task` | `startup/bootstrap.py:399` | 把 4 个 worker 挂成后台常驻任务 |
| `asyncio.to_thread` | `nodes.py:1312`；`adapters.py:234` | 同步阻塞函数丢线程池 |
| `asyncio.Lock` | `utils/common.py:597` `db_lock` | 进程内 DB 写互斥 |
| 事件循环手动管理 | `webUI/app.py:68-70` `new_event_loop` + `run_until_complete` | 见下方"跨 loop 三坑" |

**跨 loop 三坑（项目注释里写得很清楚，是绝佳面试素材）**：

1. `asyncio.Queue` 首次 `await get()` 时**绑定归属 loop**，跨 loop 复用报 `is bound to a different event loop`
   → 依据 `a2a_queue.py:63-66`（`clear_all` 直接重建队列而非清空）。
2. `asyncio` 子进程的 `wait()` Future 绑定创建时的 loop，跨 loop 启停报错
   → 依据 `bootstrap.py:50-52`（所以刻意用同步 `subprocess.Popen`）。
3. WebUI 不用 `asyncio.run()`，因为 `run()` 结束会**关闭 loop**，而 worker 要长驻
   → 依据 `webUI/app.py:68-70`。

#### P1.5 锁的两个世界（极易答错）

| 锁类型 | 保护对象 | 项目位置 | 错用后果 |
|---|---|---|---|
| `asyncio.Lock` | **协程之间** | `utils/common.py:597` `db_lock` | — |
| `threading.RLock` | **线程之间** | `memory/user_docs.py:163` `_build_lock` | 在协程里用会**阻塞事件循环** |
| SQLite 自身锁 | **跨进程** | `utils/common.py:66-67` `busy_timeout=5000` | — |

**项目里最值钱的一句注释**（`utils/common.py:63-64`）：

> `db_lock` 是【**进程内**】锁，跨进程（主进程 vs MCP server 子进程）写互斥只能靠 SQLite 层兜底。

一句话讲透：**锁的作用域必须与竞争的边界匹配**。协程间用 asyncio 锁、线程间用 threading 锁、进程间只能靠数据库/文件系统——三者混用必然出问题。

#### P1.6 多进程实现方式：`subprocess` vs `multiprocessing`

- 项目**只用 `subprocess.Popen` 拉起独立脚本**，完全不用 `multiprocessing`（无共享内存、无 pickle 进程间对象传递）。
- 依据：`bootstrap.py:97`（起 MCP 服务）、`webUI/app.py:749`（Windows 下查/启/杀进程）、`bootstrap.py:130-137`（`terminate()` → 等 5s → `kill()` 的优雅关闭三段式）。
- **取舍理由**：MCP 服务本身是**独立的 HTTP 服务**，本来就要独立进程；用 `subprocess` 还顺带绕开了 P1.4 的跨 loop 坑。

---

### 1.2 模块二：数据库与持久化（你点名的重点）

#### P2.1 关系模型与 SQL 基础

- 表/行/列/主键/索引 → 建表语句在 `database/init_db.py:42`（`CREATE TABLE IF NOT EXISTS ...`，444 行，**当字典查，不要通读**）。
- **SQL 写在哪**：`mcpGateway/bill_tools.py`——模块头注明"**SQL 全部写死在函数内**"（依据 `代码阅读路线.md:202`）。这是关键设计：不让 LLM 生成 SQL。
- 统一读入口：`utils/common.py:153` `query_sql()`，把游标结果转成 `list[dict]`（列名→值），避免下游把 tuple 当 dict 用。

#### P2.2 参数化查询（防 SQL 注入）

- 依据：`utils/common.py:131-134`
  ```python
  if params:
      cur.execute(sql, params)   # 参数绑定，不是字符串拼接
  else:
      cur.execute(sql)
  ```
- **必须能答**：为什么不写 f-string 拼 SQL？→ 注入 + 无法复用执行计划缓存。

#### P2.3 事务 ACID

- 项目三件套：`begin_transaction()`（`common.py:71`）/ `commit()`（`:84`）/ `rollback()`（`:96`）。
- 典型用例：`database/import_city_price.py:135` 批量导入"清空+插入"绑成原子事务——**中途失败不留半份数据**。
- 面试延伸：SQLite 的默认隔离级别是 SERIALIZABLE（通过文件锁实现），与 MySQL RR 不同。

#### P2.4 SQLite 的架构特点（为什么它够用）

- 嵌入式、单文件、**无独立服务进程**、零配置——所以项目能"单机单用户"直接跑（`README.md` §12）。
- 代价：写入是**全库串行**的（文件锁），没有 MySQL 的行级并发。

#### P2.5 WAL 模式 ★高频考点

- 依据：`utils/common.py:66` `PRAGMA journal_mode=WAL`
- **解决的问题**：默认 rollback journal 模式下，写操作会阻塞读；WAL 让**读不阻塞写、写不阻塞读**。
- **为什么项目必须要它**：M6 并行派发的前提（`utils/common.py:64` 注释「WAL 同时是 M6 并行派发的前提」）；
  实测"双进程各写 80 行，**零锁错误**"（`README.md` §8）。
- **副作用**：会产生 `bill.db-wal` / `bill.db-shm` 两个附属文件，备份时必须一起拷（`common.py:65` 注释）。

#### P2.6 `database is locked` 与 busy_timeout

- 成因：SQLite 写锁被别的连接持有，默认行为是**立即失败**。
- 项目双保险：
  1. `PRAGMA busy_timeout=5000`（`common.py:67`）——C 层忙等 5 秒再失败；
  2. `execute_sql(retry=3)`（`common.py:108-145`）——应用层对 `database is locked` 再重试。
- **关键取舍**（`common.py:139-140` 注释，很值钱）：改造后**删掉了 `time.sleep(0.1)`**，因为"同步 sleep 会**冻结事件循环**"——
  这正是 P1.2 的 GIL/协程阻塞知识落地的地方。

#### P2.7 连接管理

- `sqlite3.connect(db_path, check_same_thread=False)`（`common.py:61`）：允许连接跨线程复用（配合 `to_thread`）。
- **单连接 + 一把锁**，不是连接池——项目刻意不用 SQLAlchemy（`database/init_db.py:4` 注释明确"无需使用 SQLAlchemy"）。
- 面试准备：能对比"嵌入式单连接" vs "服务端连接池"的适用场景。

#### P2.8 两库分离

- `bill.db`（业务事实）+ `task_state.db`（任务状态机）——依据 `utils/common.py:45-53`，理由注释写明：
  "Durable 状态**写入频繁**，与业务库同库会**争抢写锁**"。
- 这是"按写入特征拆分存储"的典型工程判断，面试可当亮点讲。

#### P2.9 幂等

- `task_id` 幂等写入（`README.md` §7.3 状态机）：同一 `task_id` 重复投递不重复记账。
- 上位约束：**有写副作用的工具永不自动重试**（`README.md` §11），防重复记账。

---

### 1.3 模块三：数据结构与类型系统

| 知识点 | 项目依据 | 要理解什么 |
|---|---|---|
| `TypedDict` | `agents/orchestrator/state.py:4` `OrchState`；各 `agents/*/state.py` | 给 dict 加 schema，运行时不校验（LangGraph State 契约） |
| `@dataclass(frozen=True)` | `mcpGateway/tool_model.py:27` `ToolSpec`；`modelService/chat_model.py:25` `Capabilities` | 不可变对象可安全跨协程共享，无需加锁 |
| `@dataclass`（可变） | `tool_model.py:118` `ToolResult`；`chat_model.py:46` `ChatResult` | 什么时候需要可变 |
| `Enum` | `tool_model.py:171` `ToolErrorKind`；`mcpGateway/rbac_config.py:5,20,35,51` 权限枚举族 | 用枚举替代魔法字符串 |
| 类型注解 | `utils/tracer.py:48` `ContextVar[Optional[str]]`；`agentCore/parsers/json_parser.py:12` `Callable[[dict], Optional[str]]` | `Optional` / `Callable` / 泛型别名 |
| `__slots__` | `memory/user_docs.py:135` | 省内存 + 禁止动态加属性（全项目唯一一处） |
| `copy.deepcopy` | `agents/orchestrator/nodes.py:1768` | 状态"深拷贝 + 打包提交"，避免串改 |
| 浅拷贝 vs 深拷贝 | 同上 | 面试必问：嵌套结构下浅拷贝会共享内层对象 |
| `ContextVar` | `utils/tracer.py:48` | 协程"隐式上下文传递"（trace 的 run_id），且**不跨独立协程传播**（`bootstrap.py:147` 注释）——这是很好的追问点 |

---

### 1.4 模块四：面向对象与设计模式

| 模式 | 项目依据 | 一句话说明 |
|---|---|---|
| **装饰器（带参）** | `utils/retry_utils.py:9,27,37` `async_retry(...)` + `@wraps` | 装饰器工厂 + 保留元信息 |
| **装饰器（注册）** | `mcpGateway/registry.py:78,94` `ToolRegistry.tool()` | 一行把本地函数注册成工具 |
| **单例** | `registry.py:343` `get_registry()`；`startup/task_manager.py:389` `get_task_manager()`；`utils/common.py:596` `db` | 全局 `_registry` + 惰性装配 |
| **工厂** | `mcpGateway/registry.py:361` `_build_platform()` | 集中装配适配器/工具/引擎 |
| **适配器 + ABC** | `mcpGateway/adapters.py:55,68,83` `ProtocolAdapter(ABC)` → `:97` `MCPAdapter` / `:201` `LocalAdapter` | 新增协议只加子类，业务代码不感知 |
| **策略 / 接口** | `modelService/chat_model.py:116` `ChatModel(ABC)` | 本地 Ollama 与外部 API 可互换 |
| **静态方法 / property** | `task_manager.py:303` `@staticmethod`；`agents/context_manager.py:100` `@property` | 只读属性（无 setter） |
| **自定义异常** | `chat_model.py:79` `ChatError`；`tool_model.py:190` `ToolError` | 业务异常归类，便于分层重试 |

**读 `adapters.py` 和 `registry.py` 的顺序**：先 `tool_model.py`（数据结构）→ 再 `registry.py`（注册中心）→ 再 `executor.py` → `adapters.py` → `client.py`（依据 `代码阅读路线.md` §4）。

---

### 1.5 模块五：语言基础与常用库

| 知识点 | 项目依据 | 说明 |
|---|---|---|
| **上下文管理器** | `utils/tracer.py:40,141,165` `@contextmanager run_scope` / `:191` `span` | `contextlib.contextmanager` + `yield` 写打点 |
| **异步上下文管理器** | `mcpGateway/client.py:175-176` `async with stdio_client(...)`；`sql_mcp_base.py:72` `async with db_lock` | 异步资源释放 |
| **生成器 `yield`** | `tracer.py` 的 `yield rid`；`memory/user_docs.py:669` `yield c["text"]` | 惰性求值、不必一次性 load 全量 |
| **正则 `re`** | `utils/number_whitelist.py:14-19`（三个 `re.compile`）；`agentCore/parsers/json_parser.py:27,76,166,208` | 抽数字 / 从 LLM 输出捞 JSON |
| **`defaultdict`** | `mcpGateway/a2a_queue.py:39,45` | 首次访问自动建空容器，调用方无需预注册 |
| **`functools`** | `utils/retry_utils.py:6,37` `wraps`；`sql_mcp_base.py:74` `partial` | 保留元信息 / 冻结关键字参数 |
| **`pathlib`** | `utils/common.py:5`；`database/init_db.py:7` | 跨平台路径 |
| **异常处理层级** | `utils/common.py:127-151`（`try/except/finally` + `for` 重试 + `raise`）| 精细的异常分流写法 |
| **模块/包/导入** | `config/__init__.py:2` 相对导入 + `__all__`；`modelService/__init__.py:42` `__getattr__`（PEP 562 惰性导入） | 延迟加载重模型 |
| **模块级全局状态** | `memory/short_memory.py:8,10,14`（`SESSION_MEM` / `SESSION_DRAFT_CACHE` / `SESSION_ASK_ROUND`） | 会话态在进程内存（`README.md` §12 明确列为已知边界） |
| **`json` 处理** | `startup/task_manager.py:358` `json.loads(r["payload"] or "{}")`；`nodes.py:1750` `ensure_ascii=False` | 容错解析 + 中文不转义 |
| **日志 loguru** | `utils/common.py:13-29`（运行日志 + 独立审计日志） | `rotation="1 day"` / `retention="7 days"` / `logger.bind` |
| **环境变量与配置** | `README.md` §3（`BILLAGENT_*` 开关 + `load_dotenv()`） | 配置与代码分离 |

---

### 1.6 模块六：工程化基础（面试加分项，非必需但项目里有）

| 知识点 | 项目依据 |
|---|---|
| 三层测试（unit / integration / e2e） | `README.md` §10；`tests/` 目录 |
| 契约测试（设计漂移即 FAIL） | `utils/contract_check.py`（560 行，**最后看**） |
| 可观测（span 树 / run_id） | `utils/tracer.py` + `utils/trace_view.py`；落库 `trace_span` / `llm_call_stat` |
| 沙箱与资源限制 | `mcpGateway/sandbox.py`（199 行）；`bootstrap.py:105` `apply_job_object_limits` |
| RBAC 权限 | `mcpGateway/rbac_config.py`（63 行） |

---

### 1.7 项目**没有**用到的 Python 特性（别浪费时间）

依据：全项目扫描结果。

- `multiprocessing`、`concurrent.futures`、`ThreadPoolExecutor`/`ProcessPoolExecutor` —— 完全未用（并发靠 `asyncio` + `subprocess` + 零星 `threading`）。
- `Pydantic` / `BaseModel` —— 未用，数据模型统一 `@dataclass`。
- `NamedTuple`、`typing.Protocol`、`Generic`、`TypeVar`、`@overload` —— 未用，抽象靠 `abc.ABC`。
- `functools.lru_cache` / `cached_property` —— 未用，缓存靠模块级标志位。
- 自定义 `__enter__`/`__exit__`/`__iter__`/`__next__` —— 未用，统一 `contextlib.contextmanager` + 生成器。
- `asyncio.Semaphore` / `as_completed` —— 未用。
- SQLAlchemy / 连接池 —— **明确不用**（`database/init_db.py:4`）。
- 标准库 `logging` —— 业务代码不用，统一 `loguru`。

> 注意：这些"没用"本身也是面试答案——"为什么不用 Pydantic 而用 dataclass""为什么不用 SQLAlchemy"都是可能的追问，答案在对应文件注释里。

---

## 第 2 章 Agent 领域知识框架

> 岗位依据：腾讯 2027 校招 Agent 开发工程师 JD（六大事业群覆盖），来源 `腾讯PCG_Agent岗位对标评估.md` §1.1-§1.4。

### 2.1 Agent 四大机制（JD 明文要求，必须能对应到项目）

| 机制 | 定义 | 项目落地 | 成熟度 |
|---|---|---|---|
| **Planning** | 把用户意图拆成可执行步骤 | `plan_node`（`agents/orchestrator/nodes.py:920`）+ 任务白名单校验 | ★★★★☆ |
| **Memory** | 短期会话 / 长期记忆 / 用户画像 | `memory/short_memory.py`（短期）+ `memory/long_memory.py`（FAISS 长期）+ `memory/user_docs.py`（笔记库 RAG） | ★★★★☆（原评估 ★★☆☆☆ 已随 M8/M12 提升） |
| **Tool Use** | 模型调用外部能力 | `mcpGateway/`（Registry + 3 个 MCP 常驻服务 + RBAC + 审计） | ★★★★★ |
| **Reflection** | 结果自检与自我修正 | M10 已补（stat）+ M11 复用（finance）；见 `项目现状评估.md` §1.3 的 D9 ✅ | ★★★☆☆ |

> 依据修正说明：`腾讯PCG_Agent岗位对标评估.md` §2.2 给 Memory 打 ★★☆☆☆、Reflection 打 ★☆☆☆☆ 是**改造前口径**；
> `项目现状评估.md` §1.1/§1.3 显示 M8/M10/M11/M12 已完成相应补齐。**面试时必须用最新口径**，否则自曝假短板。

### 2.2 多 Agent：为什么拆、代价是什么

- **拆的正当理由**（`多Agent系统学习.md` §1.3 三条信号）：子任务信息量大需隔离 / 可并行的独立子任务 / 需要不同权限与工具边界。
  - 项目命中第 1、3 条：`bill_agent` 有写权限、`stat/price/finance` 只读（`mcpGateway/rbac_config.py`）。
- **拆的代价**（必须主动讲，否则显得没权衡）：延迟 2~10 倍、Token 翻倍、复杂度指数上升；业界经验"70% 场景单 Agent 就够"。
- **项目结论**：「1 编排 + 4 子 Agent」换来**职责内聚、失败隔离、可单独替换与测试**（`README.md` §6）。

### 2.3 编排模式与拓扑

| 模式 | 项目对照 | 依据 |
|---|---|---|
| Sequential 流水线 | bill → stat → finance 依赖链 | `TASK_CONTEXT_DEPS` |
| Parallel 并行 | bill ∥ price，之后 stat（依赖 bill） | `README.md` §7.5；`nodes.py` 批量派发 + `gather` |
| Supervisor 主管 | 中心化编排器（单次 LLM 规划，不是循环 supervisor） | `plan_node` |
| Hierarchical 分层 | 两级图的雏形 | 编排器图 + 4 个子 Agent 图 |
| Network 网状 | **无**（诚实承认） | — |

**决策权谱系（面试高分局）**：代码定框架（拓扑/依赖/权限/熔断）+ LLM 定内容（规划/文案/参数提取）。
项目就是这种混合范式——`多Agent系统学习.md` §2.2 的原话是"**决策权在代码**"。

### 2.4 通信机制

| 模式 | 原理 | 项目对照 |
|---|---|---|
| 消息传递（队列） | 点对点投递 | **本项目**：`mcpGateway/a2a_queue.py`，按 agent 分队列 + `Event` 唤醒 |
| 发布订阅 | 主题广播 | 未用 |
| 黑板/共享状态 | 共享内存 | 部分：LangGraph State（单图内） |

**必须能讲的三个总线设计点**（都在 `a2a_queue.py` 注释里）：

1. **为什么事件驱动不用轮询**：`recv_task_blocking` 的 `await queue.get()` 零空转；改造前 `get_nowait + sleep(0.1)` 导致 4 个 worker 每秒约 40 次无谓唤醒（`:161-162` 注释）；实测 P95 投递延迟 30.891ms → 0.306ms（`README.md` §8）。
2. **为什么不能把不匹配的消息放回队尾**：`await queue.get()` 会立刻再取到同一条，退化成 **100% CPU 忙循环**；所以改用旁路暂存 `_parked`（`:43-45` 注释）。
3. **为什么 `recv_task` 里有"快路径"**：直接 `await queue.get()` 会让出控制权，同 loop 的常驻 worker 会**抢走本该由调用方接收的消息**（`:102-104` 注释）。

> 这三条是"做过并发的人"才会写的注释，面试讲出来含金量极高。

### 2.5 工具：Function Calling / MCP / 工具平台三层

- **三层不能混**（`工具接入平台与MCP系统学习.md` §1.2）：
  Function Calling 管"模型怎么选工具"；MCP 管"工具怎么被接入"；工具平台管"工具怎么被管理"。
- 项目四要素架构：Registry（`registry.py`）/ Discovery（权限过滤即发现过滤）/ Runtime（`executor.py`）/ Adapter（`adapters.py`）。
- **最值钱的设计**：一次声明元数据，**鉴权 / 审计 / 计时 / 文档四件套自动生效**，业务代码一行不打点（`README.md` §7.2）。
- **权限红线**：编排器只有只读权限、**零写权限**（RBAC），写操作只可能发生在有职责的子 Agent 里。

### 2.6 记忆与 RAG

- 短期记忆（会话态、草稿、追问计数）：`memory/short_memory.py`。
- 长期记忆（向量）：`memory/long_memory.py`（FAISS + pkl 原文）。
- 用户笔记库（RAG 唯一入口）：`memory/user_docs.py`。
- 检索降级：向量不可用自动退 BM25，不崩（`README.md` §11）。
- 实测基线：hit@1 61.54% / hit@3 100% / recall@5 92.86%（`README.md` §10）。

### 2.7 可靠性与工程化（面试深挖区）

| 机制 | 项目实现 | 依据 |
|---|---|---|
| Durable / 崩溃恢复 | `submitted→running→completed/failed/interrupted` 状态机 + `task_id` 幂等 | `startup/task_manager.py`；`README.md` §7.3 |
| 防死循环 | 依赖环检测 `_has_cycle`（DFS）+ 死锁检测 + `dispatch_round > 20` 熔断 | `代码阅读路线.md` §2 |
| 超时 | 工具级 `spec.timeout`、总线 240s | `executor.py:217`；`a2a_queue.py:218` |
| 重试分类 | 可重试（格式错/5xx/429/空响应）vs 不可重试（401/400）；**有写副作用永不重试** | `README.md` §11 |
| 可观测 | run_id + span 树，自动包 span | `utils/tracer.py` |
| 沙箱 | 进程级资源限制 + 工具超时 + 危险前缀拦截 | `mcpGateway/sandbox.py` |
| 防幻觉 | 方案 C 三层（白名单约束 → 修正重试 → 回退本地确定性渲染） | `utils/number_whitelist.py`；`README.md` §7.1 |

### 2.8 评测体系（项目差异化亮点，JD 单独列条）

- L3 回答质量评测 12/12（本地渲染均分 10.0；外部模型 + 方案 C 均分 9.83）。
- 检索评测 13 条查询；记忆评测。
- 基线存档 `eval_reports/l2_baseline_userdocs_20260904.txt`。
- 测试规模：`tests/unit` 271 passed / 34s（`README.md` §10）。

### 2.9 LangGraph 核心概念（必须能对照项目讲清"用了什么、没用什么"）

背诵卡见 `LangGraph面试冲刺_腾讯PCG.md`。**关键是"没用"的部分要主动认账**：

- 用了：StateGraph、静态边、条件边（仅 stat）、`Command` 动态跳转、回环边、`ainvoke`。
- **没用**：Reducer（手动深拷贝写回）、Checkpointer（自管持久化）、Interrupt（A2A 阻塞模拟）、Streaming、Subgraph（跨 Agent 走 A2A 总线）、`send` 动态扇出。

---

## 第 3 章 面试官考察角度（他会怎么问）

### 3.1 六层漏斗：从"你会不会"到"你懂不懂"

| 层 | 面试官想验证 | 典型提问 | 你的弹药 |
|---|---|---|---|
| ① 动机层 | 是你做的还是抄的 | "为什么做这个项目？为什么记账场景？" | 设计动机三约束（`README.md` §6） |
| ② 概念层 | 基础是否扎实 | "GIL 是什么？协程和线程区别？" | §1.1 全部内容 |
| ③ 架构层 | 有没有全局观 | "整体架构讲一下""为什么 1+4 不是 1 个" | 七层架构图 + 1+4 拆分理由（`README.md` §6） |
| ④ 实现层 | 细节是不是你写的 | "统计为什么一定包含刚记的那笔？" | 依赖图 `_merge_deps` + `wait_result` 波次 |
| ⑤ 取舍层 | 有没有工程判断 | "为什么不用进程？为什么不用 SQLAlchemy？" | 见 §3.2 取舍清单 |
| ⑥ 极限层 | 系统思维深浅 | "崩了怎么办？并发写冲突怎么办？多机怎么改？" | §2.7 全部 + 已知边界诚实认账 |

**结论：①③④是保命层，⑤⑥是分水岭。**

### 3.2 高频取舍追问清单（背下来，每条都要有依据）

| 追问 | 答题要点 | 依据 |
|---|---|---|
| 为什么子 Agent 用协程不用进程？ | 主进程不做重计算，IO 等待为主；进程化带来 IPC、状态序列化、数倍内存开销 | `README.md` §11 |
| 为什么 MCP 要常驻？（原来短连接） | stdio 每次冷启动 26~29s；常驻后就绪约 6s 一次，端到端 76.6s→58.5s（-24%） | `README.md` §8 |
| 为什么用 WAL？ | 读不阻塞写，是并行派发前提；解决跨进程 `database is locked` | `utils/common.py:62-65` |
| 为什么删掉重试里的 `sleep`？ | 同步 sleep 冻结事件循环；锁竞争已由 `busy_timeout` 在 C 层吸收 | `utils/common.py:139-140` |
| 为什么不用 SQLAlchemy？ | 单机单用户、单连接足够；引入 ORM 增加抽象与依赖 | `database/init_db.py:4` |
| 为什么参数不跨通道转译？ | `num_predict`（软约束）≠ `max_tokens`（硬上限含 reasoning），曾导致推理模型正文为空 | `README.md` §11 |
| 为什么有写副作用的工具不重试？ | 防重复记账（幂等性缺失下的安全默认） | `README.md` §11 |
| 为什么总线不把消息放回队尾？ | `await get()` 会立刻取到同一条 → 100% CPU 忙循环 | `a2a_queue.py:43-45` |
| 为什么编排器不能连 DB？ | 只读走 `fact_tools`，零写权限；写只可能发生在职责 Agent 内 | `README.md` §7.2；`mcpGateway/fact_tools.py` |

### 3.3 Python 基础八股清单（面试官单独成题的部分）

按项目支撑度排序，**每条都建议直接拿项目代码当答案**：

1. GIL 是什么、影响什么 → §P1.2（有 3 处项目实证）
2. 进程/线程/协程区别与选型 → §P1.3（项目三种都用）
3. 协程原理、`await` 做了什么 → §P1.4
4. `asyncio.Queue/Event/Lock` 用法 → `a2a_queue.py`
5. 深浅拷贝区别 → `nodes.py:1768`
6. 装饰器原理与带参装饰器 / `functools.wraps` → `utils/retry_utils.py`
7. `with` 的实现（上下文管理器）→ `utils/tracer.py` 的 `@contextmanager`
8. 生成器与惰性求值 → `memory/user_docs.py:669`
9. 可变默认参数陷阱 / 闭包 / 作用域
10. `__slots__` 作用 → `memory/user_docs.py:135`
11. `dataclass` vs `dict` vs `NamedTuple`
12. 异常处理与自定义异常 → `tool_model.py:190`
13. 单例的几种实现 → `registry.py:343`
14. 数据库索引/事务/隔离级别 → §1.2
15. `ContextVar` 与上下文传播 → `utils/tracer.py:48`

### 3.4 已知短板与应对话术（**必须提前准备，不能现场卡壳**）

来源：`腾讯PCG_Agent岗位对标评估.md` §2.2/§3.2（注意其中 Memory/Reflection 评分已被 `项目现状评估.md` 修正，见 §2.1 说明）。

| 追问 | 现状 | 应对话术方向 |
|---|---|---|
| worker 有自主性吗？ | 全是确定性代码 | "场景决定收敛：财务场景**错误成本远大于灵活性收益**，所以我把确定性锁死在关键路径，只在规划与表达处放 LLM。" |
| 有 Reflection 吗？ | M10/M11 已补（stat + finance 结果自纠） | 讲具体实现 + 承认没有通用反思循环，演进方向是"结果校验失败 → 重新规划" |
| 影响力资产？ | 未开源、无博客 | "正在整理架构与踩坑文档，计划开源"（诚实 + 行动） |
| 会话态在内存？ | 重启即丢 | 主动认账 + 演进方案（Redis + TTL）——`README.md` §12 已写明 |
| 部署/高并发经验？ | 单机单用户 | 承认边界，讲 MCP 常驻 + WAL + 批量派发已为并发做的准备 |

**通用原则**：短板要**主动认账 + 给出取舍理由 + 给演进路径**，三段式。硬撑必崩。

---

## 第 4 章 学习推进路线（建议 5 期）

| 期 | 主题 | 主攻文件 | 通关标准 |
|---|---|---|---|
| **第 1 期** | 跑通 + 建立直觉 | 先跑 `start_webui.bat`（`代码阅读路线.md` §0） | 能描述"一次请求的 N 步" |
| **第 2 期** | Python 并发与执行模型 | `mcpGateway/a2a_queue.py`（167 行）→ `startup/bootstrap.py:40-140` → `utils/common.py:31-170` | 能答 L3：为什么协程/为什么 WAL/为什么删 sleep |
| **第 3 期** | 主干数据流 | `agents/orchestrator/state.py` + `graph.py` + `nodes.py` 的 4 个节点函数 | 能答 `代码阅读路线.md` §2 的 4 个问题 |
| **第 4 期** | 工具平台 + 模型层 | `mcpGateway/tool_model.py` → `registry.py` → `executor.py` → `adapters.py`；`modelService/chat_model.py` → `router.py` | 能答"为什么四件套不用写业务代码""新增工具改几处" |
| **第 5 期** | 深挖 + 面试打磨 | `utils/tracer.py`、`utils/number_whitelist.py`、`utils/contract_check.py` + `LangGraph面试冲刺_腾讯PCG.md` | 能答 §3.2 全部追问 |

**提速技巧**（`代码阅读路线.md` §8）：
1. 用 `python utils/trace_view.py <run_id>` 看真实 span 树，**拿 span 名字去搜代码**；
2. 单元测试就是文档（`tests/unit/`）；
3. IDE 大纲视图跳读 `nodes.py`，**绝不顺序翻**。

---

## 第 5 章 使用说明

1. 从**第 1 期**开始，不要跳。没有"跑通一遍"的直觉，读代码就是迷路。
2. 每学一个知识点，用 §0.2 模板问我，并明确告诉我**你卡在哪一级**（L1/L2/L3/L4）。
3. 学完一个模块，回来对照 §0.1 自查；答不到 L3 的点标记出来，集中补。
4. **不确定的结论一定让我给依据索引**——这份框架里的每条索引你都可以自己去 `文件:行号` 核对。
