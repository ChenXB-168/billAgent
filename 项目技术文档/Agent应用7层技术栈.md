# Agent 应用 7 层技术栈

> 用途：梳理一个 AI Agent 应用涉及的全部技术领域，逐层对照业界主流方案与本项目实现，
> 明确各层的成熟度与差距，作为技术选型与面试对标的参考地图。
>
> 说明：本文档中的"业界方案"指公开可见的主流选型，"本项目"指 `billAgent`（消费管家智能体）。

---

## 总览

做 Agent 应用不只是"调模型 + 写提示词"，它是一个横跨 7 个技术领域的系统工程：

| # | 层 | 解决什么问题 | 一句话概括 |
|---|---|---|---|
| 1 | **编排层** | 多步任务、多 Agent 怎么组织 | 流程引擎 |
| 2 | **工具层** | Agent 怎么调用外部能力 | 能力接入 |
| 3 | **模型层** | 怎么统一调用不同厂商的模型 | 模型抽象 |
| 4 | **记忆与检索层** | 上下文怎么管、知识怎么找 | 上下文工程 |
| 5 | **可观测层** | 链路怎么追踪、成本怎么归因 | 运行时的眼睛 |
| 6 | **评测层** | 怎么衡量 Agent 好不好 | 质量的尺子 |
| 7 | **工程与部署层** | 怎么跑起来、怎么扩、怎么恢复 | 生产化底座 |

**分层原则**：**上层依赖下层，同层内可替换。** 比如换掉模型层（LiteLLM↔自研 harness）不影响编排层；换掉编排层（LangGraph↔AutoGen）不影响工具层。

---

## 第 1 层：编排层

### 解决什么问题

一个用户请求往往需要多步完成（识别意图 → 拆任务 → 分发执行 → 汇总结果）。
**编排层决定这些步骤怎么组织、状态怎么流转、失败怎么恢复。**

两个核心问题：
1. **控制流**：谁决定下一步做什么？——图（显式）还是模型自主（隐式）
2. **协作模式**：多个 Agent 之间怎么分工与通信

### 业界方案

| 方案 | 特点 | 适用 |
|---|---|---|
| **LangGraph** | 图状态机（StateGraph），控制流显式可见 | 流程确定的场景 |
| **langgraph-supervisor** | LangGraph 官方封装的 Supervisor 模式 | 中央调度 + 多 worker |
| **AutoGen**（微软） | 多 Agent 群聊，Agent 之间自由对话 | 探索式协作 |
| **CrewAI** | 角色扮演 + 任务分工 | 分工清晰的流水线 |
| **OpenAI Agents SDK / Swarm** | 轻量 handoff（控制权转交） | 层级浅的场景 |
| **Dify Workflow** | 低代码可视化编排 | 快速搭建 |

### 核心概念

- **StateGraph**：节点 + 边 + 共享 State
- **Orchestrator-Worker**：中央编排者规划 + 多个 worker 执行（Anthropic《Building Effective Agents》定义）
- **Supervisor**：与 Orchestrator-Worker 基本同义，更强调"路由 + 监督"
- **Handoff**：Agent 之间转交控制权
- **Checkpointing**：状态持久化，支持中断恢复
- **Human-in-the-loop**：人工介入节点

### Agent 间通信的三种机制

| 机制 | 代表 | 特点 |
|---|---|---|
| **共享 State** | LangGraph 默认 | 节点读写同一对象；简单，但并发写难控 |
| **工具调用 / Handoff** | Agents SDK、langgraph-supervisor | 复用 Function Calling；同步阻塞 |
| **消息传递** | AutoGen、自建总线 | 异步解耦、可恢复；要自己写调度 |

### 本项目实现

- **编排**：LangGraph，4 节点链路 `plan → dispatch → wait_result → collect`
- **模式**：Orchestrator-Worker —— 1 个编排器 + 4 个专职子 Agent（记账 / 统计 / 比价 / 理财）
- **通信**：**自建进程内消息总线**（`mcpGateway/a2a_queue.py`），按 Agent 分队列，4 个常驻 worker 消费
- **调度**：按 DAG 依赖**分层批量派发**，同层内并行、层间串行

**为什么自建消息总线而不用共享 State**：
1. 子 Agent 需要**跨请求常驻**（作为 worker 持续运行），共享 State 只管一次图执行
2. 需要**崩溃恢复**——任务状态必须独立于执行上下文持久化
3. 需要**按 DAG 分层派发 + 乱序收集 + 每个任务独立重试**

三者分工：`LangGraph State` 管一次请求内的编排，`消息总线` 管 Agent 间通信，`task_state 表` 管任务生命周期。

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| 未使用 `langgraph-supervisor` | 自研调度换取更细的颗粒度控制（"每种任务类型至多一条"），代价是维护成本 |
| 消息总线仅进程内 | 多机部署需换成 Redis Pub/Sub / RabbitMQ |
| 无 Human-in-the-loop | 当前全自动，无人工审核节点 |

---

## 第 2 层：工具层

### 解决什么问题

Agent 本身只能"说"，不能"做"。要查数据库、调 API、读文件，必须有一套**能力接入机制**。

三个子问题：
1. **怎么声明**：工具长什么样、要什么参数
2. **怎么调用**：谁来执行、怎么传参、怎么拿结果
3. **怎么治理**：权限、审计、超时、重试

### 业界方案

| 方案 | 特点 |
|---|---|
| **MCP**（Anthropic） | 事实标准协议，生态最大；Server/Client 分离 |
| **Function Calling** | 模型侧能力，用 JSON Schema 描述工具 |
| **LangChain Tools** | 框架内工具抽象 |
| **OpenAPI / Swagger** | 传统 HTTP 接口描述规范，可作为工具来源 |
| **自研 Tool Registry** | 大厂常见，可控性最高 |

### 核心概念

- **Schema 声明**：工具的入参用 JSON Schema 描述，供模型理解
- **注册 / 发现 / 执行三段式**：声明 → 按权限筛选可见工具 → 实际执行
- **横切能力**：权限校验（RBAC）、操作审计、耗时统计、链路追踪
- **传输方式**：
  - **stdio**：客户端拉起 server 子进程，标准输入输出通信，用完关掉 → 天然"按需启动"
  - **Streamable HTTP**：server 作为 HTTP 服务常驻 → 天然"常驻"

### MCP 与 Function Calling 的关系

**两者不是一回事，是互补的**：

| | Function Calling | MCP |
|---|---|---|
| 是什么 | 模型的**能力**——输出"我要调某工具" | **协议**——工具怎么被声明、发现、调用 |
| 层级 | 模型侧 | 工程侧（服务间协议） |
| 关系 | MCP server 声明的工具，最终仍通过 FC 让模型选择 | |

### 本项目实现

- **协议**：MCP + FastMCP
- **服务**：3 个常驻 MCP 服务
  | 服务 | 端口 | 暴露的能力 |
  |---|---|---|
  | `sql_bill` | 8001 | 账单库读写（唯一碰 DB 的服务） |
  | `llm_base` | 8002 | 通用模型调用（含 FC 通道） |
  | `llm_finance` | 8003 | 理财专属模型 |
- **注册表**：自研 `ToolRegistry`（`mcpGateway/registry.py`），统一本地函数与 MCP 服务的调用契约
- **适配器层**：`adapters.py` 把协议差异收敛，上层调用无差别；新增协议只需实现一个适配器类并注册
- **治理**：RBAC 最小权限（编排器零写权限）、操作审计、耗时统计、链路追踪在网关层自动生效

**为什么拆 3 个服务**：权限隔离（不同 Agent 只能访问对应服务）+ 故障隔离 + 独立演进。

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| 未接入真实第三方 MCP 服务 | 当前 3 个服务均为自建 |
| 工具数量有限 | 18 个左右，未做工具分组检索（工具多后需要先检索再注入） |

---

## 第 3 层：模型层

### 解决什么问题

不同厂商的模型 API 细节不同（参数名、能力、错误码），业务代码不应该关心这些差异。

**模型层的目标是提供统一的调用接口，并处理路由、降级、重试。**

### 业界方案

| 方案 | 特点 |
|---|---|
| **LiteLLM** ⭐ | 统一 100+ 模型的 SDK，事实标准 |
| **OpenAI SDK + 兼容协议** | 各厂商提供 OpenAI 兼容端点，客户端不用改 |
| **LangChain ChatModel** | 框架内抽象，与 LangChain 生态绑定 |
| **vLLM** | 高性能推理服务（自部署） |
| **Ollama** | 本地模型运行工具（开发/离线） |

### 核心概念

- **OpenAI 兼容协议**：`POST /v1/chat/completions`，`{model, messages, temperature, ...}`。智谱、DeepSeek、通义、vLLM 都支持 → 换供应商不用改代码
- **参数归一化**：不同厂商参数语义不同。例：本地 Ollama 的 `num_predict`（软约束）≠ 外部 `max_tokens`（硬上限且含 reasoning 消耗）→ 必须**同义归一、异义过滤并告警**
- **结构化输出**：两种实现路径
  - **约束**：`response_format={"type":"json_schema"}` / vLLM guided decoding —— 由服务端/解码层**强制**格式
  - **容错**：提示词要求 JSON + 解析器清洗重试 —— 由**应用代码**兜底
- **重试分类**：可重试（格式错 / 5xx / 429 / 空响应）走统一预算池；不可重试（401 / 400 / 鉴权失败）立即短路
- **FC 循环**：`模型 → tool_calls → 执行工具 → 回填 observation → 再决策`，无 tool_calls 即结束

### 本项目实现

- **harness**：自研 ChatModel 抽象（`modelService/`），带**能力契约**（声明模型支持 FC / 流式 / 窗口大小）
- **双通道**：外部 OpenAI 兼容 API（默认，智谱 GLM-4-Flash）+ 本地 Ollama 小模型（离线兜底）
- **结构化输出**：**采用"容错"路径**——提示词要求 JSON + `agentCore/parsers/json_parser.py` 做清洗（剥 markdown 围栏、递归键名去空格、缺字段重试）
- **FC**：`llm.chat_fc` 通道，工具清单经 `registry.to_openai_schema()` 生成，四道护栏（可见面裁剪 / 轮数上限 / 写幂等闸门 / 失败即 observation）

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| 未对标 LiteLLM | 自研 harness 的优势在"能力契约 + 参数归一"，劣势在模型覆盖少 |
| 未使用 JSON Schema 强约束 | 当前是"解析容错"而非"输出约束"；可考虑接 `response_format` 或 guided decoding |
| 无流式输出 | 当前均为一次性返回 |

---

## 第 4 层：记忆与检索层

### 解决什么问题

1. **上下文管理**：多轮对话历史、长期偏好怎么存、怎么裁、怎么注入
2. **知识检索（RAG）**：用户资料、历史记录怎么被准确找到

### 业界方案

| 子层 | 主流选型 |
|---|---|
| **向量库** | FAISS（本地）、Milvus、Qdrant、pgvector、Chroma、Weaviate |
| **Embedding** | bge 系列、m3e、OpenAI text-embedding、Cohere |
| **检索策略** | 纯向量 / 纯 BM25 / **混合检索** / **RRF 融合** |
| **重排（Rerank）** ⭐ | bge-reranker、Cohere Rerank |
| **记忆管理** | 短期滑动窗口 / 长期摘要 / 时间衰减 / 定期裁剪 |

### 核心概念

- **混合检索（Hybrid Retrieval）**：向量路（语义）+ 词法路（BM25/关键词）互补
- **RRF（Reciprocal Rank Fusion，倒数排名融合）**：**不看绝对分数、只看排名**，把两路的排名取倒数相加。
  **解决的问题**：BM25 分数与向量距离量纲完全不同，直接加权几乎调不出好参数；RRF 免归一化
- **标准流程**：`召回 → 重排 → 取 top-k`
- **检索指标**：hit@k（命中率）、recall@k（召回率）、MRR（平均倒数排名）、NDCG

### 本项目实现

| 模块 | 实现 |
|---|---|
| **个人笔记库**（`memory/user_docs.py`） | **真 BM25**（内置 Okapi 实现，k1=1.5 / b=0.75）+ FAISS 向量 → **RRF 融合**；向量不可用时降级 BM25-only |
| **长期消费记忆**（`memory/long_memory.py`） | 向量语义路（权重 0.6）+ jieba 关键词路（0.4）+ **口语时间表达强信号**（命中 +0.15，上限 +0.3）；按 **12 个月滚动裁剪**控制索引规模 |
| **短期记忆** | 会话级滑动窗口，纯内存不落盘 |

**两个防坑设计**：
1. **检索可用集与索引构建同源**（`iter_doc_chunks()`）——防止"评测说可用但检索拿不到"
2. **时间表达翻译**：记忆文本里是 ISO 格式（`2026-08`），用户口语是"1月"，jieba 会把"1月"切成单字被过滤 → 用正则先翻译成 ISO 再参与匹配

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| **未启用重排** ⚠️ | `modelService/rerank_loader.py` 已实现但调用被注释。当前规模（摘要 12 条）不需要，规模上去后必须补 |
| 向量库为本地文件 | FAISS 本地索引，无法横向扩展；生产应换 Qdrant / Milvus |
| 无查询改写 | 未做 query rewriting / HyDE 等查询增强 |

---

## 第 5 层：可观测层

### 解决什么问题

Agent 应用是**长链路 + 多组件 + 概率性**的，出问题时"哪一步、哪个模型、花了多少 token"必须能查。

**可观测是所有优化和排障的前提**——没有它，性能问题无法定位、成本无法归因。

### 业界方案

| 方案 | 特点 |
|---|---|
| **Langfuse** ⭐ | 开源，当前最主流 |
| **LangSmith** | LangChain 官方，与 LangGraph 集成好 |
| **Phoenix** | Arize 开源 |
| **OpenTelemetry** | 通用可观测标准，正在向 LLM 场景扩展（GenAI semantic conventions） |

### 核心概念

- **Trace / Span 模型**：一次请求 = 一个 trace，链路每步 = 一个 span，span 可嵌套
- **Token 与成本统计**：按模型、按 Agent 归因
- **自动埋点**：工具调用经网关自动包 span，业务代码不写打点
- **告警与降级可见**：**任何兜底必须打 WARNING + 计数，禁止静默失败**

### 本项目实现

- **自研 Tracer**（`utils/tracer.py`）：一次请求一个 `run_id`，工具调用与 LLM 调用自动包 span
- **落库**：`trace_span`（链路）+ `llm_call_stat`（token / 耗时 / 成败 / 按 Agent 归因）
- **查看工具**：`utils/trace_view.py` 可直接打印 span 树

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| 无标准对齐验证 | span 结构与 OpenTelemetry 语义相近，但未做正式对齐 |
| 无采样与聚合 | 数据全量落 SQLite，无采样策略、无聚合看板、无告警 |
| 未接 Langfuse / LangSmith | 自研方案胜在轻量和可控，败在生态与可视化 |

---

## 第 6 层：评测层

### 解决什么问题

LLM 应用的输出是**概率性**的，改一行提示词可能让效果变好也可能变差。
**没有评测就无法判断改动是优化还是劣化。**

评测要回答两个问题：
1. **检索准不准**——该给的知识给到了吗
2. **回答对不对**——有没有幻觉、有没有答到点上

### 业界方案

| 方案 | 特点 |
|---|---|
| **RAGAS** ⭐ | RAG 评测事实标准 |
| **DeepEval** | pytest 风格，易集成 CI |
| **TruLens** | 可观测 + 评测一体 |
| **LangSmith Eval** | 云端，与 LangChain 生态集成 |
| **LLM-as-a-Judge** | 通用范式，用模型当裁判 |

### 核心概念

**检索指标**：

| 指标 | 回答什么 |
|---|---|
| hit@k | "能不能问到东西" |
| recall@k | "该给的知识给全了没有" |
| MRR@k | "排得准不准" |

**生成指标（RAGAS 体系）**：

| 指标 | 含义 | 对应本项目 |
|---|---|---|
| **faithfulness** | 回答是否忠于检索到的上下文 | **数字白名单校验** |
| answer relevancy | 回答是否切题 | 意图正确性维度 |
| context precision / recall | 检索质量 | hit@k / recall@k |

**打分方式的两条路线**：
- **规则打分**：正则 / 关键词 / 集合比对。**确定、可复现、能进 CI**，但只能判"正确性"判不了"有用性"
- **LLM 打分**：灵活，但**不稳定、会算错、贵且慢**，通常只作交叉验证

### 本项目实现

**三层评测体系**：

| 层 | 脚本 | 规模 | 方式 |
|---|---|---|---|
| **单元测试** | `tests/unit/` | 44 个文件 / 457 用例 | pytest，零外部依赖；含契约漂移检测 |
| **检索评测** | `evaluation/user_docs_recall_eval.py` | 13 条查询 | 字符串完全匹配；hit@1 / hit@3 / recall@5 / MRR |
| **回答正确性** | `evaluation/agent_answer_eval.py` | 12 条用例 | **五维规则 rubric**（意图 / 参数 / 语义 / 格式 / 无幻觉，各 0-2 分）+ **DB 硬校验** |

**两条设计原则**（写在脚本头部）：
1. **确定性优先**：能用正则/关键词判的分，绝不用 LLM 判
2. **正负双向断言**：`must_contain`（该有的有）+ `must_not_contain`（不该有的没有）

**防坑设计**：GT 缺失即**硬失败退出**（防止评测静默失效）

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| 检索评测集偏小 | 13 条查询的 GT 由样例反推，**只能验证链路，不代表泛化能力** |
| 未对标 RAGAS | 自研规则体系，需要说明与 RAGAS 指标的映射关系 |
| 评测未进 CI | `pytest tests/unit` 可进 CI，检索/回答评测需要模型，未接入 |
| 规则 rubric 的盲区 | 只能测"正确性"，测不了"建议有没有用"——后者需要 LLM 打分或人工评估 |

---

## 第 7 层：工程与部署层

### 解决什么问题

让系统稳定跑起来，并能应对崩溃、扩容、并发。

**Agent 应用的工程难点在于：长链路 + 有副作用（写库/调外部 API）+ 概率性失败。**

### 业界方案

| 子层 | 主流选型 |
|---|---|
| **服务框架** | FastAPI、Flask |
| **任务队列** | **Celery**（Redis/RabbitMQ 作 broker）、**Temporal** ⭐ |
| **消息中间件** | Redis Pub/Sub、RabbitMQ、Kafka |
| **状态持久化** | PostgreSQL、Redis、Temporal 事件历史 |
| **容器与编排** | Docker、Docker Compose、Kubernetes |
| **并发模型** | asyncio（IO 密集）、多进程（CPU 密集） |

### 核心概念

- **Durable Execution（持久化执行）**：Temporal 的核心范式——工作流状态持久化，崩溃后从事件历史**重放**恢复，保证"恰好一次"语义
- **幂等**：用业务主键（如 `task_id`）保证重复投递不重复执行
- **状态机**：任务生命周期显式建模（submitted / running / completed / failed / interrupted）
- **优雅关闭**：收到关停信号后停止接新任务、等待在途任务完成
- **WAL（Write-Ahead Logging）**：SQLite 的日志模式，**读不阻塞写、写不阻塞读**（默认模式读写互斥）
  - 原理：写操作追加到 `-wal` 文件而不碰主库，读操作先看 `-wal` 再看主库
  - 配合 `busy_timeout`：写锁竞争时由 SQLite 在 C 层等待而非立即报错

### 本项目实现

| 方面 | 实现 |
|---|---|
| **并发模型** | asyncio 单进程多协程（主进程 + 4 个常驻 worker 协程） |
| **任务状态机** | `startup/task_manager.py`，独立库 `task_state.db`，状态 `submitted / running / completed / failed / interrupted` |
| **崩溃恢复** | 启动时扫描遗留 `running` / `interrupted`，按 `task_id` 对账：未落库 → 重投；已落库 → 标完成 |
| **幂等** | `task_id` 作幂等键；有写副作用的工具**永不自动重试** |
| **并发读写** | SQLite `PRAGMA journal_mode=WAL` + `busy_timeout=5000`；写操作加进程内 `db_lock`，读操作不加锁 |
| **容器化** | 提供 Dockerfile / docker-compose，未实际部署 |

**与业界的对应关系**：

| 本项目 | 业界对应 |
|---|---|
| `task_state` 状态机 + `task_id` 对账重投 | **Temporal** 的事件重放（单机简化版） |
| 进程内消息队列 | Celery broker + backend |
| SQLite | PostgreSQL |
| 单进程多协程 | 服务化后为多实例 |

### 差距与演进方向

| 差距 | 说明 |
|---|---|
| **单机单用户** | 无用户体系、无服务端、数据在本地文件系统 |
| 消息队列在进程内 | 多机部署需换 Redis / RabbitMQ |
| 会话态在进程内存 | 重启即丢；演进方向是 Redis + TTL |
| 无 API 层 | 当前是 CLI / Gradio，未做 FastAPI 服务化 |
| 状态机是简化版 | 只支持"崩溃重投"，Temporal 支持任意时刻精确重放 |

---

## 汇总：各层成熟度自评

| 层 | 成熟度 | 主要差距 | 面试必答的对标 |
|---|---|---|---|
| 1 编排 | ⭐⭐⭐⭐ | 未用现成 Supervisor | 为什么自建消息总线而不是共享 State |
| 2 工具 | ⭐⭐⭐⭐ | 未接第三方 MCP | MCP 与 Function Calling 的关系 |
| 3 模型 | ⭐⭐⭐ | 未对标 LiteLLM | harness 比纯 SDK 多了什么（能力契约、参数归一） |
| 4 记忆检索 | ⭐⭐⭐ | **未启用重排** | 为什么不做 rerank，规模上去后怎么办 |
| 5 可观测 | ⭐⭐⭐ | 无采样聚合告警 | span 模型对齐了什么标准 |
| 6 评测 | ⭐⭐⭐ | 评测集小、未进 CI | 与 RAGAS 指标的映射；为什么用规则不用 LLM 打分 |
| 7 工程部署 | ⭐⭐ | 单机、无服务化 | 与 Celery / Temporal 的关系，演进路径 |

---

## 演进路线图

如果需要往生产化演进，优先级如下：

```
进程内队列       →  Redis Pub/Sub / RabbitMQ      （支持多机）
SQLite 状态机    →  Temporal / Celery             （支持精确重放）
FAISS 本地索引   →  Qdrant / Milvus               （支持亿级向量）
补重排            →  bge-reranker                 （召回→重排→top-k 完整链路）
自研 Tracer      →  Langfuse + OpenTelemetry      （采样、聚合、告警）
自研评测          →  对齐 RAGAS 指标 + 接入 CI
单机脚本         →  FastAPI 服务化                （多租户）
```

---

## 附：一页速记（面试用）

**问"你这个项目用了哪些技术栈"，按 7 层回答**：

> **编排层**用 LangGraph，因为我要显式控制流而不是黑盒 Agent 循环；协作模式是 **Orchestrator-Worker**，编排器负责意图识别与任务规划，4 个专职子 Agent 执行细分业务，之间通过自建的进程内消息总线通信——不用共享 State 是因为子 Agent 要常驻、要支持崩溃恢复、要按 DAG 分层派发。
>
> **工具层**用 MCP + FastMCP，3 个常驻服务（SQL / 通用模型 / 理财模型），上层是自研的 ToolRegistry，把本地函数和 MCP 服务的调用统一成同一份契约，协议差异收敛在适配器层。
>
> **模型层**是自研的 ChatModel harness，带能力契约和参数归一化——比如本地 Ollama 的 `num_predict` 和外部 `max_tokens` 语义不同，必须归一而不是直接透传。结构化输出走的是"提示词要求 JSON + 解析器容错"这条路。
>
> **记忆与检索层**是混合检索——笔记库用 BM25 + FAISS 双路召回加 **RRF 融合**，长期记忆用向量语义路 + jieba 关键词路加权，含口语时间表达强信号，配合 12 个月滚动裁剪控制规模。
>
> **可观测层**是自研 Tracer，一次请求一个 run_id，工具调用和 LLM 调用自动包 span 落库。
>
> **评测层**分三层——单元测试、检索评测（hit@k / recall@k / MRR）、回答正确性（五维规则 rubric + 数据库硬校验）。坚持能用规则判的不用 LLM 判。
>
> **工程层**是 asyncio 单进程多协程，任务状态机落独立库，支持崩溃后按 task_id 对账重投——设计理念和 Temporal 的 Durable Execution 一致，是它的单机简化版。
