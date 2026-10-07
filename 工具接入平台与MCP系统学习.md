# 工具接入平台与 MCP 系统学习手册

> 配套项目：billAgent（多 Agent 记账理财系统）
> 用途：系统性学习"工具接入平台设计"与"MCP 协议"；同时判断本项目如何规范化优化
> 学习路径：第 1-3 章打基础 → 第 4 章对照自己项目 → 第 5 章自测
> ⚠️ 更新日期：2026-09-14 —— **第 4 章已按 M1/M3/M9 落地后的实际态更正**（原文写于平台落地之前，多处已过期）；
> 契约级细节以 `设计_从零系列/09_模块契约手册.md` §4.5/§4.8 为准，变更留痕见文末修订记录。

## 目录
1. 工具为什么需要"接入平台"
2. 工具接入平台设计（注册/发现/执行引擎/协议适配）
3. MCP 协议系统学习
4. 结合 billAgent 项目剖析
5. 自测题与面试速记

---

## 第 1 章 工具为什么需要"接入平台"

### 1.1 一句话定位
LLM 只有"语言能力"，没有"行动能力"。**工具（Tool）是 Agent 连接真实世界的桥梁**：查库、调 API、算数、读文件。**工具接入平台**负责工具的完整生命周期管理——定义、注册、发现、校验、执行、审计、跨协议接入。

### 1.2 三个层次，别混为一谈（面试第一题就靠它）
| 层次 | 是什么 | 代表 | 解决什么问题 |
|---|---|---|---|
| 模型侧 | 模型如何"表达"调用意图 | Function Calling（OpenAI/Anthropic/千问） | 模型决定"调不调、调哪个、传什么参" |
| 平台侧 | 工具如何被"管理" | 工具注册中心、ToolSpec、执行引擎 | 可插拔、可管控、可观测 |
| 协议侧 | 工具如何被"接入" | MCP、gRPC、HTTP API | 生态可共享、可互操作 |

> 一句话记忆：**Function Calling 管"模型怎么选工具"，MCP 管"工具怎么被接入"，工具平台管"工具怎么被管理"。** 三者互补，不是二选一。

### 1.3 没有工具平台的痛点
- 工具散落各 agent 代码，无法复用
- 加一个工具要改多处调用代码（硬编码路由）
- 无统一参数校验 / 超时 / 重试 / 审计
- 无法接第三方生态工具
- LLM 自主选工具时没有 schema 可看

---

## 第 2 章 工具接入平台设计（核心知识）

### 2.1 设计目标
1. **统一抽象**：本地函数、MCP 工具、HTTP API 对外长一个样（ToolSpec）
2. **可插拔**：加工具 = 注册一行，不改业务代码
3. **可发现**：向 LLM 暴露工具清单（tools/list 或 function schema）
4. **可管控**：权限、校验、审计、限流、可观测
5. **可演进**：协议适配层可替换（MCP → 新协议不伤业务）

### 2.2 四要素架构（面试必画这张图）

```
                 ┌─────────────────────────────────────────┐
   LLM/Agent  ←→ │  ① 工具注册中心 Registry（统一 ToolSpec） │
                 │  ② 工具发现 Discovery（tools/list 暴露）  │
                 │  ③ 执行引擎 Runtime（校验/并发/超时/审计） │
                 │  ④ 协议适配层 Adapter（最后一公里）        │
                 └──────┬──────────────┬──────────────┬─────┘
                        │ MCP client   │ 本地函数调用   │ HTTP/gRPC
                 ┌──────▼──────┐ ┌─────▼─────┐   ┌────▼────┐
                 │ MCP Server  │ │ Python 函数│   │ 外部 API │
                 └─────────────┘ └───────────┘   └─────────┘
```

#### ① 工具注册中心 Registry
- 核心数据结构：**ToolSpec / ToolSchema**（业界统一认知）
  - `name`：唯一标识，LLM 靠它定位
  - `description`：语义描述，**决定 LLM 选不选它**——写得好坏直接影响调用准确率
  - `parameters`：JSON Schema（类型/必填/枚举/默认值/范围）
  - `executor`：实际执行函数（本地 callable 或远程封装）
  - `metadata`：权限标签、超时、限流、审计级别、版本
- 实现形态：装饰器注册（`@register_tool`）、配置驱动、插件动态加载（importlib / entry_points）

#### ② 工具发现 Discovery
- 向 LLM 暴露：`tools/list`（MCP）或 `tools[].function`（OpenAI 格式）
- **权限过滤即发现过滤**：只暴露当前 agent 有权限的工具
- 演进：硬编码 → 插件式动态加载（开源痛点：工具 PR 都要等核心维护者合并）

#### ③ 执行引擎 Runtime
- 参数校验：递归 JSON Schema 验证（返回可读错误而非裸异常）
- 结果归一：统一返回 str / 结构化 dict
- 错误处理：错误归一、重试（仅幂等工具）、超时熔断
- 并发控制：锁 / 信号量（防并发写）
- 审计 + 链路追踪：谁调的、调了什么、结果如何
- 上下文注入：session_id、user_id 自动注入

#### ④ 协议适配层 Adapter
- 同一个 ToolSpec，三种执行：本地函数直接跑 / MCP client 调 / HTTP 封装
- 关键原则：**业务代码只认 ToolSpec，不感知协议**——协议只是最后一公里

### 2.3 工具 Schema 设计细节（决定 LLM 调用质量）
- `description` 写清"何时用、何时不用、参数语义、返回值格式"
- JSON Schema 与 OpenAI Function Calling 完全兼容（子集 + strict 约束）
- 参数少而明确；枚举用 `enum`；数值给范围；必填/默认清晰
- 反面案例：`"sql": "string"` 空描述 → 模型乱传参数

### 2.4 安全设计（工具是最大的攻击面）
行业数据（2025-2026，面试引数据加分）：
- 2025 年 MCP 攻击事件激增 310%（提示注入 / 数据泄露 / 供应链劫持）
- 72% 公开 MCP server 暴露敏感功能，13% 接受不受信任输入
- Anthropic 官方 `mcp-server-git` 也被披露 3 个高危漏洞

**安全五层防线**：
1. 来源信任：供应链验证（第三方 server 默认不信任）
2. 最小权限：工具级/参数级 RBAC
3. 输入消毒：参数校验 + 内容清洗（SQL 拦截、类型识别）
4. 隔离：沙箱 / 子进程 / 容器
5. 审计监控：全链路日志 + 异常告警

### 2.5 业界实现对照（面试横向比较）
| 方案 | 形态 | 特点 |
|---|---|---|
| OpenAI Tools | 模型侧格式 + 你的代码执行 | 最简单，无平台概念 |
| LangChain Tools | 框架内工具抽象 + langchain-mcp-adapters | 生态大，抽象厚 |
| LlamaIndex | 工具定义 + JSON Schema 校验 | 偏数据侧 |
| AutoGen | 函数即工具 + 代码执行器 | 偏多 Agent |
| 自研平台 | ToolSpec + Registry + Adapter | 完全可控（你的演进目标） |

---

## 第 3 章 MCP 协议系统学习

### 3.1 背景与定位
- **MCP = Model Context Protocol**，Anthropic 2024 年 11 月提出并开源
- 比喻：**AI 世界的 USB-C**——一个标准接口，接所有设备（工具/数据源）
- 2025 年 OpenAI、微软、谷歌、亚马逊陆续宣布支持，已成 Agent 工具互操作**事实标准**
- 解决的本质问题：N 个模型 × M 个工具 = N×M 适配爆炸，MCP 统一成 1 套协议
- 定位区分：
  - MCP（接入协议）≠ Function Calling（模型能力）
  - MCP（开放标准）≠ Plugin（厂商专属生态）
  - MCP（工具协议）≠ A2A（Agent 间通信协议）

### 3.2 架构与角色
```
┌─ AI 应用 (Host) ─┐      ┌─ MCP Server ────────────┐
│  Claude/你的Agent │      │  capabilities 声明：      │
│   └── MCP Client ─┼──────┼─ tools / resources /     │
└───────────────────┘      │   prompts                │
      JSON-RPC 2.0         └──────────────────────────┘
```
- **Host**：AI 应用（Claude Desktop / 你的编排器）
- **Client**：连接器（Python/TS SDK），与 Server 一对一
- **Server**：轻量程序，暴露三种原语；经 stdio 或 HTTP 与 Client 通信
- 一个 Host 可连多个 Server（多工具源聚合）

### 3.3 通信模型（JSON-RPC 2.0）
**生命周期**：
1. `initialize`（Client→Server）：交换协议版本、capabilities、双方信息
2. `initialized`（Client→Server）：确认初始化完成
3. 会话内：`tools/list` / `tools/call` / `resources/list` / `resources/read` / `prompts/list` / `prompts/get`
4. 关闭 / 错误：JSON-RPC error 规范、notifications（logging、progress 服务端主动推）

### 3.4 三大原语（必背对比表）
| 原语 | 作用 | 类比 | 副作用 |
|---|---|---|---|
| **Tools** | 可执行函数，让模型做事 | 函数 / API | 有（写库、调接口） |
| **Resources** | 可读数据，让模型看内容 | 文件 / 数据库查询 | 无 |
| **Prompts** | 可复用模板，标准化交互 | 提示词模板 | 无 |

### 3.5 传输层选型（踩坑高频）
| 维度 | stdio | Streamable HTTP | SSE（旧） |
|---|---|---|---|
| 位置 | 本地子进程（stdin/stdout） | 远程 HTTP + SSE | 远程 HTTP |
| 场景 | 本地开发 / 本地工具 | 云端部署 / 跨网络 | 已被 Streamable HTTP 取代 |
| 认证 | 不需要 | OAuth 2.0 / Bearer | 同左 |
| 特点 | 启动即用、进程隔离 | 可服务多客户端 | 连接管理复杂 |

> 你的项目用 stdio + 短连接（每次调用起子进程），是本地开发的合理选择。

### 3.6 开发实战（SDK 层）
- 服务端：FastMCP（Python）/ MCP TS SDK，`@tool` 装饰器即可暴露
- 客户端：`ClientSession` + `stdio_client`（Python SDK）
- 你的项目就是完整实证（第 4 章详述）

### 3.7 生态与演进
- 规模：公开 MCP server 已从数千涨到 12000+（2026 年初）
- 框架支持：LangChain（langchain-mcp-adapters）、LlamaIndex、Spring AI、AgentScope 均原生/适配支持
- 厂商：OpenAI（兼容层）、微软 Copilot、谷歌、亚马逊 Bedrock 均已接入
- 演进方向：认证标准化（OAuth）、远程部署成熟、安全规范、与 A2A 互补

---

## 第 4 章 结合你的项目：billAgent 的 MCP 落地剖析

### 4.1 现状盘点（先看实证，全都能指到代码）
| 组件 | 代码位置 | 职责 |
|---|---|---|
| sql_bill Server | `mcpGateway/sql_server/server_bill.py` + `sql_common/` | `exec_sql`：唯一 SQL 入口，DDL 拦截 + SQL 类型识别 + 读写权限 + `db_lock` 并发控制 + 审计 |
| llm_base Server | `mcpGateway/server_llm_base.py` | `llm_chat`：通用基座模型 |
| llm_finance Server | `mcpGateway/server_llm_finance.py` | `finance_chat`：理财专属（外部 API 优先 + 本地 QLoRA 回退） |
| 统一客户端 | `mcpGateway/client.py` | `SERVER_CMD_MAP`（server 启动命令）+ **3 个业务兼容层入口**（`call_bill_sql` / `call_llm_base` / `call_llm_finance`，内部转调执行引擎）；传输：**streamable-http 常驻主路径**（M3，冷启动只付一次）+ `BILLAGENT_MCP_STDIO=1` 回滚 stdio 短连接 |
| RBAC | `mcpGateway/rbac_config.py` | `AGENT_PERMISSION_MAP`：5 个 agent 最小权限 |
| 审计 | `mcpGateway/audit_recorder.py` | 双端审计（客户端 + 服务端） |
| **工具平台（M1 新增）** | `mcpGateway/{tool_model,registry,executor,adapters}.py` | ToolSpec 声明式注册 / 权限过滤发现 / 9 步执行流水线（鉴权·校验·审计·超时·重试·埋点）/ 协议适配（MCP·local） |

> ⚠️ **本章 2026-09-14 已按 M1/M9 落地后的实际态更正**（原文写于 M1 之前，多处已过期）。
> 契约级细节（工具清单、schema、权限矩阵）的唯一权威是 `设计_从零系列/09_模块契约手册.md` §4.5/§4.8，本文只讲"怎么讲、怎么判断"。

### 4.2 对照四要素自查表

| 平台要素 | 你项目现状（2026-09-14 实测） | 评估 |
|---|---|---|
| **注册中心** | ✅ **有**：`ToolRegistry` + `ToolSpec` 声明式注册，`registry._build_platform` 启动即装配 **10 个工具**（4 协议级：`sql.read`/`sql.write`/`llm.chat`/`llm.finance`；6 业务：`config.get_latest`/`bill.sum_by_month`/`bill.sum_by_category`/`city_price.get_avg`/`habit.list`/`habit.upsert`），注册期 fail-fast 校验（名/描述/schema 必填） | **已从"缺"变为"有"**（M1 建平台、M9 收口 D15） |
| **发现机制** | ⚠️ **地基已就绪，但生产零接线**：`registry.to_openai_schema(agent_id, tags=...)` 能按权限+tags 导出 function-calling schema，但**唯一引用在测试**（`tests/unit/test_tool_platform.py`）；`to_mcp_schema()` 无人调用；`ToolSpec.description` 从未进入任何提示词 | **平台当"执行治理层"用上了，当"能力暴露层"没用上**——这是"LLM 不自主选工具"的真正落点（不是缺平台） |
| **执行引擎** | ✅ **有且高频在用**：`ExecutionEngine` 9 步流水线——解析→鉴权→参数校验(JSON Schema)→前置审计→超时→重试→结果→后置审计→埋点；生产侧所有 DB/LLM 访问必经（orchestrator 5 处事实工具、bill_agent 的 habit.upsert、`client.py` 3 个兼容层入口内部转调） | 已覆盖最贵的部分（治理与安全），且是**每日运行**的核心资产 |
| **协议适配** | ✅ **有（双协议）**：`MCPAdapter`（streamable-http 常驻主路径 + stdio 回滚，M3）与 `LocalAdapter`（本地函数，默认 `to_thread`；async 函数直接 await 以保住协程级锁） | **已从"单向 stdio"变为"双协议统一抽象"** |
| **安全** | ✅✅ 强：RBAC 双层（客户端前置 + 服务端二次）+ DDL 危险拦截 + 双端审计 + 写操作禁用自动重试（防重复记账） | 超多数生产项目 |

> **一句话结论（面试可直接用）**：**四要素里"注册/执行/协议适配"已落地并在跑，"发现"是唯一空着的一格**——
> 空的方式很具体：导出函数写好了、schema 能生成、`tags` 过滤都支持，**只差把它接到模型请求的 `tools` 字段上**（`设计/12` §4.3 的 M15 P2 就是接这根线）。

### 4.3 你的决策为什么是对的（论证链，面试逐条讲）
1. **资源访问 MCP 化**：`exec_sql` / `llm_chat` / `finance_chat` 有副作用、需隔离、需权限审计 → 命中 MCP 化判定（第 2.4 节五层防线）
2. **业务计算本地化**：`_fix_category` / `_fix_date` / 统计 / 物价计算是纯函数、高频、强耦合 state → 命中本地 SDK 判定（MCP 有子进程 + 序列化开销）
3. **短连接**：规避 cancel scope 跨 task 崩溃（踩坑实证）；子进程隔离 + 每次独立握手
4. **双层 RBAC**：客户端前置校验 + 服务端二次校验 → 纵深防御
5. **暂不让 LLM 自主选工具**：在**派发与落库**环节，确定性收益 > 自主收益（记账误判成本极高）
   —— **但此结论有边界**：在"**目标消解**"环节（"改哪一条账"）不成立，那里模糊、需观察、需多轮，**该用 agent**
   （判据：**步骤不确定 → agent；步骤确定 → workflow**，详见 `设计/12` §2.3）

### 4.4 规范化优化路径（三阶段，按 ROI 排序）

**阶段 0：不动（历史状态，已越过）**——原文写"MCP 化 + 本地纯函数双轨已够用"，该阶段在 M1 之前成立。

**阶段 1：加 `ToolRegistry` 统一抽象 —— ✅ 已完成（M1 建平台 / M9 收口）**。实际落地形态与下面的草图有差异：

| 草图中的写法 | 实际落地 |
|---|---|
| `ToolSpec(name, description, params_schema, executor, via_mcp)` | `ToolSpec` 17 字段：`protocol`（mcp/local）+ `binding`（server_key/tool 或 func）+ `required_perm` + `side_effect` + `auditable` + `idempotent` + `timeout` + `max_retry` + `tags` 等 |
| `invoke()` 里手写 `if via_mcp` 分支 | 分支收敛进 **执行引擎 + 适配器**（业务代码只认 ToolSpec，协议在最后一公里） |
| 无鉴权/审计 | 引擎内建：RBAC 鉴权 + `agent_id` 由适配器注入（不进 schema，防伪造提权）+ 双段审计 + span 埋点 |

收益已兑现：加工具 = 注册一行；所有调用统一走鉴权/校验/审计/超时；`to_openai_schema` 为自主选工具留好接口。

**阶段 2（接下来）**：把发现层接上——`to_openai_schema(agent_id, tags=...)` → `llm.chat` 的 `tools` 参数，在 **bill_agent 内部**做自主目标消解循环（设计见 `设计/12` §4，施工序见 `设计/11` §3 M15 的 P2/P3）。
接入生态（`langgraph.prebuilt` / `langchain-mcp-adapters`）仍不急。

### 4.5 当初值得做阶段 1 的判据（现已满足，留作方法记）
- 要做 LLM 自主 function calling（让模型选工具）→ **必须做**（需要 list_tools + schema）——**正在做**（M15）
- 工具数量接近/超过 10 且还在增长 → 建议做 —— **已到临界**（现 10 个工具；M15 落地账单改删后将增至 15 个）
- 第三方 agent / 系统也要复用你的工具 → 必须做（MCP 化暴露）
- 仅当前形态 → 可缓，但保留设计文档

---

## 第 5 章 自测题与面试速记

### 5.1 自测 20 题（先自己答，再看文档）
1. MCP 全称？谁提出？何时？——Model Context Protocol，Anthropic，2024.11
2. MCP 解决什么本质问题？——工具接入标准化，避免 N×M 适配爆炸
3. Host / Client / Server 三者关系？
4. 三大原语及区别？各自有无副作用？
5. 两种主流传输？各自适用场景？
6. 一次 tools/call 的完整消息流？
7. capabilities 是什么？
8. MCP 与 Function Calling 的本质区别？
9. MCP 与 A2A 的区别？
10. 你的项目哪 3 个 MCP server？各暴露什么？
11. 你的项目为什么短连接？踩了什么坑？
12. 你的 RBAC 怎么做到双层校验？
13. exec_sql 有哪些安全措施？为什么 SQL 要 MCP 化？
14. _fix_category 为什么不该 MCP 化？
15. 工具平台四要素是什么？——**注册 / 发现 / 执行引擎 / 协议适配**（安全是横切要求，不单列为一要素）
16. ToolSpec 有哪些字段？description 为什么重要？——本项目为 **17 字段**，关键几项：`protocol` + `binding`（协议在最后一公里）、`required_perm`（权限即"可见性 + 可调用性"）、`side_effect`（**决定是否自动重试**：写操作永不重试）、`auditable`、`timeout`、`max_retry`、`tags`（能力过滤）；`description` 是**给 LLM 看的**、决定模型选不选它——⚠️ 本项目 description 至今**未进入任何提示词**，这正是 M15 要接的线
17. 发现机制何时必需？——LLM 自主选工具时
18. 执行引擎要处理哪些问题？——校验/超时/重试/审计/并发（本项目为 9 步流水线）
19. 你的项目四要素分别是什么状态？——**注册 ✅ / 执行 ✅（高频在用，所有 DB·LLM 访问必经） / 协议适配 ✅（双协议：MCP + local） / 发现 ⚠️ 地基已就绪但生产零接线**；安全 ✅✅（RBAC 双层 + DDL 拦截 + 双端审计 + 写操作禁重试）
20. 阶段 1 的 ToolRegistry 解决什么问题？——**✅ 已完成（M1 建平台 / M9 收口）**：工具统一入口 → 统一鉴权/校验/审计/超时/埋点；并产出 `to_openai_schema()` 作为"LLM 自主选工具"的地基（当前闲置，M15 P2 接线）

### 5.2 面试话术模板

**"讲一下你的 MCP 实践"**：
> "我的项目把资源访问类能力（SQL 执行、模型推理）做成了 3 个 MCP server（sql_bill / llm_base / llm_finance），统一走短连接网关，双层 RBAC + 双端审计。SQL server 有 DDL 拦截、读写权限分离、并发锁。业务计算类（正则修正、统计）保留为本地纯函数——MCP 有子进程和序列化开销，纯计算高频调用不划算。踩过一个坑：常驻长连接在 LLM 调用超时触发 cancel scope 后，错误会跨 task 传播崩溃整个事件循环，改成短连接后错误完全隔离。"

**"你项目有没有工具注册中心？"**：
> "有，而且是工具的**唯一真相源**。`ToolRegistry` + `ToolSpec` 声明式注册 10 个工具（4 个协议级：`sql.read`/`sql.write`/`llm.chat`/`llm.finance` + 6 个业务工具：配置读取、账单聚合、城市均价、月度习惯读写），注册期 fail-fast 校验，调用侧统一走执行引擎的 9 步流水线（解析→鉴权→JSON Schema 校验→前置审计→超时→重试→后置审计→埋点），协议适配放最后一公里——同一个 ToolSpec 可以是 MCP 工具也可以是本地函数，业务代码不感知协议。
> 但我得诚实说清边界：**我只用了一半**。执行/治理侧天天在跑；**发现侧（把工具清单暴露给模型）是空着的**——`to_openai_schema(agent_id, tags=…)` 能按权限导出 function-calling schema，但生产代码零调用，所以模型不知道有哪些工具，工具调用是代码写死的。这不是"缺平台"，是**最后一根线没接**。
> 下一步就是接这根线：在 bill_agent 内做工具自主调用，并用 `tags` 实现'**工具可见面即操作锁**'——edit 任务下模型根本看不到 delete 工具，操作类型不靠提示词约束。"

**"什么工具该 MCP 化？"**：
> "四个判定：跨进程/异构技术栈、有副作用需统一管控、需要沙箱隔离、需要被多系统复用——命中即 MCP 化。纯计算、高频调用、强耦合内存状态的就留本地。我的项目就是按这个划分的：exec_sql 和模型推理走 MCP，正则修正和统计留本地。"

### 5.3 高频考点浓缩
- MCP 是**接入协议**，不是模型能力（别和 function calling 混淆）
- 三大原语：**Tools（有副作用）/ Resources（只读）/ Prompts（模板）**
- 传输：**stdio（本地）/ Streamable HTTP（远程）**，SSE 是过渡方案
- 工具平台四要素：**注册 / 发现 / 执行引擎 / 协议适配**
- 安全五层：**来源信任 / 最小权限 / 输入消毒 / 隔离 / 审计**
- 你的差异化实证：**短连接踩坑 + 双层 RBAC + DDL 拦截 + 双端审计**
- 你的平台现状：**注册/执行/协议适配在用，发现未接线**（`to_openai_schema` 生产零调用）——这才是"没有自主选工具"的根因（不是缺平台）

---

## 修订记录

| 日期 | 变更 |
|---|---|
| 2026-09-14 | **第 4 章按 M1/M3/M9 落地后的实际态更正**（原文写于 M1 之前，多处自曝）：§4.1 补"工具平台"组件行、修正 client 传输描述（短连接 → streamable-http 主路径 + stdio 回滚、兼容层转调引擎）；§4.2 四要素自查表重写（注册中心由"△ 弱/缺 ToolSpec"改为"✅ 有"、协议适配由"△ 单向 stdio"改为"✅ 双协议"、发现机制改为"⚠️ 地基就绪但生产零接线"、执行引擎补 9 步流水线并标注高频在用）；§4.3 第 5 条补**结论边界**（派发/落库确定性优先成立，"目标消解"环节该用 agent，判据见 `设计/12` §2.3）；§4.4 阶段 0/1 标注已完成 + 补"草图 vs 实际落地"差异表 + 阶段 2 指向 M15 P2/P3；§4.5 改作"当初判据（已满足）"；§5.1 第 15/16/19/20 题补答案口径；§5.2"有没有工具注册中心"话术重写为实际态（含"只用了一半"的诚实口径）；§5.3 补"平台现状"一行 |



