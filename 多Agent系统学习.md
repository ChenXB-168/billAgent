# 多 Agent 系统学习（结合 billAgent 项目）

> 定位：系统性学习多 Agent 知识 + 结合本项目理解"为什么我的项目是多 Agent，但又不是标准多 Agent"。适合腾讯 PCG Agent 应用开发岗项目深挖。

---

## 第 1 章 为什么需要多 Agent？——单 Agent 的瓶颈

### 1.1 单 Agent 的四大瓶颈（面试先答这个）

| 瓶颈 | 表现 | 拆分的解法 |
|---|---|---|
| 上下文污染 | 一个 prompt 里塞太多职责，指令互相冲突，模型"降智" | 每个 agent 只装一个角色的 prompt，上下文干净 |
| 工具过载 | 一个 agent 挂几十个工具，工具选择冲突、误调用 | 按角色分发工具子集，各管各的 |
| 无法并行 | 串行执行，N 个步骤延迟线性累加 | 独立子任务并行执行，fan-out/gather |
| 权限不分 | 一个 agent 什么都能干，无法做最小权限隔离 | 每个 agent 独立授权，边界清晰 |

> 一句话：**多 Agent 的价值不是"看起来高级"，而是解决"上下文污染、工具冲突、串行慢、权限不分"四个工程问题。**

### 1.2 多 Agent 的代价（必须知道，面试问"为什么不全用多 Agent"）

- **延迟**：多链路 + 协调开销，端到端延迟是单 Agent 的 **2~10 倍**（一次消息传递就要一次 LLM 调用）
- **成本**：Token 翻倍、多次调用
- **复杂度**：通信协议、状态同步、错误处理、冲突消解——开发成本是指数级上升
- **数据**：70% 的项目单 Agent 就够；研究显示多 Agent 平均只提升 2.1% 准确率

### 1.3 什么时候该拆（3 个信号，来自实践）

1. **子任务会产生大量无关信息**：如 codex 排查 bug，让子 Agent 读完整堆栈，只回传几十字根因——避免主 Agent context 被稀释
2. **可拆为互不依赖的子任务**：如调研拆成政策/市场/竞品/技术四路并行
3. **需要不同的权限/工具/记忆边界**：如记账 agent 只能写库，理财 agent 才能看理财配置

**不该拆**：任务简单、串行依赖强、纯计算——拆了只是多付一次 LLM 调用和消息延迟。

### 1.4 结合 billAgent：你为什么拆了 4 个？

你的 4 个 worker 命中第 1 和第 3 条信号：

| Worker | 职责 | 拆分的正当理由 |
|---|---|---|
| bill_agent | 记账（增删改查） | 有 DB 写权限，需要严格校验 |
| stat_agent | 统计查询 | 只读，专注 SQL 生成与数字校验 |
| price_agent | 物价对标/保费试算 | 纯计算 + 外部价格数据 |
| finance_agent | 理财分析 | 依赖前三者结果，独立角色 prompt |

四个角色 prompt 各自纯净（无指令冲突）、权限可分（RBAC 最小权限）。**这个拆分是对的。** 注意依赖事实：stat/price **依赖 bill**（`TASK_CONTEXT_DEPS`），bill 完成后二者才具备并行条件；且当前调度实现为**串行单派**（`dispatch_node` 一次只派 `ready[0]`）——并行收益尚未兑现，已列入 To-Be 并行派发（`03_逻辑架构.md` §9.4 ③，取舍依据见 §8.2 T2/T3）。

---

## 第 2 章 多 Agent 核心概念（先建立坐标系）

### 2.1 五个基本要素

| 要素 | 含义 | 面试问法 |
|---|---|---|
| **角色 Role** | 每个 agent 的 system prompt 与职责边界 | "agent 怎么分工？prompt 怎么设计？" |
| **自主性 Autonomy** | 决策权在 LLM 还是代码（谱系：全自主↔确定性） | "你的 worker 是自主决策还是代码驱动？" |
| **通信 Communication** | 消息传递 / 共享状态 / 黑板 | "agent 之间怎么传消息？" |
| **记忆 Memory** | 共享记忆 vs 独立记忆，上下文隔离 | "上下文怎么管理？会不会互相污染？" |
| **协调 Coordination** | 谁来编排、冲突怎么解、死循环怎么防 | "谁来分配任务？冲突怎么办？" |

### 2.2 自主性谱系（关键认知：多 Agent ≠ 全 LLM）

```
全自主                                   全确定性
  ↓                                         ↓
自由对话式    半自主(LLM规划+代码执行)    确定性流水线(代码控制)
AutoGen/Swarm   LangGraph编排器+Worker     DAG/FSM
   你的 orchestrator 在这里 ↕  你的 worker 在这里
```

- **标准多 Agent 范式**（AutoGen / OpenAI Swarm / MetaGPT）：agent 之间用自然语言对话，互相派活、辩论、协商——**决策权在 LLM**
- **工程化多 Agent 范式**（你的项目）：LLM 只负责"规划"，执行、路由、汇总都是确定性代码——**决策权在代码**

> 面试金句：*"多 Agent 的核心不是'有几个 agent'，而是'决策权怎么分配'——LLM 和代码的边界划在哪，决定了它是自主协作还是确定性流水线。"*

### 2.3 记忆模型（多 Agent 必考）

| 模式 | 说明 | 适用 |
|---|---|---|
| 独立记忆 | 每个 agent 只看到自己的上下文，结果经编排器汇总 | 隔离好，防污染 |
| 共享记忆/黑板 | 所有 agent 读写同一份状态 | 需要全局事实一致 |
| 混合 | 全局共享 + 角色私有 | 生产主流 |

**你的项目**：混合模式。每个 worker 有独立 `state`（TypedDict，只装自己需要的字段，见 `bootstrap.py` 的手工 state 映射——这是**上下文隔离**）；跨 agent 的结果通过 `task_data` 显式传递（如 finance 依赖 bill/stat/price 结果），不靠共享全局内存——**显式数据依赖，无隐式污染**。

### 2.4 冲突与失控（高级考点）

- **冲突**：两个 agent 改同一份数据 → 用依赖图（你的 `TASK_CONTEXT_DEPS`）+ 串行化（`wait_result` 按 task 等待）
- **死循环**：编排器回环边无限转 → 轮次熔断（你的 `dispatch_round > 20`）
- **超时**：worker 卡死 → 结果超时熔断（你的 `wait_result` 240s 超时）
- **并发**：DB 并发写 → 全局 `db_lock`

> 这三个是面试官最爱深挖的"工程细节"，你项目全有实证。

---

## 第 3 章 编排模式全景（多 Agent 的"组织架构"）

### 3.1 按拓扑分类（必背图）

```
① 中心化 Hub-and-Spoke          ② 分层 Hierarchical
   [Supervisor]                    [顶层编排器]
    /  |  \                           /      \
 [W1][W2][W3]                 [中层Agent A]  [中层Agent B]
  只与中心通信                        /  \        |
                              [W][W]  [W]   [W]
③ 对等 Peer-to-Peer           ④ 网状 Network
  [A]↔[B] 互相直接通信          agent 全连接自由对话
```

### 3.2 Anthropic 官方 5 种协调模式（2026 实践文，直接引）

| 模式 | 一句话 | 你的项目对照 |
|---|---|---|
| Sequential 流水线 | A→B→C 串行传递 | bill→stat→finance 的依赖链 |
| Parallel 并行 | 扇出 fan-out 后聚合 gather | stat/price 在 bill 完成后的并行（依赖图就绪；**当前串行单派，To-Be 已规划并行派发**） |
| Supervisor 主管 | 一个 LLM agent 负责路由和任务分配 | 你的 plan_node（但它是单次 LLM，不是循环 supervisor） |
| Hierarchical 分层 | 顶层拆解，下层再拆解 | 两级架构的雏形 |
| Network 网状 | 自由对话协商 | ✗ 你没有 |

### 3.3 决策权分类（工程视角，比背模式更高级）

- **代码决策（Workflow）**：路由、依赖、重试全部写死——可预测、可测试
- **LLM 决策（Agentic）**：模型自己决定下一步——灵活、不可控
- **混合**：代码定框架（图结构/依赖/熔断），LLM 定内容（任务规划/文本生成）——**你的项目就是这个，这是生产最佳实践**

> 面试高分局：*"我用代码锁死'框架正确性'（拓扑、依赖、权限、熔断），用 LLM 负责'内容灵活性'（任务规划、文案、SQL 意图解析）。该确定性的一定确定性，该智能的才交给模型。"*

---

## 第 4 章 通信机制详解（多 Agent 的"神经系统"）

### 4.1 三种主流通信模式（面试必问，含代码）

| 模式 | 原理 | 延迟 | 解耦 | 典型框架 |
|---|---|---|---|---|
| 消息传递（点对点/队列） | 发件人→收件人，消息入队 | 低 | 中 | 你的 A2AMessageBus、Celery |
| 发布订阅 Pub/Sub | 发件人广播主题，订阅者自取 | 低 | 高 | Redis pub/sub、Kafka |
| 黑板/共享内存 | 所有 agent 读写同一份状态 | 高（要读上下文） | 低 | LangGraph State、MetaGPT |

**你的项目：中心化 + 消息传递（队列模式）**——`mcpGateway/a2a_queue.py`：

```python
class A2AMessageBus:
    _task_queues:  Dict[str, asyncio.Queue]  # 按 agent 分队列（点对点投递）
    _result_events: Dict[str, asyncio.Event] # task_id → 结果就绪事件
    _result_store:  Dict[str, Any]           # task_id → 结果缓存

    def send_task(target_agent, session_id, task_id, task_data): ...  # 投递
    async def recv_task_blocking(agent_id): ...   # worker 常驻消费
    def send_result(target_agent, task_id, result):  # 先存 store 再 set Event（防竞态）
    async def wait_result(task_id, timeout=240): ...  # 编排器阻塞等待 + 超时熔断
```

这是**教科书级的进程内消息总线**：
- 按 agent 分队列 → 点对点解耦，worker 只关心自己的队列
- `Event + store` 双结构 → 先写结果再唤醒，杜绝"唤醒后拿不到数据"的竞态
- 240s 超时熔断 → 防 worker 卡死拖死编排器
- 会话不匹配的消息放回队尾 → 支持多 session 并发隔离

### 4.2 面试加分：黑板模式手写（展示"懂原理"）

```python
# 黑板模式：agent 不直接通信，共享一份状态
class Blackboard:
    def __init__(self):
        self.data = {}
    def read(self, key): return self.data.get(key)
    def write(self, key, value): self.data[key] = value

bb = Blackboard()
# Agent A 写结果，Agent B 直接读——谁写谁读都解耦，只认 key
```

> 话术：*"消息总线适合'明确谁干活的中心化场景'（我的项目）；黑板适合'职责边界模糊、多 agent 看同一事实'的场景（如编排器 + 评审团）。我项目选队列因为拓扑是中心化的，依赖是显式的。"*

### 4.3 通信内容设计（工程细节）

- **消息结构**：`{session_id, task_id, task_data}`——用 task_id 关联结果，session_id 隔离并发会话（你的做法，正确）
- **传递什么**：只传"下一个 agent 需要的最小上下文"，不传全量（你的 `bootstrap.py` 手工 state 映射就是在做这件事）
- **防丢消息**：`send_result` 先写 store 再 set Event；`clear_all` 启动时清残留（你都有）

---

## 第 5 章 A2A 协议专题（Google Agent2Agent）

### 5.1 A2A 是什么

- **全称**：Agent2Agent（Agent-to-Agent）
- **发布**：Google 2025 年 4 月 9 日提出，后捐给 **Linux 基金会**，目标是跨厂商、跨框架的 Agent 间协作开放标准
- **解决的本质问题**：**Agent 与 Agent 之间如何"发现彼此、交换任务、协作完成"**
- 类比：MCP 是"AI 世界的 USB-C"（接设备/工具），**A2A 是"AI 世界的邮局"**（人与人/Agent 与 Agent 通信）

### 5.2 核心概念（必背 4 个）

| 概念 | 含义 |
|---|---|
| **AgentCard** | Agent 的"名片"（JSON-LD 格式）：名字、能力描述、端点 URL、认证方式——用于**发现**（像 DNS） |
| **Task** | Agent 间协作的最小工作单元，有生命周期：submitted → working → input-required → completed / failed / canceled |
| **Message / Part** | Task 内传递的内容，支持文本、图片、音视频多模态 |
| **能力协商** | 发起方读对方 AgentCard，判断对方能否胜任再派任务 |

### 5.3 A2A 与 MCP 的区别（2026 年最高频对比题）

| 维度 | MCP | A2A |
|---|---|---|
| 连接对象 | **Agent ↔ 工具/数据** | **Agent ↔ Agent** |
| 解决 | 工具怎么被接入和调用 | Agent 怎么协作 |
| 角色 | 客户端-服务器 | 对等（发起方/响应方） |
| 协议 | JSON-RPC 2.0 | JSON-RPC 2.0（借） |
| 生态 | 12000+ servers | 仍在建设，大厂陆续接入 |
| 类比 | 手（干活） | 人脉（找人干活） |

> 一句话：**MCP 让 Agent"有手可用"，A2A 让 Agent"有同事可找"。两者互补，不是竞争。**

### 5.4 你的项目里" A2AMessageBus"是什么？（重要澄清）

你的 `mcpGateway/a2a_queue.py` 命名为 `A2AMessageBus`，**但它是"借用 A2A 这个名字的进程内消息总线"，不是标准 A2A 协议**：

| 标准 A2A 要素 | 你的 A2AMessageBus |
|---|---|
| AgentCard（名片+发现） | ✗ 没有，靠 `TASK_AGENT_MAP` 硬编码路由 |
| Task 生命周期协议 | △ 只有 task_id + Event，无状态机 |
| 跨进程/跨厂商 JSON-RPC | ✗ 纯进程内 asyncio.Queue |
| 能力协商 | ✗ 无 |

**但这不是缺陷**——你的场景是单进程、拓扑固定、agent 固定，不需要"发现和协商"这层成本。准确的说法是：**"你实现了 A2A 思想的进程内简化版（消息总线 + 任务关联），把'A2A 协议'里最有价值的部分（异步任务、结果关联、超时熔断）落地了。** 面试这样讲，既诚实又显深度。

---

## 第 6 章 结合 billAgent 剖析：你的项目是什么多 Agent？

### 6.1 你的真实架构（一句话版）

> **一个 LLM 规划的编排器（orchestrator）通过 A2A 消息总线，把任务派给 4 个 worker 子 Agent（LLM 解析意图 + 确定性代码执行），按依赖图串行/并行执行，结果回传汇总。**

```
用户输入
   │
   ▼
┌─ orchestrator_graph（4 节点 StateGraph）──────────────┐
│ plan_node ──▶ dispatch_node ──▶ wait_result_node ◀──┐ │
│                 ▲                    │               │ │
│                 └──── 回环边(多轮派发) ┘               │ │
└─────────────────┬──────────────────────────────────┘ │
                  │ a2a_bus.send_task / wait_result     │
     ┌────────────┼────────────┬────────────┐           │
     ▼            ▼            ▼            ▼           │
 bill_agent  stat_agent   price_agent  finance_agent    │
 (execute→reply) (parse→exec→reply) (2节点)   (2节点)     │
     │            │            │            │           │
     └────────────┴─────┬──────┴────────────┘           │
                  collect_node ◀───────────────────────┘
```

关键实证（面试全可指代码）：
- 编排器 4 节点 + 回环边：`agents/orchestrator/graph.py`（`wait_result→dispatch` 回环 = 多轮派发）
- 任务依赖图：`nodes.py` 的 `TASK_CONTEXT_DEPS`（stat/price 依赖 bill，finance 依赖三者）
- 消息总线：`mcpGateway/a2a_queue.py`
- worker 常驻消费：`startup/bootstrap.py`（4 个 `asyncio.create_task`）
- 熔断三件套：`_has_cycle` 防环 + `dispatch_round > 20` 轮次熔断 + `wait_result` 240s 超时

### 6.2 为什么"不是标准多 Agent"？（5 条，直接背）

| 标准多 Agent 特征 | 你的项目 | 说明 |
|---|---|---|
| ① worker 有 LLM 自主决策循环 | ✗ 没有 | bill_agent 只有 `execute→reply` 两个节点，几乎全是确定性代码，LLM 只做局部解析/生成 |
| ② agent 之间直接对话 | ✗ 没有 | worker 之间零通信，全靠 orchestrator 中转结果（TASK_CONTEXT_DEPS 是数据依赖不是对话） |
| ③ LLM 动态选人/路由 | ✗ 没有 | `TASK_AGENT_MAP` 是写死的映射 + `validate_task_list` 白名单校验，不是模型自主挑 agent |
| ④ 动态拓扑/动态创建 agent | ✗ 没有 | 5 个 graph 编译期固定，worker 数固定 |
| ⑤ agent 共享/协商全局目标 | △ 半有 | 目标由 plan_node 单次 LLM 拆分，worker 不参与协商 |

**结论**：你的项目是**"Orchestrator-Workers 模式 + 消息总线"的工程化落地**，处在多 Agent 谱系中偏"确定性/工程化"的一端。它是**真·多 Agent**（5 个独立 graph + 独立角色 + 消息通信），但不是**"自主协商式多 Agent"**（AutoGen/Swarm 那种）。

### 6.3 但为什么这么设计是对的？（面试论证链，逐条讲）

1. **记账场景要确定性**：账目增删改、金额统计不允许模型自由发挥 → worker 用代码锁死执行逻辑，LLM 只做"意图解析/表达"
2. **防幻觉闸门需要代码边界**：数字白名单、`_fix_category` 修正等确定性兜底，必须是本地代码而非 agent 对话
3. **权限最小化**：bill 有写库权限，如果走自由对话，模型可能绕过校验 → 中心化调度 + 显式 task_data 传递，每个 worker 只拿最小数据
4. **可观测可调试**：确定性拓扑 + 显式消息，链路可复现；全自主对话式多 Agent 出问题极难排查
5. **成本可控**：worker 不来回对话，每任务只调 1-2 次 LLM，无协商开销

> 面试金句：*"标准多 Agent（对话协商式）解决的是'开放性问题'，我的场景是'确定性业务流水线'——所以我把自主性收敛到编排器的规划环节，worker 做成工具化的确定性 agent。这是场景决定的架构选择，不是做不到，是不该做。"*

### 6.4 四要素自查表（与"工具接入平台"呼应）

| 要素 | 你的状态 | 评估 |
|---|---|---|
| 角色分工 | ✓✓ 4 角色职责清晰，prompt 纯净 | 好 |
| 通信机制 | ✓✓ 进程内消息总线 + Event 防竞态 | 好 |
| 记忆隔离 | ✓ 独立 state + 显式 task_data | 好 |
| 协调/编排 | ✓✓ 依赖图 + 回环 + 三件套熔断 | 强 |
| 自主性 | △ 只有 plan_node 用 LLM | 场景决定的合理收敛 |
| 标准 A2A 化 | ✗ 进程内简化版 | 单进程场景不需要 |

---

## 第 7 章 面试考察点清单（多 Agent 一般问什么）

### 7.1 八大必考块

1. **为什么用多 Agent**（单 Agent 瓶颈 + 代价 + 拆分信号）
2. **编排模式**（Supervisor / Orchestrator-Workers / Sequential / Parallel / 分层 / Network 画图对比）
3. **通信机制**（消息传递 / 黑板 / 发布订阅 + 各自适用场景）
4. **记忆与上下文**（独立 vs 共享，如何防污染）
5. **冲突与失控**（死循环熔断 / 超时 / 依赖环 / 并发锁）
6. **安全与权限**（最小权限、角色隔离、提示注入）
7. **成本与延迟**（为什么不全自主、多 Agent 的 2-10 倍延迟）
8. **协议与生态**（A2A vs MCP、LangGraph Send/Command/子图、AutoGen/Swarm/CrewAI/MetaGPT 对比）

### 7.2 高频 Q&A（8 题必背）

**Q1：你的项目为什么是多个 agent 而不是一个？**
4 个角色 prompt 纯净、权限可分（bill 可写、stat/price 只读）、上下文隔离防污染；依赖图上 stat/price 在 bill 完成后具备并行条件（**当前串行单派，To-Be 规划并行派发**）。

**Q2：你这是什么编排模式？**
Orchestrator-Workers（中心化）：编排器负责规划+派发+汇总，worker 只干自己的活。加了一张依赖图（TASK_CONTEXT_DEPS）：stat/price 依赖 bill，finance 依赖前三者；**当前串行单派，To-Be 规划并行派发**。

**Q3：worker 之间怎么通信？为什么不直接对话？**
不走对话，走消息总线（A2AMessageBus）：orchestrator 派任务、worker 回结果，task_id 关联、session_id 隔离。直接对话引入不可控性和成本，确定性流水线用显式数据依赖更可靠。

**Q4：你的 agent 是真的自主吗？**
只有编排器用 LLM 规划，worker 是确定性执行。这是刻意的：记账要确定性，防幻觉闸门必须是代码。自主性在谱系上收敛到"LLM 规划 + 代码执行"的半自主点。

**Q5：怎么防止死循环？**
三件套：DFS 防环（_has_cycle）+ 轮次熔断（dispatch_round > 20）+ 结果超时（wait_result 240s）。

**Q6：上下文怎么隔离的？**
每个 worker 独立 TypedDict state，bootstrap 手工映射只塞必要字段；跨 agent 只传 task_data 最小集。

**Q7：你的 A2A 是标准 A2A 协议吗？**
不是。是借用名字的进程内消息总线（asyncio.Queue + Event），标准 A2A 是跨进程跨厂商协议（AgentCard 发现 + Task 生命周期 + JSON-RPC）。单进程场景不需要标准 A2A 的发现和协商层。

**Q8：如果任务更复杂，怎么演进？**
（见第 8 章）三个方向：worker 间 peer 通信 → LLM 动态路由 → 标准 A2A 化。

### 7.3 高频考点浓缩（30 秒速记）

- 多 Agent 四大动机：**上下文污染 / 工具过载 / 无法并行 / 权限不分**
- 五大协调模式：**Sequential / Parallel / Supervisor / Hierarchical / Network**
- 三大通信：**消息队列（中心化）/ 黑板（共享）/ Pub-Sub（解耦）**
- 核心矛盾：**自主性 vs 确定性**——生产实践是"代码锁框架，LLM 定内容"
- 三大熔断：**防环 / 轮次上限 / 超时**
- 你的定位：**Orchestrator-Workers + 消息总线的工程化多 Agent**，不是自主协商式

---

## 第 8 章 规范化演进路径（想展示视野就讲这个）

### 8.1 三阶段演进（按 ROI 排序）

**阶段 1（低成本，推荐）**：worker 内加入 LLM 反思节点
- 现状：worker 是"LLM 解析一次 + 代码执行"
- 演进：execute 后加一个 verify 节点（LLM 校验结果合理性），失败回退执行节点
- 收益：把"确定性兜底"升级为"确定性 + 智能校验"双保险

**阶段 2（中成本）**：LLM 动态路由（真正的半自主）
- 现状：`TASK_AGENT_MAP` 硬编码路由
- 演进：plan_node 用 function calling 选 worker（工具 = 4 个 agent 的调用入口），模型自主决定派给谁、派几轮
- 收益：任务类型扩展时不用改代码；这正好接上"工具接入平台"的 ToolRegistry（第 4 章讲过）

**阶段 3（高成本，跨进程时才需要）**：标准 A2A 化
- 现状：进程内总线
- 演进：若 agent 要独立部署/跨团队复用，把 A2AMessageBus 换成标准 A2A（AgentCard 发现 + JSON-RPC 传输），或直接上 LangGraph 官方 MCP/A2A 支持
- 判据：**有没有第二个独立部署的 agent 系统要协作？没有就不做。**

### 8.2 面试收尾话术（直接背）

> *"我的项目是工程化的 Orchestrator-Workers 多 Agent：编排器负责 LLM 规划，4 个 worker 做确定性执行，A2A 消息总线做进程内通信，依赖图 + 三件套熔断保证可靠。它偏确定性是因为记账场景要防幻觉和保权限，自主性收敛是场景决定的。如果将来要支持开放任务或跨系统协作，我会按'worker 反思 → LLM 动态路由 → 标准 A2A'三阶段演进。"*



