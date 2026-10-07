# LangGraph 面试冲刺背诵文档 —— 腾讯 PCG Agent 应用开发

> 目标：经得住面试官对项目的逐层深挖，冲击 SSP。
> 项目一句话：**基于 LangGraph 两级图 + 自研 A2A 消息总线，构建的多 Agent 记账理财智能体**，底层对接 1.8B 本地小模型 + MCP 网关 + 记忆系统。

---

## 第〇部分：30 秒自我介绍模板（背熟）

> "我的项目是一个**多 Agent 记账理财智能体**。整体架构用 **LangGraph 搭了两级图**：上层是一个**编排器状态机**（plan→dispatch→wait→collect 四个节点），负责意图识别、任务规划、依赖调度和结果汇总；下层是 **bill / stat / price / finance 四个常驻 worker 子 Agent**，各自是独立的 2~3 节点小图。跨 Agent 通信我**没有用 LangGraph 原生的 Supervisor 子图机制，而是自研了一条基于 asyncio.Queue + Event 的 A2A 消息总线**，因为我要 worker 常驻、要子 Agent 可脱离编排器独立测试。另外底层是 1.8B 本地小模型，所以我做了一整套**确定性规则兜底体系**（类别/日期/金额/统计 IR 的代码层修正、数字白名单防幻觉闸门），保证小模型输出 100% 落库正确。"

记忆锚点：**两级图 / 4 节点编排环 / A2A 自研总线 / 小模型兜底体系** —— 这四词是整场面试主线。

---

## 第一部分：核心概念背诵卡（20 张卡）

> 每张卡三行：**一句话定义 → 项目怎么用 → 可能的追问点**。

### 卡 1：LangGraph 是什么
- **定义**：把 Agent 工作流显式建模为**有向图**的低层编排框架——State 是共享数据，Node 是执行单元，Edge 是控制流，运行时负责调度、循环、持久化。
- **项目**：编排器是 4 节点图，4 个子 Agent 各是独立小图，全部 `graph.ainvoke()` 驱动。
- **追问**：和 LangChain AgentExecutor / AutoGen / CrewAI 区别？→ LangGraph 更底层、控制流显式、原生支持循环与状态持久化，不绑定特定 Agent 范式。

### 卡 2：State（状态/Schema）
- **定义**：TypedDict 或 Pydantic 模型，定义图内共享数据结构；节点返回的 dict 被运行时**合并**进 State。
- **项目**：`OrchState`（user_input / task_plan / dispatch_round / agent_result_cache...），每个子 Agent 各自 `BillState`、`StatState`，**两级状态完全隔离**。
- **追问**：和普通 dict 传参区别？→ State 有 Schema 约束、有合并规则（reducer）、可被 checkpointer 快照、所有节点可见；且禁止节点就地改状态。

### 卡 3：Node（节点）
- **定义**：接收 State、返回**部分更新 dict 或 Command** 的普通函数；一次执行叫一个 superstep。
- **项目**：编排器 plan/dispatch/wait_result/collect 四节点，子 Agent 的 execute/reply 等，全部 `async def`。
- **追问**：节点内部随便写？→ 能。LLM 调用、SQL、正则、启发式都是普通代码，LangGraph 只关心输入输出契约。

### 卡 4：Edge（静态边）
- **定义**：`add_edge(A, B)` 表示 A 完必去 B，无判断；`START`/`END` 是虚拟起止点。
- **项目**：子 Agent 图全是静态边；编排器关键一条**回环边** `wait_result_node → dispatch_node` 实现调度循环。
- **追问**：图可以成环吗？→ **可以，环是多轮迭代的根基**，但必须设计终止条件（你的 `dispatch_round` 熔断）。

### 卡 5：Conditional Edge（条件边）
- **定义**：`add_conditional_edges(source, router_fn, path_map)`——路由是**普通 Python 函数**，运行时按返回值查表决定走向。
- **项目**：全项目唯一一处——`stat_agent` 的 `parse_query_node`：解析成功→execute，失败→reply（lambda 判 `raw_ir` 是否 dict）。
- **追问**：为什么只有 stat 用了？→ 因为 stat 有"解析失败仍要正常回话"的真实分支需求。**原则：分支静态已知→条件边；目标动态才知→Command。**

### 卡 6：Command（动态跳转）
- **定义**：`Command(update={...}, goto="node")` 返回状态更新的同时直接指定下一节点，无需图上声明边。
- **项目**：编排器**全部**动态跳转靠它——拦截/失败走 collect、有就绪任务走 wait、全 done 走 collect、轮次超限熔断。子 Agent reply_node 用 `Command(update={}, goto=END)` 收尾。
- **追问**：和条件边冲突谁说了算？→ **条件边源节点里 Command 的 goto 被忽略，以条件边为准**——这是 stat_agent 踩过的真实坑（追问链 4 细讲）。

### 卡 7：START / END
- **定义**：图的虚拟入口/出口。
- **项目**：每个图都有；编排器 `START→plan_node`、`collect_node→END`；子 Agent `reply_node→END`（回传结果后收尾）。
- **追问**：能没有 END？→ 必须至少一条到达 END 的路径，否则执行不完。

### 卡 8：执行模型（invoke / superstep / 环）
- **定义**：`graph.ainvoke(input_state)` 异步执行直到 END；每轮可并发节点集合算一个 superstep；环靠反复回到节点实现多轮。
- **项目**：`chat_loop.py` 每轮输入构造 `init_state` → `orchestrator_graph.ainvoke(init_state)` → 取 `final_reply`；worker 循环消费队列后各自 `ainvoke`。
- **追问**：一次 invoke 调度循环跑几轮？→ 由任务数决定，每任务一轮 dispatch→wait，全 done 进 collect。

### 卡 9：Reducer（状态归约器）
- **定义**：默认字段"覆盖"，`Annotated[type, reducer]` 自定义合并，如 `Annotated[int, operator.add]` 累加、`Annotated[list, operator.add]` 追加。
- **项目**：**没用 reducer**——`dispatch_round` 手动 `+1` 整体写回；所有状态更新"深拷贝 + 打包提交"。
- **追问**：为什么不用？→ 显式优于隐式，多 Agent 场景字段语义必须完全可控、可审计。什么场景必须用？→ 多轮对话消息列表累积（官方模板）。

### 卡 10：Checkpointer（检查点/持久化）
- **定义**：`compile(checkpointer=MemorySaver())` 后每次 superstep 结束快照 State；支持断点恢复、时间旅行、按 thread_id 多轮共享、配合 interrupt 人机协同。
- **项目**：**没配置**——状态只在单次 invoke 内存中流转；多轮记忆用自研 `ShortSessionMemory`（进程级 dict）+ 草稿缓存。
- **追问**：为什么不配？→ 明确取舍：A2A + 240s 阻塞已手工模拟"挂起等子任务"。**必须诚实认账：进程崩溃不可恢复、无时间旅行、无官方人机协同**（主动认账加分，演进见第五部分）。

### 卡 11：Interrupt（中断/人机协同）
- **定义**：`interrupt(payload)` 挂起图，等外部 `Command(resume=...)` 恢复；配合 checkpointer。
- **项目**：等价物是 `a2a_bus.wait_result(task_id, timeout=240)`——编排器挂起等子 Agent；子 Agent 返回 `need_more_info` 时 collect 转成对用户追问。**本质：用 A2A 总线手工实现了 interrupt 的"挂起-恢复"语义。**
- **追问**：差别？→ 官方可持久化、支持多轮恢复、线程内自动续跑；我的纯内存 + 超时熔断。

### 卡 12：Streaming（流式输出）
- **定义**：`graph.stream` / `astream_events` 逐节点、逐 token 流出进度。
- **项目**：没用官方流式，`_log()` 打印节点日志 + collect 一次性生成完整回复。
- **追问**：接 Web UI 打字机怎么做？→ `astream_events` 监听 `on_chat_model_stream`；节点进度用 `stream_mode="updates"`。

### 卡 13：子图（Subgraph）
- **定义**：编译后的图可作为另一图的节点（`add_node("sub", compiled_sub)`），子图 State 与父图自动字段映射。
- **项目**：**没用子图**——跨 Agent 协作走 A2A 总线（最大架构决策，追问链 1）。
- **追问**：差异？→ 子图状态自动传递、单进程同步、可检查点；A2A 进程解耦、worker 常驻复用、可独立部署测试，代价是自管协议与可靠性。

### 卡 14：send（动态扇出/并行）
- **定义**：`send("node", state)` 动态生成多个并行分支（Map-Reduce）；多分支同一 superstep 并行执行。
- **项目**：没用 send；并行靠"编排器派发 + 4 个常驻 worker 各跑各队列"，且**一次只派一个任务**。
- **追问**：允许并行怎么改？→ 就绪任务批量投递到不同 worker 队列（A2A 天然支持，队列按 agent 分离）。

### 卡 15：任务调度与 DAG（调度核心）
- **定义**：task_plan 每个任务有 `pending→running→done` 状态 + `deps` 依赖表。
- **项目**：`dispatch_node` 筛"依赖全 done"就绪任务，**一次只派一个**；全 done→collect。依赖表：stat/price 依赖 bill，finance 依赖 bill+stat+price。
- **追问**：为什么不一次派所有就绪任务？→ 本地单库单写更安全、240s 窗口内串行可控；上生产可放开并行。

### 卡 16：防死循环三件套（亮点）
- **定义**：图循环必须有终止保证。
- **项目**：① 依赖环检测 `_has_cycle`（DFS）② 依赖死锁检测（有 pending 但就绪集空）③ `dispatch_round > 20` 熔断。
- **追问**：为什么 20？→ 任务数×最坏轮次上界的经验值；熔断后带 error_msg 进 collect，不丢回复。

### 卡 17：统一载荷协议（跨 Agent 契约）
- **定义**：所有子 Agent 返回同一 JSON 结构，编排器唯一渲染函数生成文案。
- **项目**：`{"success", "agent_type", "msg", "data", "error"}`；`error="need_more_info"` + `data.prompt` = 多轮追问。
- **追问**：为什么子 Agent 禁止拼文案？→ 文案是表现层，统一渲染保证语气/格式一致；子 Agent 只回结构化事实，可单测、可复用到任何前端。

### 卡 18：确定性规则兜底（小模型适配核心）
- **定义**：LLM 输出不可信 → 代码规则每层二次修正，保证落库/计算正确。
- **项目**：bill 的 `_fix_category`（"打车"→交通关键词表）、`_fix_date`（原文时间线索优先，杜绝照抄示例日期）、**金额白名单修正**（LLM 金额不在原文数字集合→回退原文首个数字）；stat 的 `_fix_stat_ir`；编排层 `prune_tasks_by_heuristics`（裁剪脑补任务、强制补记账）。
- **追问**：为什么不用更大模型？→ 成本与离线可控；用兜底体系把"模型不聪明"的代价转移给确定性规则，准确率可量化。

### 卡 19：数字白名单防幻觉闸门（最强亮点，必讲）
- **定义**：LLM 生成文案时可能编数字 → 事实数字做成白名单，生成后校验。
- **项目**：collect_node 先确定性算好事实（金额、预算、预警 L1/L2/L3），LLM 只组织语言；文案出现白名单外数字 → 修正重试 1 次 → 仍失败**回退本地确定性渲染**。**纯计算不进 LLM，LLM 只做表达。**
- **追问**：怎么校验？→ 正则提取文案数字，与事实集合比对，多出的即幻觉。

### 卡 20：MCP 网关 + 记忆系统（支撑组件）
- **定义**：工具/模型访问统一收口 + 多轮记忆。
- **项目**：`mcp_client`（SQL/LLM 全走 MCP 子进程，**短连接**规避跨 task 崩溃，带 RBAC + 审计）；短期 `ShortSessionMemory`（对话历史 + bill 草稿缓存）+ 长期 `monthly_habit` 事实表 + FAISS 语义检索（规划时注入）。
- **追问**：为什么短连接？→ 本地异步任务跨 cancel scope 长连接崩溃，短连接稳定优先于性能。

---

## 第二部分：项目深挖追问链预演（面试核心战场）

> 按面试官连续追问的真实节奏编排。【回答】可直接背，【深挖】是被追问时的第二层弹药。

### 追问链 1：总体架构（开场必问）

**Q1：先介绍项目整体架构？**
【回答】两级：① 编排器 LangGraph 主图（plan→dispatch→wait_result→collect，含回环边实现调度循环）；② 4 个常驻 worker 子 Agent（bill/stat/price/finance），各自独立小图。中间是自研 A2A 总线（按 agent 分的 asyncio.Queue + task_id→Event 结果注册表）。底层：MCP 网关（SQL/LLM 统一入口 + 权限审计）、记忆系统（短期对话 + 长期习惯/FAISS）、1.8B 本地模型。
【深挖】数据流一句话：`chat_loop 构造 init_state → orchestrator.ainvoke → plan 出 DAG → dispatch 逐任务投递队列 → worker ainvoke → reply_node send_result 回传 → wait_result 收结果 → 全 done → collect 确定性渲染 + LLM 组织文案 → final_reply`。

**Q2：为什么不用 LangGraph 原生多 Agent（Supervisor/子图）？**
【回答】三个理由：① **进程解耦与常驻**——worker 是 `asyncio.create_task` 启动的独立消费者，天然任务级并发，崩溃只影响单个子 Agent；② **可独立测试**——子 Agent 可脱离编排器单独喂 state 跑通；③ **状态隔离**——各自独立 State 互不污染。代价也清楚：自管协议、可靠性、超时，状态不可持久化。
【深挖】若面试官说"子图也能做到"→ 答：官方 Supervisor 状态自动传递、可检查点、开发效率高；我自研 A2A 本质是**用工程复杂度换进程解耦**，适合子 Agent 将来独立部署成微服务的演进路径——A2A 总线未来可无缝替换成 Redis Stream / NATS，这也是它叫 A2A（Agent-to-Agent）的原因。

**Q3：A2A 总线内部数据结构？**
【回答】全局单例 `A2AMessageBus`，三块：① `_task_queues: Dict[str, asyncio.Queue]` 按 agent 分队列，`send_task` 投递 `{session_id, task_id, task_data}`；② `_result_events: Dict[task_id, Event]` + `_result_store: Dict[task_id, result]`——worker `reply_node` 调 `send_result`（先存 store 再 set Event），编排器 `wait_result` 用 `asyncio.wait_for(event.wait(), timeout=240)`；③ worker 端 `recv_task_blocking` 无限轮询，会话不匹配的消息**放回队尾**（FIFO 公平）。
【深挖】细节考点：`send_result` 先存结果再 set Event，避免"事件已 set 但 wait 才建 Event"的竞态（代码显式 `if task_id not in _result_events: 创建`）；`wait_result` 超时会 pop 残留，防内存泄漏。

**Q4：编排器和子 Agent 的数据怎么传？**
【回答】**数据经 State 流转，A2A 只做投递+回传的传输层**。dispatch 时组装 `task_data` 进消息，worker 消费后**手工 state 映射**（task_data 字段映射到子 Agent TypedDict）再 `graph.ainvoke(init_state)`；结果走子 Agent 的 `result` state 字段，reply_node 序列化回传。刻意设计：避免大对象走队列、子 Agent 输入输出有 Schema 约束。

### 追问链 2：状态管理（最爱深挖）

**Q5：State 怎么设计的？为什么到处深拷贝？**
【回答】OrchState 是 TypedDict，节点返回 `Command(update=...)` 提交变更。**LangGraph 哲学：节点只能返回更新提案，运行时统一合并**。但 Python dict 是引用类型，节点直接改嵌套结构（如 task_plan）在多节点回环下会产生共享污染——所以每个节点先 `copy.deepcopy` 再改再提交，保证**单向数据流**。
【深挖】第一性原理：**把状态当不可变快照，不当可变对象**。性能？→ 任务量级小，深拷贝成本可忽略，正确性优先。

**Q6：为什么不用 reducer？**
【回答】reducer 是隐式合并规则，我用显式打包提交。多 Agent 场景字段语义必须完全可控（如 agent_result_cache 是"追加 key"不是"覆盖 dict"），显式代码可审计可调试。**不是能力问题，是"隐式魔法 vs 显式控制"的权衡**——官方在消息累积场景推荐 reducer，业务状态也推荐显式。

**Q7：多轮记忆为什么不直接用 thread + checkpointer？**
【回答】我的图**每次 invoke 都是全新执行**（worker 常驻消费，非同一 thread 连续调用），记忆需跨多次独立 invoke 共享，而 checkpointer 的 thread 是给"同线程多轮"设计的。我的 `ShortSessionMemory` 用进程级 `SESSION_MEM[session_id]` 存对话历史 + 草稿缓存（bill_add/edit/delete），`get_history(limit=3)` 注入 prompt。**记忆外置方案：图只管当次执行，记忆由外层会话层管理**，分层更清晰。

### 追问链 3：调度循环（核心亮点）

**Q8：调度循环怎么转起来的？**
【回答】plan 产出 task_plan（任务+依赖 DAG）→ dispatch 选就绪任务派发（pending→running）→ wait_result 阻塞收结果（running→done，写缓存）→ **Command(goto="dispatch_node") 回环** → 直到"无就绪任务且全 done"才 `Command(goto="collect_node")`。循环动力是回环边，终止条件在 dispatch 判断。
【深挖】图论视角：带终止条件的可达性循环——每过回环，状态向"更多任务 done"单调推进，有限步内终止（20 轮熔断兜底）。

**Q9：任务依赖怎么表达？想过并行吗？**
【回答】每任务 `deps: [任务名]`，每轮只派"依赖全 done"的就绪任务，**当前一次只派一个**。依赖：stat/price 依赖 bill（先有账才能统计/比价），finance 依赖 bill+stat+price。**一次一个的理由**：本地单 SQLite 单写安全 + 240s 窗口内串行可控。演进：多库或写冲突可忽略时，就绪任务**批量投递**到不同 worker 队列并行（A2A 天然支持）。

**Q10：怎么防死循环？**
【回答】三层：① 依赖环检测 `_has_cycle`（DFS）；② 依赖死锁检测——有 pending 但就绪集空，带 error_msg 进 collect；③ `dispatch_round > 20` 熔断。覆盖"环/死锁/异常滞留"三类问题。

### 追问链 4：条件边 vs Command（考察踩坑）

**Q11：条件边和 Command 怎么分工？**
【回答】原则：**分支静态已知→条件边；目标动态才知→Command**。全项目只有 stat_agent 用条件边（parse 成败二选一，图结构层面就固定）；编排器"下一步去 wait 还是 collect"取决于运行时 DAG 状态，目标不确定，用 Command。
【深挖】**必讲踩坑**：最初 stat_agent 用静态边 + 节点内 `Command(goto="reply_node")` 处理解析失败，结果 **LangGraph 忽略条件边源节点内 Command 的 goto**，解析失败仍进 execute_query_node，`raw_ir=None` 触发 AttributeError。结论：**条件边源节点里的跳转由条件边决定**。改成 `add_conditional_edges` + lambda 判断修好。这证明：① 懂两者优先级；② 踩过真实版本行为。

**Q12：为什么编排器全用 Command 不用条件边？**
【回答】Command 的 goto 把"状态更新 + 路由决策"放进**同一个返回**，事务性强；编排器路由依赖整个 task_plan 运行时状态，条件边需在图定义处写复杂 router，可读性反而不如节点内判断。**图保持静态，动态逻辑收敛节点内**——这是我定义的"图可读性"。

### 追问链 5：跨 Agent 协议（工程规范）

**Q13：统一载荷协议怎么设计的？**
【回答】五字段 `success / agent_type / msg / data / error`。`success=false + error="need_more_info" + data.prompt` = 多轮追问；其余 error 是业务错误。所有子 Agent 强制返回，编排器唯一渲染函数 `build_user_display_text` 生成文案。
【深挖】动机：① **表现层与逻辑层分离**——文案全收敛编排器，子 Agent 只回事实，换前端不用改子 Agent；② **可测试**——输出是结构化 JSON 可单测断言；③ **可演进**——加新 Agent 只要遵守契约，编排器零改动。

**Q14：追问（need_more_info）怎么闭环？**
【回答】子 Agent 信息不足→返回 `error="need_more_info"` + prompt→collect 渲染成追问→**用户补充后草稿（bill_add）存进 ShortSessionMemory**→下一轮 plan 注入草稿继续记账。**多轮对话 + 部分信息累积**的闭环，靠草稿缓存而非每次从头解析。

### 追问链 6：小模型适配（差异化亮点，面试官会非常感兴趣）

**Q15：为什么用 1.8B 小模型？怎么保证质量？**
【回答】① 成本——免 API、可离线、数据不出域；② 挑战——小模型幻觉多，逼我把质量保障做成系统性工程，**三层兜底**：输入层（prompt 模板 + schema_example + 结构化 JSON 约束）；解析层（`parse_json_output` 带 validate + max_self_retry=3 自动重试）；修正层（类别关键词表、日期以原文为准、金额白名单、统计 IR 修正）。结果：**LLM 只负责理解和抽取，正确性由确定性规则兜底**，落库字段 100% 符合约束。

**Q16：数字白名单闸门具体怎么做？（值得展开）**
【回答】场景在 collect_node：预算状态、预警链 L1/L2/L3、汇总数字都是**确定性计算**（不进 LLM）。LLM 只拿结构化事实组织语言。生成后做**数字校验**：正则提取文案所有数字与事实白名单比对——白名单外即幻觉→修正重试 1 次→仍失败**回退本地确定性渲染模板**。**核心原则：计算不进 LLM，LLM 只做表达。**
【深挖】为什么先算后说？→ 让 LLM 直接算汇总再写文案，1.8B 的算术+幻觉双重出错且错误不可检测；先算后说把错误面压缩到表达层，且表达层有闸门可兜。

### 追问链 7：健壮性与失败处理

**Q17：LLM 失败/超时怎么办？**
【回答】三层：① `parse_json_output` 校验失败重试 3 次；② bill 解析整体异常→**兜底转追问**（"请提供金额、内容和日期"）而非硬失败；③ 编排层 `wait_result` 240s 超时熔断（对齐 `MODEL_TIMEOUT=240`，本地 CPU 单次 LLM 最坏 120s，熔断必须比 LLM 慢否则误杀）。异常路径全带 error 进 collect，用户永远收到一句人话。

**Q18：240s 超时为什么这么设计？**
【回答】超时值必须**大于最坏单次 LLM 耗时**。本地 CPU 跑 1.8B 最坏 120s + 排队 + 重试，240s 是安全上界。太小误杀正常执行，太大异常拖死会话。**超时是对齐资源画像后的工程决策，不是拍脑袋。**

**Q19：幂等性考虑过吗？**
【回答】当前单机单会话、每次一条链，重复投递概率低；但已留空间：`task_id` 全程透传（session_id+task_id 是 A2A 消息唯一键），未来加 Redis Stream 可直接做幂等键去重。诚实说明：当前没做重试投递，靠异常兜底保证"失败不脏数据"（bill 入库失败不会写半条）。

### 追问链 8：记忆系统

**Q20：短期记忆和长期记忆分别是什么？**
【回答】短期：`ShortSessionMemory`（进程级 dict）——① 对话历史 `get_history(limit=3)` 拼文本注入 prompt；② 草稿缓存 bill_add/edit/delete（多轮追问的信息累积）。长期：`monthly_habit` 事实表（记账/统计后沉淀月度习惯）+ FAISS(pkl) 语义层（规划时检索注入，如"这个月预算还剩多少"的上下文）。
【深挖】为什么长期记忆用 FAISS？→ 习惯/事实是语义查询（"我最近花钱趋势"），向量检索比 SQL like 更鲁棒；但结构化事实（预算数字）仍用 SQL 表保证精确——**结构化走 SQL、语义走向量，两条腿**。

---

## 第三部分：架构决策论证表（每个决策都能自证）

> 面试官问"为什么这么选"时，按「方案对比 → 我的选择 → 代价」三层回答。

| # | 决策点 | 备选方案 | 我的选择 | 为什么 | 代价 |
|---|---|---|---|---|---|
| 1 | 跨 Agent 协作 | LangGraph Supervisor/子图 | 自研 A2A 总线 | 进程解耦、worker 常驻、独立可测、可演进微服务 | 自管协议/可靠性，状态不可持久化 |
| 2 | 分支实现 | 全用条件边 | 编排器用 Command，stat 用条件边 | 分支静态→条件边；目标动态→Command；图保持静态 | 图上"看不出分支" |
| 3 | 状态合并 | 全用 reducer | 深拷贝+打包提交 | 显式可控可审计，防引用污染 | 样板代码略多 |
| 4 | 多轮记忆 | LangGraph thread+checkpointer | 记忆外置 ShortSessionMemory | 每次 invoke 全新执行，记忆归会话层管 | 无时间旅行/断点恢复 |
| 5 | 结果等待 | 官方 interrupt | wait_result 240s 阻塞 | A2A 已承担传输层，Event 语义足够 | 崩溃不可恢复 |
| 6 | 模型选型 | 大模型 API | 1.8B 本地 | 成本/离线/数据不出域 | 需整套确定性兜底体系 |
| 7 | MCP 连接 | 长连接复用 | 短连接（每次起子进程） | 规避跨 task cancel scope 崩溃 | 启动开销 |
| 8 | 并发派发 | 就绪任务全量并行 | 一次只派一个 | 单库单写安全、超时窗口可控 | 吞吐受限（可演进） |

---

## 第四部分：深度加分项（主动展示，不要等问）

> 面试中主动抛 1-2 个，瞬间拉开和"只会调 API"的候选人的差距。

**1. 数字白名单防幻觉闸门**（首选）——先讲"确定性计算 vs LLM 表达"的分离原则，再讲闸门实现。这是所有面试官都会眼前一亮的点：大部分人只会"让 LLM 自己算"，你展示了工程化防幻觉思路。

**2. 超时对齐的资源画像思维**——`wait_result 240s = MODEL_TIMEOUT 对齐 = 最坏单次 LLM 120s × 安全系数`。展示你不是拍脑袋定参数，而是**根据底层资源画像推导约束**。

**3. 状态不可变快照哲学**——所有节点深拷贝 + 打包提交，对齐 LangGraph"节点只提更新提案"的底层设计，展示你读懂了框架哲学而非会用 API。

**4. 两级状态隔离**——编排器状态与子 Agent 状态物理隔离，A2A 只做传输层、数据经 State 流转。展示你理解"状态边界 = 系统边界"。

**5. A2A 的演进预留**——task_id 全程透传、消息带 session_id 可放回队尾、未来可换 Redis Stream。展示你为分布式演进留了后门。

**6. 先想清楚再写代码**——`need_more_info` 契约、草稿缓存闭环、防死循环三件套，都是"多轮对话 / 循环终止 / 失败兜底"这些 Agent 系统核心难点的显式设计。

---

## 第五部分：演进方向（体现视野，背 2 个即可）

**演进 1：Checkpointer 化改造**
把 orchestrator 图 `compile(checkpointer=...)`，用 thread_id 管理会话：
- 获得断点恢复、时间旅行、官方 interrupt 人机协同
- 240s 阻塞可替换为 `interrupt()`——派发任务后图挂起持久化，结果回来 `Command(resume=...)` 恢复
- 代价：引入 thread_id 管理、worker 与编排器状态同步需要设计

**演进 2：A2A 总线网络化**
把 `asyncio.Queue` 替换为 Redis Stream / NATS：
- 消息持久化 + 消费者组 + 重试投递
- task_id 直接做幂等键，解决 Q19 的重试问题
- 子 Agent 无缝升级为独立服务（代码改动集中在 a2a_queue 层）

**演进 3：就绪任务并行派发**
依赖 DAG 不变，dispatch 阶段把就绪任务**批量投递**到不同 worker 队列（A2A 天然支持），配合结果收集的顺序化归并，吞吐翻倍。

**演进 4：官方 send() Map-Reduce 替代部分 A2A 场景**
如果某些子任务不需要进程隔离（如纯计算），可用 `send()` 在编排器图内并行扇出，省去队列开销——**按"是否要进程边界"决定用哪套**。

> 面试话术：*"当前阶段我优先保证了正确性和可调试性，但架构上已经为这三个演进留了接口：A2A 的传输层可替换、task_id 可做幂等、状态字段边界清晰。如果要上生产，我会先做 checkpointer 化，再做总线网络化。"*

---

## 第六部分：SSP 级面试建议

### 1. 面试官深挖时的心态模型（最重要）
腾讯 PCG 的 Agent 岗面试官深挖项目，**不是考你 API 背得全不全，是考你有没有"系统思维"**。每次被追问，心里过三关：
- **Why**（为什么这么设计）→ 讲约束与取舍，不讲"当时顺手"
- **Trade-off**（代价是什么）→ 主动承认代价（如"状态不可持久化"），比被逼问出来强 10 倍
- **How to evolve**（怎么演进）→ 给出 1 条演进路径，证明你在思考边界

### 2. 项目讲解的 STAR 结构（背熟）
- **S**：多 Agent 记账理财智能体，LangGraph 两级图 + A2A 总线，1.8B 本地模型
- **T**：小模型质量保障 + 多 Agent 协作 + 调度循环可控
- **A**：两级图架构、A2A 自研总线、三层兜底体系、数字白名单闸门、防死循环三件套
- **R**：落库字段 100% 符合约束、多轮追问闭环、异常全兜底不白屏

### 3. 主动展示深度的 3 个时机
- 讲完架构后主动说一句：*"这里有个我刻意的取舍——没用官方 Supervisor，代价是状态不可持久化，演进方案是 checkpointer 化"*
- 讲完 collect 后主动说：*"这里我加了数字白名单闸门，因为小模型会编数字"*
- 讲完调度后主动说：*"这里我做了三层防死循环"*

### 4. 知识盲区兜底话术
被问到完全不会的（如"LangGraph 和 xx 框架的 xx 特性对比"）：
- *"这个特性我还没在生产里用过，但按我的理解它是解决 xx 问题，如果让我做我会先这样验证……"*——**把问题拉回你的系统思维**，而不是硬编。

### 5. 高频必背清单（面试前 1 小时过一遍）
- [ ] 30 秒自我介绍（第〇部分）
- [ ] Q2 为什么不用 Supervisor（三个理由 + 演进）
- [ ] Q11 条件边 vs Command + 踩坑故事
- [ ] Q15/Q16 小模型三层兜底 + 数字白名单
- [ ] Q8/Q10 调度循环 + 防死循环三件套
- [ ] Q5 状态深拷贝哲学
- [ ] 演进 1（checkpointer 化）+ 演进 2（总线网络化）

### 6. 冲刺 SSP 的差异化策略
PCG Agent 岗 SSP 候选人共同点：**能把"工程实现"讲成"系统设计"**。具体到你的项目：
- 所有设计都有"约束 → 决策 → 代价 → 演进"的完整链条，杜绝"我这样写了"
- 每个决策都能和 LangGraph 官方机制对照（subgraph/interrupt/checkpointer/send），证明你懂框架而非会用
- 主动暴露 1-2 个已知短板并给出演进方案（checkpointer 化），这是资深工程师的标志——**敢认代价、有路线图**
- 准备 1 个"如果重做"的回答：*"如果重做，我会第一版就配 checkpointer，把 interrupt 用于等待子任务，A2A 只保留在跨进程场景"*

---

## 附录：代码位置速查（面试官要看代码时快速定位）

| 内容 | 文件 |
|---|---|
| 编排器 State | `agents/orchestrator/state.py` |
| 编排器四节点（Command 跳转） | `agents/orchestrator/nodes.py` |
| 编排器图（回环边） | `agents/orchestrator/graph.py` |
| 条件边（stat 唯一一处） | `agents/stat_agent/graph.py` |
| A2A 总线（Queue+Event） | `mcpGateway/a2a_queue.py` |
| worker 装配（常驻消费+state 映射） | `startup/bootstrap.py` |
| 对话入口（ainvoke 调用） | `startup/chat_loop.py` |
| 短期记忆 + 草稿缓存 | `memory/short_memory.py` |
| 金额/类别/日期兜底修正 | `agents/bill_agent/nodes.py` |
| MCP 网关（短连接+RBAC） | `mcpGateway/client.py` |

