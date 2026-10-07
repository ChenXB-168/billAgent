# ==============================================
# 启动编排模块 - 系统一键初始化与优雅关闭
# 说明：bootstrap_all() 按序完成 DB 连通校验 → 用户文档库索引加载 → MCP 常驻服务拉起
#       → A2A 总线清理 + Durable 崩溃恢复 → 拉起 4 个子 Agent 常驻 worker。
# ==============================================
import asyncio
import os
import subprocess
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from utils.common import logger, db
# RAG 检索源为用户文档库：启动时对账加载 userDocs 索引（设计 `10` §5.5 A 方案）
from memory.user_docs import load_user_docs
from mcpGateway.a2a_queue import a2a_bus
from mcpGateway.client import SERVER_CMD_MAP
# 沙箱：Windows 用 Job Object 给常驻 server 挂内存硬上限（POSIX 由 server self-prison）
from mcpGateway.sandbox import apply_job_object_limits
# Durable 任务状态机（崩溃恢复 + 重投，落点见 `03` §8.1）
from startup.task_manager import get_task_manager
from config.config import (MCP_HOST, MCP_SERVER_PORTS, MCP_HTTP_PATH, MCP_STDIO_FALLBACK,
                           MCP_HEALTH_INTERVAL)
# worker 执行 span + run_id 恢复（接入点见 `11` §3 M7）
from utils.tracer import run_scope, span

# ========= 导入全部Agent Graph实例 =========
from agents.orchestrator.graph import orchestrator_graph
from agents.bill_agent.graph import bill_agent_graph
from agents.stat_agent.graph import stat_agent_graph
from agents.price_agent.graph import price_agent_graph
from agents.finance_agent.graph import finance_agent_graph

# 全局统一Agent注册表
AGENT_GRAPH: Dict[str, StateGraph] = {
    "orchestrator_agent": orchestrator_graph,
    "bill_agent": bill_agent_graph,
    "stat_agent": stat_agent_graph,
    "price_agent": price_agent_graph,
    "finance_agent": finance_agent_graph,
}

# 全局关闭标记
SHUTDOWN_FLAG = False
# 保存所有worker任务句柄，用于退出时cancel
WORKER_TASKS: List[asyncio.Task] = []
# 常驻 MCP 服务进程句柄（`03` §9.8.2 / `07` §3.2 To-Be 步骤3）
# 用同步 subprocess.Popen 而非 asyncio.create_subprocess_exec：
# asyncio.Process.wait() 的 Future 绑定创建时的 event loop，跨 loop 启停（如 e2e
# conftest 用两个 asyncio.run）会抛 "attached to a different loop"；Popen 无此问题。
MCP_SERVER_PROCESSES: Dict[str, subprocess.Popen] = {}


async def _probe_mcp_server(server_key: str) -> bool:
    """
    函数功能与逻辑描述：
        健康探活：按 config 的 host/port/path 拼出该 MCP 服务的 streamable-http 地址，
        建立连接并完成一次 MCP initialize 握手，握手成功即视为服务就绪
        （依据 `03` §9.8.2 ④ 治理清单）。任何异常（连接被拒 / 超时 / 协议错误）一律
        吞掉并返回 False，由调用方决定是否重试；无副作用、不写任何状态。
    入参说明：
        server_key (str)：MCP 服务键，取值 sql_bill / llm_base / llm_finance，
            用于索引 MCP_SERVER_PORTS 得到端口。
    返回值说明：
        bool：True = initialize 握手成功；False = 探活失败（含任何异常）。
    """
    url = f"http://{MCP_HOST}:{MCP_SERVER_PORTS[server_key]}{MCP_HTTP_PATH}"
    try:
        async with streamablehttp_client(url) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
        return True
    except Exception:
        return False


async def start_mcp_servers():
    """
    函数功能与逻辑描述：
        遍历 SERVER_CMD_MAP 拉起 3 个 MCP 常驻服务（sql_bill / llm_base / llm_finance）
        并逐个健康检查，冷启动成本只付一次（`07` §3.2 步骤3）。每个服务用同步
        subprocess.Popen 拉起（不绑定 event loop，避免跨 loop 启停报错），立即挂
        Job Object 内存硬上限，随后每 1s 探活一次、最多 120 次：
        进程提前退出即 raise RuntimeError，120s 内探不通同样 raise RuntimeError。
        开关 BILLAGENT_MCP_STDIO=1（MCP_STDIO_FALLBACK）时直接告警返回，不拉常驻服务。
    入参说明：
        无。
    返回值说明：
        无（成功返回 None；失败抛 RuntimeError，由调用方决定是否中止启动）。
    """
    if MCP_STDIO_FALLBACK:
        logger.warning("[M3] MCP_STDIO_FALLBACK=1：跳过常驻服务拉起（stdio 回滚模式）")
        return
    # creationflags 是 **Windows 专有**语义（仅用于隐藏控制台窗口），
    #   Linux/macOS 下传非 0 值会抛错，故非 Windows 平台统一传 0，
    #   保证 Docker（`python:3.10-slim`，见 utils/Dockerfile）等 Linux 环境可正常拉起。
    _creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for key, cmd in SERVER_CMD_MAP.items():
        # 幂等保护：若该服务已在运行（如 e2e 场景下顶层 tests/conftest.py 已拉起，
        #   或上次崩溃残留的进程），直接复用而不重复拉起——否则新子进程会因端口占用而退出，
        #   随即触发下方 `proc.poll() is not None` 抛 RuntimeError。
        #   非本进程拉起的服务**不进** MCP_SERVER_PROCESSES，
        #   故 stop_mcp_servers 也不会去关它（所有权对称，不会误杀别人的进程）。
        if await _probe_mcp_server(key):
            logger.info(f"[M3] {key} 常驻服务已在运行，跳过拉起（复用现有进程）")
            continue
        proc = subprocess.Popen(
            cmd,
            env=dict(os.environ),
            creationflags=_creationflags,
        )
        MCP_SERVER_PROCESSES[key] = proc
        # 子进程刚拉起、内存占用尚小时挂 Job 最严密（竞态窗口仅启动初期，
        # 远低于内存上限，实际不会触发；实现见 `mcpGateway/sandbox.py`）
        apply_job_object_limits(proc)
        logger.info(f"[M3] 已拉起 {key} 常驻服务（pid={proc.pid}），等待健康检查...")
        # 健康检查：120s 内每 1s 探活一次
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"[M3] {key} 常驻服务启动失败，退出码 {proc.returncode}")
            if await _probe_mcp_server(key):
                logger.info(f"[M3] {key} 常驻服务就绪：http://{MCP_HOST}:{MCP_SERVER_PORTS[key]}{MCP_HTTP_PATH}")
                break
            await asyncio.sleep(1)
        else:
            raise RuntimeError(f"[M3] {key} 常驻服务健康检查超时（120s）")


async def stop_mcp_servers():
    """
    函数功能与逻辑描述：
        M3：遍历 MCP_SERVER_PROCESSES 关闭全部常驻服务，避免留下孤儿进程
        （`07` §3.3 优雅关闭）。仍在运行的进程先 terminate() 并最多等 5s，
        超时则 kill() 强制结束；最后清空进程字典，保证可重复调用。
    入参说明：
        无。
    返回值说明：
        无（仅终止子进程并清空全局字典 MCP_SERVER_PROCESSES）。
    """
    for key, proc in MCP_SERVER_PROCESSES.items():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            logger.info(f"[M3] {key} 常驻服务已关闭")
    MCP_SERVER_PROCESSES.clear()


async def _mcp_health_watchdog():
    """
    函数功能与逻辑描述：
        M21（T-4）MCP 服务**运行期**心跳看门狗：每 `MCP_HEALTH_INTERVAL` 秒对 3 个常驻服务
        各做一次探活（建连接 + MCP `initialize` 握手，复用启动期的 `_probe_mcp_server`），
        失败即打 WARNING 并累计"连续失败次数"，恢复时打 INFO。

        为什么需要：**启动期健康检查只覆盖"服务能不能起来"**；服务在**运行中**挂掉时，
        主 Agent 唯一的感知途径是"工具调用失败 → 编排器盲等 240s 超时"，且**分不清
        "慢"还是"死"**。本看门狗把"240s 盲等才发现"缩短为"≤1 个探测周期发现"，
        并给出可告警的明确信号。

        边界三条：
        ① **只告警、不摘除通道** —— 自动摘除会引出"何时恢复、要不要重探"的新状态机，
           属后续演进；当前先把"可观测"补齐（健康治理的第一步）；
        ② **探活失败不抛异常**（内部吞掉）—— 看门狗自身绝不能成为新的故障源；
        ③ **stdio 回退模式 / 间隔 <= 0 时直接返回** —— 此时无常驻服务或已显式关闭。

        挂载方式：由 `bootstrap_all` 以 `create_task` 挂入 `WORKER_TASKS`，故
        `shutdown_system` 的 `cancel()` + `gather` 会一并收敛它（无需单独登记）。
    入参说明：
        无。
    返回值说明：
        无（常驻协程；`SHUTDOWN_FLAG` 置位后被 cancel 或自然退出，不返回值）。
    """
    if MCP_STDIO_FALLBACK or MCP_HEALTH_INTERVAL <= 0:
        logger.info("[M21] MCP 运行期心跳未启用（stdio 回退模式或间隔 <= 0）")
        return
    logger.info(f"[M21] MCP 运行期心跳看门狗启动，间隔 {MCP_HEALTH_INTERVAL}s")
    consecutive: Dict[str, int] = {}
    while not SHUTDOWN_FLAG:
        await asyncio.sleep(MCP_HEALTH_INTERVAL)
        if SHUTDOWN_FLAG:
            break
        for key in MCP_SERVER_PORTS:
            try:
                ok = await _probe_mcp_server(key)
            except Exception as e:  # noqa: BLE001 —— 探活异常一律按失败计，绝不外抛
                ok = False
                logger.warning(f"[M21] MCP 探活异常 {key}: {e}")
            if ok:
                if consecutive.get(key):
                    logger.info(
                        f"[M21] MCP 服务已恢复：{key}（此前连续失败 {consecutive[key]} 次）")
                consecutive[key] = 0
            else:
                consecutive[key] = consecutive.get(key, 0) + 1
                logger.warning(
                    f"[M21] MCP 服务探活失败：{key}（连续 {consecutive[key]} 次）"
                    f"—— 该通道工具调用将失败，请检查进程 / 端口")


async def _trace_agent_run(agent_name: str, msg: dict, runner) -> Any:
    """
    函数功能与逻辑描述：
        worker 侧统一执行包装——以 `{agent}.run` span（kind="agent"）包裹子 Agent 图的
        执行，并从 task_data 恢复 trace 上下文。
        contextvars 不跨独立协程传播：worker 常驻循环与 orchestrator 请求是不同协程，
        故必须从 msg["task_data"] 显式取回 run_id 与 parent_span_id（编排入口生成、
        dispatch_node 注入 task_data 下发，见 `03`:546 与 `11` §3 M7）。
        本函数只做埋点，不写 task_state，也不吞异常——runner() 异常原样抛出，
        由各 worker 的 except 分支兜底记录日志（`utils/tracer.span` 内部自行吞异常，
        保证可观测失败不阻断主链路）。
    入参说明：
        agent_name (str)：子 Agent 名，取值 bill_agent / stat_agent / price_agent /
            finance_agent，用作 span 名与 agent_id。
        msg (dict)：A2A 消息，需含 session_id 与 task_data（task_data 内可选
            run_id / parent_span_id，缺失即按 None 处理，span 自建根/孤儿 span）。
        runner：无参可等待对象（协程函数），真正执行 Agent 图，如
            `lambda: bill_agent_graph.ainvoke(init_state)`。
    返回值说明：
        Any：原样透传 runner() 的返回值（LangGraph 输出状态 dict）；runner() 抛出的
            异常不捕获、直接向上传播。
    """
    rid = (msg.get("task_data") or {}).get("run_id")
    pid = (msg.get("task_data") or {}).get("parent_span_id")
    with run_scope(session_id=msg.get("session_id"), run_id=rid):
        with span(f"{agent_name}.run", kind="agent", agent_id=agent_name,
                  parent_span_id=pid):
            return await runner()


async def _bill_agent_worker():
    """
    函数功能与逻辑描述：
        bill_agent（记账）常驻 worker：无限阻塞从 A2A 总线领取本 agent 的任务，
        把 task_data 映射为 bill 图所需的 init_state（bill 专属 state 口径，与
        orchestrator dispatch_node 下发的字段对齐）后执行 bill_agent_graph。
        该图会写账单库。本循环经 task_manager.track() 包状态流转
        （running → completed / failed / interrupted），与 dispatch_node 建单、
        bootstrap_all 启动恢复共同构成完整 Durable 链路。bill_agent 有写副作用 → retriable=0，
        崩溃后禁止盲重投，须先按 task_id 查 bill 表对账（有记录 → completed，无记录 → 重投）。
        异常在循环内被捕获记日志并 sleep 0.5s 后继续消费，保证消费循环不中断。
    入参说明：
        无（agent_name 取函数内固定字面量 "bill_agent"）。
    返回值说明：
        无（常驻协程，SHUTDOWN_FLAG 置位后循环自然结束，不返回值）。
    """
    agent_name = "bill_agent"
    logger.info(f"[{agent_name}] Worker消费循环启动成功")
    while not SHUTDOWN_FLAG:
        try:
            msg = await a2a_bus.recv_task_blocking(agent_name)
            logger.info(f"[{agent_name}] 获取任务 task_id={msg['task_id']}, session_id={msg['session_id']}")
            # bill 专属state映射，与 orchestrator dispatch_node 派发的 task_data 对齐
            init_state = {
                "session_id": msg["session_id"],
                "task_id": msg["task_id"],
                "raw_segments": msg["task_data"].get("raw_segments", []),
                "operate_sub_type": msg["task_data"].get("operate_sub_type", ""),
                # session_memory 承载草稿与历史（删除二次确认依赖它），
                #   缺失时回落空字典，与 stat / finance worker 的口径一致。
                "session_memory": msg["task_data"].get("session_memory", {}),
                "city": msg["task_data"].get("city"),
                "month": msg["task_data"].get("month"),
                "result": None
            }
            # 以 `bill_agent.run` span 包裹本次图执行，run_id 从 task_data 恢复
            #   （编排入口生成、dispatch_node 注入下发，见 `11` §3 M7）。
            # 经 track() 包状态流转 running → completed / failed / interrupted，
            #   使 recover_orphans 的按 task_id 对账生效。
            #   外层 track（生命周期）→ 中层 span（可观测）→ 内层图执行。
            await get_task_manager().track(
                msg,
                lambda: _trace_agent_run(
                    "bill_agent",
                    msg,
                    lambda: bill_agent_graph.ainvoke(init_state)))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def _stat_agent_worker():
    """
    函数功能与逻辑描述：
        stat_agent（统计）常驻 worker：无限阻塞领取本 agent 的 A2A 任务，把 task_data
        映射为 stat 图所需的 init_state 后执行 stat_agent_graph。user_input 优先取
        orchestrator 显式下发的原始用户输入，为空时退回 raw_segments 首条（防模型改写
        后的 raw_segments 失真）。该 Agent 无落库副作用 → retriable=1，崩溃恢复时可安全重投。
        本循环经 task_manager.track() 落 running / completed / failed / interrupted。
        异常在循环内被捕获记日志并 sleep 0.5s 后继续消费，保证消费循环不中断。
    入参说明：
        无（agent_name 取函数内固定字面量 "stat_agent"）。
    返回值说明：
        无（常驻协程，SHUTDOWN_FLAG 置位后循环自然结束，不返回值）。
    """
    agent_name = "stat_agent"
    logger.info(f"[{agent_name}] Worker消费循环启动成功")
    while not SHUTDOWN_FLAG:
        try:
            msg = await a2a_bus.recv_task_blocking(agent_name)
            logger.info(f"[{agent_name}] 获取任务 task_id={msg['task_id']}, session_id={msg['session_id']}")
            raw_segments = msg["task_data"].get("raw_segments", [])
            # 优先使用 orchestrator 显式下发的原始用户输入（防模型改写的 raw_segments 失真）
            user_input = msg["task_data"].get("user_input") or (raw_segments[0] if raw_segments else "")
            init_state = {
                "session_id": msg["session_id"],
                "task_id": msg["task_id"],
                "user_input": user_input,
                "session_memory": msg["task_data"].get("session_memory", {}),
                "target_control": msg["task_data"].get("target_control"),
                "raw_ir": None,
                "final_struct": None
            }
            # stat 无落库副作用 → retriable=1，崩溃后可安全重投；
            #   经 track() 包状态流转 running → completed / failed / interrupted
            await get_task_manager().track(
                msg,
                lambda: _trace_agent_run("stat_agent", msg,
                                         lambda: stat_agent_graph.ainvoke(init_state)))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def _price_agent_worker():
    """
    函数功能与逻辑描述：
        price_agent（物价对标）常驻 worker：无限阻塞领取本 agent 的 A2A 任务，把 task_data
        映射为 price 图所需的 init_state 后执行 price_agent_graph。raw_segments 用直接下标
        取值（与 price TypedDict 严格对齐，缺失即 KeyError → 由循环的 except 记录并继续消费）。
        该 Agent 无落库副作用 → retriable=1，崩溃恢复时可安全重投。
        本循环经 task_manager.track() 落 running / completed / failed / interrupted。
        异常在循环内被捕获记日志并 sleep 0.5s 后继续消费，保证消费循环不中断。
    入参说明：
        无（agent_name 取函数内固定字面量 "price_agent"）。
    返回值说明：
        无（常驻协程，SHUTDOWN_FLAG 置位后循环自然结束，不返回值）。
    """
    agent_name = "price_agent"
    logger.info(f"[{agent_name}] Worker消费循环启动成功")
    while not SHUTDOWN_FLAG:
        try:
            msg = await a2a_bus.recv_task_blocking(agent_name)
            logger.info(f"[{agent_name}] 获取任务 task_id={msg['task_id']}, session_id={msg['session_id']}")
            # 和price TypedDict严格对齐
            init_state = {
                "session_id": msg["session_id"],
                "task_id": msg["task_id"],
                "raw_segments": msg["task_data"]["raw_segments"],
                "extract_info": None,
                "final_struct": None,
                "error_stack": None
            }
            # 经 track() 包状态流转 running → completed / failed / interrupted
            await get_task_manager().track(
                msg,
                lambda: _trace_agent_run("price_agent", msg,
                                         lambda: price_agent_graph.ainvoke(init_state)))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def _finance_agent_worker():
    """
    函数功能与逻辑描述：
        finance_agent（理财分析）常驻 worker：无限阻塞领取本 agent 的 A2A 任务，把 task_data
        映射为 finance 图所需的 init_state 后执行 finance_agent_graph；除本任务的输入外，
        还透传 orchestrator 已归集的账单/统计/物价上游结果（bill_result / stat_result /
        price_result）、历史习惯（habit_data）与用户文档（user_docs），供理财分析聚合使用。
        该 Agent 无落库副作用 → retriable=1，崩溃恢复时可安全重投。
        本循环经 task_manager.track() 落 running / completed / failed / interrupted。
        异常在循环内被捕获记日志并 sleep 0.5s 后继续消费，保证消费循环不中断。
    入参说明：
        无（agent_name 取函数内固定字面量 "finance_agent"）。
    返回值说明：
        无（常驻协程，SHUTDOWN_FLAG 置位后循环自然结束，不返回值）。
    """
    agent_name = "finance_agent"
    logger.info(f"[{agent_name}] Worker消费循环启动成功")
    while not SHUTDOWN_FLAG:
        try:
            msg = await a2a_bus.recv_task_blocking(agent_name)
            logger.info(f"[{agent_name}] 获取任务 task_id={msg['task_id']}, session_id={msg['session_id']}")
            init_state = {
                "session_id": msg["session_id"],
                "task_id": msg["task_id"],
                "raw_segments": msg["task_data"].get("raw_segments", []),
                "operate_sub_type": msg["task_data"].get("operate_sub_type", "analyse"),
                "session_memory": msg["task_data"].get("session_memory", {}),
                "target_control": msg["task_data"].get("target_control"),
                "bill_result": msg["task_data"].get("bill_result"),
                "stat_result": msg["task_data"].get("stat_result"),
                "price_result": msg["task_data"].get("price_result"),
                "habit_data": msg["task_data"].get("habit_data"),
                "user_docs": msg["task_data"].get("user_docs", [])
            }
            # 经 track() 包状态流转 running → completed / failed / interrupted
            await get_task_manager().track(
                msg,
                lambda: _trace_agent_run("finance_agent", msg,
                                         lambda: finance_agent_graph.ainvoke(init_state)))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def bootstrap_all():
    """
    函数功能与逻辑描述：
        系统一键初始化总入口，按固定顺序完成 5 步：① 数据库连通校验（`SELECT 1;`）；
        ② 加载用户文档库索引（启动对账，文件增删改自动重建；失败仅告警、不阻断启动，
        首次检索时自动重建）；③ 拉起全部 MCP 常驻服务并健康检查；④ 清空 A2A
        总线残留消息后执行 Durable 崩溃恢复——顺序不可颠倒：先由 DB 改状态
        （recover_orphans()），再把 submitted 任务重投回队列（resubmit_submitted()），
        且只在启动路径调用一次，失败只告警不阻断启动；⑤ 打印已注册的 Agent 列表，
        并把 4 个子 Agent 常驻 worker **与 MCP 运行期心跳看门狗**（`_mcp_health_watchdog`，
        M21/T-4）以 asyncio.create_task 挂到全局 WORKER_TASKS。
        副作用：重置全局 SHUTDOWN_FLAG、写 DB、拉起 MCP 子进程、创建常驻协程任务。
    入参说明：
        无。
    返回值说明：
        无（初始化完成后返回 None；MCP 拉起/健康检查失败会向上抛 Exception，
            用户文档库索引与崩溃恢复失败已内部降级为告警）。
    """
    global SHUTDOWN_FLAG
    SHUTDOWN_FLAG = False
    logger.info("======== 多Agent系统启动初始化开始 ========")

    # 1. 数据库连通校验
    db.execute_sql("SELECT 1;")
    logger.info("数据库初始化完成")

    # 2. 加载用户文档库索引（启动对账，文件增删改自动重建；失败不阻断启动，检索时自动重建）
    try:
        load_user_docs()
        logger.info("用户文档库索引加载完成")
    except Exception as e:
        logger.warning(f"用户文档库索引加载失败（将在首次检索时重建）：{e}")

    # 3. MCP 网关就绪（拉起 3 个常驻服务 + 健康检查，冷启动只付一次）
    await start_mcp_servers()
    logger.info("MCP 网关就绪（常驻服务已拉起）")

    # 4. A2A消息总线清理残留任务
    a2a_bus.clear_all()
    logger.info("A2A消息总线清理历史消息完成")

    # 4.1 崩溃恢复——recover_orphans（改状态）→ 重建队列（重投 submitted）
    #   顺序不可颠倒：DB 是任务事实的唯一来源，队列只是下游投影。
    #   只在启动路径调用一次：此时队列刚 clear_all，submitted 全部是上次进程遗留任务
    #   （运行期重复调用会把"正在排队等待领取"的新任务重复投递）。
    #   失败只告警不阻断启动：Durable 是加固，不能成为启动的单点故障。
    try:
        recovered = get_task_manager().recover_orphans()
        requeued = get_task_manager().resubmit_submitted(a2a_bus.send_task)
        logger.info(f"崩溃恢复：{recovered}；重建队列重投 {requeued} 条")
    except Exception as e:  # noqa: BLE001
        logger.error(f"崩溃恢复失败（降级：不影响启动）: {e}")

    # 5. 加载注册所有Agent Graph
    logger.info(f"加载Agent列表：{list(AGENT_GRAPH.keys())}")
    logger.info("全部Agent Graph实例加载注册完成")

    # 启动各个子Agent常驻worker
    worker_coroutines = [
        _bill_agent_worker(),
        _stat_agent_worker(),
        _price_agent_worker(),
        _finance_agent_worker()
    ]
    for coro in worker_coroutines:
        task = asyncio.create_task(coro)
        WORKER_TASKS.append(task)

    # M21（T-4）：运行期心跳看门狗——与 worker 同批挂载，退出时由 shutdown_system 一并 cancel
    WORKER_TASKS.append(asyncio.create_task(_mcp_health_watchdog()))

    logger.info("======== 全部底层初始化完毕，可以启动聊天交互 ========")


async def shutdown_system():
    """
    函数功能与逻辑描述：
        优雅关闭系统（main 程序退出 / WebUI 退出路径调用）：置全局 SHUTDOWN_FLAG=True
        让各 worker 消费循环自然退出，随后 cancel 全部 WORKER_TASKS 并
        asyncio.gather(..., return_exceptions=True) 等待收敛（取消异常不向上抛），
        最后关闭 MCP 常驻服务避免孤儿进程。
        副作用：修改全局标记、取消后台任务、终止 MCP 子进程。
    入参说明：
        无。
    返回值说明：
        无（全部 worker 与常驻服务收敛后返回 None）。
    """
    global SHUTDOWN_FLAG
    logger.warning("收到关闭信号，准备停止所有Worker...")
    SHUTDOWN_FLAG = True
    for task in WORKER_TASKS:
        task.cancel()
    await asyncio.gather(*WORKER_TASKS, return_exceptions=True)
    logger.info("所有后台Worker已正常退出")
    # 关闭 MCP 常驻服务，避免孤儿进程
    await stop_mcp_servers()