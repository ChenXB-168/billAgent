import asyncio
import os
import subprocess
from typing import Any, Dict, List, Tuple

from langgraph.graph import StateGraph
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from utils.common import logger, db
# M8：RAG 由「用户文档库」接管（`10` §5.5 A 方案）：启动对账加载 userDocs 索引，
#     替换原 ragKnowledge.load_rag_index（旧 RAG 模块已退役，见 `11` §3 M8）
from memory.user_docs import load_user_docs
from mcpGateway.a2a_queue import a2a_bus
from mcpGateway.client import SERVER_CMD_MAP
# ★M4（D8 沙箱）：Windows 用 Job Object 给常驻 server 挂内存硬上限（POSIX 由 server self-prison）
from mcpGateway.sandbox import apply_job_object_limits
# ★M5（D2）：Durable 任务状态机（崩溃恢复 + 重投，落点见 `03` §8.1）
from startup.task_manager import get_task_manager
from config.config import MCP_HOST, MCP_SERVER_PORTS, MCP_HTTP_PATH, MCP_STDIO_FALLBACK
# M7（D3 接入点②）：worker 执行 span + run_id 恢复（见 `11` §3 M7）
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
# ★M3：常驻 MCP 服务进程句柄（`03` §9.8.2 / `07` §3.2 To-Be 步骤3）
# 用同步 subprocess.Popen 而非 asyncio.create_subprocess_exec：
# asyncio.Process.wait() 的 Future 绑定创建时的 event loop，跨 loop 启停（如 e2e
# conftest 用两个 asyncio.run）会抛 "attached to a different loop"；Popen 无此问题。
MCP_SERVER_PROCESSES: Dict[str, subprocess.Popen] = {}


async def _probe_mcp_server(server_key: str) -> bool:
    """M3 健康探活：MCP initialize 握手成功即视为服务就绪（`03` §9.8.2 ④ 治理清单）"""
    url = f"http://{MCP_HOST}:{MCP_SERVER_PORTS[server_key]}{MCP_HTTP_PATH}"
    try:
        async with streamablehttp_client(url) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
        return True
    except Exception:
        return False


async def start_mcp_servers():
    """M3：拉起 3 个 MCP 常驻服务并健康检查（冷启动只付一次，`07` §3.2 步骤3）"""
    if MCP_STDIO_FALLBACK:
        logger.warning("[M3] MCP_STDIO_FALLBACK=1：跳过常驻服务拉起（stdio 回滚模式）")
        return
    for key, cmd in SERVER_CMD_MAP.items():
        proc = subprocess.Popen(
            cmd,
            env=dict(os.environ),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        MCP_SERVER_PROCESSES[key] = proc
        # ★M4：子进程刚拉起、内存占用尚小时挂 Job 最严密（竞态窗口仅启动初期，
        # 远低于内存上限，实际不会触发；`mcpGateway/sandbox.py`）
        apply_job_object_limits(proc)
        logger.info(f"[M3] 已拉起 {key} 常驻服务（pid={proc.pid}），等待健康检查...")
        # 健康检查：120s 内每 1s 探活一次（对齐旧 stdio 握手超时 120s）
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
    """M3：关闭 3 个常驻服务，避免孤儿进程（`07` §3.3 优雅关闭）"""
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


async def _trace_agent_run(agent_name: str, msg: dict, runner) -> Any:
    """M7（D3 接入点②）：worker 执行包 `{agent}.run` span 并恢复 run_id。

    ★contextvars 不跨独立协程传播：worker 常驻循环与 orchestrator 请求是**不同协程**，
    必须从 task_data 显式恢复 run_id（编排入口生成、dispatch_node 注入 task_data 下发，
    见 `03`:546 接入点②与 `11` §3 M7）。异常由 span 标 error 后照常抛出（task_manager 兜底）。
    """
    rid = (msg.get("task_data") or {}).get("run_id")
    pid = (msg.get("task_data") or {}).get("parent_span_id")
    with run_scope(session_id=msg.get("session_id"), run_id=rid):
        with span(f"{agent_name}.run", kind="agent", agent_id=agent_name,
                  parent_span_id=pid):
            return await runner()


async def _bill_agent_worker():
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
                "city": msg["task_data"].get("city"),
                "month": msg["task_data"].get("month"),
                "result": None
            }
            # ★M5（D2）：执行前后打状态（running → completed / failed / interrupted）
            await _trace_agent_run("bill_agent", msg,
                                   lambda: bill_agent_graph.ainvoke(init_state))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def _stat_agent_worker():
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
            # ★M5（D2）：stat 无落库副作用 → retriable=1，崩溃后可安全重投
            await _trace_agent_run("stat_agent", msg,
                                   lambda: stat_agent_graph.ainvoke(init_state))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def _price_agent_worker():
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
            await _trace_agent_run("price_agent", msg,
                                   lambda: price_agent_graph.ainvoke(init_state))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def _finance_agent_worker():
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
            await _trace_agent_run("finance_agent", msg,
                                   lambda: finance_agent_graph.ainvoke(init_state))
            logger.info(f"[{agent_name}] task {msg['task_id']} 执行完成")
        except Exception as e:
            logger.exception(f"[{agent_name}] 任务执行异常")
            await asyncio.sleep(0.5)


async def bootstrap_all():
    """系统一键初始化"""
    global SHUTDOWN_FLAG
    SHUTDOWN_FLAG = False
    logger.info("======== 多Agent系统启动初始化开始 ========")

    # 1. 数据库连通校验
    db.execute_sql("SELECT 1;")
    logger.info("数据库初始化完成")

    # 2. 加载用户文档库索引（M8：启动对账，文件增删改自动重建；失败不阻断启动，检索时自动重建）
    try:
        load_user_docs()
        logger.info("用户文档库索引加载完成")
    except Exception as e:
        logger.warning(f"用户文档库索引加载失败（将在首次检索时重建）：{e}")

    # 3. MCP 网关就绪（★M3：拉起 3 个常驻服务 + 健康检查，冷启动只付一次）
    await start_mcp_servers()
    logger.info("MCP 网关就绪（常驻服务已拉起）")

    # 4. A2A消息总线清理残留任务
    a2a_bus.clear_all()
    logger.info("A2A消息总线清理历史消息完成")

    # 4.1 ★M5（D2）：崩溃恢复——recover_orphans（改状态）→ 重建队列（重投 submitted）
    #   顺序不可颠倒（`04` §3.1 补注）：DB 是任务事实的唯一来源，队列只是下游投影，
    #   故先由 DB 把状态改对，再把 `submitted` 任务重投回队列。
    #   ★只在启动路径调用一次：此时队列刚 clear_all，submitted 全部是上次进程遗留任务
    #   （运行期重复调用会把"正在排队等待领取"的新任务重复投递）。
    #   ★失败只告警不阻断启动：Durable 是加固，不能成为启动的单点故障。
    try:
        recovered = get_task_manager().recover_orphans()
        requeued = get_task_manager().resubmit_submitted(a2a_bus.send_task)
        logger.info(f"[M5] 崩溃恢复：{recovered}；重建队列重投 {requeued} 条")
    except Exception as e:  # noqa: BLE001
        logger.error(f"[M5] 崩溃恢复失败（降级：不影响启动）: {e}")

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

    logger.info("======== 全部底层初始化完毕，可以启动聊天交互 ========")


async def shutdown_system():
    """优雅关闭系统，main程序退出时调用"""
    global SHUTDOWN_FLAG
    logger.warning("收到关闭信号，准备停止所有Worker...")
    SHUTDOWN_FLAG = True
    for task in WORKER_TASKS:
        task.cancel()
    await asyncio.gather(*WORKER_TASKS, return_exceptions=True)
    logger.info("所有后台Worker已正常退出")
    # ★M3：关闭 MCP 常驻服务，避免孤儿进程
    await stop_mcp_servers()