# ==============================================
# 命令行交互主循环 - 终端多轮对话入口
# 说明：main.py 调用 run_chat()；WebUI（webUI/app.py）走另一条入口，不经本文件
# ==============================================
import asyncio
import uuid
from startup.bootstrap import bootstrap_all, shutdown_system, AGENT_GRAPH
from memory.short_memory import get_session_memory
from utils.tracer import run_scope, span

# 进程级固定会话ID：多轮对话复用同一session，保证短时记忆与草稿缓存生效
SESSION_ID = str(uuid.uuid4())


async def run_chat():
    """
    函数功能与逻辑描述：
        终端 REPL 主循环：一次性初始化全部底层资源后，反复读取用户输入并驱动编排器全流程。
        每轮流程为：① 先把用户提问写入短期记忆（与 collect_node 写入的 Agent 回复成对，
        构成完整会话上下文）；② 组装编排器初始 state；③ 在 run_scope + orchestrator.run span
        观测范围内 await 编排图（M7/D3 接入点①：每轮建 run_id + root span，覆盖 plan/dispatch/wait/collect
        全程，run_id 由 dispatch_node 下发 worker）；④ 打印 final_reply。
        退出条件：输入 "exit"（不区分大小写）；输入为空则提示并重新读取。
        单轮异常被捕获后仅打印，不终止循环，以保证交互不中断。
        ★M23（D26）：整个循环包在 `try/finally` 中 —— **任何退出路径**（正常 `exit` /
        `input()` 上的 Ctrl+C（KeyboardInterrupt）/ 异常）都会 `await shutdown_system()`
        收敛 4 个 worker 与 3 个 MCP 常驻子进程。修复改造前"输入 exit 后 MCP 子进程
        必然泄漏（靠下次启动 `_cleanup_stale_mcp_ports` 事后清理兜底）"的缺陷。
    入参说明：
        无。
    返回值说明：
        无（结果直接打印到标准输出；函数在用户输入 exit 后返回）。
    """
    # 全局一次性初始化底层资源
    await bootstrap_all()
    # 从全局注册表拿到编排器，统一入口
    orchestrator_graph = AGENT_GRAPH["orchestrator_agent"]

    print("===== 消费管家智能体启动完成，输入exit退出对话 =====")
    # ★M23（D26）L2：CLI 入口必须收尾 ——
    #   4 个 agent worker 是**协程**（`asyncio.run` 结束时会被取消，天然收敛），
    #   但 3 个 MCP 常驻服务是 `Popen` 起的**独立子进程**，不显式关闭必然泄漏
    #   （改造前靠下次启动时 `_cleanup_stale_mcp_ports` 事后清理兜底）。
    #   `input()` 上的 Ctrl+C 会抛 KeyboardInterrupt，同样由本 finally 兜住。
    try:
        while True:
            user_input = input("用户：")
            if user_input.strip().lower() == "exit":
                print("程序退出")
                break
            if not user_input.strip():
                print("请输入有效内容")
                continue

            # 每轮先落库用户提问，与 collect_node 写入的 agent 回复成对，构成完整会话上下文
            get_session_memory(SESSION_ID).add_msg("user", user_input)

            # 初始化编排Agent状态，复用进程级固定会话（记忆多轮生效）
            init_state = {
                "user_input": user_input,
                "session_id": SESSION_ID,
                "session_history": "",
                "task_plan": {},
                "current_task": None,
                "all_task_results": [],
                "final_reply": None,
                "error_msg": None,
                "current_agent": "orchestrator_agent",
                # M16/M19：本轮终止标志与原因——入口统一初始化，不依赖节点侧兜底默认值
                "cancelled": False,
                "abort_reason": None,
            }

            try:
                # 执行编排Agent全流程（M7：D3 接入点①——每轮建 run_id + root span，
                #   覆盖 plan/dispatch/wait/collect 全程；run_id 由 dispatch_node 下发 worker）
                with run_scope(session_id=SESSION_ID):
                    with span("orchestrator.run", kind="orchestrator",
                              agent_id="orchestrator_agent"):
                        result_state = await orchestrator_graph.ainvoke(init_state)
                reply = result_state.get("final_reply", "无返回结果")
                print(f"智能体：{reply}\n")
            except Exception as e:
                print(f"【系统异常】执行失败：{str(e)}\n")
    finally:
        # M23（D26）：无论正常 `exit`、Ctrl+C 还是异常退出，都收敛后台资源
        await shutdown_system()
