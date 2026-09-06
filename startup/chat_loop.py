import asyncio
import uuid
from startup.bootstrap import bootstrap_all, AGENT_GRAPH
from memory.short_memory import get_session_memory
from utils.tracer import run_scope, span

# 进程级固定会话ID：多轮对话复用同一session，保证短时记忆与草稿缓存生效
SESSION_ID = str(uuid.uuid4())


async def run_chat():
    # 全局一次性初始化底层资源
    await bootstrap_all()
    # 从全局注册表拿到编排器，统一入口
    orchestrator_graph = AGENT_GRAPH["orchestrator_agent"]

    print("===== 消费管家智能体启动完成，输入exit退出对话 =====")
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
            "current_agent": "orchestrator_agent"
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

