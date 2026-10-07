# -*- coding: utf-8 -*-
"""
端到端联调冒烟脚本（临时文件，联调完成后删除）
用法：python smoke_test.py "用户输入" [session_id]

注意：本脚本会真实拉起 bootstrap（含 MCP 子进程与模型服务），
      不依赖 pytest，用于人工在终端快速验证单条输入的全链路表现。
"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from startup.bootstrap import bootstrap_all, AGENT_GRAPH
from mcpGateway.client import mcp_client


async def run_case(text: str, session_id: str):
    """
    函数功能与逻辑描述：
        构造一份完整的编排器初始 state 并跑一次端到端对话，随后打印回复与关键状态：
        final_reply（最终回复）、error_msg（若存在）、task_plan 各任务的 status 汇总。
        异常不向上抛：捕获后打印 traceback 与异常摘要，保证冒烟脚本本身不因单条用例失败而中断。
    入参说明：
        text (str)：模拟的用户输入文本。
        session_id (str)：会话标识，用于短期记忆与草稿缓存的隔离。
    返回值说明：
        无（结果全部打印到标准输出，包含 "=" * 90 分隔线）。
    """
    init_state = {
        "user_input": text,
        "session_id": session_id,
        "session_history": None,
        "task_plan": {},
        "current_task": None,
        "all_task_results": [],
        "final_reply": None,
        "error_msg": None,
        "current_agent": "orchestrator_agent",
        "dispatch_round": 0,
        "current_sub_agent_struct": None,
        "agent_result_cache": {},
    }
    print("=" * 90)
    print(f"【场景】{text}")
    try:
        result = await AGENT_GRAPH["orchestrator_agent"].ainvoke(init_state)
        reply = result.get("final_reply") or "(无 final_reply)"
        print(f"【回复】{reply}")
        if result.get("error_msg"):
            print(f"【error_msg】{result['error_msg']}")
        tp = result.get("task_plan", {})
        if tp:
            print("【task_plan 状态】" + ", ".join(f"{k}={v['status']}" for k, v in tp.items()))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"【异常】{type(e).__name__}: {e}")
    print("=" * 90)


async def main():
    """
    函数功能与逻辑描述：
        脚本入口：从命令行取用户输入与会话标识（缺省分别为 "晚餐52元" 与 "smoke-001"），
        先 bootstrap 全量资源，再执行单条用例，最后在 finally 中关闭全部 MCP 客户端连接，
        保证进程不会因遗留子进程而挂住。
    入参说明：
        无（从 sys.argv 读取：argv[1] 为输入文本，argv[2] 为 session_id，均可省略）。
    返回值说明：
        无（结果由 run_case 打印；资源清理在 finally 中完成）。
    """
    text = sys.argv[1] if len(sys.argv) > 1 else "晚餐52元"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "smoke-001"
    try:
        await bootstrap_all()
        await run_case(text, session_id)
    finally:
        await mcp_client.shutdown_all()


if __name__ == "__main__":
    asyncio.run(main())
