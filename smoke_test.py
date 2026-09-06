# -*- coding: utf-8 -*-
"""
端到端联调冒烟脚本（临时文件，联调完成后删除）
用法：python smoke_test.py "用户输入" [session_id]
"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from startup.bootstrap import bootstrap_all, AGENT_GRAPH
from mcpGateway.client import mcp_client


async def run_case(text: str, session_id: str):
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
    text = sys.argv[1] if len(sys.argv) > 1 else "晚餐52元"
    session_id = sys.argv[2] if len(sys.argv) > 2 else "smoke-001"
    try:
        await bootstrap_all()
        await run_case(text, session_id)
    finally:
        await mcp_client.shutdown_all()


if __name__ == "__main__":
    asyncio.run(main())
