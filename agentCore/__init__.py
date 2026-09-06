# ============================================================
# agentCore 包（2026-09-04 单图遗留清理后：仅 parsers/json_parser.py 现役）
# ------------------------------------------------------------
# 当前采用 多智能体架构（LangGraph 编排 + A2A 进程内消息总线）：
#   main.py → startup/chat_loop.py → startup/bootstrap.py
#           → agents/orchestrator（编排） + agents/{bill,stat,price,finance}（执行）
#
# 单图遗留（graph/nodes/state/skills/guardrails/session_manager/intent_parser）已于
# 2026-09-04 随 LEGACY 引用链（offline_debug.py / memory/task_memory.py / 根级 prompts/）
# 一并删除。本包仅保留 parsers/json_parser.py —— `parse_json_output` 被 5 个现役
# agent + orchestrator 共用（带自校验+重试的 JSON 解析器），【必须保留】。
# ============================================================
