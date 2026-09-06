# -*- coding: utf-8 -*-
"""M7（D3）span 树查看脚本：`python utils/trace_view.py [run_id]`

- 无参：列出最近 10 次有记录的 run（trace_span 表），便于回忆/挑选 run_id
- 带 run_id：ASCII 打印 span 树（orchestrator.run → {agent}.run → tool.*），
  并附该 run 的 LLM 调用明细（llm_call_stat，token/耗时/成败）

设计依据：`设计/08_架构演进方案与决策记录.md` §3.4 Q10（offline_debug.py 风格命令行）；
          D3 验收"任一失败 L3 用例可导出 span 树定位环节"（`08` §3.1 / `11` §3 M7）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.tracer import load_recent_runs, load_spans, summary_by_run  # noqa: E402

# 树连接线（ASCII，规避 Windows 控制台 GBK 对 ├└│ 的编码问题）
_TEE = "+-- "
_LAST = "`-- "
_VLINE = "|   "
_SPACE = "    "


def _fmt_span(s: dict) -> str:
    """单行摘要：`{name} [kind:agent_id] {status} {duration_ms}ms`"""
    tag = f"[{s.get('kind') or '-'}:{s.get('agent_id') or '-'}]" if (s.get("kind") or s.get("agent_id")) else ""
    line = f"{s['name']} {tag} {s['status']} {int(s.get('duration_ms') or 0)}ms"
    if s.get("error"):
        line += f"\n{_SPACE}   ^ error: {s['error']}"
    return line


def render_span_tree(spans: list) -> str:
    """把 span 行渲染为 ASCII 树（CLI 与单测共用）。spans 须按 id 升序。

    多根场景（如无 run_id 的孤儿 span）按创建序并列渲染；真正的树根 = parent
    不在本 run 内的 span（编排入口正常会落一条 orchestrator.run 根）。
    """
    if not spans:
        return "（无 span 记录）"
    ids = {s["span_id"] for s in spans}
    children: dict = {}
    for s in spans:
        children.setdefault(s["parent_span_id"], []).append(s)
    roots = [s for s in spans if not s["parent_span_id"] or s["parent_span_id"] not in ids]
    lines: list = []
    for i, root in enumerate(roots):
        is_last = i == len(roots) - 1
        lines.append((_LAST if is_last else _TEE) + _fmt_span(root))
        _walk(lines, children, root["span_id"], "" if is_last else _VLINE)
    return "\n".join(lines)


def _walk(lines: list, children: dict, span_id: str, prefix: str) -> None:
    kids = children.get(span_id) or []
    for i, kid in enumerate(kids):
        is_last = i == len(kids) - 1
        lines.append(prefix + (_LAST if is_last else _TEE) + _fmt_span(kid))
        _walk(lines, children, kid["span_id"], prefix + (_SPACE if is_last else _VLINE))


def print_run(run_id: str) -> None:
    spans = load_spans(run_id)
    print(f"===== run_id: {run_id} =====")
    print(render_span_tree(spans))
    llms = summary_by_run(run_id)
    if llms:
        print("\n-- LLM 调用明细（llm_call_stat）--")
        for r in llms:
            tokens = int(r.get("total_tokens") or 0)
            ok = "ok" if r.get("success") else f"FAIL({r.get('error_type') or '?'})"
            print(
                f"  {r['created_at']} agent={r['agent_tag']:<16} {r['channel']:<8} "
                f"{r['model']:<28} token={tokens:<7} {int(r.get('latency_ms') or 0)}ms {ok}"
            )
    else:
        print("\n（无 LLM 调用记录——该 run 未产生模型调用或 run_id 未跨进程贯通）")


def main() -> None:
    if len(sys.argv) < 2:
        rows = load_recent_runs()
        if not rows:
            print("（暂无 span 记录——先跑一次真实对话，或确认 TRACER_ENABLED=1）")
        else:
            print("最近 run（span 有记录，按创建倒序）:")
            for r in rows:
                print(f"  {r['run_id']}  {r['started_at']}  "
                      f"spans={r['span_cnt']}  errors={r['error_cnt']}")
        print("\n用法: python utils/trace_view.py <run_id>   # 查看某次 run 的 span 树")
        return
    print_run(sys.argv[1])


if __name__ == "__main__":
    main()
