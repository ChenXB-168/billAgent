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
    """
    函数功能与逻辑描述：
        把一条 span 记录格式化为单行 CLI 摘要文本，形如
        `{name} [kind:agent_id] {status} {duration_ms}ms`；当 kind 与 agent_id 均为空时省略方括号段。
        若该 span 带 error 字段，则追加第二行，以缩进 "^ error: ..." 呈现错误详情。
        持续时间取整为整数毫秒；duration_ms 缺失或为 None 时按 0 处理。
    入参说明：
        s (dict)：单条 span 记录字典，必含 name、status，
            可选 kind、agent_id、duration_ms、error。
    返回值说明：
        str：单行摘要文本；含 error 时为两行（第二行带缩进前缀）。
    """
    tag = f"[{s.get('kind') or '-'}:{s.get('agent_id') or '-'}]" if (s.get("kind") or s.get("agent_id")) else ""
    line = f"{s['name']} {tag} {s['status']} {int(s.get('duration_ms') or 0)}ms"
    if s.get("error"):
        line += f"\n{_SPACE}   ^ error: {s['error']}"
    return line


def render_span_tree(spans: list) -> str:
    """
    函数功能与逻辑描述：
        把扁平的 span 列表还原成 ASCII 树形文本，供 CLI 与单测共用，是对外唯一的渲染入口。
        先按 parent_span_id 建邻接表，再取「根」——即 parent_span_id 为空或父节点不在本批
        span 内的记录（正常一次 run 会落一条 orchestrator.run 作为根；无 run_id 的孤儿 span
        会产生多根）。多根场景按传入顺序并列渲染，兄弟节点用 ├/└ 连接线区分末节点，
        递归时按是否为末节点决定子层前缀续竖线还是空格。
        连接线采用 ASCII 字符（+-- / `-- / |），用于规避 Windows 控制台 GBK 编码对
        ├└│ 的乱码问题。调用方须保证 spans 已按 id 升序，否则同级顺序不稳定。
    入参说明：
        spans (list)：span 记录字典列表，每条需含 span_id、parent_span_id 及 _fmt_span 所需列；
            传入空列表表示无记录。
    返回值说明：
        str：渲染结果。spans 为空时返回固定文案 "（无 span 记录）"；否则为多行树形文本。
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
    """
    函数功能与逻辑描述：
        render_span_tree 的递归辅助：深度优先展开某一节点的全部子 span，把格式化后的行按顺序
        追加进 lines。每层依据「是否为最后一个子节点」决定续前缀（|   ）还是空白前缀，
        使竖线只在仍有后续兄弟的层级继续向下延伸。原地修改 lines，属副作用式设计；无子节点即空操作。
    入参说明：
        lines (list)：结果行列表，由调用方持有并被本函数就地追加。
        children (dict)：邻接表，结构为 parent_span_id -> 子 span 列表。
        span_id (str)：当前要展开其子节点的父 span ID。
        prefix (str)：当前层级已累积的前缀串（竖线与空格的组合）。
    返回值说明：
        无（结果通过就地修改 lines 输出，无其它副作用）。
    """
    kids = children.get(span_id) or []
    for i, kid in enumerate(kids):
        is_last = i == len(kids) - 1
        lines.append(prefix + (_LAST if is_last else _TEE) + _fmt_span(kid))
        _walk(lines, children, kid["span_id"], prefix + (_SPACE if is_last else _VLINE))


def print_run(run_id: str) -> None:
    """
    函数功能与逻辑描述：
        打印指定 run 的完整排查视图：先输出 span 树（load_spans + render_span_tree），
        再输出该 run 的 LLM 调用明细（summary_by_run，取自 llm_call_stat），字段含 created_at /
        agent_tag / channel / model / total_tokens / latency_ms 与成败。失败调用标注为
        "FAIL({error_type})"；token 与耗时缺失时按 0 处理，并按固定宽度对齐各列。
        无 LLM 记录时打印提示，并区分「该 run 未产生模型调用」与「run_id 未跨进程贯通」两种可能。
    入参说明：
        run_id (str)：要查看的 run 标识。
    返回值说明：
        无（结果直接打印到标准输出，不做返回）。
    """
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
    """
    函数功能与逻辑描述：
        脚本入口，按命令行参数分两种行为：
        1) 无参：列出最近 10 次有 span 记录的 run（run_id / started_at / spans / errors）供人工挑选，
           随后打印用法提示并结束；
        2) 带参：取第一个参数作为 run_id，交由 print_run 渲染 span 树与 LLM 调用明细。
        无记录时不抛异常，而是打印明确提示（提示先跑一次真实对话，或确认 TRACER_ENABLED=1）。
    入参说明：
        无（从 sys.argv 读取，argv[1] 为可选 run_id）。
    返回值说明：
        无（结果直接打印到标准输出，不做返回）。
    """
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
