# -*- coding: utf-8 -*-
"""端到端耗时基准（benchmark）

【测什么】
    一次完整对话的耗时：从编排器接收用户输入，到产出 final_reply。
    链路 = 意图识别 → 任务派发 → 子 Agent 执行（含工具调用与模型调用）→ 结果汇总。
    ★不含 bootstrap 启动时间（启动单独计时并打印），便于和"端到端耗时"口径区分。

【为什么分开测启动与对话】
    MCP 服务常驻化之前，启动阶段要付 26~29s/服务的冷启动；
    常驻化之后冷启动只在启动时付一次。混在一起测会把"一次性成本"和"每次成本"混淆。

【怎么读结果】
    P50 / P95 / P99 为分位数；同时给平均与最大。
    单次对话耗时受外部模型响应波动影响较大，故用分位数而非只看平均。

用法：
    python bench/bench_e2e.py                  # 每个场景默认 3 次
    python bench/bench_e2e.py --n 5            # 自定义每场景次数
    python bench/bench_e2e.py --case multi     # 只跑多任务场景（multi|single|all）

注意：
    - 会真实拉起 bootstrap（含 3 个 MCP 子进程）并调用外部/本地模型；
    - 记账类用例会真实写库（bill.db），如需干净数据请自行清理；
    - 每轮使用独立 session_id，避免多轮历史污染。
"""
import argparse
import asyncio
import math
import statistics
import sys
import time
from pathlib import Path

# 支持脚本放在 bench/ 子目录下直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Windows 控制台中文编码根治
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from startup.bootstrap import bootstrap_all, AGENT_GRAPH  # noqa: E402
from mcpGateway.client import mcp_client  # noqa: E402

RUN_ID = "bench-e2e"

# 场景清单：key -> (展示名, 用户输入)
CASES = {
    # README §6 的"满汉全席"场景：一句话含 2 笔记账 + 统计 + 比价
    "multi": ("多任务（记账×2 + 统计 + 比价）",
              "记午饭30，记打车20，这个月花了多少，打车贵不贵"),
    # 单任务基线：只有记账，用于对比"调度开销"有多大
    "single": ("单任务（仅记账）",
               "记晚饭45元 [e2e基准]"),
}


def _build_state(text: str, session_id: str) -> dict:
    """构造一份完整的编排器初始 state（与 smoke_test.py 同构）。"""
    return {
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


async def run_once(text: str, session_id: str) -> dict:
    """跑一次完整对话，返回 {elapsed, reply, error}。"""
    state = _build_state(text, session_id)
    t0 = time.perf_counter()
    result = await AGENT_GRAPH["orchestrator_agent"].ainvoke(state)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed": elapsed,
        "reply": (result.get("final_reply") or "")[:80],
        "error": result.get("error_msg"),
    }


def _pct(sorted_samples, p):
    """最近秩法（nearest-rank）取分位数，返回秒。"""
    k = max(0, math.ceil(p / 100 * len(sorted_samples)) - 1)
    return sorted_samples[k]


def _summarize(samples: list) -> dict:
    """对样本列表做统计。"""
    s = sorted(samples)
    return {
        "n": len(s),
        "p50": _pct(s, 50),
        "p95": _pct(s, 95),
        "p99": _pct(s, 99),
        "mean": statistics.mean(s),
        "maxv": s[-1],
        "minv": s[0],
    }


async def main():
    parser = argparse.ArgumentParser(description="端到端耗时基准")
    parser.add_argument("--n", type=int, default=3, help="每个场景的样本数（默认 3）")
    parser.add_argument("--case", type=str, default="all",
                        choices=["all", "multi", "single"], help="跑哪个场景")
    parser.add_argument("--out", type=str, default="bench/bench_e2e_result.txt",
                        help="结果输出文件，默认 bench/bench_e2e_result.txt")
    args = parser.parse_args()

    keys = list(CASES.keys()) if args.case == "all" else [args.case]

    lines = []

    def emit(s=""):
        """同时输出到控制台与结果文件（避免 Windows 控制台编码问题）。"""
        lines.append(s)
        try:
            print(s)
        except Exception:
            pass

    emit("=" * 78)
    emit(f"端到端耗时基准 | 每场景 {args.n} 次 | run_id={RUN_ID}")
    emit("=" * 78)

    # ── 启动计时（一次性成本，与"每次对话耗时"分开统计）──
    t_boot = time.perf_counter()
    try:
        await bootstrap_all()
    except Exception as e:
        emit(f"[启动失败] {type(e).__name__}: {e}")
        return
    boot_elapsed = time.perf_counter() - t_boot
    emit(f"启动耗时（bootstrap + 3 个 MCP 服务就绪）: {boot_elapsed:.2f} s")
    emit("-" * 78)

    try:
        for key in keys:
            label, text = CASES[key]
            emit(f"\n【场景】{label}")
            emit(f"【输入】{text}")
            samples = []
            for i in range(args.n):
                sid = f"{RUN_ID}-{key}-{i}"
                r = await run_once(text, sid)
                samples.append(r["elapsed"])
                flag = f"  <error: {r['error']}>" if r["error"] else ""
                emit(f"  第 {i + 1} 次: {r['elapsed']:.2f} s{flag}")
                if i == 0:
                    emit(f"  首次回复: {r['reply']}")

            st = _summarize(samples)
            emit(f"  --- P50 {st['p50']:.2f}s | P95 {st['p95']:.2f}s | "
                 f"平均 {st['mean']:.2f}s | 最大 {st['maxv']:.2f}s | 最小 {st['minv']:.2f}s")
    finally:
        await mcp_client.shutdown_all()

    emit("\n" + "=" * 78)
    emit("说明：以上耗时为「单次完整对话」，不含启动时间。")
    emit("=" * 78)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
