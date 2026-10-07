# -*- coding: utf-8 -*-
"""
A2A 消息总线投递延迟基准测试（benchmark）

【测什么】
    投递延迟 = 编排器调用 send_task() 把任务放进队列的时刻
               -> 目标 Agent 的接收循环把它取出来的时刻。
    这是任务派发链路上最纯粹的一段开销，不涉及业务逻辑、数据库与模型调用，
    因此结果稳定、可反复复现。

【为什么要 A/B 对照】
    本项目 M2 改造把接收端从「轮询」改成「事件驱动」：
      改造前：get_nowait() 取不到就 sleep(0.1) 再试 —— 消息到达后最多要等 100ms 才被取走；
      改造后：await queue.get() 直接挂起      —— 消息到达即被唤醒，零等待且零空转。
    本脚本把这两种接收方式放在同一台机器、同一个队列上各跑一遍，得到可复现的对比数据。
    轮询模式的复刻依据见 mcpGateway/a2a_queue.py 中 recv_task_blocking 的 docstring。

【怎么保证公平】
    每条消息都等接收端确认收到后才发下一条（同步往返），
    因此测的是「空队列上单条消息的响应延迟」，不受消息积压影响。

【怎么读结果】
    P50 / P95 / P99 是分位数：把 N 次投递的耗时从小到大排序后取第 50 / 95 / 99 个，
    含义是「有 95% 的投递在此时间内完成」。用分位数而非平均值，
    是因为平均值会被极端值掩盖，分位数更能反映真实体验。

用法：
    python bench/bench_a2a_latency.py               # 默认每种模式 500 次
    python bench/bench_a2a_latency.py --n 1000      # 自定义样本数
"""
import argparse
import asyncio
import contextlib
import io
import math
import random
import statistics
import sys
import time
from pathlib import Path

# 支持脚本放在 bench/ 子目录下直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcpGateway.a2a_queue import A2AMessageBus  # noqa: E402

AGENT_ID = "bench_agent"
SESSION_ID = "bench_session"
T0_KEY = "_t0"

# 改造前生产接收口的轮询间隔；与 a2a_queue.py 中 recv_task_blocking 的
# docstring 记载一致（原实现为 get_nowait + sleep(0.1) 轮询）。
POLLING_INTERVAL = 0.1


async def _consume_event_driven(bus, n, samples, ack):
    """改造后接收方式：调用生产的 recv_task_blocking（内部 await queue.get()）。"""
    for _ in range(n):
        msg = await bus.recv_task_blocking(AGENT_ID, SESSION_ID)
        samples.append(time.perf_counter() - msg["task_data"][T0_KEY])
        ack.set()


async def _consume_polling(bus, n, samples, ack):
    """
    改造前接收方式复刻：get_nowait() 取不到就 sleep(POLLING_INTERVAL) 再试。
    现有代码已移除该实现，此处按 a2a_queue.py docstring 记载的原始逻辑复刻用于 A/B 对照。
    直接访问 _task_queues 属 benchmark 专用，业务代码不得效仿。
    """
    queue = bus._task_queues[AGENT_ID]
    for _ in range(n):
        while True:
            try:
                msg = queue.get_nowait()
                break
            except asyncio.QueueEmpty:
                await asyncio.sleep(POLLING_INTERVAL)
        samples.append(time.perf_counter() - msg["task_data"][T0_KEY])
        ack.set()


async def _produce(bus, n, ack, jitter=0.0):
    """
    发送端：调用真实的 send_task 接口投递消息，每条等接收端确认后才发下一条。
    jitter：发送前随机延时（0~jitter 秒）。用于模拟真实场景中「任务到达时刻相对
    接收端轮询周期是随机的」——否则消息会固定落在轮询间隙起点，使轮询端被系统性低估。
    两种接收模式使用相同的 jitter，保证对比公平。
    """
    ack.set()
    for i in range(n):
        await ack.wait()
        ack.clear()
        if jitter > 0:
            await asyncio.sleep(random.uniform(0, jitter))
        t0 = time.perf_counter()
        bus.send_task(AGENT_ID, SESSION_ID, f"bench-{i}", {T0_KEY: t0})


def _pct(sorted_samples, p):
    """最近秩法（nearest-rank）取分位数，返回毫秒。"""
    k = max(0, math.ceil(p / 100 * len(sorted_samples)) - 1)
    return sorted_samples[k] * 1000.0


def _summarize(samples):
    s = sorted(samples)
    return {
        "n": len(s),
        "p50": _pct(s, 50),
        "p95": _pct(s, 95),
        "p99": _pct(s, 99),
        "mean": statistics.mean(s) * 1000.0,
        "maxv": s[-1] * 1000.0,
    }


async def _run(mode, n):
    """跑一轮基准，返回统计结果。mode: 'polling' | 'event'。"""
    bus = A2AMessageBus()
    samples, ack = [], asyncio.Event()
    consumer = _consume_event_driven if mode == "event" else _consume_polling

    # 屏蔽总线内部调试日志：避免刷屏，同时排除 stdout I/O 对计时的干扰
    with contextlib.redirect_stdout(io.StringIO()):
        consumer_task = asyncio.create_task(consumer(bus, n, samples, ack))
        await asyncio.sleep(0)          # 让接收端先进入等待
        await _produce(bus, n, ack, jitter=POLLING_INTERVAL)
        await consumer_task

    return _summarize(samples)


async def main():
    parser = argparse.ArgumentParser(description="A2A 总线投递延迟基准")
    parser.add_argument("--n", type=int, default=300,
                        help="每种模式的样本数（默认 300；每条样本约耗时 150ms，300 条约 45 秒）")
    parser.add_argument("--out", type=str, default="bench/bench_result.txt",
                        help="结果输出文件（UTF-8），默认 bench/bench_result.txt")
    args = parser.parse_args()

    lines = []

    def emit(s=""):
        """同时输出到控制台与结果文件，避免 Windows 控制台编码问题导致结果丢失。"""
        lines.append(s)
        try:
            print(s)
        except Exception:
            pass

    emit("=" * 72)
    emit(f"A2A 投递延迟基准 | 样本数 {args.n} | 轮询间隔 {POLLING_INTERVAL * 1000:.0f} ms"
         f" | 同步往返 + 随机到达时刻")
    emit("=" * 72)

    results = {}
    for mode, label in (("polling", "轮询 (0.1s)"), ("event", "事件驱动")):
        r = await _run(mode, args.n)
        results[mode] = r
        emit(f"{label:<10}  P50 {r['p50']:>9.3f} ms"
             f"   P95 {r['p95']:>9.3f} ms"
             f"   P99 {r['p99']:>9.3f} ms"
             f"   平均 {r['mean']:>9.3f} ms")

    emit("-" * 72)
    for key, name in (("p50", "P50"), ("p95", "P95"), ("p99", "P99")):
        old, new = results["polling"][key], results["event"][key]
        emit(f"{name} 提升：{old / new:>7.1f}x   （{old:.3f} ms -> {new:.3f} ms）")
    emit("=" * 72)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())
