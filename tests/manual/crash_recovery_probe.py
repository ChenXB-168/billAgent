# -*- coding: utf-8 -*-
"""M5 强杀验证：覆盖三个中间态。

★为什么放在 `tests/manual/` 且**改名**（不再叫 `*_test.py`）：
  本脚本会**真实 SIGKILL 进程**。若被 pytest 自动收集，它会在每次回归时杀掉自己的子进程、
  破坏测试环境。故置于 `tests/manual/` 下并命名为 `crash_recovery_probe.py` ——
  **不匹配 pytest 默认的 `test_*.py` / `*_test.py` 收集规则**，
  仅在需要人工取证（如面试佐证、发布前验证）时手动执行。

★它验证的三个中间态（正是简历"崩溃恢复"可证据化的部分）：
  - A：任务未开始执行          → 重投执行
  - B：执行中、未写库          → 重投执行
  - C：账单已写、状态未更新    → 对账判 completed，**不重复记账**

为什么这样设计：
  - 中间态「任务被 worker 领取、正在执行中」窗口较宽 → 用**真实 SIGKILL** 验证；
  - 中间态「账单已写、状态未更新」窗口极窄（毫秒级），真实 kill 几乎打不中 →
    **构造该状态**后走**真实的 recover_orphans**，才能稳定验证对账分支；
  - 中间态「未落库、状态 running」同理构造。
  两者结合 = 既有真实强杀证据，又有对每个对账分支的确定覆盖。

用法（须在仓库根目录执行）：
    python tests\\manual\\crash_recovery_probe.py          # 主流程
    python tests\\manual\\crash_recovery_probe.py child    # 内部用：子进程跑一次多任务请求（等待被杀）
"""
import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

# ★本文件位于 tests/manual/ 下，故需上溯两级才是仓库根：
#   parents[0]=manual, parents[1]=tests, parents[2]=仓库根
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TASK_DB = ROOT / "database" / "task_state.db"
BILL_DB = ROOT / "database" / "bill.db"
TEXT = "记午饭30，记打车20，这个月花了多少，打车贵不贵"
SID = "kill-test"


def _state(text, sid):
    return {
        "user_input": text, "session_id": sid, "session_history": None,
        "task_plan": {}, "current_task": None, "all_task_results": [],
        "final_reply": None, "error_msg": None,
        "current_agent": "orchestrator_agent", "dispatch_round": 0,
        "current_sub_agent_struct": None, "agent_result_cache": {},
        "cancelled": False, "abort_reason": None,
    }


# ────────────────────────── 子进程：跑一次请求，等待被强杀 ──────────────────────────
async def _child():
    from startup.bootstrap import AGENT_GRAPH, bootstrap_all
    from utils.tracer import run_scope

    await bootstrap_all()
    print("[child] bootstrap 完成，开始执行请求", flush=True)
    with run_scope(session_id=SID):
        await AGENT_GRAPH["orchestrator_agent"].ainvoke(_state(TEXT, SID))
    print("[child] 请求执行完毕（若未被杀，说明杀晚了）", flush=True)


# ────────────────────────── 场景 A：真实 SIGKILL ──────────────────────────
def scenario_real_kill() -> dict:
    print("\n" + "=" * 72)
    print("场景 A · 真实 SIGKILL（覆盖中间态：任务已被 worker 领取、执行中）")
    print("=" * 72)

    # stdout 走 DEVNULL：子进程日志量大，用 PIPE 不读会填满缓冲区把子进程堵死
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "child"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 等 bootstrap(~40s) + 派发(~10s) → 此刻任务应处于 running
    waited = 0
    hit_running = False
    while waited < 75:
        time.sleep(5)
        waited += 5
        c = sqlite3.connect(TASK_DB)
        n = c.execute("SELECT COUNT(*) FROM task_state WHERE session_id=? AND status='running'",
                      (SID,)).fetchone()[0]
        c.close()
        if n:
            hit_running = True
            print(f"  t={waited}s  检测到 {n} 个 running 任务 → 立即 SIGKILL")
            break
        print(f"  t={waited}s  尚无 running 任务，继续等…")

    proc.kill()                       # Popen.kill() 即 SIGKILL（Windows 走 TerminateProcess）
    proc.wait(timeout=10)
    print(f"  已发送 SIGKILL（pid={proc.pid}），返回码={proc.returncode}")

    c = sqlite3.connect(TASK_DB)
    c.row_factory = sqlite3.Row
    rows = c.execute("SELECT task_id, agent_id, status FROM task_state WHERE session_id=?",
                     (SID,)).fetchall()
    c.close()
    print("  强杀后 task_state 残留：")
    for r in rows:
        print(f"    {r['task_id'][:16]}  {r['agent_id']:<15} {r['status']}")

    # ★重启验证：调用 bootstrap 启动时同一个 recover_orphans，看残留任务被如何处理。
    #   严格说"重启"应是新进程，但该函数是纯 DB 查询 + 状态修改，同进程调用逻辑等价。
    from startup.task_manager import get_task_manager

    print("\n  模拟重启（调用 bootstrap 同款 recover_orphans）：")
    st = get_task_manager().recover_orphans()
    print(f"    对账统计：{st}")

    c = sqlite3.connect(TASK_DB)
    c.row_factory = sqlite3.Row
    after = c.execute("SELECT task_id, agent_id, status FROM task_state WHERE session_id=?",
                      (SID,)).fetchall()
    c.close()
    print("    对账后状态：")
    for r in after:
        print(f"    {r['task_id'][:16]}  {r['agent_id']:<15} {r['status']}")
    print("    （bill_agent 不可重投：账单已写→completed；未写→failed。二者都不重复记账）")

    return {"hit_running": hit_running, "leftover": [dict(r) for r in rows],
            "after_restart": [dict(r) for r in after], "recover_stat": st}


# ────────────────────────── 场景 B/C/D：构造中间态 + 走真实对账 ──────────────────────────
def scenario_constructed() -> dict:
    print("\n" + "=" * 72)
    print("场景 B/C/D · 构造中间态后走真实 recover_orphans")
    print("=" * 72)

    from startup.task_manager import get_task_manager

    tm = get_task_manager()
    results = {}

    # 场景 B：任务在队列里（submitted），进程被杀 → 应重投
    tid_b = "t-killtest-submitted"
    tm.create_task(tid_b, SID, "stat_agent", {})          # 只读任务 → retriable=1

    # 场景 C：★最关键——账单已写、状态未更新 → 应对账判定 completed（不是重投！）
    tid_c = "t-killtest-bill-written"
    tm.create_task(tid_c, SID, "bill_agent", {})          # 写任务 → retriable=0
    tm.mark_running(tid_c)
    c = sqlite3.connect(BILL_DB)
    try:
        c.execute("INSERT INTO bill (amount, category, consume_time, remark, create_time, task_id) "
                  "VALUES (?,?,?,?,?,?)",
                  (12.5, "餐饮", "2026-10-08 12:00:00", "[kill-test] 构造：账单已写", time.time(), tid_c))
        c.commit()
        print(f"  构造 C：往 bill 表写入 task_id={tid_c[:16]} 的记录（模拟「账单已写」）")
    except Exception as e:                                # noqa: BLE001
        print(f"  ! 构造 C 插入 bill 失败：{e}")
    c.close()

    # 场景 D：状态 running 但账单没写（写任务且不可重投）→ 应判 failed（宁可告警不重复记账）
    tid_d = "t-killtest-no-write"
    tm.create_task(tid_d, SID, "bill_agent", {})
    tm.mark_running(tid_d)

    # 场景 E：stat 任务 running 但无副作用 → 应重投
    tid_e = "t-killtest-stat-running"
    tm.create_task(tid_e, SID, "stat_agent", {})
    tm.mark_running(tid_e)

    print("\n  构造完成，四个任务状态：")
    for tid, desc in ((tid_b, "submitted(stat)"), (tid_c, "running(bill,账单已写)"),
                      (tid_d, "running(bill,未写)"), (tid_e, "running(stat,未写)")):
        row = tm.get_task(tid)
        print(f"    {tid[:26]:<28} {row['status']:<10} retriable={row['retriable']}  {desc}")

    print("\n  调用 recover_orphans（真实对账逻辑）：")
    stat = tm.recover_orphans()
    print(f"    {stat}")

    print("\n  对账后状态：")
    for tid, expect in ((tid_b, "submitted"), (tid_c, "completed"),
                        (tid_d, "failed"), (tid_e, "submitted")):
        actual = tm.get_task(tid)["status"]
        flag = "✅" if actual == expect else "❌"
        print(f"    {tid[:26]:<28} {actual:<10} (期望 {expect}) {flag}")
        results[tid] = (actual, expect)

    return {"stat": stat, "results": results,
            "tids": (tid_b, tid_c, tid_d, tid_e)}


def _cleanup(tids=None):
    """清理本次验证写入的数据，不污染真实账单。"""
    if tids:
        c = sqlite3.connect(TASK_DB)
        c.execute("DELETE FROM task_state WHERE task_id IN (%s)" % ",".join("?" * len(tids)),
                  list(tids))
        c.commit()
        c.close()
        c = sqlite3.connect(BILL_DB)
        c.execute("DELETE FROM bill WHERE task_id LIKE 't-killtest-%'")
        c.commit()
        c.close()
    # 清掉场景 A 在 kill-test 会话下可能残留的任务
    c = sqlite3.connect(TASK_DB)
    c.execute("DELETE FROM task_state WHERE session_id=?", (SID,))
    c.commit()
    c.close()
    print("\n（已清理本次验证产生的 task_state / bill 记录）")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        asyncio.run(_child())
        return

    print("=" * 72)
    print("M5 强杀验证 · 覆盖三个中间态")
    print("=" * 72)

    a = scenario_real_kill()
    bcd = scenario_constructed()

    ok_bcd = all(actual == expect for actual, expect in bcd["results"].values())

    print("\n" + "=" * 72)
    print("结论")
    print("=" * 72)
    print(f"  场景 A 真实 SIGKILL：{'✅ 在任务执行中（running）成功打出 SIGKILL' if a['hit_running'] else '⚠️ 未在 running 窗口内命中'}")
    print(f"                       残留任务 {len(a['leftover'])} 个，状态：{[r['status'] for r in a['leftover']]}")
    print(f"  场景 B/C/D（构造 + 真实对账）：{'✅ 四个分支全部符合预期' if ok_bcd else '❌ 有分支不符'}")
    print(f"  对账统计：{bcd['stat']}")
    print("=" * 72)

    _cleanup(bcd["tids"])


if __name__ == "__main__":
    main()
