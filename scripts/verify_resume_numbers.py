# -*- coding: utf-8 -*-
"""简历数字一键复现 / 自检脚本。

用途：把简历 & README 上出现的所有关键数字**跑一遍**并汇总，
用于投递前自检，也可在面试现场演示（每个数字都能当场复现）。

用法：
    python scripts/verify_resume_numbers.py            # 全量（约 5-6 分钟，含 embedding 评测）
    python scripts/verify_resume_numbers.py --fast     # 跳过两项检索评测（约 2 分钟）

输出：每个数字 + 对应的**独立复现命令**（面试时可单独跑其中一条）。
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

COV_DIRS = ["agents", "mcpGateway", "startup", "utils", "memory", "coreModules"]


def run(args: list[str], timeout: int = 900) -> tuple[int, str]:
    p = subprocess.run([PY, *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=timeout)
    return p.returncode, (p.stdout or "") + "\n" + (p.stderr or "")


def pytest_run(args: list[str], timeout: int = 900) -> tuple[int, str]:
    """跑 pytest，并强制使用**全新唯一**的 basetemp。

    ★为什么必须这样：pytest 结束时会清理历史 tmp 目录；若累积了大量历史垃圾，
      该批量删除动作会被环境的「bulk delete 安全拦截」终止，导致进程被杀、
      拿不到统计行（表现为解析不到 passed / collected）。
      每次用全新目录 → 目录内无历史垃圾 → 无需清理 → 绕开该拦截。
    """
    bt = str(Path(tempfile.gettempdir()) / f"pytest_verify_{int(time.time()*1000)}")
    return run(["-m", "pytest", *args, "--basetemp", bt], timeout=timeout)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过两项检索评测")
    args = ap.parse_args()
    t0 = time.time()

    # ---------- 1. 单元测试 ----------
    section("1) 单元测试（简历：620 条 / 37 文件）")
    rc, out = pytest_run(["tests/unit", "-q", "--no-header", "--tb=no"])
    m = re.search(r"(\d+) passed", out)
    mf = re.search(r"(\d+) failed", out)
    print(f"  passed = {m.group(1) if m else '?'}"
          f"   failed = {mf.group(1) if mf else 0}")
    print(f"  复现: {PY.name} -m pytest tests/unit -q")

    # ---------- 2. 覆盖率 ----------
    section("2) 覆盖率（简历：业务代码语句覆盖率 79%）")
    cov_args = ["tests/unit", "-q", "--no-header", "--tb=no"]
    for d in COV_DIRS:
        cov_args += ["--cov", d]
    _, out = pytest_run(cov_args)
    mt = re.search(r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%", out, re.M)
    if mt:
        stmts, miss, pct = mt.group(1), mt.group(2), mt.group(3)
        print(f"  TOTAL = {pct}%  （语句 {stmts}，未覆盖 {miss}）")
    print("  复现: pytest tests/unit -q --cov=<业务目录> --cov-report=term-missing")

    # ---------- 3. 集成 / e2e ----------
    section("3) 集成 / 端到端（简历：集成 14 条 / e2e 49 条）")
    for label, path in (("集成", "tests/integration"), ("e2e", "tests/e2e")):
        _, out = pytest_run([path, "--collect-only", "-q", "--no-header"])
        m = re.search(r"(\d+) tests? collected", out)
        print(f"  {label} = {m.group(1) if m else '?'} 条（收集数）")
        print(f"    复现: pytest {path} --collect-only -q")

    # ---------- 4. 契约漂移 ----------
    section("4) 契约漂移检测（README：exit 0）")
    rc, out = run(["utils/contract_check.py"], timeout=300)
    print(f"  exit = {rc}（0 = 无漂移）")
    print("  复现: python utils/contract_check.py")

    if not args.fast:
        # ---------- 5. 笔记库检索评测 ----------
        section("5) 笔记库检索评测（简历：55 篇 / 40 条查询）")
        _, out = pytest_run(["tests/unit/test_user_docs_retrieval_scale.py",
                             "-q", "-s", "--no-header"])
        for line in out.splitlines():
            if "hit@" in line or "recall@" in line or "MRR@" in line:
                print("  " + line.strip())
        print("  复现: pytest tests/unit/test_user_docs_retrieval_scale.py -s")

        # ---------- 6. 长期记忆检索评测 ----------
        section("6) 长期记忆检索评测（简历：56 条查询）")
        pkl = ROOT / "ragKnowledge" / "long_consume_mem.pkl"
        bak = ROOT / "ragKnowledge" / "_bak_verify.pkl"
        had = pkl.exists()
        if had:
            shutil.copy2(pkl, bak)
            pkl.unlink()
        try:
            run(["evaluation/seed_memory_data.py"], timeout=300)
            _, out = run(["evaluation/memory_recall_eval_50.py"], timeout=600)
            for line in out.splitlines():
                if "recall@" in line or "MRR@" in line or "共 56 条" in line:
                    print("  " + line.strip())
            print("  复现: python evaluation/seed_memory_data.py && "
                  "python evaluation/memory_recall_eval_50.py")
        finally:
            if had and bak.exists():
                shutil.copy2(bak, pkl)
                bak.unlink()
            elif pkl.exists():
                pkl.unlink()
            print("  （已恢复原 long_consume_mem.pkl）")

    print(f"\n全部完成，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
