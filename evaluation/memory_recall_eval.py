# -*- coding: utf-8 -*-
"""
长期记忆 RAG 召回率评测脚本（离线、无需 LLM 在线）

衡量方法与 evaluation/rag_recall_eval.py 完全一致：
  - hit@k    命中率  ：查询的 top-k 里有没有出现期望记忆
  - recall@k 召回率  ：多个期望记忆时找回了几条
  - MRR@k    平均倒数排名：首个命中的位置有多靠前

评测对象：memory.long_memory.search_history_consume
  注意其两个特性（与静态知识库检索不同）：
    1. 返回的是 "\n" 拼接的字符串，评测时按 "\n" 拆分还原排名列表；
    2. 无匹配时"兜底返回最新 N 条"，这是记忆场景的设计行为，评测如实统计。

前置条件：先运行 evaluation/seed_memory_data.py 写入模拟数据。

用法：
    python evaluation/memory_recall_eval.py            # 评估 top1/3/5
    python evaluation/memory_recall_eval.py --max-k 3  # 指定最大考察深度
"""
import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.long_memory import LONG_MEM_PATH, search_history_consume  # noqa: E402
from evaluation.seed_memory_data import MOCK_SUMMARIES as M  # noqa: E402

# ===================== 评测集（查询 -> 期望命中的记忆原文） =====================
# 期望记忆直接引用 seed_memory_data.MOCK_SUMMARIES 的元素，保证与写入内容完全一致
# 查询模拟"用户原始输入"（plan_node 里 search_history_consume(user_text) 直接收原始语句，
# 不经过 LLM 改写），因此保留口语化表达，如实反映真实检索难度。
GT_CASES = OrderedDict([
    ("我1月总支出多少", [M[0]]),                 # 精确月份
    ("哪个月份餐饮花的多", [M[1]]),             # 餐饮最高：2月(1500)
    ("我6月消费了多少", [M[6]]),                 # 精确月份
    ("去年12月购物花了多少", [M[4]]),           # 购物最高：12月(3200)
    ("我每个月房租多少", [M[3]]),                # 单笔最大3000元(房租)
    ("娱乐花的最多的月份", [M[4]]),             # 娱乐最高：12月(1100)
    ("我交通费是不是很高", [M[6]]),             # 交通最高：6月(300)
    ("哪个月总支出最少", [M[5]]),               # 总支出最低：5月(2800)
    ("我最近一个月的消费情况", [M[7]]),          # 最新一条：7月（测试相对时间语义）
    ("餐饮和交通大概花了多少", [M[0]]),          # 常见查询：任意一条含餐饮交通的
])


def _hit_positions(results, expected, available):
    """返回 (positions, unevaluable)，逻辑与 rag_recall_eval 一致。"""
    positions, unevaluable = [], []
    for exp in expected:
        if exp not in available:
            positions.append(None)
            unevaluable.append(True)
        elif exp in results:
            positions.append(results.index(exp) + 1)
            unevaluable.append(False)
        else:
            positions.append(None)
            unevaluable.append(False)
    return positions, unevaluable


def main(max_k: int = 5):
    ks = [k for k in (1, 3, 5) if k <= max_k]
    if not ks:
        ks = [max_k]

    if not os.path.exists(LONG_MEM_PATH):
        print(f"[错误] 长期记忆文件不存在: {LONG_MEM_PATH}")
        print("请先运行: python evaluation/seed_memory_data.py")
        return

    available = set(M)
    missing = {d for _, exps in GT_CASES.items() for d in exps} - available
    if missing:
        print(f"[警告] {len(missing)} 条期望记忆不在模拟数据中（seed 与 eval 不同步?），将标注为不可评估:")
        for d in missing:
            print(f"    - {d[:40]}...")
        print()

    rows = []  # (query, expected, positions, unevaluable, results)
    for query, expected in GT_CASES.items():
        raw = search_history_consume(query, top_k=max_k)
        results = [r for r in raw.split("\n") if r.strip()] if raw else []
        positions, unevaluable = _hit_positions(results, expected, available)
        rows.append((query, expected, positions, unevaluable, results))

    # ===================== 逐查询明细 =====================
    print("=" * 80)
    print(f"{'#':<3}{'查询':<24}{'期望':<5}{'命中位置':<26}top1 结果")
    print("-" * 80)
    for i, (query, expected, positions, unevaluable, results) in enumerate(rows, 1):
        n_eval = sum(1 for u in unevaluable if not u)
        mark = []
        for k in ks:
            if n_eval == 0:
                mark.append("不可评估")
                break
            hit = any(p is not None and not u and p <= k for p, u in zip(positions, unevaluable))
            mark.append(f"[{'Y' if hit else 'X'}]@{k}")
        mark_str = " ".join(mark)
        q = query if len(query) <= 22 else query[:21] + "~"
        top1 = results[0][:22] + "..." if results and len(results[0]) > 22 else (results[0] if results else "(空)")
        print(f"{i:<3}{q:<24}{n_eval:<5}{mark_str:<26}{top1}")
    print("=" * 80)

    # ===================== 汇总指标 =====================
    evals = [(q, e, p, u) for q, e, p, u, _ in rows]
    n_uneval_all = sum(1 for _, _, _, u in evals if all(u))
    total_exp = sum(1 for _, _, _, u in evals for uu in u if not uu)
    print(f"可评估查询: {len(evals) - n_uneval_all} 条（共 {len(evals)} 条，期望记忆全部可评估）")
    print(f"可评估期望记忆共 {total_exp} 条\n")
    print(f"{'指标':<10}{' '.join(f'k={k:<6}' for k in ks)}")
    print("-" * 40)

    for metric in ["hit@k", "recall@k", "MRR@k"]:
        vals = []
        for k in ks:
            if metric == "hit@k":
                n = sum(1 for _, _, _, u in evals if not all(u))
                v = sum(1 for _, _, ps, u in evals
                        if not all(u) and any(p is not None and not uu and p <= k for p, uu in zip(ps, u))) / n if n else 0.0
            elif metric == "recall@k":
                hit_exp = sum(1 for _, _, ps, u in evals
                              for p, uu in zip(ps, u) if not uu and p is not None and p <= k)
                v = hit_exp / total_exp if total_exp else 0.0
            else:  # MRR@k
                n = sum(1 for _, _, _, u in evals if not all(u))
                s = 0.0
                for _, _, ps, u in evals:
                    first = min((p for p, uu in zip(ps, u) if not uu and p is not None), default=None)
                    s += 1.0 / first if (first is not None and first <= k) else 0.0
                v = s / n if n else 0.0
            vals.append(v)
        print(f"{metric:<10}{' '.join(f'{v:.2%}' if metric != 'MRR@k' else f'{v:.4f}' for v in vals)}")
    print("-" * 40)

    # ===================== 定位问题 =====================
    print("\n[定位] 在 k=5 下仍未命中的查询（建议重点检查）：")
    bad = [q for q, _, ps, u in evals if not all(u) and not any(p is not None and not uu and p <= 5 for p, uu in zip(ps, u))]
    if bad:
        for q in bad:
            print(f"    - {q}")
    else:
        print("    （无，全部可评估查询在 k=5 内命中）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="长期记忆 RAG 召回率评测")
    parser.add_argument("--max-k", type=int, default=5, help="最大考察深度（默认 5）")
    args = parser.parse_args()
    main(args.max_k)
