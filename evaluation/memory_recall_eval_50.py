# -*- coding: utf-8 -*-
"""
扩大样本的长期记忆召回评测（56 条）。

为什么需要它：memory_recall_eval.py 只有 10 条查询，样本太小，100% 的召回率
置信度不足。本脚本构造 56 条查询，覆盖 7 大类表达，期望答案严格对齐
evaluation/seed_memory_data.py 中 8 条记忆的真实数据（金额/极值/月份）。

类别说明（用于定位盲区，不影响整体统计）：
  normal   规则可检索（精确月份/相对时间/分类关键词），应全部命中
  compare  跨月对比，"哪个更高"语义
  extreme  极值语义（最多/最少/最高/最低），向量检索天然比不了精确极值
  vague    无锚点宽泛查询（该返回多条，评测集定单期望本身勉强）
  hanzi    汉字数字月份（"一月/十二月"），_extract_time_tokens 目前只支持阿拉伯数字

用法：
    python evaluation/seed_memory_data.py          # 先确保 8 条记忆已落盘
    python evaluation/memory_recall_eval_50.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import OrderedDict
from memory.long_memory import search_history_consume
from evaluation.seed_memory_data import MOCK_SUMMARIES as M

# (查询, 期望记忆, 类别)
CASES = [
    # ========== 精确月份 + 消费意图（11条） ==========
    ("我1月总支出多少", M[0], "normal"),
    ("我2月消费了多少", M[1], "normal"),
    ("我3月花了多少钱", M[2], "normal"),
    ("我4月开销是多少", M[3], "normal"),
    ("我5月总支出多少", M[5], "normal"),
    ("我6月消费了多少", M[6], "normal"),
    ("我7月花了多少钱", M[7], "normal"),
    ("我2026年1月消费情况", M[0], "normal"),
    ("我2025年12月总支出多少", M[4], "normal"),
    ("我12月消费了多少", M[4], "normal"),
    ("今年1月的开销是多少", M[0], "normal"),
    # ========== 分类 + 月份组合（10条） ==========
    ("我2月餐饮花了多少", M[1], "normal"),
    ("我6月交通费多少", M[6], "normal"),
    ("我1月娱乐支出了多少", M[0], "normal"),
    ("我5月房租是多少", M[5], "normal"),
    ("我4月购物花了多少钱", M[3], "normal"),
    ("我3月餐饮消费多少", M[2], "normal"),
    ("我2025年12月购物开销", M[4], "normal"),
    ("我7月娱乐花了多少", M[7], "normal"),
    ("我2月购物用了多少", M[1], "normal"),
    ("我6月餐饮多少钱", M[6], "normal"),
    # ========== 分类宽泛（不过滤月份）（4条） ==========
    ("我每个月房租多少", M[0], "normal"),
    ("我的交通费是不是很高", M[6], "normal"),
    ("餐饮一个月大概花多少", M[1], "normal"),
    ("哪个月房租最低", M[5], "extreme"),
    # ========== 相对时间（8条） ==========
    ("我上个月消费了多少", M[7], "normal"),
    ("我最近一个月的消费情况", M[7], "normal"),
    ("我去年12月花了多少", M[4], "normal"),
    ("我今年1月支出多少", M[0], "normal"),
    ("上上个月我花了多少", M[6], "normal"),
    ("最近三个月的消费总额", M[6], "vague"),
    ("我去年12月的开销", M[4], "normal"),
    ("今年6月的消费", M[6], "normal"),
    # ========== 极值语义（10条） ==========
    ("哪个月总支出最少", M[5], "extreme"),
    ("哪个月总支出最多", M[4], "extreme"),
    ("哪个月份餐饮花的多", M[1], "extreme"),
    ("娱乐花的最多的月份", M[4], "extreme"),
    ("交通费最高的月份", M[6], "extreme"),
    ("哪个月交通费最低", M[2], "extreme"),
    ("购物花的最多的月份", M[4], "extreme"),
    ("哪个月光吃饭就花最多", M[1], "extreme"),
    ("哪个月消费最多", M[4], "extreme"),
    ("哪个月花钱最少", M[5], "extreme"),
    # ========== 跨月对比（4条） ==========
    ("2月和1月哪个餐饮花的多", M[1], "compare"),
    ("6月和3月哪个月总支出高", M[6], "compare"),
    ("12月和7月哪个购物花的多", M[4], "compare"),
    ("5月和4月哪个月总支出少", M[5], "compare"),
    # ========== 双分类宽泛（3条） ==========
    ("餐饮和交通大概花了多少", M[0], "vague"),
    ("我的餐饮和娱乐开销", M[0], "vague"),
    ("房租和餐饮哪个占比高", M[0], "vague"),
    # ========== 汉字数字月份（3条，可能暴露新盲区） ==========
    ("我一月总支出多少", M[0], "hanzi"),
    ("去年十二月购物花了多少", M[4], "hanzi"),
    ("七月我用了多少钱", M[7], "hanzi"),
    # ========== 同义改写（3条） ==========
    ("我6月份消费了多少", M[6], "normal"),
    ("1月份的开销", M[0], "normal"),
    ("7月我的支出是多少", M[7], "normal"),
]

# 期望记忆必须真实存在于 seed 数据，防手误
for q, exp, cat in CASES:
    assert exp in M, f"期望记忆不在 seed 数据中: {q}"


def run(queries, max_k=5):
    pos_map = {}
    for q, exp, _ in queries:
        raw = search_history_consume(q, top_k=max_k)
        results = [r for r in raw.split("\n") if r.strip()] if raw else []
        pos_map[q] = results.index(exp) + 1 if exp in results else None
    return pos_map


def stats(queries, pos_map, title):
    n = len(queries)
    print(f"\n{title}（{n} 条）")
    for k in (1, 3, 5):
        hits = sum(1 for q, _, _ in queries if pos_map[q] is not None and pos_map[q] <= k)
        print(f"  recall@{k} = {hits}/{n} = {hits/n:.1%}")

    # MRR@1
    mrr1 = sum(1.0 / pos_map[q] for q, _, _ in queries if pos_map[q] is not None) / n
    print(f"  MRR@1     = {mrr1:.4f}")


def main():
    pos_map = run(CASES)
    total = len(CASES)

    print("=" * 72)
    print(f"扩大评测集：共 {total} 条查询，8 条记忆")
    print(f"类别分布: " + ", ".join(
        f"{cat}×{sum(1 for _,_,c in CASES if c==cat)}"
        for cat in ["normal", "compare", "extreme", "vague", "hanzi"]
    ))

    stats(CASES, pos_map, "\n【整体】")

    by_cat = OrderedDict()
    for q, exp, cat in CASES:
        by_cat.setdefault(cat, []).append((q, exp, cat))
    for cat, subset in by_cat.items():
        stats(subset, pos_map, f"【类别 {cat}】")

    print("\n" + "=" * 72)
    print("失败清单（@5 内未命中期望记忆）:")
    fails = [c for c in CASES if pos_map[c[0]] is None]
    if not fails:
        print("  无，全部 56 条均在 top5 内命中")
    for q, exp, cat in fails:
        print(f"  [{cat:<7}] {q}")

    print("\n@1 未命中但 top5 内找回（排序问题）:")
    for q, exp, cat in CASES:
        p = pos_map[q]
        if p is not None and p > 1:
            print(f"  [{cat:<7}] {q}  @{p}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
