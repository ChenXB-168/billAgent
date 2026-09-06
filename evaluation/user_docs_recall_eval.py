# -*- coding: utf-8 -*-
"""
用户文档库（userDocs/）召回率评测（L2，离线、无需 LLM 在线）——M8 专用评测入口。

历史：M0（2026-08-30）数据源解耦时，本组 GT 曾放在 rag_recall_eval.py 的 userdocs 源；
M8 交付 `memory/user_docs.py`（`10` §5.5 A 方案）后，旧 RAG（ragKnowledge/）整体退役，
评测收口为本文件（userdocs 唯一数据源），旧文件 rag_recall_eval.py 随 RAW_KNOWLEDGE 下线删除。

衡量方法（三个指标，分别回答三个问题）：
  - hit@k  命中率  ：查询的 top-k 结果里有没有出现期望文档。回答"用户问得对，能不能问到东西"
  - recall@k 召回率 ：多个期望文档时，期望文档被找回的比例。回答"该给的知识给全了没有"
  - MRR@k  平均倒数排名：第一个命中文档排得越靠前越好。回答"排得准不准、答案质量高不高"

评测集（ground truth）：查询 -> 期望命中的用户笔记原文（与 userDocs/ 样例**逐字一致**）。
判定方式：期望文档 == 召回结果字符串（完全匹配）。
可用集：iter_doc_chunks()（与索引构建同源——防"评测可用但检索不可得"，走查结论第 4 行）。

【GT 缺失即失败】期望文档不在数据源中 → 硬失败退出（M0 起；原为静默不计入导致评测静默失效，
见 09 §7.1）。确为诊断需要时 --allow-missing 降级为警告。

用法：
    python evaluation/user_docs_recall_eval.py               # 默认 userdocs 源
    python evaluation/user_docs_recall_eval.py --max-k 5     # 指定最大考察深度
    python evaluation/user_docs_recall_eval.py --allow-missing
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

# 项目根目录加入 sys.path，保证直接运行可用
sys.path.insert(0, str(Path(__file__).parent.parent))


def _retrieve_userdocs(query, top_k):
    """新数据源：用户文档库 userDocs/（`memory/user_docs.py`，M8 交付）"""
    from memory.user_docs import search_user_docs, iter_doc_chunks
    results = search_user_docs(query, top_k=top_k)
    return [r["text"] for r in results], set(iter_doc_chunks())


# ===================== 评测集（查询 -> 期望命中的用户笔记原文） =====================
# 内容必须与 userDocs/ 下样例文件的原文**完全一致**（判定为字符串完全匹配）。
# 各 GT 语义与旧 RAG 内置规则（已随 M8 下线）一一对应，衡量用户资料检索质量。
GT_CASES_USERDOCS = OrderedDict([
    ("我平时午饭花多少钱", ["我午饭一般在公司楼下的快餐店吃，人均 25 到 30 元，偶尔和同事下馆子会到 50 元以上。"]),
    ("吃饭大概什么价位", ["我午饭一般在公司楼下的快餐店吃，人均 25 到 30 元，偶尔和同事下馆子会到 50 元以上。"]),
    ("通勤怎么省钱", ["工作日通勤坐地铁，单程 4 元，一天来回 8 元左右，偶尔赶时间才打车。"]),
    ("每天交通花多少", ["工作日通勤坐地铁，单程 4 元，一天来回 8 元左右，偶尔赶时间才打车。"]),
    ("出差住酒店什么标准", ["出差住酒店我一般选 200 到 300 元的连锁，太贵了报销麻烦，太便宜的住着不舒服。"]),
    ("怎么控制冲动消费", ["买东西我习惯先加购物车放一晚，第二天还想要再买，冲动消费少了很多。"]),
    ("娱乐方面我怎么花的", ["娱乐方面我主要花在游戏和电影上，一个月大概 200 元，超过这个数我会控制一下。"]),
    ("我怎么给消费分档", ["消费上我分三档：能省则省的日常开销、必要的固定支出、偶尔才有的享受型消费。"]),
    ("预算超支了怎么办", ["我给自己定的月度预算是 4000 元，超支了就从下个月的娱乐和购物里扣回来。"]),
    ("我希望的消费占比是多少", ["我希望的月度消费占比是：餐饮 35%、交通 10%、购物 20%、住宿 15%、娱乐 10%，剩下的存起来。"]),
    ("多少钱算大额消费", ["对我来说，单笔超过 800 元就算大额消费了，付款前会多想一下是不是真的需要。"]),
    ("我想存下多少收入", ["我的目标是一个月存下收入的 20%，先攒够 6 个月的生活费作为应急金。"]),
    ("购物和娱乐怎么分配", [
        "买东西我习惯先加购物车放一晚，第二天还想要再买，冲动消费少了很多。",
        "娱乐方面我主要花在游戏和电影上，一个月大概 200 元，超过这个数我会控制一下。",
    ]),
])


def _hit_positions(results, expected, available):
    """返回 (positions, unevaluable)。

    positions: 每个期望文档的 1-based 命中位置；未命中为 None。
    unevaluable: 期望文档不在数据源中时为 True（不可评估，不计入指标）。
    """
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


def _collect(gt_cases, retrieve, max_k):
    """按评测集逐条检索，返回 (rows, available)"""
    rows, available = [], set()
    for query, expected in gt_cases.items():
        results, docs = retrieve(query, max_k)
        available = set(docs)
        positions, unevaluable = _hit_positions(results, expected, available)
        rows.append((query, expected, positions, unevaluable, results))
    return rows, available


def _report(rows, max_k):
    """打印明细 + 汇总指标 + 问题定位（输出格式与 M0 版保持一致，便于跨版本对比）"""
    ks = [k for k in (1, 3, 5) if k <= max_k]
    if not ks:
        ks = [max_k]

    # ===================== 逐查询明细 =====================
    print("=" * 80)
    print(f"{'#':<3}{'查询':<22}{'期望':<5}{'命中位置':<24}top1 结果")
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
        q = query if len(query) <= 20 else query[:19] + "~"
        top1 = results[0][:20] + "..." if results and len(results[0]) > 20 else (results[0] if results else "(空)")
        print(f"{i:<3}{q:<22}{n_eval:<5}{mark_str:<24}{top1}")
    print("=" * 80)

    # ===================== 汇总指标 =====================
    evals = [(q, e, ps, u) for q, e, ps, u, _ in rows]
    n_uneval_all = sum(1 for _, _, _, u in evals if all(u))
    total_exp = sum(1 for _, _, _, u in evals for uu in u if not uu)
    print(f"可评估查询: {len(evals) - n_uneval_all} 条（共 {len(evals)} 条，{n_uneval_all} 条期望不可评估，"
          f"可评估期望文档共 {total_exp} 条）\n")
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


def run(max_k, allow_missing):
    gt_cases = GT_CASES_USERDOCS
    print()
    print("#" * 80)
    print(f"# 数据源: userdocs（用户文档库）      用例数: {len(gt_cases)}")
    print("#" * 80)

    rows, available = _collect(gt_cases, _retrieve_userdocs, max_k)
    print(f"当前索引文档数(chunk 数): {len(available)}")

    # ===================== GT 缺失硬断言 =====================
    missing = {d for _, exps in gt_cases.items() for d in exps} - available
    if missing:
        print(f"[警告] {len(missing)} 条期望文档不在数据源中：")
        for d in missing:
            print(f"    - {d[:40]}...")
        if not allow_missing:
            raise SystemExit(
                f"\n[失败] {len(missing)} 条 GT 不在数据源中 → **GT 与 userDocs/ 样例不同步**。\n"
                "       M0 起此为硬失败（原为静默不计入，导致评测静默失效，见 09 §7.1）。\n"
                "       确为诊断需要时，加 --allow-missing 降级为警告。"
            )
        print("[提示] --allow-missing 已开启：上述缺失不计入指标，结果为降级统计。\n")

    _report(rows, max_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用户文档库召回率评测（L2，M8 专用入口）")
    parser.add_argument("--max-k", type=int, default=5, help="最大考察深度（默认 5）")
    parser.add_argument("--allow-missing", action="store_true",
                        help="诊断用：GT 缺失时仅警告不退出（默认硬失败）")
    args = parser.parse_args()
    run(args.max_k, args.allow_missing)
