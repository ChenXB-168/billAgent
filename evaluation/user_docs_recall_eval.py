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
    """
    函数功能与逻辑描述：
        评测专用检索适配器：新数据源为用户文档库 userDocs/（`memory/user_docs.py`，M8 交付）。
        从该库检索与 query 相关的文档块，并返回检索结果文本列表与全量可用块集合；
        检索委托 search_user_docs，可用集来自 iter_doc_chunks()（与索引构建同源，
        保证「评测可用即检索可得」）。函数内延迟 import，避免模块加载期强依赖 memory 子系统。
    入参说明：
        query (str)：待检索的用户自然语言查询。
        top_k (int)：检索返回的最大文档块数量，透传为 search_user_docs 的 top_k。
    返回值说明：
        tuple：二元组 (texts, available)
            - texts (list[str])：检索命中的文档块正文，按相关度排序，长度 <= top_k。
            - available (set[str])：用户文档库全量文档块集合，用于判定 GT 是否可评估。
    """
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

    # ===================== 2026-10-09 扩充：覆盖并入的 55 篇语料 =====================
    # 起因：原 13 条 GT 只覆盖最初的 10 篇笔记，**完全不覆盖物价类**；语料增至 65 篇后，
    # 旧 GT 的规模已不足以反映真实检索质量（小样本虚高）。
    # 判定口径不变（原文完全匹配）；期望原文取各篇**首行**，与 `tests/fixtures/user_docs_notes/` 同步
    # （语料由 `tests/fixtures/_gen_user_docs_notes.py` 生成，勿手改）。
    # ---- 物价类（12 条）----
    ("苹果多少钱一斤", ["楼下水果店的苹果 5.8 元一斤，比上个月贵了 6 毛。"]),
    ("香蕉什么价格", ["香蕉一直是 3.5 元一斤，进口蕉要 5 元以上。"]),
    ("西瓜怎么卖", ["夏天西瓜 1.2 元一斤，一个十斤的西瓜 12 元左右。"]),
    ("青菜一斤多少钱", ["青菜、油麦菜一般 3 到 4 元一斤，雨天会涨到 6 元。"]),
    ("猪肉多少钱一斤", ["前腿肉 14 元一斤，五花肉 18 元一斤。"]),
    ("大米多少钱", ["常吃的东北大米 5 公斤装 42 元，折合 4.2 元一斤。"]),
    ("一顿早点花多少", ["公司楼下的包子 2 元一个，豆浆 2.5 元一杯。"]),
    ("外卖配送费多少", ["外卖配送费一般 3 到 5 元，雨天会涨到 8 元。"]),
    ("打车起步价多少", ["起步价 10 元含 3 公里，之后每公里 2.3 元。"]),
    ("地铁单程票价", ["地铁单程 4 元，公交 2 元，刷卡打九折。"]),
    ("房租一个月多少钱", ["我租的一居室 2200 元一个月，押一付三。"]),
    ("电费一度多少钱", ["电费每度 0.56 元，夏天开空调一个月要 150 元。"]),
    # ---- 消费习惯类（12 条）----
    ("怎么防止冲动消费", ["想买的非必需品我先加购物车放一晚，第二天还想要再买。"]),
    ("买东西怎么比价", ["买大件前我会在三个平台比价，再用比价插件看历史低价。"]),
    ("哪些东西适合囤货", ["纸巾、洗衣液这类消耗品我会在大促时囤半年的量。"]),
    ("视频会员怎么处理", ["视频会员我只在追剧时开一个月，看完就关自动续费。"]),
    ("外卖优惠券怎么领", ["点外卖前先领券，一般能省 3 到 8 元。"]),
    ("闲置怎么卖二手", ["闲置的电子产品我会挂二手平台，一般能回本四成。"]),
    ("分期付款划不划算", ["超过 2000 元的支出我会考虑免息分期，把手头现金流留住。"]),
    ("信用卡怎么还款", ["信用卡我只用一张，账单日设发薪日后三天，方便还款。"]),
    ("去超市前列清单吗", ["去超市前我会列清单，按清单买能少花三成。"]),
    ("促销活动有什么套路", ["「满 199 减 30」这类活动容易让人多买，我算过并不划算。"]),
    ("为了包邮凑单吗", ["为了凑包邮我常买一些用不上的小东西，后来改成直接付运费。"]),
    ("一周点几次外卖", ["我一周点三次外卖，其余自己做或者吃食堂。"]),
    # ---- 其他消费类（13 条）----
    ("每月想存多少收入", ["我的目标是一个月存下收入的 20%。"]),
    ("应急金准备多少", ["应急金我按每月必要开支 3500 元算，目标是 21000 元。"]),
    ("多少钱算大额消费", ["对我来说，单笔超过 800 元就算大额消费。"]),
    ("送礼预算多少", ["给家人的礼物我一般控制在 300 元以内。"]),
    ("旅游大概花多少", ["短途周边游两天一夜，预算 800 元左右。"]),
    ("养猫一个月多少钱", ["养猫一个月猫粮 150 元，猫砂 60 元。"]),
    ("健身卡多少钱", ["健身房年卡 1800 元，折合每月 150 元。"]),
    ("买书和课程预算", ["我每年留 1000 元买书和课程。"]),
    ("买了什么保险", ["我买了百万医疗险，一年保费 300 多元。"]),
    ("每天通勤怎么走", ["工作日通勤坐地铁，单程 4 元，一天来回 8 元。"]),
    ("过年花多少钱", ["过年给长辈红包一共 2000 元，年货 800 元。"]),
    ("换手机预算多少", ["我的手机用了三年，换机预算 2500 元。"]),
    ("月度消费占比是多少", ["我希望的月度消费占比是：餐饮 35%、交通 10%、购物 20%、住宿 15%、娱乐 10%。"]),
])


def _hit_positions(results, expected, available):
    """
    函数功能与逻辑描述：
        逐条判定期望文档的命中位置：若期望文档不在数据源可用集 available 中，则标记为
        不可评估（positions 记 None、unevaluable 记 True）；否则在检索结果 results 中
        按字符串完全匹配查找其 1-based 位置，未命中则 positions 记 None、unevaluable 记 False。
    入参说明：
        results (list[str])：单条查询的检索结果正文列表，按相关度降序。
        expected (list[str])：该查询期望命中的用户笔记原文列表（GT）。
        available (set[str])：用户文档库全量文档块集合，用于判定 GT 是否可评估。
    返回值说明：
        tuple：二元组 (positions, unevaluable)，两者等长且与 expected 一一对应
            - positions (list[Optional[int]])：每个期望文档的 1-based 命中位置；未命中或不可评估为 None。
            - unevaluable (list[bool])：True 表示该期望文档不在数据源中、不计入指标。
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
    """
    函数功能与逻辑描述：
        按评测集 gt_cases 逐条调用检索函数 retrieve(query, max_k)，对每条查询用
        _hit_positions 计算命中位置明细，汇总为 rows；同时返回全量可用文档集合
        available。可用集由检索函数返回的 docs 覆盖式取得（各查询共享同一数据源）。
    入参说明：
        gt_cases (OrderedDict[str, list[str]])：评测集，查询 -> 期望命中文档原文列表。
        retrieve (Callable[[str, int], tuple[list[str], set[str]]])：检索适配器，
            签名 (query, top_k) -> (结果文本列表, 全量可用块集合)。
        max_k (int)：单条查询的检索深度，作为 top_k 透传给 retrieve。
    返回值说明：
        tuple：二元组 (rows, available)
            - rows (list[tuple])：每条为 (query, expected, positions, unevaluable, results)。
            - available (set[str])：最后一次检索得到的全量可用文档块集合。
    """
    rows, available = [], set()
    for query, expected in gt_cases.items():
        results, docs = retrieve(query, max_k)
        available = set(docs)
        positions, unevaluable = _hit_positions(results, expected, available)
        rows.append((query, expected, positions, unevaluable, results))
    return rows, available


def _report(rows, max_k):
    """
    函数功能与逻辑描述：
        打印评测报告，分三段：逐查询明细（命中位置与 top1 结果）、汇总指标
        （hit@k / recall@k / MRR@k）、k=5 下仍未命中的问题定位。考察深度 ks 取
        {1,3,5} 中不超过 max_k 的值，若为空则退化为 [max_k]。hit@k 与 MRR@k 仅在
        可评估查询上计算，recall@k 以可评估期望文档总数为分母；输出格式与 M0 版
        保持一致，便于跨版本对比。
    入参说明：
        rows (list[tuple])：_collect 返回的逐查询结果
            (query, expected, positions, unevaluable, results)。
        max_k (int)：最大考察深度，用于确定报告展示的 k 值集合。
    返回值说明：
        无（仅打印评测明细、汇总指标与问题定位，不写文件、无副作用）。
    """
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


# ===================== 索引健康前置检查（M25 / T-4） =====================
def _report_index_mode(require_vector: bool) -> None:
    """
    函数功能与逻辑描述：
        ★M25（T-4）评测前置声明：打印本次评测**实际使用的检索模式**（含向量 / BM25-only）
        与索引规模，并在索引"缺向量"时给出**显式告警**。

        为什么必须有：检索质量**强依赖索引是否含向量** —— 实测（`设计/23` §2.2 三场景对照）
        同一 50 条 GT 下，含向量 hit@1 = **75.51%**、BM25-only = **51.02%**（差 **24.5pp**）。
        而"残缺索引"曾经是**静默**的，导致跨模式的指标被误当"可比"，
        进而得出"改动使指标变差"的**错误结论**。本函数把该状态**显式暴露**在报告开头。

        落点：必须在 `_collect`（首次检索）**之前**调用 —— 检索会触发索引构建 / 对账
        （含 M25 的"缺向量自愈"），此处的状态即"评测将实际使用"的状态。
    入参说明：
        require_vector (bool)：True 时若索引不含向量则**硬失败退出**（CI / 严格回归用）；
            False 时仅打印警告并继续（诊断用）。
    返回值说明：
        无（仅打印；`require_vector=True` 且缺向量时以 SystemExit 终止）。
    """
    from memory.user_docs import _ensure_state
    st = _ensure_state()
    mode = "BM25 + 向量（RRF 融合）" if st.vector_ok else "BM25-only（无向量）"
    print(f"检索模式: {mode}  |  chunk={len(st.chunks)}  |  dim={st.meta.get('dim')}")
    if st.vector_ok:
        return
    print("[警告] 当前索引**不含向量** → 本结果**不代表完整链路**（hit@1 将低约 24pp）！")
    print("       成因多为构建索引时 embedding 不可用；请确认 BILLAGENT_EMBEDDING=1 且模型可加载，")
    print("       再删 userDocs/.index 后重跑（或依赖 M25 的自动自愈：下次启动会重建）。")
    if require_vector:
        raise SystemExit(
            "\n[失败] --require-vector 已开启，但当前索引不含向量。\n"
            "       该模式下指标**不可**与「含向量」基线比较（见 `设计/23` §2.2）。\n"
        )


def run(max_k, allow_missing, require_vector=False):
    """
    函数功能与逻辑描述：
        评测主入口：加载 userdocs 评测集（GT_CASES_USERDOCS），调用 _collect 执行检索
        与指标收集，对「期望文档不在数据源中」做硬断言——存在缺失且未开启 allow_missing
        时抛出 SystemExit 终止（避免评测静默失效，见 09 §7.1），否则仅警告并降级统计；
        最后调用 _report 打印报告。前置依赖：需可 import memory.user_docs 且 userDocs/
        样例与 GT 同步。
    入参说明：
        max_k (int)：最大考察深度，透传给 _collect 作为检索 top_k。
        allow_missing (bool)：True 时 GT 缺失仅告警不退出（诊断用）；False 时硬失败。
        require_vector (bool)：True 时要求索引含向量，否则硬失败（**防跨模式指标被误比**，
            M25/T-4；详见 `_report_index_mode`）。
    返回值说明：
        无（仅打印报告；GT 缺失且不允许降级时以 SystemExit 终止进程）。
    """
    gt_cases = GT_CASES_USERDOCS
    print()
    print("#" * 80)
    print(f"# 数据源: userdocs（用户文档库）      用例数: {len(gt_cases)}")
    print("#" * 80)
    # ★M25（T-4）：先声明检索模式（含向量 / BM25-only），再跑检索 ——
    #   防"跨模式指标被误当可比"（该坑实际发生过，见 `设计/23` §六 R-1）
    _report_index_mode(require_vector)

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
    parser.add_argument("--require-vector", action="store_true",
                        help="严格模式：索引不含向量时直接失败（防跨模式指标被误比，M25/T-4）")
    args = parser.parse_args()
    run(args.max_k, args.allow_missing, args.require_vector)
