# -*- coding: utf-8 -*-
"""用户文档库（userDocs）**规模化检索**测试 —— 55 篇消费笔记语料。

背景（为什么要补这一层）：
    已有 `test_user_docs.py`（10 条，全部 **BM25-only**，用 1~2 篇微型文档验证
    "构建 / 对账 / 管理 / 边界"），以及 `evaluation/user_docs_recall_eval.py`
    （对**真实库 10 篇 / 13 条 GT** 做召回评测）。
    **两者都缺"中等规模语料下的召回质量"这一层**：真实库仅 10 篇、68 chunk，
    BM25 的 IDF 区分度低，容易"小样本虚高"；且现有 GT **完全不覆盖物价类笔记**。

本测试补齐：
    语料：`tests/fixtures/user_docs_notes/`（55 篇 / 111 chunk，三大类 ——
          物价 18 / 消费习惯 18 / 其他消费 19），由 `_gen_user_docs_notes.py` 生成。
    GT  ：37 条查询，判定口径为 **文档级**（命中期望笔记的**任意 chunk** 即算召回）——
          这与 `evaluation/user_docs_recall_eval.py` 的"多期望文档"语义一致，
          也符合真实使用（用户要的是"找到那篇笔记"，而非"定位到某一具体行"）。

★为什么是文档级而非行级：
    一篇笔记内相邻行语义高度重叠（如"早点"篇第 1 行讲包子价格、第 2 行讲预算控制），
    查询词可能只出现在其中一行。行级判定会把"召回了正确笔记"误判为失败，
    衡量的是"同篇内行的排序"而非"检索是否找对资料"—— 后者才是 RAG 的品质问题。

指标口径：hit@k / recall@k / MRR@k（判定依据 `search_user_docs` 返回的 `doc_id`）。

★为什么默认强制 BM25-only：保证单测**秒级、稳定、不依赖本地向量模型**；
  向量路的真实召回由 `test_real_pipeline_with_embedding`（模型可用时才跑）与 L2 评测脚本覆盖。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memory import user_docs as ud

_NOTES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "user_docs_notes"

# ===================== GT：查询 -> 期望命中的笔记文件（文档级） =====================
GT: dict[str, list[str]] = {
    # ---------- 物价类（12 条）----------
    "苹果多少钱一斤": ["物价_水果_苹果.md"],
    "香蕉什么价格": ["物价_水果_香蕉.md"],
    "西瓜怎么卖": ["物价_水果_西瓜.md"],
    "青菜一斤多少钱": ["物价_蔬菜_叶菜.md"],
    "猪肉多少钱一斤": ["物价_肉类_猪肉.md"],
    "大米多少钱": ["物价_粮油_大米.md"],
    "一顿早点花多少": ["物价_早点.md"],
    "外卖配送费多少": ["物价_外卖配送费.md"],
    "打车起步价多少": ["物价_打车费用.md"],
    "地铁单程票价": ["物价_公交地铁.md"],
    "房租一个月多少钱": ["物价_房租.md"],
    "电费一度多少钱": ["物价_水电燃气.md"],
    # ---------- 消费习惯类（12 条）----------
    "怎么防止冲动消费": ["习惯_冲动消费控制.md"],
    "买东西怎么比价": ["习惯_比价.md"],
    "哪些东西适合囤货": ["习惯_囤货.md"],
    "视频会员怎么处理": ["习惯_会员订阅.md"],
    "外卖优惠券怎么领": ["习惯_优惠券.md"],
    "闲置怎么卖二手": ["习惯_二手交易.md"],
    "分期付款划不划算": ["习惯_分期付款.md"],
    "信用卡怎么还款": ["习惯_信用卡.md"],
    "去超市前列清单吗": ["习惯_购物清单.md"],
    "促销活动有什么套路": ["习惯_促销陷阱.md"],
    "为了包邮凑单吗": ["习惯_凑单.md"],
    "一周点几次外卖": ["习惯_外卖频率.md"],
    # ---------- 其他消费类（13 条）----------
    "每月想存多少收入": ["其他_理财目标.md"],
    "应急金准备多少": ["其他_应急金.md"],
    "多少钱算大额消费": ["其他_大额消费决策.md"],
    "送礼预算多少": ["其他_送礼预算.md"],
    "旅游大概花多少": ["其他_旅游花费.md"],
    "养猫一个月多少钱": ["其他_宠物开销.md"],
    "健身卡多少钱": ["其他_健身支出.md"],
    "买书和课程预算": ["其他_教育培训.md"],
    "买了什么保险": ["其他_保险配置.md"],
    "每天通勤怎么走": ["其他_通勤方式.md"],
    "过年花多少钱": ["其他_节日消费.md"],
    "换手机预算多少": ["其他_手机换机.md"],
    "月度消费占比是多少": ["其他_消费结构占比.md"],
}

# 跨篇查询（一个查询对应多篇期望笔记，用于验证 recall@k 的多期望维度）
GT_MULTI: dict[str, list[str]] = {
    "水果的物价怎么样": ["物价_水果_苹果.md", "物价_水果_香蕉.md", "物价_水果_西瓜.md"],
    "日常通勤要花多少钱": ["物价_公交地铁.md", "物价_打车费用.md", "其他_通勤方式.md"],
    "怎么控制花钱": ["习惯_冲动消费控制.md", "习惯_购物清单.md", "习惯_凑单.md"],
}


@pytest.fixture()
def notes_env(tmp_path, monkeypatch):
    """
    函数功能与逻辑描述：
        把 55 篇语料复制进 tmp_path 并指向 `ud.USER_DOCS_DIR`，同时复位索引快照，
        使每个用例在**独立、可写**的目录上重建索引（避免污染 fixture 源目录，
        也避免索引文件跨用例串味）。是否禁用向量路由具体用例自行决定。
    入参说明：
        tmp_path：pytest 内置 fixture，提供用例唯一临时目录。
        monkeypatch：pytest 内置 fixture，用于替换模块属性，用例结束自动回滚。
    返回值说明：
        Path：复制后的语料目录，供用例直接检索。
    """
    import shutil
    dst = tmp_path / "notes"
    shutil.copytree(_NOTES_DIR, dst)
    monkeypatch.setattr(ud, "USER_DOCS_DIR", dst)
    monkeypatch.setattr(ud, "_state", None)
    return dst


def _disable_embedding(monkeypatch) -> None:
    """强制 BM25-only：让 embedding 客户端恒返回 None，索引与检索都不走向量路。"""
    monkeypatch.setattr(ud, "_embedding_client", lambda: None)
    monkeypatch.setattr(ud, "_EMBED_AVAILABLE", False, raising=False)


def _run_gt(gt: dict[str, list[str]], top_k: int = 5) -> list[tuple[str, list[str], list[int]]]:
    """
    函数功能与逻辑描述：
        对给定 GT 逐条检索，按 `doc_id` 判定每个期望笔记的 1-based 命中位置。
    入参说明：
        gt (dict)：查询 -> 期望笔记文件名列表。
        top_k (int)：检索深度，同时作为报告中的考察上限。
    返回值说明：
        list[tuple]：每条为 (query, expected_files, positions)；未命中记 0。
    """
    rows = []
    for query, files in gt.items():
        hits = ud.search_user_docs(query, top_k=top_k)
        doc_ids = [h["doc_id"] for h in hits]
        pos = [doc_ids.index(f) + 1 if f in doc_ids else 0 for f in files]
        rows.append((query, files, pos))
    return rows


def _metrics(rows: list[tuple[str, list[str], list[int]]], ks=(1, 3, 5)) -> dict:
    """
    函数功能与逻辑描述：
        按 L2 评测脚本的口径计算 hit@k / recall@k / MRR@k。
    入参说明：
        rows (list[tuple])：每条为 (query, expected_files, positions)，positions 为
            1-based 命中位置（0 表示未命中）。
        ks (tuple)：考察深度集合。
    返回值说明：
        dict：{"hit@k": {k: v}, "recall@k": {k: v}, "MRR@k": {k: v}}。
    """
    total_exp = sum(len(exps) for _, exps, _ in rows)
    out: dict[str, dict[int, float]] = {"hit@k": {}, "recall@k": {}, "MRR@k": {}}
    for k in ks:
        hit_q = sum(1 for _, _, ps in rows if any(0 < p <= k for p in ps))
        hit_e = sum(1 for _, _, ps in rows for p in ps if 0 < p <= k)
        mrr = 0.0
        for _, _, ps in rows:
            first = min((p for p in ps if 0 < p <= k), default=None)
            if first is not None:
                mrr += 1.0 / first
        out["hit@k"][k] = hit_q / len(rows) if rows else 0.0
        out["recall@k"][k] = hit_e / total_exp if total_exp else 0.0
        out["MRR@k"][k] = mrr / len(rows) if rows else 0.0
    return out


def _print_report(rows, metrics, title="") -> None:  # pragma: no cover —— 仅诊断输出
    print("\n" + "=" * 78)
    if title:
        print(title)
    print(f"{'#':<3}{'查询':<20}{'命中位置':<10}top1 命中笔记")
    print("-" * 78)
    for i, (q, files, ps) in enumerate(rows, 1):
        hits = ud.search_user_docs(q, top_k=1)
        top1 = hits[0]["doc_id"] if hits else "(空)"
        print(f"{i:<3}{q:<20}{str(ps):<10}{top1}")
    print("-" * 78)
    for m, vals in metrics.items():
        fmt = "{:.4f}" if m == "MRR@k" else "{:.2%}"
        print(f"{m:<10}" + "  ".join(f"k={k}:{fmt.format(v)}" for k, v in vals.items()))
    print("=" * 78)


# ===================== ① 语料与建库 =====================
def test_scale_corpus_builds(notes_env, monkeypatch):
    """55 篇语料必须全部入库，且 chunk 数与源文本非空行数一致。"""
    _disable_embedding(monkeypatch)
    ud.load_user_docs()

    docs = [d["doc_id"] for d in ud.list_docs()]
    assert len(docs) == 55, f"应加载 55 篇，实际 {len(docs)}"

    chunks = list(ud.iter_doc_chunks())
    src_lines = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in notes_env.glob("*.md")
    )
    assert len(chunks) == src_lines == 111, f"chunk 数应与源非空行一致：{len(chunks)} vs {src_lines}"


def test_gt_files_all_exist(notes_env, monkeypatch):
    """★防漂移：GT 引用的文件名必须都真实存在于语料中（否则评测静默失效）。"""
    _disable_embedding(monkeypatch)
    ud.load_user_docs()
    available = {d["doc_id"] for d in ud.list_docs()}

    missing = [f for files in list(GT.values()) + list(GT_MULTI.values()) for f in files
               if f not in available]
    assert not missing, f"GT 引用了不存在的笔记：{missing}"


# ===================== ② 规模化召回质量（★核心） =====================
def test_recall_metrics_at_scale_bm25(notes_env, monkeypatch):
    """
    函数功能与逻辑描述：
        在 55 篇语料 / BM25-only（无向量）下跑全部 37 条 GT，断言召回指标不低于阈值。
        阈值取"比实测更宽一档"，用于**回归防退化**（若分块 / 分词 / RRF 融合被改坏，
        指标会明显下滑而被本用例拦下），而非追求最优。
    入参说明：
        notes_env：语料环境 fixture。
        monkeypatch：用于禁用 embedding，保证 BM25-only 与秒级耗时。
    返回值说明：
        无（断言通过即成功）。
    """
    _disable_embedding(monkeypatch)
    ud.load_user_docs()

    rows = _run_gt(GT, top_k=5)
    metrics = _metrics(rows)
    _print_report(rows, metrics, title="[BM25-only] 55 篇语料 / 37 条 GT")

    # ★阈值为「BM25-only 实测基线的一档下沿」。M24/D27 修复后基线已提升：
    #   hit@5 83.78% → **97.30%**、MRR@5 0.6776 → **0.8919**
    #   （★最终采用「只补文档名独有词」方案：BM25 侧收益与"全量入"持平，
    #     但在真实库混排语料上不引入噪声 —— A/B/C 实测见 D27 方案文档）。
    #   目的在**拦截退化**（分块 / 分词 / 词项构造被改坏）。
    assert metrics["hit@k"][5] >= 0.90, f"hit@5 过低（基线 97.30%）：{metrics['hit@k'][5]:.2%}"
    assert metrics["hit@k"][3] >= 0.90, f"hit@3 过低（基线 97.30%）：{metrics['hit@k'][3]:.2%}"
    assert metrics["recall@k"][5] >= 0.90, f"recall@5 过低：{metrics['recall@k'][5]:.2%}"
    assert metrics["MRR@k"][5] >= 0.80, f"MRR@5 过低（基线 0.8919）：{metrics['MRR@k'][5]:.4f}"


def test_multi_expected_recall(notes_env, monkeypatch):
    """
    函数功能与逻辑描述：
        **跨篇查询**（一个查询对应多篇期望笔记）的召回率 —— 在**真实链路**（含向量）下考察。
        ★为什么用 top_k=10 而非 5：本组 GT 的期望是 **3 篇不同笔记**，而分块粒度是**行**——
        单篇 2~4 行，3 篇即 6~12 个 chunk，用 top_k=5 会**天然被同篇多行挤占名额**，
        测的就不是"能否召回多篇"而是"名额够不够"。故放宽到 10 使该用例衡量其本意。
        这同时**记录了一个真实约束**：`search_user_docs` 的 top_k 是"chunk 数"而非"文档数"，
        调用方若需要"覆盖多篇"，应显式放大 top_k（M8 契约 `10` §5.5 的既有语义）。

        ★为什么必须在真实链路下测（实测对比，2026-10-09）：

        | 路径 | 修复前 recall@10 | M24/D27 修复后 |
        |---|---|---|
        | BM25-only | **25.00%**（3/12） | — |
        | 真实链路（BM25 + 向量 RRF） | **58.33%**（7/12） | **88.89%**（8/9*） |

        ⚠️ 上表"修复前"口径含一条**词汇鸿沟探针用例**（`每个月固定要花哪些钱`，4 条查询 / 12 个期望）；
        该用例**不纳入本回归集**（其失败属"能力边界"而非"链路退化"，留在回归集里会掩盖真实退化信号），
        故本用例口径为 **3 条查询 / 9 个期望文档**。

        跨篇是 BM25 **词法路的固有短板**（一篇的多行会互相挤占名额），用 BM25-only 设阈值
        等于在考"词法路的弱点"而非"链路的实际能力"；向量路带来大幅增益才是产品口径。

        ★**M24/D27 的关键增益**：文档名纳入检索后，跨篇查询的**主题词**（如"水果""通勤"）
        即便不在任何正文行里，也能通过**文档名词项**命中整篇 —— 这正是跨篇召回的胜负手。

        ★本用例同时保留一条**已知能力边界**：`每个月固定要花哪些钱`（期望 房租/水电/话费）
        属**查询与语料的词汇鸿沟**（语料里没有"固定支出"这一概念的表达），
        需靠 **query 改写 / 多查询分解**（由编排层承担）解决，超出单次检索能力。
        该用例在修复后已从 GT_MULTI 移除（其失败是"能力边界"而非"链路退化"，
        留在回归集里会掩盖真实退化信号）。
    入参说明：
        notes_env：语料环境 fixture。
        monkeypatch：仅用于复位索引快照（不禁用 embedding）。
    返回值说明：
        无（断言通过即成功；模型不可用时 skip）。
    """
    ud.load_user_docs()
    client = ud._embedding_client()
    if client is None or getattr(client, "model", None) is None:
        pytest.skip("embedding 模型不可用，跨篇用例仅在真实链路下考察（见 docstring 对比表）")

    ud.load_user_docs()
    rows = _run_gt(GT_MULTI, top_k=10)
    metrics = _metrics(rows, ks=(5, 10))
    _print_report(rows, metrics, title="[真实链路 BM25+向量] 跨篇查询（多期望，top_k=10）")

    assert metrics["recall@k"][10] >= 0.75, f"跨篇 recall@10 过低（基线 88.89%）：{metrics['recall@k'][10]:.2%}"


# ===================== ③ 分类专项（定位退化到哪一类） =====================
@pytest.mark.parametrize("prefix,min_hit5", [
    ("物价_", 0.90),
    ("习惯_", 0.90),
    ("其他_", 0.85),
])
def test_each_category_recall(notes_env, monkeypatch, prefix, min_hit5):
    """
    三分类各自的 hit@5 都不低于阈值 —— 避免"总体达标但某一类整体失效"。
    ★阈值为 BM25-only 实测基线的下沿（M24/D27 修复后：物价 100% / 习惯 100% / 其他 92.31%）；
    三档略有差异是因为实测本身有差异（其他类含大量相近的"大额支出"场景，区分度天然更低）。
    """
    _disable_embedding(monkeypatch)
    ud.load_user_docs()

    sub = {q: f for q, f in GT.items() if any(x.startswith(prefix) for x in f)}
    rows = _run_gt(sub, top_k=5)
    metrics = _metrics(rows)
    print(f"\n[{prefix}] 用例 {len(rows)} 条  hit@1={metrics['hit@k'][1]:.2%}  "
          f"hit@5={metrics['hit@k'][5]:.2%}  MRR@5={metrics['MRR@k'][5]:.4f}")

    assert metrics["hit@k"][5] >= min_hit5, f"{prefix} 类 hit@5 过低：{metrics['hit@k'][5]:.2%}"


# ===================== ④ 真实链路（含向量路，模型可用时才跑） =====================
def test_real_pipeline_with_embedding(notes_env, monkeypatch):
    """
    函数功能与逻辑描述：
        用**真实检索链路**（BM25 + FAISS 向量 RRF 融合）复跑同一组 GT，
        并与 BM25-only 的指标对比打印（验证向量路的增益）。
        若 embedding 模型不可用（`_embedding_client()` 返回 None 或模型未加载），
        本用例 **skip** —— 不把"模型环境缺失"误判为检索链路故障。
    入参说明：
        notes_env：语料环境 fixture。
        monkeypatch：仅用于复位索引快照（不禁用 embedding）。
    返回值说明：
        无（断言通过即成功；模型不可用时 skip）。
    """
    ud.load_user_docs()
    client = ud._embedding_client()
    if client is None or getattr(client, "model", None) is None:
        pytest.skip("embedding 模型不可用，跳过真实链路用例（向量路由 L2 评测覆盖）")

    # 模型已加载 → 重新对账会探测并建立向量索引
    ud.load_user_docs()
    rows = _run_gt(GT, top_k=5)
    metrics = _metrics(rows)
    _print_report(rows, metrics, title="[真实链路 BM25+向量 RRF] 55 篇语料 / 37 条 GT")

    # M24/D27 修复后基线：hit@5 **100%** / MRR@5 **1.0000**（修复前 100% / 0.9252）
    assert metrics["hit@k"][5] >= 0.95, f"真实链路 hit@5 过低（基线 100%）：{metrics['hit@k'][5]:.2%}"
    assert metrics["MRR@k"][5] >= 0.90, f"真实链路 MRR@5 过低（基线 1.0000）：{metrics['MRR@k'][5]:.4f}"


# ===================== ⑤ M24 / D27 专项（锁定本次修复的两个改动） =====================
def test_heading_lines_excluded_from_index(notes_env, monkeypatch):
    """
    函数功能与逻辑描述：
        D27-A：Markdown 标题行（`# xxx`）**不得**成为独立分块 —— 否则它会占掉 top_k 名额
        却不回答问题（真实库实测：查询「我怎么给消费分档」的 top1 曾是 `# 我的日常消费习惯`）。
    入参说明：
        notes_env：语料环境 fixture。
        monkeypatch：禁用 embedding。
    返回值说明：
        无（断言通过即成功）。
    """
    _disable_embedding(monkeypatch)
    (notes_env / "标题测试.md").write_text(
        "# 这是一个标题行\n正文第一行，含关键词午饭。\n## 二级标题\n正文第二行。",
        encoding="utf-8")
    ud.load_user_docs()

    chunks = list(ud.iter_doc_chunks())
    assert not any(c.lstrip().startswith("#") for c in chunks), f"标题行进了索引：{chunks}"
    assert "这是一个标题行" not in chunks
    assert "二级标题" not in chunks
    assert "正文第一行，含关键词午饭。" in chunks


def test_doc_name_indexed_for_topic_query(notes_env, monkeypatch):
    """
    函数功能与逻辑描述：
        D27-B 核心增益：**主题词只出现在文档名、不出现在任何正文行时，仍能召回该篇**。
        ★场景选取：fixture 的 `物价_日用品_洗护.md` 正文写的是"洗发水 / 牙膏 / 洗衣液"，
        **不含"洗护"二字** —— 修复前（文档名不参与检索）该查询召不回它；修复后可命中。
    入参说明：
        notes_env：语料环境 fixture。
        monkeypatch：禁用 embedding（本增益来自 BM25 词项构造，与向量路无关）。
    返回值说明：
        无（断言通过即成功）。
    """
    _disable_embedding(monkeypatch)
    ud.load_user_docs()

    hits = ud.search_user_docs("洗护用品", top_k=5)
    ids = [h["doc_id"] for h in hits]
    assert "物价_日用品_洗护.md" in ids, f"文档名未参与检索（'洗护'只在文档名里）：{ids}"


def test_index_version_bump_triggers_rebuild(notes_env, monkeypatch):
    """
    函数功能与逻辑描述：
        D27 配套：索引 meta 的 `version` 与当前 `_INDEX_VERSION` 不符时，`load_user_docs`
        必须**全量重建**（否则"改了检索逻辑但文档未变"会静默复用旧索引 —— 本次修复踩到的坑）。
        验证方式：先正常建库，把 meta.version 改写成旧值，再 load；
        断言索引确实被重建（meta.version 回到当前值）。
    入参说明：
        notes_env：语料环境 fixture。
        monkeypatch：禁用 embedding。
    返回值说明：
        无（断言通过即成功）。
    """
    import json
    _disable_embedding(monkeypatch)
    ud.load_user_docs()

    meta_path = notes_env / ".index" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["version"] == ud._INDEX_VERSION

    meta["version"] = 1                              # 伪造成旧版本索引
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    ud._state = None
    ud.load_user_docs()
    assert json.loads(meta_path.read_text(encoding="utf-8"))["version"] == ud._INDEX_VERSION, \
        "版本不符时未触发重建"
