# -*- coding: utf-8 -*-
"""用户文档库 `memory/user_docs.py` 的 L1 单测（M8）。

覆盖：构建/检索、内容哈希变更对账、doc:* 管理、上限红线、索引落盘、异常安全。
测试强制 BM25-only（monkeypatch 禁用 embedding），不依赖本地向量模型；
向量路径由 L2 `evaluation/user_docs_recall_eval.py` 覆盖真实检索。
"""
import pytest

from memory import user_docs as ud
from memory.user_docs import _BM25

D1 = "我午饭一般在公司楼下的快餐店吃，人均 25 到 30 元。"
D2 = "我的目标是存下收入的 20%。"


@pytest.fixture()
def doc_env(tmp_path, monkeypatch):
    """隔离环境：USER_DOCS_DIR 指向 tmp，禁用 embedding，复位模块级索引状态。"""
    monkeypatch.setattr(ud, "USER_DOCS_DIR", tmp_path)
    monkeypatch.setattr(ud, "_state", None)
    monkeypatch.setattr(ud, "_embedding_client", lambda: None)
    return tmp_path


def _write(doc_env, name, text):
    p = doc_env / name
    p.write_text(text, encoding="utf-8")
    return p


def test_build_and_search(doc_env):
    _write(doc_env, "午饭.md", D1)
    ud.load_user_docs()
    hits = ud.search_user_docs("快餐店人均多少钱", top_k=3)
    assert hits and D1 in hits[0]["text"], hits
    assert hits[0]["doc_id"] == "午饭.md"
    assert isinstance(hits[0]["score"], float)


def test_empty_query_returns_empty(doc_env):
    _write(doc_env, "目标.md", D2)
    assert ud.search_user_docs("") == []
    assert ud.search_user_docs("   ") == []


def test_empty_library_safe(doc_env):
    ud.load_user_docs()
    assert ud.search_user_docs("随便问问") == []


def test_reconcile_on_content_change(doc_env):
    p = _write(doc_env, "目标.md", D2)
    ud.load_user_docs()
    assert ud.search_user_docs("存下收入比例", top_k=3), "首版应命中 D2"
    # 修改文件内容（哈希变化）→ 再次 load 必须重建
    p.write_text("我的目标是三个月不买游戏。", encoding="utf-8")
    ud.load_user_docs()
    hits = ud.search_user_docs("游戏还买吗", top_k=3)
    assert hits and "不买游戏" in hits[0]["text"], hits


def test_add_delete_lifecycle(doc_env):
    ud.load_user_docs()
    added = ud.add_doc("购物笔记", "冲动消费前先加购物车放一晚。", category="日常")
    assert added["doc_id"] == "日常/购物笔记.md"
    assert added["chunk_count"] == 1
    doc_ids = [d["doc_id"] for d in ud.list_docs()]
    assert added["doc_id"] in doc_ids
    hits = ud.search_user_docs("怎么防冲动消费", top_k=3)
    assert hits and "加购物车" in hits[0]["text"], hits
    assert ud.delete_doc(added["doc_id"]) is True
    assert added["doc_id"] not in [d["doc_id"] for d in ud.list_docs()]
    assert not (doc_env / "日常" / "购物笔记.md").exists()


def test_add_rejects_bad_components(doc_env):
    ud.load_user_docs()
    with pytest.raises(ValueError):
        ud.add_doc("../逃逸", "内容")
    with pytest.raises(ValueError):
        ud.add_doc("合法标题", "", category="正常")  # 空内容
    with pytest.raises(ValueError):
        ud.add_doc("合法标题", "内容", category="../坏类别")


def test_max_docs_enforced(doc_env, monkeypatch):
    monkeypatch.setattr(ud, "MAX_DOCS", 2)
    _write(doc_env, "a.md", "内容A")
    _write(doc_env, "b.md", "内容B")
    _write(doc_env, "c.md", "内容C")  # 第 3 篇超上限
    with pytest.raises(ValueError, match="上限"):
        ud.load_user_docs()
    # add 路径也要挡：删掉 b/c 只留 1 篇，add 1 篇即满 2 篇上限，再 add 应报错
    (doc_env / "b.md").unlink()
    (doc_env / "c.md").unlink()
    ud.load_user_docs()
    ud.add_doc("一篇", "内容1")
    with pytest.raises(ValueError, match="上限"):
        ud.add_doc("二篇", "内容2")


def test_iter_chunks_same_source(doc_env):
    _write(doc_env, "a.md", D1)
    _write(doc_env, "b.md", D2)
    ud.load_user_docs()
    all_chunks = list(ud.iter_doc_chunks())
    assert D1 in all_chunks and D2 in all_chunks
    # 检索返回的 text 必须来自同一 chunk 源
    for h in ud.search_user_docs("存钱目标", top_k=5):
        assert h["text"] in all_chunks


def test_index_files_persisted(doc_env):
    _write(doc_env, "a.md", D1)
    ud.load_user_docs()
    assert (doc_env / ".index" / "meta.json").exists()
    assert (doc_env / ".index" / "store.json").exists()
    assert (doc_env / ".index" / "bm25.pkl").exists()
    # 二次 load 幂等：索引文件被复用（内容哈希一致，不重建）
    mtime = (doc_env / ".index" / "meta.json").stat().st_mtime
    ud.load_user_docs()
    assert (doc_env / ".index" / "meta.json").stat().st_mtime == mtime


def test_bm25_unit():
    bm = _BM25([["午饭", "快餐店"], ["存钱", "目标"], ["游戏", "娱乐"]])
    scores = bm.get_scores(["快餐店"])
    assert scores[0] > 0 and scores[1] == 0.0
    empty = _BM25([])
    assert empty.get_scores(["任意"]) == []
