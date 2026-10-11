# -*- coding: utf-8 -*-
"""用户文档库 `memory/user_docs.py` 的 L1 单测（M8）。

覆盖：构建/检索、内容哈希变更对账、doc:* 管理（add/list/delete）、上限红线、索引落盘与幂等、
非法入参拒绝（路径穿越 / 空内容）与空库兜底。
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
    """
    函数功能与逻辑描述：
        构建 user_docs 的隔离测试环境，提供一个可自由读写的临时「真相源」目录：
        monkeypatch 把 ud.USER_DOCS_DIR 指向 tmp_path、把模块级索引快照 ud._state 复位为 None
        （迫使 _ensure_state 重新对账而非复用旧快照），并把 ud._embedding_client 替换为
        恒返回 None 的 lambda —— 从而禁用向量路，使全文件走 BM25-only，不依赖本地向量模型。
        作用域为 function（默认）：每个用例拿到独立 tmp_path 与全新状态；清理由 pytest 负责——
        monkeypatch 在用例结束时自动回滚三处被替换的属性，tmp_path 目录由 pytest 自动删除。
    入参说明：
        tmp_path：pytest 内置 fixture，为本用例提供唯一临时目录（Path），充当 USER_DOCS_DIR。
        monkeypatch：pytest 内置 fixture，用于本用例内替换模块属性，用例结束后自动回滚。
    返回值说明：
        Path：即 tmp_path，供用例直接写入文档、拼路径断言 .index/ 下索引文件。
    """
    monkeypatch.setattr(ud, "USER_DOCS_DIR", tmp_path)
    monkeypatch.setattr(ud, "_state", None)
    monkeypatch.setattr(ud, "_embedding_client", lambda: None)
    return tmp_path


def _write(doc_env, name, text):
    """
    函数功能与逻辑描述：
        测试辅助：在隔离目录下写入一篇文档（UTF-8），用于模拟 userDocs/ 真相源中的用户文档，
        供 load_user_docs 对账与 search_user_docs 检索；直接覆盖同名文件，不做任何校验。
    入参说明：
        doc_env：doc_env fixture 提供的临时 USER_DOCS_DIR（Path）。
        name (str)：文档文件名（含 .md 扩展名，可含子目录）。
        text (str)：文档正文。
    返回值说明：
        Path：写入后的文件路径，即 doc_env / name，便于用例后续改写该文件。
    """
    p = doc_env / name
    p.write_text(text, encoding="utf-8")
    return p


def test_build_and_search(doc_env):
    """
    函数功能与逻辑描述：
        验证「构建 + 检索」主链路：先写入一篇午饭开销文档并 load_user_docs 建库，
        再用口语化问句 "快餐店人均多少钱" 检索；断言有命中且首条 text 含原文 D1、
        doc_id 为文件名 "午饭.md"、score 为 float。
        覆盖场景：单文档建库后的 BM25 命中（embedding 已被 doc_env fixture 禁用）。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR（Patch 掉的临时目录 + 复位索引状态）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    _write(doc_env, "午饭.md", D1)
    ud.load_user_docs()
    hits = ud.search_user_docs("快餐店人均多少钱", top_k=3)
    assert hits and D1 in hits[0]["text"], hits
    assert hits[0]["doc_id"] == "午饭.md"
    assert isinstance(hits[0]["score"], float)


def test_empty_query_returns_empty(doc_env):
    """
    函数功能与逻辑描述：
        验证空查询短路：库中已有 D2 文档时，空串 "" 与纯空白串 "   " 两种查询都直接返回 []，
        不进入 BM25 / 向量检索流程（否则可能命中无意义的全量结果）。
        覆盖场景：查询文本为空的输入边界。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    _write(doc_env, "目标.md", D2)
    assert ud.search_user_docs("") == []
    assert ud.search_user_docs("   ") == []


def test_empty_library_safe(doc_env):
    """
    函数功能与逻辑描述：
        验证空库兜底：隔离目录不写入任何文档即直接 load_user_docs（空库），
        再检索任意关键词应返回 [] 而不抛异常。
        覆盖场景：userDocs/ 为空（无文档）时的安全返回。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR（此时目录为空）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ud.load_user_docs()
    assert ud.search_user_docs("随便问问") == []


def test_reconcile_on_content_change(doc_env):
    """
    函数功能与逻辑描述：
        验证按内容哈希的对账与重建：先写入 D2 并 load，确认可检索到（首版应命中 D2）；
        随后改写同一文件内容（sha256 变化），再次 load 必须触发全量重建，
        使新内容（"不买游戏"）可被检索命中。
        覆盖场景：已有文档内容被修改（meta.files 与扫描结果不一致）。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    p = _write(doc_env, "目标.md", D2)
    ud.load_user_docs()
    assert ud.search_user_docs("存下收入比例", top_k=3), "首版应命中 D2"
    # 修改文件内容（哈希变化）→ 再次 load 必须重建
    p.write_text("我的目标是三个月不买游戏。", encoding="utf-8")
    ud.load_user_docs()
    hits = ud.search_user_docs("游戏还买吗", top_k=3)
    assert hits and "不买游戏" in hits[0]["text"], hits


def test_add_delete_lifecycle(doc_env):
    """
    函数功能与逻辑描述：
        验证 doc:add / doc:delete 完整生命周期：建空库后 add_doc 带 category 写入一篇文档，
        断言返回的 doc_id 为 "类别/标题.md"、chunk_count 为 1、list_docs 能列出该 doc_id、
        且新内容可被检索命中；再 delete_doc 后断言 list_docs 不再包含该 doc_id，
        且磁盘文件（<category>/<title>.md）已被 unlink（文件系统是真相源）。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证 add_doc 的入参校验红线：标题含 "../"（路径穿越）、内容为空白、类别含 "../"
        三种非法输入都抛 ValueError，确保任何路径组件都无法逃逸出 USER_DOCS_DIR。
        覆盖场景：_safe_component 对标题 / 类别的校验分支与「内容为空拒绝写入」分支。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    ud.load_user_docs()
    with pytest.raises(ValueError):
        ud.add_doc("../逃逸", "内容")
    with pytest.raises(ValueError):
        ud.add_doc("合法标题", "", category="正常")  # 空内容
    with pytest.raises(ValueError):
        ud.add_doc("合法标题", "内容", category="../坏类别")


def test_max_docs_enforced(doc_env, monkeypatch):
    """
    函数功能与逻辑描述：
        验证文档数上限红线：把模块级 MAX_DOCS（来自 USER_DOCS_MAX_DOCS）monkeypatch 为 2，
        写入 3 篇后 load_user_docs 抛含 "上限" 的 ValueError；随后删掉 2 篇只留 1 篇并重新 load，
        add_doc 补到 2 篇后再 add 也抛 "上限" 错误——确认对账与新增两条路径都被拦截。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
        monkeypatch：pytest 内置 fixture，用于把 ud.MAX_DOCS 临时替换为 2，用例结束自动回滚。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证 iter_doc_chunks 与检索结果同源：两篇文档建库后，iter_doc_chunks 产出的集合含 D1、D2；
        且 search_user_docs 命中的每条 text 都能在该集合中找到。
        覆盖场景：保证「评测判可用」与「索引实际收录」口径一致，避免 chunk 源漂移。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    _write(doc_env, "a.md", D1)
    _write(doc_env, "b.md", D2)
    ud.load_user_docs()
    all_chunks = list(ud.iter_doc_chunks())
    assert D1 in all_chunks and D2 in all_chunks
    # 检索返回的 text 必须来自同一 chunk 源
    for h in ud.search_user_docs("存钱目标", top_k=5):
        assert h["text"] in all_chunks


def test_index_files_persisted(doc_env):
    """
    函数功能与逻辑描述：
        验证索引落盘与二次 load 幂等：首次 load 后 .index/ 下 meta.json、store.json、bm25.pkl
        均存在；再次 load 时因内容哈希一致、且 embedding 被禁用（vector_ok=False）而跳过向量
        能力探测，直接复用磁盘索引，meta.json 的 mtime 保持不变（即未重建）。
    入参说明：
        doc_env：doc_env fixture 提供的隔离 USER_DOCS_DIR。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
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
    """
    函数功能与逻辑描述：
        验证内置 _BM25（不引入 rank_bm25 依赖）的打分语义：对三篇已分词文档、查询 ["快餐店"]，
        命中文档（下标 0）得分 > 0，未命中文档（下标 1）得分为 0.0；
        另断言空文档集合构造出的实例 get_scores 返回 []。
        覆盖场景：正常打分与空库（n_docs=0）两条路径。
    入参说明：
        无（pytest 自动发现并调用，无需外部入参）。
    返回值说明：
        无（断言通过即用例成功，抛出断言异常即失败）。
    """
    bm = _BM25([["午饭", "快餐店"], ["存钱", "目标"], ["游戏", "娱乐"]])
    scores = bm.get_scores(["快餐店"])
    assert scores[0] > 0 and scores[1] == 0.0
    empty = _BM25([])
    assert empty.get_scores(["任意"]) == []
