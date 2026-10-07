# -*- coding: utf-8 -*-
"""M25（D28）用户文档库「融合权重」与「残缺索引自愈」单元测试。

覆盖（对齐 `设计/23` §七）：

| 用例 | 验证点 |
|---|---|
| `test_rrf_weights_favor_vector_path` | ★两路给出**相反名次**时，加权后**向量路的候选胜出**（现状等权是平局） |
| `test_missing_vector_index_self_heals` | 磁盘索引标 `vector_ok=False` 且 embedding 可用 → **触发重建修复** |
| `test_missing_vector_index_kept_when_embedding_disabled` | `EMBEDDING_ENABLED=False` 时**不**强制重建（尊重"纯词法"开关） |
| `test_rrf_weight_constants_ordering` | 红线：`_RRF_VEC_WEIGHT > _RRF_BM25_WEIGHT`（向量路为主力） |

★不依赖本地向量模型：融合用例以**假 `_BM25` / 假 FAISS index** 直接构造 `_DocState`，
自愈用例以 monkeypatch 拦截 `_rebuild` 观测副作用。故全部秒级、可离线跑。
"""
from __future__ import annotations

import pytest

from memory import user_docs as ud


# ===================== ① 融合权重（纯逻辑，无模型） =====================
class _FakeBM25:
    """按预定分数返回打分：分数越高名次越前。"""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def get_scores(self, tokens):  # noqa: ARG002
        return list(self._scores)


class _FakeIndex:
    """按预定顺序返回近邻 id（模拟 FAISS `search` 的 (distances, ids) 返回）。"""

    def __init__(self, order: list[int]):
        self._order = order

    def search(self, vec, k):  # noqa: ARG002
        return [[0.0] * len(self._order)], [self._order]


class _FakeEmbeddingClient:
    """仅需 `get_embedding` 返回与 meta.dim 匹配的向量。"""
    model = object()          # 非 None 即视为"模型已加载"

    def get_embedding(self, text):  # noqa: ARG002
        return [0.1, 0.2]


def _install_fake_state(monkeypatch, *, bm25_scores, vec_order, dim=2):
    """构造一个受控的索引快照：BM25 与向量路给出**相反**的名次。"""
    chunks = [{"id": 0, "doc_id": "a.md", "text": "甲"},
              {"id": 1, "doc_id": "b.md", "text": "乙"}]
    st = ud._DocState(meta={"dim": dim, "vector_ok": True},
                      chunks=chunks, bm25=_FakeBM25(bm25_scores),
                      index=_FakeIndex(vec_order), ids=None, vector_ok=True)
    monkeypatch.setattr(ud, "_ensure_state", lambda: st)
    monkeypatch.setattr(ud, "_embedding_client", lambda: _FakeEmbeddingClient())
    return st


def test_rrf_weights_favor_vector_path(monkeypatch):
    """
    函数功能与逻辑描述：
        ★核心：构造 BM25 与向量路**名次相反**的受控场景，验证加权后**向量路的候选胜出**。
        · BM25：chunk0 > chunk1（chunk0 名次 1）
        · 向量：chunk1 在前（chunk1 名次 1）
        现状等权（vec_w=1.0）下两者 RRF 分**完全相等** → 按 chunk id 升序 → chunk0 胜出；
        vec_w=1.5 时 chunk1 的向量名次权重更高 → **chunk1 胜出**。
    入参说明：
        monkeypatch：pytest 内置 fixture，用于替换模块属性，用例结束自动回滚。
    返回值说明：
        无（断言通过即成功）。
    """
    _install_fake_state(monkeypatch, bm25_scores=[1.0, 0.5], vec_order=[1, 0])
    top = ud.search_user_docs("任意查询", top_k=2)
    assert top[0]["doc_id"] == "b.md", f"加权后应由向量路的候选胜出，实际 {[r['doc_id'] for r in top]}"

    # 反向对照：等权时两者 RRF 分数相同 → 退化为 chunk id 升序（a.md 在前）
    monkeypatch.setattr(ud, "_RRF_VEC_WEIGHT", 1.0)
    top2 = ud.search_user_docs("任意查询", top_k=2)
    assert top2[0]["doc_id"] == "a.md", "等权时两者应同分，按 chunk id 升序"

    scores = {r["doc_id"]: r["score"] for r in top2}
    assert scores["a.md"] == scores["b.md"], "等权下两路相反的候选应当同分（这是对照的前提）"


def test_rrf_weight_constants_ordering():
    """红线：向量路权重必须 **>=** 词法路 —— 实证 vec_w<1 会让 hit@1 崩（见 `设计/23` §2.2）。"""
    assert ud._RRF_VEC_WEIGHT > ud._RRF_BM25_WEIGHT, \
        "向量路是主力召回通道，其 RRF 权重不得低于词法路"


# ===================== ② 残缺索引自愈 =====================
@pytest.fixture()
def heal_env(tmp_path, monkeypatch):
    """
    函数功能与逻辑描述：
        隔离环境：临时语料目录 + 复位快照；`_rebuild` 替换为**记录调用**的桩，
        使"是否触发重建"成为可观测副作用。
        ★同时桩掉 `_load_existing` 与 `_install_state`：
        本 fixture 只关注"对账路径是否决定重建"。若不桩 `_load_existing`，
        则因临时目录下缺 `store.json` / `bm25.pkl` 而**必然**返回 None → **必然重建**，
        从而掩盖真实断言（这是该用例第一版失败的原因）。
    入参说明：
        tmp_path：pytest 内置 fixture，提供用例唯一临时目录。
        monkeypatch：pytest 内置 fixture，用于替换模块属性，用例结束自动回滚。
    返回值说明：
        tuple：二元组 (临时目录, `_rebuild` 的调用记录列表)。
    """
    (tmp_path / "a.md").write_text("午饭人均 25 元。", encoding="utf-8")
    monkeypatch.setattr(ud, "USER_DOCS_DIR", tmp_path)
    monkeypatch.setattr(ud, "_state", None)
    calls: list[int] = []
    monkeypatch.setattr(ud, "_rebuild", lambda files: calls.append(len(files)))
    monkeypatch.setattr(ud, "_load_existing", lambda meta: object())   # 非 None 即视为"加载成功"
    monkeypatch.setattr(ud, "_install_state", lambda st: None)
    return tmp_path, calls


def _write_meta(tmp_path, *, vector_ok: bool) -> None:
    """写入一份与当前语料匹配、但向量状态可控的 meta.json。"""
    import hashlib
    import json
    idx = tmp_path / ".index"
    idx.mkdir(exist_ok=True)
    sha = hashlib.sha256((tmp_path / "a.md").read_bytes()).hexdigest()
    (idx / "meta.json").write_text(json.dumps({
        "version": ud._INDEX_VERSION,
        "model_name": "fake", "dim": 512 if vector_ok else 0,
        "vector_ok": vector_ok, "next_id": 1, "files": {"a.md": sha},
    }), encoding="utf-8")


def test_missing_vector_index_self_heals(heal_env, monkeypatch):
    """
    函数功能与逻辑描述：
        ★核心：磁盘索引标 `vector_ok=False`（残缺：无向量）而 embedding 功能开启时，
        `load_user_docs` 必须**触发重建** —— 修复"懒加载导致校验被跳过、残缺索引长期复用"
        的自愈盲区（实测代价：hit@1 75.51% → 51.02%）。
        实现断言：`_rebuild` 桩被调用即可（真重建逻辑由其他用例覆盖）。
    入参说明：
        heal_env：隔离环境 fixture（返回 (临时目录, 调用记录)）。
        monkeypatch：用于让 `_probe_embedding` 报告"模型可用"。
    返回值说明：
        无（断言通过即成功）。
    """
    tmp_path, calls = heal_env
    _write_meta(tmp_path, vector_ok=False)
    monkeypatch.setattr(ud, "EMBEDDING_ENABLED", True)
    monkeypatch.setattr(ud, "_probe_embedding", lambda: (512, True))

    ud.load_user_docs()
    assert calls, "缺向量索引 + embedding 可用时必须触发重建（否则检索静默退化为 BM25-only）"


def test_missing_vector_index_kept_when_embedding_disabled(heal_env, monkeypatch):
    """
    函数功能与逻辑描述：
        `EMBEDDING_ENABLED=False`（"纯词法极速"开关，见 `config.py` 注释）**且模型从未加载**时，
        **不得**为修复向量而强制加载模型 —— 必须尊重该开关，不做无谓重建。
        ★为何要显式打桩 `_embedding_client`：真实客户端是**进程级单例**，
        同进程内若前面的用例已加载过模型，`client.model` 会非 None，从而走"模型已加载"分支 ——
        那是另一条既有路径（改动前就存在：模型加载了就校验）。本用例要验证的是
        **"纯词法模式"下新增的自愈逻辑不越权**，故须模拟"模型未加载"。
    入参说明：
        heal_env：隔离环境 fixture。
        monkeypatch：关闭 EMBEDDING_ENABLED，并模拟模型未加载。
    返回值说明：
        无（断言通过即成功）。
    """
    tmp_path, calls = heal_env
    _write_meta(tmp_path, vector_ok=False)
    monkeypatch.setattr(ud, "EMBEDDING_ENABLED", False)
    monkeypatch.setattr(ud, "_embedding_client", lambda: None)   # 模拟"模型从未加载"

    ud.load_user_docs()
    assert not calls, "embedding 功能关闭时不得因'缺向量'触发重建"
