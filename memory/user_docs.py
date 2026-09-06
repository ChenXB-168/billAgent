# -*- coding: utf-8 -*-
"""用户文档库（userDocs/）——「用户资料」唯一检索模块（`10` §5.5 A 方案，M8）

职责：
  1. userDocs/（文件系统 = 真相源）→ 按行分块 → 本地索引落盘；
  2. 启动对账 `load_user_docs()`：按**内容哈希**对照文件系统，增/删/改自动重建；
  3. 检索唯一入口 `search_user_docs()`：BM25 + FAISS 向量两路 RRF 融合；
  4. doc:* 管理指令（`add_doc` / `delete_doc` / `list_docs`），供 UI / 手工维护。

红线（`10` §5.5，M8 走查结论第 6 行）：
  - 不做"用户问什么就注入什么"的白名单——内容裁剪由调用方负责；
  - 文档数上限 `USER_DOCS_MAX_DOCS`（默认 1000），超出**明确报错**；
  - 分块按行（复用 split_text 语义），超长行按 `USER_DOCS_CHUNK_SIZE` 硬切；
  - 冲突时确定性优先、向用户说明（R9）——由渲染方负责。

生命周期（走查结论第 5 行）：
  - 索引对象整体替换引用（`_state` 指向不可变快照），检索不持锁读快照；
  - 构建 / 对账用 `_build_lock` 串行化，避免并发重复构建；
  - 落盘一律临时文件 + `os.replace` 原子替换；
  - embedding 模型不可用时自动降级 BM25（`vector_ok=False` 记入 meta），
    模型恢复 / 更换（dim 变化）后下一次对账自动全量重建。

仅依赖标准库；jieba / faiss / numpy / embedding 均在用到时才惰性导入，
避免 `import memory.user_docs` 就拉起 torch + SentenceTransformer。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Iterator

from config.config import USER_DOCS_PATH, USER_DOCS_TOP_K, USER_DOCS_MAX_DOCS, USER_DOCS_CHUNK_SIZE

# ===================== 模块级配置（测试可注入） =====================
USER_DOCS_DIR: Path = USER_DOCS_PATH          # 真相源目录
MAX_DOCS: int = USER_DOCS_MAX_DOCS            # 文档数上限，超出明确报错
TOP_K_DEFAULT: int = USER_DOCS_TOP_K          # search_user_docs 默认返回条数
CHUNK_SIZE: int = USER_DOCS_CHUNK_SIZE        # 超长行硬切长度
_DOC_EXTS = (".md", ".txt")
_EMBED_AVAILABLE: bool | None = None          # None=未探测；由对账时探测一次

# ===================== 索引落盘（userDocs/.index/ 下，四件套原子写） =====================
def _index_dir() -> Path:
    return USER_DOCS_DIR / ".index"


def _meta_path() -> Path:
    return _index_dir() / "meta.json"


def _store_path() -> Path:
    return _index_dir() / "store.json"


def _bm25_path() -> Path:
    return _index_dir() / "bm25.pkl"


def _faiss_path() -> Path:
    return _index_dir() / "faiss.index"


# ===================== 内存状态（整体替换，检索读快照） =====================
class _DocState:
    """一次对账 / 重建后的不可变索引快照。"""

    __slots__ = ("meta", "chunks", "bm25", "index", "ids", "vector_ok", "last_error")

    def __init__(self, meta: dict, chunks: list[dict], bm25, index, ids,
                 vector_ok: bool, last_error: str = ""):
        self.meta = meta                      # {"model_name","dim","vector_ok","next_id","files":{rel:sha}}
        self.chunks = chunks                  # [{"id":int,"doc_id":str,"text":str}]
        self.bm25 = bm25                      # BM25Okapi 或 None（空库）
        self.index = index                    # faiss.IndexIDMap 或 None
        self.ids = ids                        # np.ndarray(int64) chunk ids（与 faiss 内 ids 一致）或 None
        self.vector_ok = vector_ok            # 向量检索是否可用
        self.last_error = last_error          # 最近一次构建/检索失败原因（诊断用）


_state: _DocState | None = None
_build_lock = threading.RLock()


def _scan_files() -> dict[str, str]:
    """扫描真相源：relpath -> sha256（字节级哈希，不用 mtime——契约要求内容哈希对账）。"""
    out: dict[str, str] = {}
    if not USER_DOCS_DIR.is_dir():
        return out
    for p in sorted(USER_DOCS_DIR.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _DOC_EXTS:
            continue
        if ".index" in p.parts:
            continue
        rel = p.relative_to(USER_DOCS_DIR).as_posix()
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _chunk_lines(text: str) -> list[str]:
    """按行分块：空行剔除；超长行按 CHUNK_SIZE 硬切（复用 split_text 语义）。"""
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) <= CHUNK_SIZE:
            chunks.append(line)
        else:
            for i in range(0, len(line), CHUNK_SIZE):
                chunks.append(line[i:i + CHUNK_SIZE])
    return chunks


def _tokenize(text: str) -> list[str]:
    """jieba 分词（惰性导入；失败退化为按空白拆分，保证离线可用）。"""
    try:
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    except Exception:
        return [t for t in re.split(r"[^\w\u4e00-\u9fa5]+", text) if t]


class _BM25:
    """轻量 BM25（Okapi，k1=1.5 / b=0.75）——不引入 rank_bm25 依赖（环境未装）。

    倒排索引（term -> {doc_i: freq}）构建后 scoring；与 rank_bm25 行为对齐，
    支持纯词法检索。加载时由落盘 tokenized 列表直接 fit，无需重新分词。
    """

    def __init__(self, docs_tokens: list[list[str]]):
        self.docs_tokens = docs_tokens
        self.n_docs = len(docs_tokens)
        self.avgdl = (sum(len(d) for d in docs_tokens) / self.n_docs) if self.n_docs else 0.0
        self.doc_len = [len(d) for d in docs_tokens]
        self.postings: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(docs_tokens):
            cnt: dict[str, int] = {}
            for t in toks:
                cnt[t] = cnt.get(t, 0) + 1
            for t, f in cnt.items():
                self.postings.setdefault(t, {})[i] = f

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        n = self.n_docs
        if n == 0:
            return []
        k1, b = 1.5, 0.75
        scores = [0.0] * n
        for t in set(query_tokens):
            post = self.postings.get(t)
            if not post:
                continue
            idf = math.log(1 + (n - len(post) + 0.5) / (len(post) + 0.5))
            for i, freq in post.items():
                dl = self.doc_len[i]
                tf = freq * (k1 + 1) / (freq + k1 * (1 - b + b * dl / self.avgdl)) if self.avgdl else 0.0
                scores[i] += idf * tf
        return scores


def _embedding_client():
    """惰性获取 embedding 单例；模型加载失败返回 None。"""
    global _EMBED_AVAILABLE
    try:
        from modelService.embedding_loader import embedding_client
        if _EMBED_AVAILABLE is None:
            _EMBED_AVAILABLE = getattr(embedding_client, "model", None) is not None
        return embedding_client
    except Exception as e:                     # noqa: BLE001  import 失败（无 torch 等）
        _EMBED_AVAILABLE = False
        return None


def _batch_embed(texts: list[str]) -> tuple[list[list[float]] | None, int]:
    """批量向量化（分批 256）。返回 (vectors|None, dim|0)；任一批失败 → None。"""
    client = _embedding_client()
    if client is None:
        return None, 0
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 256):
        try:
            batch = client.get_batch_embedding(texts[i:i + 256]) or []
        except Exception:                      # noqa: BLE001
            return None, 0
        if len(batch) != len(texts[i:i + 256]) or any(not v for v in batch):
            return None, 0
        vectors.extend(batch)
    return vectors, len(vectors[0]) if vectors else 0


# ===================== 原子写 =====================
def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ===================== 全量重建 =====================
def _rebuild(files: dict[str, str]) -> None:
    """全量重建索引并原子落盘。embedding 不可用时降级 BM25-only。"""
    _index_dir().mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    next_id = 0
    for rel in sorted(files):
        text = (USER_DOCS_DIR / rel).read_text(encoding="utf-8", errors="replace")
        for line_text in _chunk_lines(text):
            chunks.append({"id": next_id, "doc_id": rel, "text": line_text})
            next_id += 1

    # --- BM25（jieba 分词，全量；内置 _BM25 不依赖 rank_bm25） ---
    tokenized = [_tokenize(c["text"]) for c in chunks]
    bm25 = _BM25(tokenized) if tokenized else None

    # --- 向量（分批 embedding；任一条失败即整库降级，保证"文本↔向量"不漂移） ---
    vector_ok = False
    dim = 0
    model_name = ""
    faiss = index = ids = None
    if chunks:
        vectors, dim = _batch_embed([c["text"] for c in chunks])
        if vectors is not None:
            import numpy as np
            import faiss
            ids = np.arange(len(chunks), dtype="int64")
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            index.add_with_ids(np.asarray(vectors, dtype="float32"), ids)
            vector_ok = True
            model_name = _embedding_model_name()
    _atomic_write_bytes(_faiss_path(), faiss.serialize_index(index) if index is not None else b"")
    import pickle
    _atomic_write_bytes(_bm25_path(), pickle.dumps({"tokenized": tokenized}))
    _atomic_write_json(_store_path(), {"chunks": chunks})
    _atomic_write_json(_meta_path(), {
        "version": 1,
        "model_name": model_name,
        "dim": dim,
        "vector_ok": vector_ok,
        "next_id": next_id,
        "files": files,
    })
    _install_state(_DocState(
        meta={"model_name": model_name, "dim": dim, "vector_ok": vector_ok,
              "next_id": next_id, "files": files},
        chunks=chunks, bm25=bm25, index=index, ids=ids, vector_ok=vector_ok,
    ))


def _embedding_model_name() -> str:
    try:
        from modelService import embedding_loader as el
        return Path(getattr(el, "EMB_MODEL_PATH", "")).name or "unknown"
    except Exception:                          # noqa: BLE001
        return "unknown"


def _install_state(state: _DocState | None) -> None:
    global _state
    _state = state


def _load_existing(meta: dict) -> _DocState | None:
    """按 meta 加载落盘索引。任一文件损坏/缺失 → None（触发全量重建）。"""
    try:
        store = json.loads(_store_path().read_text(encoding="utf-8"))
        chunks = store["chunks"]
        import pickle
        tokenized = pickle.loads(_bm25_path().read_bytes())["tokenized"]
        bm25 = _BM25(tokenized) if tokenized else None
        index = ids = None
        if meta.get("vector_ok"):
            data = _faiss_path().read_bytes()
            if not data:
                return None
            import faiss
            index = faiss.deserialize_index(data)
            ids = None
        return _DocState(meta=meta, chunks=chunks, bm25=bm25,
                         index=index, ids=ids, vector_ok=bool(meta.get("vector_ok")))
    except Exception:                          # noqa: BLE001  损坏/版本不符 → 全量重建
        return None


# ===================== 对外：启动对账 =====================
def load_user_docs() -> None:
    """启动对账（幂等）：索引与 userDocs/ 内容哈希对齐，必要时全量重建。

    - 模型不可用但索引为向量模式（或反之）→ 重建，保证 vector_ok 与现状一致；
    - 文件超上限 → 抛 ValueError（调用方自行决定记录还是退出）。
    """
    with _build_lock:
        files = _scan_files()
        if len(files) > MAX_DOCS:
            raise ValueError(
                f"用户文档数 {len(files)} 超过上限 {MAX_DOCS}（`10` §5.5 红线），请先清理再继续")
        meta = None
        try:
            meta = json.loads(_meta_path().read_text(encoding="utf-8"))
        except Exception:                      # noqa: BLE001
            meta = None
        if meta is not None and meta.get("files") == files:
            # 向量能力状态一致性校验（模型装了/换了/坏了 → 全量重建对齐）仅在模型**已加载**
            # 时进行。模型走懒加载（首次编码才加载，见 embedding_loader）：启动对账若强制
            # _probe_embedding 会"为校验而启动即加载模型"拖慢冷启动，故未加载时信任磁盘索引
            # （vector_ok 记于 meta 直接载入）。模型不可用/换代的兜底：检索向量路按 dim 不符
            # 自动跳过（见 search_user_docs），文档增删改（files 变化）自然触发全量重建对齐。
            client = _embedding_client()
            if client is not None and getattr(client, "model", None) is not None:
                dim, ok = _probe_embedding()
                if ok != bool(meta.get("vector_ok")) or (ok and dim != meta.get("dim")):
                    _rebuild(files)
                    return
            state = _load_existing(meta)
            if state is None:
                _rebuild(files)
                return
            _install_state(state)
            return
        _rebuild(files)


def _probe_embedding() -> tuple[int, bool]:
    """探测 embedding 是否可用并返回维度。不可用 → (0, False)。"""
    client = _embedding_client()
    if client is None:
        return 0, False
    try:
        vec = client.get_embedding("探测")
        return (len(vec), True) if vec else (0, False)
    except Exception:                          # noqa: BLE001
        return 0, False


# ===================== 对外：检索唯一入口 =====================
def _ensure_state() -> _DocState | None:
    if _state is None:
        with _build_lock:
            if _state is None:
                try:
                    load_user_docs()
                except ValueError as e:
                    st = _DocState({}, [], None, None, None, False, last_error=str(e))
                    _install_state(st)
                except Exception as e:         # noqa: BLE001  其它构建异常不炸调用方
                    st = _DocState({}, [], None, None, None, False, last_error=str(e))
                    _install_state(st)
    return _state


def search_user_docs(query: str, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    """用户文档库**唯一检索入口**（`10` §5.5 / §5.6.6）。

    :param query: 自然语言查询
    :param top_k: 返回条数
    :return: [{"doc_id": 相对路径, "text": 分块原文, "score": 融合分}, ...]；空库/异常 → []
    """
    if not query or not query.strip():
        return []
    state = _ensure_state()
    if state is None or not state.chunks:
        return []
    top_k = max(1, min(int(top_k), len(state.chunks)))

    # --- 两路召回，各记名次，RRF 融合 ---
    ranks: dict[int, list[int]] = {}          # chunk id -> 命中过的检索路名次（1-based）
    pool_size = min(top_k * 3, len(state.chunks))

    # 路 1：BM25（分数降序名次）
    bm_ids: list[int] = []
    if state.bm25 is not None:
        try:
            q_tokens = _tokenize(query)
            scores = state.bm25.get_scores(q_tokens)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            bm_ids = [i for i in order if scores[i] > 0][:pool_size]
        except Exception:                      # noqa: BLE001
            state.last_error = "BM25 检索失败"
    for rank, cid in enumerate(bm_ids, 1):
        ranks.setdefault(cid, []).append(rank)

    # 路 2：FAISS 向量（L2 距离升序名次）
    vec_ids: list[int] = []
    if state.index is not None:
        client = _embedding_client()
        q_vec = client.get_embedding(query) if client else []
        if q_vec and len(q_vec) == state.meta.get("dim"):
            import numpy as np
            try:
                _dists, ids_found = state.index.search(
                    np.asarray([q_vec], dtype="float32"), pool_size)
                vec_ids = [int(c) for c in ids_found[0]
                           if int(c) >= 0 and int(c) < len(state.chunks)]
            except Exception:                  # noqa: BLE001
                state.last_error = "向量检索失败，已用 BM25 结果"
    for rank, cid in enumerate(vec_ids, 1):
        ranks.setdefault(cid, []).append(rank)

    # RRF：score = Σ 1/(60 + rank)
    fused = {cid: sum(1.0 / (60 + r) for r in rs) for cid, rs in ranks.items()}
    picked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [{"doc_id": state.chunks[cid]["doc_id"],
             "text": state.chunks[cid]["text"],
             "score": round(float(sc), 6)} for cid, sc in picked]


def iter_doc_chunks() -> Iterator[str]:
    """迭代当前索引全部 chunk 原文（评测 available 与索引构建同源，防"评测可用但检索不可得"）。"""
    state = _ensure_state()
    if state is None:
        return
    for c in state.chunks:
        yield c["text"]


# ===================== 对外：doc:* 管理指令 =====================
_TITLE_RE = re.compile(r"^[\w\u4e00-\u9fa5-]{1,60}$")


def _safe_component(name: str, field: str) -> str:
    name = (name or "").strip()
    if not _TITLE_RE.match(name) or name in (".", ".."):
        raise ValueError(f"{field} 非法：仅允许中文/字母/数字/下划线/中划线，长度 1-60（当前={name!r}）")
    return name


def add_doc(title: str, content: str, category: str = "") -> dict:
    """doc:add——新增/覆盖一篇用户文档（低频管理指令，调用方先做权限与审计记录）。

    :return: {"doc_id": 相对路径, "chunk_count": 分块数}
    :raises ValueError: 标题/类别非法、文档数超上限、内容为空
    """
    title = _safe_component(title, "标题")
    category = _safe_component(category, "类别") if category else ""
    if not content or not content.strip():
        raise ValueError("内容为空：拒绝写入空文档")
    if len(_scan_files()) >= MAX_DOCS:
        raise ValueError(f"用户文档数已达上限 {MAX_DOCS}（`10` §5.5 红线），请先 delete_doc 再添加")

    rel = f"{category}/{title}.md" if category else f"{title}.md"
    target = USER_DOCS_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    # 先写文件（真相源）再重建索引；写失败不碰索引
    target.write_text(content, encoding="utf-8")
    try:
        with _build_lock:
            load_user_docs()
    except Exception as e:                     # noqa: BLE001
        raise ValueError(f"文档已写入但索引重建失败（下次 load_user_docs 自动修复）：{e}") from e
    _audit("doc:add", f"doc_id={rel} title={title} category={category or '(根目录)'}")
    n = sum(1 for c in (_state.chunks or []) if c["doc_id"] == rel)
    return {"doc_id": rel, "chunk_count": n}


def delete_doc(doc_id: str) -> bool:
    """doc:delete——按 doc_id（userDocs/ 下的相对路径）删除文档并重建索引。

    :raises ValueError: doc_id 非法或文件不存在
    """
    if not doc_id:
        raise ValueError("doc_id 为空")
    target = (USER_DOCS_DIR / doc_id).resolve()
    base = USER_DOCS_DIR.resolve()
    if not str(target).startswith(str(base)) or not target.is_file():
        raise ValueError(f"doc_id 不存在或越界：{doc_id}")
    target.unlink()
    with _build_lock:
        load_user_docs()
    _audit("doc:delete", f"doc_id={doc_id}")
    return True


def list_docs(category: str = "") -> list[dict]:
    """doc:list——列出文档元数据（不建索引，直接读文件系统）。

    :return: [{"doc_id","category","size","modified_at"}, ...]（按 doc_id 排序）
    """
    out = []
    for rel in sorted(_scan_files()):
        if category and not rel.startswith(category + "/"):
            continue
        p = USER_DOCS_DIR / rel
        out.append({
            "doc_id": rel,
            "category": rel.split("/", 1)[0] if "/" in rel else "",
            "size": p.stat().st_size,
            "modified_at": int(p.stat().st_mtime),
        })
    return out


# ===================== 审计（doc:add/delete 记录；doc:list/search 不入） =====================
# 直接追加 AUDIT_LOG_PATH（不自带 import utils.common——那会连带初始化 DB 连接）。
def _audit(op: str, detail: str) -> None:
    try:
        from datetime import datetime
        from config.config import AUDIT_LOG_PATH
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} | INFO | "
                    f"user_docs {op} | {detail}\n")
    except OSError:
        pass                                    # 审计不可用不阻断管理指令


# ===================== 自测（离线跑：python memory/user_docs.py） =====================
def _selftest() -> int:
    """轻量自检：构建 → 检索 → doc:* → 删除临时文档。返回退出码。"""
    n_before = len(_scan_files())
    try:
        load_user_docs()
    except Exception as e:                     # noqa: BLE001
        print(f"[自检] 对账失败：{e}")
        return 1
    hits = search_user_docs("我午饭一般在公司楼下的快餐店吃", top_k=3)
    print(f"[自检] 检索 top3 命中 {len(hits)} 条：")
    for h in hits:
        print(f"    - doc_id={h['doc_id']} score={h['score']} text={h['text'][:30]}...")
    tmp_title = f"自检临时文档_{n_before}"
    try:
        added = add_doc(tmp_title, "自检专用内容：每周一晨会，地点 2 楼会议室。")
        print(f"[自检] doc:add -> {added}")
        listed = [d["doc_id"] for d in list_docs()]
        assert added["doc_id"] in listed, "doc:list 应包含刚添加的文档"
        r2 = search_user_docs("晨会在哪里开", top_k=1)
        assert r2 and "2 楼会议室" in r2[0]["text"], f"doc:add 后应可检索到新内容（实际 {r2}）"
        delete_doc(added["doc_id"])
        assert added["doc_id"] not in [d["doc_id"] for d in list_docs()], "doc:delete 应删除文档"
        print("[自检] doc:add / search / doc:delete 全通过")
        return 0
    finally:
        # 兜底清理：临时文档必须删干净（文件系统是真相源，不残留测试数据）
        for rel in list(_scan_files()):
            if rel == f"{tmp_title}.md":
                (USER_DOCS_DIR / rel).unlink()


if __name__ == "__main__":
    sys.exit(_selftest())
