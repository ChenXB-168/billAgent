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
  - 分块按行（沿用已退役 `ragKnowledge.split_text` 的历史行切分语义），
    超长行按 `USER_DOCS_CHUNK_SIZE`（默认 128）硬切；
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

from config.config import (USER_DOCS_PATH, USER_DOCS_TOP_K, USER_DOCS_MAX_DOCS,
                           USER_DOCS_CHUNK_SIZE, EMBEDDING_ENABLED)

# ===================== 模块级配置（测试可注入） =====================
USER_DOCS_DIR: Path = USER_DOCS_PATH          # 真相源目录
MAX_DOCS: int = USER_DOCS_MAX_DOCS            # 文档数上限，超出明确报错
TOP_K_DEFAULT: int = USER_DOCS_TOP_K          # search_user_docs 默认返回条数
CHUNK_SIZE: int = USER_DOCS_CHUNK_SIZE        # 超长行硬切长度
_DOC_EXTS = (".md", ".txt")
# ★M24（D27）索引格式版本：**分块规则或词项构造变更时必须 +1**。
#   为什么必须有：`load_user_docs` 的对账依据是**文件内容哈希** —— 只要文档文件没改，
#   即便分块 / 分词逻辑变了也会**复用旧索引**，使改动**静默不生效**（本次修复即踩到该坑）。
#   v1 → v2：分块过滤 Markdown 标题行 + 文档名纳入 BM25 词项与向量输入。
_INDEX_VERSION = 2

# ★M25（D28）RRF 两路权重：**向量路 1.5 / 词法路 1.0**。
#   为什么向量路要加权：消费笔记是**口语化表述**，同一意图的措辞差异极大
#   （查询"每天交通花多少" vs 笔记"工作日通勤坐地铁"——**零词重叠**，纯词法路无法召回）；
#   向量路能跨表述匹配，是主力召回通道，词法路只作补充。
#   ★实证（两套形态不同的语料交叉验证，2026-10-10）：
#     · 真实库（65 篇）：hit@5 97.96% → **100.00%**、recall@5 96% → **98%**；
#     · fixture（55 篇专题）：持平（已饱和于 100%），**无副作用**；
#     · 反向验证：`vec_w=0.5`（偏词法）在两套语料上**分别崩到 95.92% / 67.57%**。
#   取 1.5 而非更大值：实测 1.5 与 2.0/3.0 结果相同（已达饱和），取温和值更稳。
_RRF_BM25_WEIGHT = 1.0
_RRF_VEC_WEIGHT = 1.5
_EMBED_AVAILABLE: bool | None = None          # None=未探测；首次 _embedding_client() 时定值（对账或检索路径）

# ===================== 索引落盘（userDocs/.index/ 下，四件套原子写） =====================
def _index_dir() -> Path:
    """
    函数功能与逻辑描述：
        返回用户文档索引目录（`userDocs/.index/`），作为四件套索引文件的统一父目录；
        纯路径拼接、无副作用，不创建目录（目录创建由 `_rebuild` 负责）。
    入参说明：
        无。
    返回值说明：
        Path：`USER_DOCS_DIR / ".index"` 路径对象（不保证已存在）。
    """
    return USER_DOCS_DIR / ".index"


def _meta_path() -> Path:
    """
    函数功能与逻辑描述：
        返回索引元信息文件路径（`.index/meta.json`），内容含 version / model_name / dim /
        vector_ok / next_id / files(rel→sha256)；纯路径拼接、无副作用。
    入参说明：
        无。
    返回值说明：
        Path：`_index_dir() / "meta.json"` 路径对象（不保证已存在）。
    """
    return _index_dir() / "meta.json"


def _store_path() -> Path:
    """
    函数功能与逻辑描述：
        返回分块原文存储文件路径（`.index/store.json`），落盘结构为 `{"chunks": [...]}`，
        每个分块含 id / doc_id / text；纯路径拼接、无副作用。
    入参说明：
        无。
    返回值说明：
        Path：`_index_dir() / "store.json"` 路径对象（不保证已存在）。
    """
    return _index_dir() / "store.json"


def _bm25_path() -> Path:
    """
    函数功能与逻辑描述：
        返回 BM25 分词结果落盘文件路径（`.index/bm25.pkl`），pickle 结构为
        `{"tokenized": [[token, ...], ...]}`，加载时直接据此重建 `_BM25` 免重新分词；
        纯路径拼接、无副作用。
    入参说明：
        无。
    返回值说明：
        Path：`_index_dir() / "bm25.pkl"` 路径对象（不保证已存在）。
    """
    return _index_dir() / "bm25.pkl"


def _faiss_path() -> Path:
    """
    函数功能与逻辑描述：
        返回 FAISS 向量索引落盘文件路径（`.index/faiss.index`）；向量不可用时写空字节，
        加载侧据 meta.vector_ok 决定是否读取。纯路径拼接、无副作用。
    入参说明：
        无。
    返回值说明：
        Path：`_index_dir() / "faiss.index"` 路径对象（不保证已存在）。
    """
    return _index_dir() / "faiss.index"


# ===================== 内存状态（整体替换，检索读快照） =====================
class _DocState:
    """
    函数/类功能与逻辑描述：
        一次对账 / 重建后的不可变索引快照；检索侧只读已安装的快照引用、不加锁，构建侧用
        `_build_lock` 串行化后整体替换引用，以此规避读写竞争。空库与构建失败也会以
        「空列表 / 全 None」的兜底快照形态存在（失败原因记入 last_error）。
    构造入参说明：
        meta (dict)：索引元信息，含 model_name / dim / vector_ok / next_id / files(rel→sha256)。
        chunks (list[dict])：分块列表，元素为 {"id": int, "doc_id": str, "text": str}。
        bm25 (object)：`_BM25` 实例；空库为 None。
        index (object)：faiss.IndexIDMap 或 None。
        ids (object)：分块 id 数组（np.ndarray/int64）；当前重建与加载路径中检索均不依赖它。
        vector_ok (bool)：向量检索是否可用。
        last_error (str)：最近一次构建 / 检索失败原因（诊断用），默认 ""。
    返回值说明：
        构造返回 _DocState 实例，属性与 `__slots__` 声明的 7 项一一对应。
    """

    __slots__ = ("meta", "chunks", "bm25", "index", "ids", "vector_ok", "last_error")

    def __init__(self, meta: dict, chunks: list[dict], bm25, index, ids,
                 vector_ok: bool, last_error: str = ""):
        """
        函数功能与逻辑描述：
            逐字段保存索引快照引用，不做任何校验、拷贝与 I/O；空库 / 失败兜底快照同样走本构造。
        入参说明：
            meta (dict)：索引元信息字典。
            chunks (list[dict])：分块列表。
            bm25 (object)：BM25 实例或 None。
            index (object)：FAISS 索引或 None。
            ids (object)：分块 id 数组或 None。
            vector_ok (bool)：向量检索是否可用。
            last_error (str)：失败原因，默认 ""。
        返回值说明：
            无（仅初始化实例属性，无副作用）。
        """
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
    """
    函数功能与逻辑描述：
        递归扫描真相源目录，对 `.md` / `.txt` 文件按**字节级 sha256** 建立 relpath → 摘要
        映射，用于启动对账比对（刻意不用 mtime——契约要求内容哈希对账）；跳过 `.index`
        目录与其它后缀。目录不存在时返回空字典（视为空库，不报错、不创建目录）。
    入参说明：
        无。
    返回值说明：
        dict[str, str]：{相对路径(posix 形式): sha256 十六进制摘要}；目录缺失 / 无匹配文件 → {}。
    """
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


# ★M24（D27）：Markdown ATX 标题行（`# 标题` ~ `###### 标题`）**不作为独立分块**。
#   理由：标题的职责是**组织文档**而非承载内容，单独成块会占掉 top_k 名额却不回答问题 ——
#   实测证据：真实库中查询「我怎么给消费分档」的 top1 曾是 `# 我的日常消费习惯`（纯标题）。
#   代价：标题里的主题词（如 `# 青龙镇水果物价记录`）不再被索引 ——
#   由**同批的「文档名纳入索引」**（`_doc_tokens`）补回，文档名通常比标题更规范稳定。
_HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def _chunk_lines(text: str) -> list[str]:
    """
    函数功能与逻辑描述：
        把单篇文档原文按行切分为分块：逐行 strip 后剔除空行与 **Markdown 标题行**
        （`^#{1,6}\\s`，M24/D27）；长度 ≤ `CHUNK_SIZE` 直接成块，超长行按 `CHUNK_SIZE`
        定长硬切为多块（沿用已退役 `ragKnowledge.split_text` 的历史行切分语义）。
        纯字符串处理，无 I/O、无副作用。
    入参说明：
        text (str)：文档完整原文。
    返回值说明：
        list[str]：分块文本列表；原文全为空白行 / 仅标题行时返回 []。
    """
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _HEADING_RE.match(line):   # ★标题行不进索引（理由见 _HEADING_RE 注释）
            continue
        if len(line) <= CHUNK_SIZE:
            chunks.append(line)
        else:
            for i in range(0, len(line), CHUNK_SIZE):
                chunks.append(line[i:i + CHUNK_SIZE])
    return chunks


def _doc_tokens(doc_id: str) -> list[str]:
    """
    函数功能与逻辑描述：
        M24（D27）：把**文档名**转成 BM25 的「文档级主题词项」。取文件名主干（去扩展名）、
        将 `_` 换成空格后走 `_tokenize`，使 `物价_水果_苹果.md` → `["物价","水果","苹果"]`。
        为什么需要：分块粒度是**行**，而行内往往不含文档主题词（如"水果"可能只出现在文档名里），
        导致「水果的物价怎么样」这类**主题级查询**召不回正确文档（实测 top1 曾落到无关笔记）。
        把文档名作为每个 chunk 的**公共词项**加入，等价于给"整篇"补一个主题信号 ——
        与 D27 的**标题行过滤**互补：标题承载的主题信息由文档名补回。
        注意：**只影响 BM25 词项，不改变 `chunk["text"]`** —— 返回给用户的仍是原文，
        评测 GT 的"字符串完全匹配"口径不受影响。
    入参说明：
        doc_id (str)：文档相对路径（如 "物价_水果_苹果.md" 或 "日常/购物.md"）。
    返回值说明：
        list[str]：文档级主题词项列表（文件名无有效字符时可能为空列表）。
    """
    stem = Path(doc_id).stem            # 去扩展名（保留纯文件名，不含子目录）
    return _tokenize(stem.replace("_", " "))


def _chunk_tokens(chunk: dict) -> list[str]:
    """
    函数功能与逻辑描述：
        M24（D27）：单个 chunk 的 BM25 词项 = **「文档名独有词」** + 正文本词。
        ★为什么只加"文档名独有词"（而非文档名全量）—— 实测 A/B 证据：
        把文档名**全量**加入 BM25，在 fixture（55 篇**专题**笔记）上 hit@5 83.78% → 97.30%，
        但在**真实库**（10 篇**综合**笔记 + 55 篇专题混排）上 hit@1 反而 46.15% → 38.46%。
        根因：综合笔记的文件名偏泛化（`消费习惯.md`），与专题文件名（`习惯_冲动消费控制.md`）
        **语义重叠** —— 泛化词同时出现在多篇文档名里 → df 抬高、IDF 下降 → 区分度反被稀释。
        故收敛为「**只补信息缺口**」：仅取"文档名里有、而该 chunk 正文里没有"的主题词。
        效果：`洗护`（正文写的是"洗发水"）仍被补入 ✓；`消费`（正文早已出现）不重复放大 ✓。
    入参说明：
        chunk (dict)：分块字典，需含 "doc_id"（文档名）与 "text"（正文）。
    返回值说明：
        list[str]：该 chunk 的 BM25 词项（文档名独有词在前，正文词在后）。
    """
    text_tokens = _tokenize(chunk["text"])
    text_set = set(text_tokens)
    doc_only = [t for t in _doc_tokens(chunk["doc_id"]) if t not in text_set]
    return doc_only + text_tokens


def _tokenize(text: str) -> list[str]:
    """
    函数功能与逻辑描述：
        分词入口：优先惰性导入 jieba 做中文分词（按 jieba.lcut 结果过滤纯空白 token）；
        导入失败或分词抛异常时，降级为正则切分——按连续「非单词字符且非中日韩汉字」串
        （即标点与空白）切断并丢弃空片段，保证离线无 jieba 时仍可用。
    入参说明：
        text (str)：待分词文本。
    返回值说明：
        list[str]：分词结果（jieba 路径已过滤空白 token；降级路径为按标点 / 空白切分后的片段）。
    """
    try:
        import jieba
        return [t for t in jieba.lcut(text) if t.strip()]
    except Exception:
        return [t for t in re.split(r"[^\w\u4e00-\u9fa5]+", text) if t]


class _BM25:
    """
    函数/类功能与逻辑描述：
        轻量 BM25（Okapi，k1=1.5 / b=0.75）实现，不引入 rank_bm25 依赖（环境未安装）；
        构造时即建倒排索引（term -> {doc_i: freq}）并预计算平均文档长度，`get_scores` 只做
        纯词法打分，行为与 rank_bm25 对齐。加载落盘索引时由已保存的 tokenized 列表直接重建，
        无需重新分词。
    构造入参说明：
        docs_tokens (list[list[str]])：已分词文档集合，每个元素是一篇（块）的 token 列表。
    返回值说明：
        构造返回 _BM25 实例；打分接口见 `get_scores`。
    """

    def __init__(self, docs_tokens: list[list[str]]):
        """
        函数功能与逻辑描述：
            保存 token 列表并预计算文档数 n_docs、平均长度 avgdl、逐文档长度 doc_len，
            同时构建 term → {文档下标: 词频} 的倒排索引（postings），供打分阶段取用；
            空文档集合时 avgdl 记 0.0、postings 为空。
        入参说明：
            docs_tokens (list[list[str]])：已分词文档集合。
        返回值说明：
            无（仅初始化实例属性，无副作用）。
        """
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
        """
        函数功能与逻辑描述：
            对查询 token **去重**后逐个查倒排索引，按 BM25 公式累加各文档得分：
            idf = ln(1 + (n - df + 0.5)/(df + 0.5))，tf 分子 freq*(k1+1) 并做长度归一化
            （k1=1.5 / b=0.75）；未命中的 token 直接跳过，无 I/O、无副作用。
        入参说明：
            query_tokens (list[str])：查询文本的分词结果。
        返回值说明：
            list[float]：长度等于文档数的得分列表，下标即文档下标（未命中为 0.0）；
                空库（n_docs=0）返回 []；avgdl 为 0 时 tf 记 0.0。
        """
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
    """
    函数功能与逻辑描述：
        惰性获取 embedding 单例（`modelService.embedding_loader.embedding_client`），
        使 `import memory.user_docs` 不拉起 torch / SentenceTransformer；首次调用时把
        「模型是否已加载」记忆到模块级 `_EMBED_AVAILABLE`（None 时才探测）。导入失败或
        异常时置 `_EMBED_AVAILABLE=False` 并返回 None，由调用方降级 BM25-only。
    入参说明：
        无。
    返回值说明：
        单例对象（其 `model` 属性可能尚未加载，需调用方自行判空）；导入失败 / 异常 → None。
    """
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
    """
    函数功能与逻辑描述：
        批量向量化：按每批 256 条调用 `get_batch_embedding`；任一批抛异常、返回条数与请求
        不符或含空向量，即整体判定失败并返回 (None, 0)，保证「文本↔向量」一一对应不漂移，
        由调用方降级 BM25-only。全部成功时返回向量列表与单条维度。
    入参说明：
        texts (list[str])：待向量化文本列表。
    返回值说明：
        tuple[list[list[float]] | None, int]：
            - 成功：(vectors, dim)，dim 为单条向量维度（取首条长度）。
            - 无 embedding 客户端 / 任一批失败：(None, 0)。
            - 空输入：( [], 0)。
    """
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
    """
    函数功能与逻辑描述：
        原子写 JSON：先写同目录临时文件（原文件名追加 `.tmp` 后缀），再 `os.replace` 覆盖
        目标，避免写入中途崩溃 / 被杀留下半文件；序列化用 UTF-8 且 `ensure_ascii=False`
        以保留中文原文。
    入参说明：
        path (Path)：目标文件路径（其父目录需已存在）。
        obj：任意可被 json 序列化的对象。
    返回值说明：
        无（副作用：写入临时文件并原子替换目标文件）。
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """
    函数功能与逻辑描述：
        原子写二进制：先写临时文件（原文件名追加 `.tmp` 后缀），再 `os.replace` 覆盖目标，
        避免半文件；用于 FAISS 索引与 BM25 pickle 落盘。
    入参说明：
        path (Path)：目标文件路径（其父目录需已存在）。
        data (bytes)：待写入的二进制内容；空库 / 向量不可用时可为 b""（写空文件）。
    返回值说明：
        无（副作用：写入临时文件并原子替换目标文件）。
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ===================== 全量重建 =====================
def _rebuild(files: dict[str, str]) -> None:
    """
    函数功能与逻辑描述：
        全量重建索引：按 `_chunk_lines` 分块（chunk id 从 0 递增）→ jieba 分词构建 `_BM25`
        → 分批 embedding 后建 FAISS（L2 + IndexIDMap）；embedding 任一步失败即整库降级
        BM25-only（vector_ok=False、dim=0）。随后原子落盘四件套（faiss.index / bm25.pkl /
        store.json / meta.json），最后整体替换内存快照 `_state`。调用方须自行持有 `_build_lock`。
    入参说明：
        files (dict[str, str])：内容哈希对账结果 {relpath: sha256}，同时原样写入 meta.files。
    返回值说明：
        无（副作用：创建 `.index/` 目录、写 4 个索引文件、替换模块级 `_state`）。
    """
    _index_dir().mkdir(parents=True, exist_ok=True)

    chunks: list[dict] = []
    next_id = 0
    for rel in sorted(files):
        text = (USER_DOCS_DIR / rel).read_text(encoding="utf-8", errors="replace")
        for line_text in _chunk_lines(text):
            chunks.append({"id": next_id, "doc_id": rel, "text": line_text})
            next_id += 1

    # --- BM25（jieba 分词，全量；内置 _BM25 不依赖 rank_bm25） ---
    # ★M24（D27）：每个 chunk 的词项 = **文档名独有词** + 正文词（构造见 `_chunk_tokens`）。
    #   只补"正文里没有的文档名主题词"，用于救"主题词只在文档名出现"的查询；
    #   `chunk["text"]` 保持**原文不变** —— 返回给用户的内容与评测 GT 的
    #   "字符串完全匹配"口径均不受影响。
    tokenized = [_chunk_tokens(c) for c in chunks]
    bm25 = _BM25(tokenized) if tokenized else None

    # --- 向量（分批 embedding；任一条失败即整库降级，保证"文本↔向量"不漂移） ---
    vector_ok = False
    dim = 0
    model_name = ""
    faiss = index = ids = None
    if chunks:
        # ★M24（D27）：向量输入同样带上文档名（与 BM25 侧对称）—— 主题级查询
        #   （如"水果的物价怎么样"）在正文行里可能不出现主题词，文档名是唯一线索。
        vectors, dim = _batch_embed([f"{Path(c['doc_id']).stem} {c['text']}" for c in chunks])
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
        "version": _INDEX_VERSION,
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
    """
    函数功能与逻辑描述：
        读取 embedding 模型标识（取 `modelService.embedding_loader.EMB_MODEL_PATH` 的
        basename），记入 meta 用于「模型换代识别」；导入失败 / 属性为空时返回 "unknown"，
        不抛异常。
    入参说明：
        无。
    返回值说明：
        str：模型目录名；无法获取时返回 "unknown"。
    """
    try:
        from modelService import embedding_loader as el
        return Path(getattr(el, "EMB_MODEL_PATH", "")).name or "unknown"
    except Exception:                          # noqa: BLE001
        return "unknown"


def _install_state(state: _DocState | None) -> None:
    """
    函数功能与逻辑描述：
        以整体引用替换（而非就地修改）的方式安装最新索引快照，使检索侧读到的始终是完整一致的
        `_DocState`；构建 / 对账路径须在 `_build_lock` 内调用。
    入参说明：
        state (_DocState | None)：新快照；None 表示清空当前状态（现有调用方均传实例）。
    返回值说明：
        无（副作用：改写模块级全局 `_state`）。
    """
    global _state
    _state = state


def _load_existing(meta: dict) -> _DocState | None:
    """
    函数功能与逻辑描述：
        按 meta 从落盘索引反向加载内存快照：store.json 取分块、bm25.pkl 取 tokenized 重建
        `_BM25`、meta.vector_ok 为真时反序列化 faiss.index（空文件视为缺失）；任一文件缺失 /
        损坏 / 结构不符一律返回 None，交由调用方触发全量重建，不向外抛异常。
        该加载路径下快照 ids 传 None（检索不依赖它）。
    入参说明：
        meta (dict)：`.index/meta.json` 的解析结果，至少需含 vector_ok 键。
    返回值说明：
        _DocState：加载成功的快照；任一环节异常 / 文件缺失 → None。
    """
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
    """
    函数功能与逻辑描述：
        启动对账（幂等）：扫描 userDocs/ 内容哈希，与 `.index/meta.json` 的 files 比对——
        一致则尝试加载落盘索引（加载失败仍全量重建），不一致（增 / 删 / 改）则全量重建。
        向量能力一致性校验（vector_ok 或 dim 与现状不符 → 重建）**仅在 embedding 模型已加载
        时**进行，避免为校验而强制加载模型拖慢冷启动；模型未加载时信任磁盘索引（见函数内注释）。
        全过程由 `_build_lock` 串行化，可重入。
    入参说明：
        无。
    返回值说明：
        无（副作用：可能全量重建并落盘索引、替换 `_state`）；
            文档数超 `MAX_DOCS` 时抛 `ValueError`（调用方自行决定记录还是退出）。
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
            # ★M24（D27）索引格式版本校验：分块规则 / 词项构造变更后必须**全量重建**。
            #   对账依据是**文件内容哈希**，与索引格式无关 —— 若只比对 files，则
            #   "改了检索逻辑但文档未变"会静默复用旧索引（本次修复踩到的坑）。
            if meta.get("version") != _INDEX_VERSION:
                _rebuild(files)
                return
            # 向量能力状态一致性校验（模型装了/换了/坏了 → 全量重建对齐）**仅在模型已加载**
            # 时进行。模型走懒加载（首次编码才加载，见 embedding_loader）：启动对账若强制
            # _probe_embedding 会"为校验而启动即加载模型"拖慢冷启动，故未加载时信任磁盘索引
            # （vector_ok 记于 meta 直接载入）。模型不可用/换代的兜底：检索向量路按 dim 不符
            # 自动跳过（见 search_user_docs），文档增删改（files 变化）自然触发全量重建对齐。
            client = _embedding_client()
            model_ready = client is not None and getattr(client, "model", None) is not None
            # ★M25（D28）索引"缺向量"必须自愈 —— 上述"仅模型已加载时校验"有一个**自愈盲区**：
            #   模型是**懒加载**的，而对账发生在**启动期**，此时模型往往尚未加载 → 校验被跳过
            #   → **残缺索引（vector_ok=False）被长期复用**，检索静默退化为 BM25-only。
            #   实测代价极大：同一 50 条 GT 下 hit@1 **75.51% → 51.02%**、recall@5 96% → 90%
            #   （三场景对照见 `设计/23` §2.2）。
            #   故：当磁盘索引标记为"无向量"且 embedding 功能开启时，**强制探测一次**
            #   （代价 = 一次模型加载）以触发重建、修复残缺索引。
            #   ★这是**单向**的：vector_ok=True 的索引仍走原懒加载路径，不拖慢冷启动。
            if not meta.get("vector_ok") and EMBEDDING_ENABLED:
                model_ready = True
            if model_ready:
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
    """
    函数功能与逻辑描述：
        用固定文本「探测」试编码一次，探测 embedding 是否可用并取得向量维度，供对账时判断
        是否需因模型装 / 换 / 坏而全量重建；不改变全局状态，异常被吞。
    入参说明：
        无。
    返回值说明：
        tuple[int, bool]：(维度, 是否可用)；
            编码成功且向量非空 → (len(vec), True)；无客户端 / 编码抛异常 / 空向量 → (0, False)。
    """
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
    """
    函数功能与逻辑描述：
        获取当前索引快照；未初始化时在 `_build_lock` 内**双重检查**后调用 `load_user_docs()`
        懒加载。加载抛 `ValueError`（超上限）或其它异常时**不向外传播**，改为安装一个空快照
        （空 meta / chunks / bm25 / index / ids，vector_ok=False）并把原因写入 `last_error`，
        保证检索调用方不因构建失败而崩溃。
    入参说明：
        无。
    返回值说明：
        _DocState：当前快照；构建失败时为含 `last_error` 的空快照（不返回 None）。
    """
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
    """
    函数功能与逻辑描述：
        用户文档库**唯一检索入口**（`10` §5.5 / §5.6.6）：BM25 词法路与 FAISS 向量路各召回
        `top_k*3` 个候选并记 1-based 名次，按 RRF（score = Σ 1/(60 + rank)）融合后取前 top_k。
        向量路仅在 embedding 客户端存在且查询向量维度等于 meta.dim 时参与；BM25 检索失败 /
        向量检索失败只记入 `last_error` 并继续（不阻断）；空 query 或空库直接返回 []，
        任一异常路径均不外抛。
    入参说明：
        query (str)：自然语言查询文本；None / 纯空白视为无效查询。
        top_k (int)：返回条数，默认 `TOP_K_DEFAULT`；内部夹紧到 [1, 当前分块总数]。
    返回值说明：
        list[dict]：[{"doc_id": 相对路径, "text": 分块原文, "score": 融合分(6 位小数)}, ...]，
            按融合分降序、同分按 chunk id 升序；空 query / 空库 / 无命中 → []。
    """
    if not query or not query.strip():
        return []
    state = _ensure_state()
    if state is None or not state.chunks:
        return []
    top_k = max(1, min(int(top_k), len(state.chunks)))

    # --- 两路召回，各记名次，RRF 融合 ---
    # chunk id -> 命中过的检索路 (权重, 名次)；名次为 1-based
    ranks: dict[int, list[tuple[float, int]]] = {}
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
        ranks.setdefault(cid, []).append((_RRF_BM25_WEIGHT, rank))

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
        ranks.setdefault(cid, []).append((_RRF_VEC_WEIGHT, rank))

    # RRF：score = Σ w / (60 + rank)（w 取值见 `_RRF_BM25_WEIGHT` / `_RRF_VEC_WEIGHT`）
    fused = {cid: sum(w / (60 + r) for w, r in rs) for cid, rs in ranks.items()}
    picked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return [{"doc_id": state.chunks[cid]["doc_id"],
             "text": state.chunks[cid]["text"],
             "score": round(float(sc), 6)} for cid, sc in picked]


def iter_doc_chunks() -> Iterator[str]:
    """
    函数功能与逻辑描述：
        惰性迭代当前索引内全部分块原文（与检索读同一快照），供评测侧取 available 集合，
        保证「评测判可用」与「索引实际收录」同源，防止口径漂移；快照为空时不产出任何元素。
    入参说明：
        无。
    返回值说明：
        Iterator[str]：分块原文生成器；空快照为空生成器（无副作用）。
    """
    state = _ensure_state()
    if state is None:
        return
    for c in state.chunks:
        yield c["text"]


# ===================== 对外：doc:* 管理指令 =====================
_TITLE_RE = re.compile(r"^[\w\u4e00-\u9fa5-]{1,60}$")


def _safe_component(name: str, field: str) -> str:
    """
    函数功能与逻辑描述：
        校验并归一化路径组件（标题 / 类别）：先做 `(name or "").strip()`，再要求匹配
        `_TITLE_RE`（仅中文 / 字母 / 数字 / 下划线 / 中划线，长度 1-60）且不等于 "." / ".."，
        以防路径穿越与非法文件名；校验失败抛 `ValueError`，错误信息带上 `field` 便于定位。
    入参说明：
        name (str)：待校验名称，允许 None / 空串（空串会因不匹配正则而报错）。
        field (str)：字段中文名，仅用于拼装错误信息。
    返回值说明：
        str：strip 后的合法名称；不合法则抛 `ValueError`（无正常返回值）。
    """
    name = (name or "").strip()
    if not _TITLE_RE.match(name) or name in (".", ".."):
        raise ValueError(f"{field} 非法：仅允许中文/字母/数字/下划线/中划线，长度 1-60（当前={name!r}）")
    return name


def add_doc(title: str, content: str, category: str = "") -> dict:
    """
    函数功能与逻辑描述：
        doc:add——新增 / 覆盖一篇用户文档（低频管理指令，调用方先做权限与审计）：
        校验标题与类别 → 校验内容非空 → 现有文档数不得已达 `MAX_DOCS` 上限 → **先写文件
        （真相源）再重建索引**（写失败不碰索引；索引重建失败则抛 `ValueError` 提示下次
        `load_user_docs` 自动修复）→ 记审计 → 统计该文档分块数返回。
        `category` 为空时文档落在 userDocs/ 根目录，否则落在 "<category>/" 子目录，正文以
        UTF-8 写入且文件名固定补 ".md"。
    入参说明：
        title (str)：文档标题，同时作为文件名（不含扩展名），须满足 `_safe_component` 约束。
        content (str)：文档正文，strip 后为空则拒绝写入。
        category (str)：类别（子目录名），默认 ""（根目录）；非空时须满足 `_safe_component` 约束。
    返回值说明：
        dict：{"doc_id": 相对路径, "chunk_count": 该文档分块数}；
            标题 / 类别非法、内容为空、文档数超上限、索引重建失败 → 抛 `ValueError`（无返回值）。
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
    """
    函数功能与逻辑描述：
        doc:delete——按 doc_id（userDocs/ 下的相对路径）删除文档并重建索引：先校验 doc_id
        非空、`resolve()` 后的绝对路径仍位于 `USER_DOCS_DIR` 之内（防目录穿越）且确为文件，
        再 `unlink`，然后持 `_build_lock` 调 `load_user_docs()` 重建，最后记审计。
    入参说明：
        doc_id (str)：userDocs/ 下的相对路径，如 "餐饮/口味偏好.md"。
    返回值说明：
        bool：删除成功恒返回 True；
            doc_id 为空 / 越界 / 目标不存在（非文件）→ 抛 `ValueError`（无返回值）。
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
    """
    函数功能与逻辑描述：
        doc:list——列出文档元数据，**不触发索引构建**，直接扫描文件系统取信息；
        `category` 非空时按 "<category>/" 前缀过滤（只匹配一级子目录），结果按 doc_id 升序。
    入参说明：
        category (str)：类别（子目录名）过滤条件，默认 ""（不过滤，列出全部文档）。
    返回值说明：
        list[dict]：[{"doc_id","category","size","modified_at"}, ...]；
            category 取相对路径首段（根目录文档为 ""），size 为字节数，modified_at 为
            mtime 的秒级时间戳；无匹配文档 → []。
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
    """
    函数功能与逻辑描述：
        审计落日志：向 `AUDIT_LOG_PATH` 追加一行
        "时间(秒级 ISO) | INFO | user_docs <op> | <detail>"。为免 `import utils.common`
        连带初始化 DB 连接，这里直接 open 追加；仅捕获 `OSError`（日志不可写时静默忽略，
        不阻断管理指令）。`doc:add` / `doc:delete` 调用本函数；`doc:list` / `search_user_docs`
        不入审计（高频只读）。
    入参说明：
        op (str)：操作名，如 "doc:add" / "doc:delete"。
        detail (str)：操作明细文本（doc_id / title / category 等）。
    返回值说明：
        无（副作用：尽力追加一行审计日志；写失败静默丢弃）。
    """
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
    """
    函数功能与逻辑描述：
        轻量离线自检（`python memory/user_docs.py` 触发）：对账 → 检索 top3 打印 → doc:add
        → doc:list 校验包含 → 检索新内容校验可命中 → doc:delete 校验已删除；`finally` 中以
        `_scan_files()` 计数命名兜底删除临时文档，确保文件系统真相源不残留测试数据
        （步骤中的 assert 失败会向上抛出，不被吞）。对账失败时打印原因并提前返回。
    入参说明：
        无。
    返回值说明：
        int：进程退出码，0 = 全部通过；对账失败返回 1（异常路径由 assert / 抛错体现）。
    """
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
