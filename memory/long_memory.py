import os
import pickle
import re
import datetime
import numpy as np
import faiss
import jieba
from modelService.embedding_loader import embedding_client
from config.config import RAG_PATH

# 长期记忆持久化文件
LONG_MEM_PATH = os.path.join(RAG_PATH, "long_consume_mem.pkl")

# 内存缓存所有记忆文本
_long_mem_texts: list[str] = []
# 长期记忆专属FAISS向量索引
_long_mem_index = None
# 向量维度
_embedding_dim = None


def _init_index(dim: int):
    """初始化FAISS索引，L2距离计算相似度"""
    global _long_mem_index
    _long_mem_index = faiss.IndexFlatL2(dim)


# ─────────── 惰性加载（2026-09-05 重构：import / 进程启动不再立即加载 embedding） ───────────
# 读文本是纯 IO（毫秒级）；向量索引需要 embedding 模型（首次 ~10s，见 embedding_loader 懒加载）。
# 拆成两步后：只有真正写/查向量时才触发模型加载一次并常驻；纯记账 / 关键词检索不拖慢启动。
_TEXTS_LOADED = False


def _load_texts() -> None:
    """读盘记忆文本（快，不触发 embedding）。"""
    global _long_mem_texts
    if os.path.exists(LONG_MEM_PATH):
        try:
            with open(LONG_MEM_PATH, "rb") as f:
                _long_mem_texts = pickle.load(f)
        except Exception:  # pragma: no cover —— 持久化损坏按"无记忆"处理
            _long_mem_texts = []


def _ensure_loaded() -> None:
    """首次需要记忆前的惰性文本加载（幂等）。"""
    global _TEXTS_LOADED
    if _TEXTS_LOADED:
        return
    _TEXTS_LOADED = True
    _load_texts()


def _ensure_index() -> None:
    """文本已存在但向量索引未建时构建（触发 embedding 懒加载一次；不可用则保持 None 走 jieba 路）。"""
    global _long_mem_index, _embedding_dim
    if _long_mem_index is not None or not _long_mem_texts:
        return
    try:
        vectors = []
        for text in _long_mem_texts:
            vec = embedding_client.get_embedding(text)
            if vec:
                vectors.append(vec)
        if vectors:
            _embedding_dim = len(vectors[0])
            _init_index(_embedding_dim)
            _long_mem_index.add(np.array(vectors, dtype="float32"))
    except Exception:  # pragma: no cover —— 模型故障时保持 jieba 路可用
        _long_mem_index = None


def _load_mem():
    """强制从磁盘重读文本并重建向量索引（兼容旧入口/测试；常规启动已不再自动调用）。"""
    global _TEXTS_LOADED, _long_mem_index, _embedding_dim
    _TEXTS_LOADED = False
    _long_mem_index = None
    _embedding_dim = None
    _ensure_loaded()
    _ensure_index()


def save_consume_memory(summary_text: str):
    """月度报表生成后，存入长期向量记忆，同步更新索引"""
    global _embedding_dim
    _ensure_loaded()
    _ensure_index()

    vec = embedding_client.get_embedding(summary_text)
    if vec is None:
        _long_mem_texts.append(summary_text)
    else:
        _long_mem_texts.append(summary_text)
        if _long_mem_index is None:
            _embedding_dim = len(vec)
            _init_index(_embedding_dim)
        vec_array = np.array([vec], dtype="float32")
        _long_mem_index.add(vec_array)

    # 持久化到本地文件
    with open(LONG_MEM_PATH, "wb") as f:
        pickle.dump(_long_mem_texts, f)

# 月度习惯/摘要文本的月份前缀提取（【YYYY-MM 开头），供按月份替换与滚动裁剪
_MONTH_PREFIX = re.compile(r"^【(\d{4}-\d{2})")


def build_month_habit_text(month: str, habits: list) -> str:
    """根据月度消费习惯聚合数据生成标准摘要文本（长期记忆语义层格式）

    :param month: 月份 YYYY-MM
    :param habits: 该月聚合结果列表，元素为 {"category","amount_sum","count","avg_amount"}
    :return: 如 【2026-08 消费习惯】餐饮12笔1560.00元, 交通8笔420.00元; 总45笔6280.00元
    """
    if not habits:
        return ""
    cat_parts = [
        f"{h['category']}{int(h['count'])}笔{float(h['amount_sum']):.2f}元"
        for h in habits
    ]
    total_amount = sum(float(h["amount_sum"]) for h in habits)
    total_count = sum(int(h["count"]) for h in habits)
    return f"【{month} 消费习惯】" + ", ".join(cat_parts) + f"; 总{total_count}笔{total_amount:.2f}元"


def _keep_recent_months_boundary(keep_months: int = 12) -> str:
    """滚动保留边界：含当月往前共 keep_months 个自然月的最早月份 YYYY-MM"""
    now = datetime.datetime.now()
    y, m = now.year, now.month
    for _ in range(keep_months - 1):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return f"{y:04d}-{m:02d}"


def _rebuild_index_and_persist():
    """整体重建FAISS索引并落盘（月度粒度下文本量小，12个月最多约60条，开销可接受）"""
    global _long_mem_index, _embedding_dim
    _long_mem_index = None
    _embedding_dim = None
    if _long_mem_texts:
        vectors = []
        for text in _long_mem_texts:
            vec = embedding_client.get_embedding(text)
            if vec is not None:
                vectors.append(vec)
        if vectors:
            _embedding_dim = len(vectors[0])
            _init_index(_embedding_dim)
            _long_mem_index.add(np.array(vectors, dtype="float32"))
    with open(LONG_MEM_PATH, "wb") as f:
        pickle.dump(_long_mem_texts, f)


def upsert_month_habit_memory(month: str, habits: list) -> bool:
    """记账成功后更新某月消费习惯到长期记忆（语义层）

    - 按月份前缀替换该月旧摘要文本，不存在则新增
    - 12个月滚动清理早于边界月份的记忆文本
    - 整体重建FAISS索引并落盘 pkl
    静默失败返回 False，不影响主流程。
    """
    global _long_mem_texts
    try:
        _ensure_loaded()
        text = build_month_habit_text(month, habits)
        if not text:
            return False
        replaced = False
        for i, t in enumerate(_long_mem_texts):
            m = _MONTH_PREFIX.match(t)
            if m and m.group(1) == month:
                _long_mem_texts[i] = text
                replaced = True
                break
        if not replaced:
            _long_mem_texts.append(text)
        # 12个月滚动：仅保留边界月份及之后
        boundary = _keep_recent_months_boundary()
        _long_mem_texts = [
            t for t in _long_mem_texts
            if not (m := _MONTH_PREFIX.match(t)) or m.group(1) >= boundary
        ]
        _rebuild_index_and_persist()
        return True
    except Exception:
        return False


# 汉字数字月份 -> 阿拉伯数字（"一月"~"十二月"），供 _extract_time_tokens 归一化
_CN_MONTH = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "十一": 11, "十二": 12,
}


def _cn_month_to_arabic(query: str) -> str:
    """把汉字数字月份归一化为阿拉伯数字（"去年十二月" -> "去年12月"）。

    与 jieba 切碎阿拉伯数字同理："一月" 会被切成 "一"+"月" 两个单字后蒸发，
    且记忆文本里只存在 ISO 格式。这里在时间提取前统一归一化，让后续所有
    \d{1,2} 月 的正则分支直接复用。
    """
    def repl(m):
        return f"{_CN_MONTH.get(m.group(1), 0)}月" if m.group(1) in _CN_MONTH else m.group(0)
    return re.sub(r"([一二三四五六七八九十]{1,2})月", repl, query)


def _extract_time_tokens(query: str) -> list:
    """从用户查询中提取时间表达，翻译成能匹配记忆文本 ISO 日期(YYYY-MM-DD)的 token。

    背景：记忆文本里的月份是 "2026-01" 这类 ISO 格式，而用户口语是 "1月"。
    jieba 会把 "1月" 切成单字 "1" + "月"，单字又被 len>=2 过滤，导致关键词路
    对月份零贡献；向量路对具体数值又不敏感。这里用正则把口语时间翻译成
    记忆文本里实际存在的格式，作为强信号单独加分。

    返回 [(type, value)]：
      ("date",  "YYYY-MM")  子串匹配记忆文本（如 "2026-01"）
      ("month", 1)          边界匹配 ISO 日期中的 "-01-"（只匹配月份位置）
    """
    query = _cn_month_to_arabic(query)  # 汉字数字月份先归一化，其余分支全部复用
    tokens = []
    now = datetime.datetime.now()

    # 1) 带年份："2026年1月" / "2026年01月"
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月", query):
        tokens.append(("date", f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"))

    # 2) 去年/今年 X月（隐含年份）
    m = re.search(r"去年\s*(\d{1,2})\s*月", query)
    if m:
        tokens.append(("date", f"{now.year - 1}-{int(m.group(1)):02d}"))
    m = re.search(r"今年\s*(\d{1,2})\s*月", query)
    if m:
        tokens.append(("date", f"{now.year}-{int(m.group(1)):02d}"))

    # 3) 单独月份："1月" / "12月"
    #    (?<!\d年)(?<!\d) 排除已带年份的情况（避免与 1) 重复），"去年12月" 仍能命中
    for m in re.finditer(r"(?<!\d年)(?<!\d)(\d{1,2})\s*月", query):
        tokens.append(("month", int(m.group(1))))

    # 4) 相对时间："上个月" / "最近一个月" -> 当前日期的上个月
    if re.search(r"上个月|上上月|最近.{0,2}个月|近.{0,2}个月", query):
        first = now.replace(day=1)
        prev = first - datetime.timedelta(days=1)
        tokens.append(("date", f"{prev.year:04d}-{prev.month:02d}"))

    return tokens


def search_history_consume(query: str, top_k: int = 3) -> str:
    """
    轻量混合检索：向量语义 + jieba分词关键词精确匹配 融合排序
    :param query: 用户查询文本
    :param top_k: 返回最匹配的条数
    :return: 拼接好的历史记忆文本
    """
    _ensure_loaded()
    _ensure_index()
    if not _long_mem_texts:
        return "暂无历史消费记录"

    # 存储每条记忆的最终得分，下标对应记忆列表下标
    scores = [0.0] * len(_long_mem_texts)

    # ========== 第一路：向量语义检索打分 ==========
    query_vec = embedding_client.get_embedding(query)
    if query_vec is not None and _long_mem_index is not None:
        real_k = min(top_k * 2, len(_long_mem_texts))
        query_array = np.array([query_vec], dtype="float32")
        distances, indices = _long_mem_index.search(query_array, real_k)

        # L2距离转相似度分：距离越小，相似度越高，得分越高
        max_dist = max(distances[0]) if max(distances[0]) > 0 else 1
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(_long_mem_texts):
                # 语义相似度权重 0.6
                sim_score = (1 - distances[0][i] / max_dist) * 0.6
                scores[idx] += sim_score

    # ========== 第二路：jieba中文分词关键词精确匹配打分 ==========
    # 精准切分中文词汇，解决中文无空格问题
    raw_words = jieba.lcut(query)
    # 过滤单字、空字符，只保留有效关键词
    keywords = [word.strip() for word in raw_words if len(word.strip()) >= 2]
    # 额外提取时间表达（口语月份 -> 记忆文本 ISO 日期格式），弥补 jieba 切碎数字
    time_tokens = _extract_time_tokens(query)
    if keywords or time_tokens:
        for idx, text in enumerate(_long_mem_texts):
            hit_count = 0
            for kw in keywords:
                if kw.isdigit() and len(kw) >= 2:
                    # 纯数字用边界匹配，避免 "12" 误命中金额里的 "1200"
                    if re.search(rf"(?<!\d){re.escape(kw)}(?!\d)", text):
                        hit_count += 1
                else:
                    if kw in text:
                        hit_count += 1
            # 关键词权重0.4
            keyword_score = (hit_count / len(keywords)) * 0.4 if keywords else 0.0
            scores[idx] += keyword_score
            # 时间表达命中：强信号，单独加权（每个 +0.15，上限 +0.3，不稀释关键词权重）
            time_hits = 0
            for ttype, tval in time_tokens:
                if ttype == "date" and tval in text:
                    time_hits += 1
                elif ttype == "month" and re.search(rf"-{tval:02d}-", text):
                    time_hits += 1
            scores[idx] += min(time_hits, 2) * 0.15

    # ========== 按最终得分排序，取top_k ==========
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    result_texts = []
    for idx, score in ranked:
        if score <= 0:
            break
        result_texts.append(_long_mem_texts[idx])
        if len(result_texts) >= top_k:
            break

    # 兜底：无匹配返回最新N条
    if not result_texts:
        result_texts = _long_mem_texts[-top_k:]

    return "\n".join(result_texts)


def clear_all_long_mem():
    """清空全部长期记忆（测试/重置用）"""
    global _long_mem_texts, _long_mem_index, _embedding_dim, _TEXTS_LOADED
    _long_mem_texts = []
    _long_mem_index = None
    _embedding_dim = None
    _TEXTS_LOADED = True
    if os.path.exists(LONG_MEM_PATH):
        os.remove(LONG_MEM_PATH)