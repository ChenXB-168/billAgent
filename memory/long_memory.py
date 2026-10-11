# ==============================================
# 长期记忆 - 月度消费习惯的向量语义层 + 混合检索
# 说明：文本清单持久化在 RAG_PATH/long_consume_mem.pkl，向量索引（FAISS IndexFlatL2）内存态
#       采用惰性加载：读文本为纯 IO，向量索引才需要 embedding 模型（首次约 10s）
# ==============================================
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
    """
    函数功能与逻辑描述：
        用给定维度新建 FAISS 平铺 L2 索引（IndexFlatL2）并替换全局 _long_mem_index；
        仅创建索引对象、不加入任何向量，亦不做异常处理（由调用方守卫）。
    入参说明：
        dim (int)：向量维度，须与 embedding 模型输出维度一致。
    返回值说明：
        无（副作用：改写全局 _long_mem_index）。
    """
    global _long_mem_index
    _long_mem_index = faiss.IndexFlatL2(dim)


# ─────────── 惰性加载（2026-09-05 重构：import / 进程启动不再立即加载 embedding） ───────────
# 读文本是纯 IO（毫秒级）；向量索引需要 embedding 模型（首次 ~10s，见 embedding_loader 懒加载）。
# 拆成两步后：只有真正写/查向量时才触发模型加载一次并常驻；纯记账 / 关键词检索不拖慢启动。
_TEXTS_LOADED = False


def _load_texts() -> None:
    """
    函数功能与逻辑描述：
        从 LONG_MEM_PATH 反序列化（pickle）记忆文本到 _long_mem_texts。纯磁盘 IO、毫秒级，
        不触发 embedding 模型加载；文件不存在时保持 _long_mem_texts 现状（不覆盖、不置空），
        文件损坏 / 读取失败按「无记忆」处理（置空列表，不抛出）。
    入参说明：
        无。
    返回值说明：
        无（副作用：改写全局 _long_mem_texts；异常时置 []）。
    """
    global _long_mem_texts
    if os.path.exists(LONG_MEM_PATH):
        try:
            with open(LONG_MEM_PATH, "rb") as f:
                _long_mem_texts = pickle.load(f)
        except Exception:  # pragma: no cover —— 持久化损坏按"无记忆"处理
            _long_mem_texts = []


def _ensure_loaded() -> None:
    """
    函数功能与逻辑描述：
        首次需要记忆前的惰性文本加载（幂等）：以 _TEXTS_LOADED 为标志，仅首次调用时置位并执行
        _load_texts()，之后直接返回；不涉及向量索引，也不触发 embedding 模型加载。
    入参说明：
        无。
    返回值说明：
        无（副作用：首次调用触发一次磁盘读取，并改写 _TEXTS_LOADED / _long_mem_texts）。
    """
    global _TEXTS_LOADED
    if _TEXTS_LOADED:
        return
    _TEXTS_LOADED = True
    _load_texts()


def _ensure_index() -> None:
    """
    函数功能与逻辑描述：
        文本已加载但向量索引未建时按需构建：若索引已存在或文本为空则直接返回；否则逐条求
        embedding 后建 FAISS（IndexFlatL2）索引，首次调用会触发 embedding 模型懒加载。
        异常处理：任一步失败（模型不可用 / 编码失败）都被吞掉，索引保持 None，检索自动降级
        jieba 关键词路；取不到向量的单条文本被跳过，不写入索引。
    入参说明：
        无。
    返回值说明：
        无（副作用：可能改写全局 _long_mem_index / _embedding_dim；异常时置 None）。
    """
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
    """
    函数功能与逻辑描述：
        强制从磁盘重读文本并重建向量索引：先把 _TEXTS_LOADED / _long_mem_index / _embedding_dim
        全部复位，再依次执行 _ensure_loaded() 与 _ensure_index()，从而绕过幂等短路。
        兼容旧入口与测试用；常规启动路径已不再自动调用本函数（见 _ensure_loaded 惰性改造）。
        注意本函数会丢弃内存中尚未落盘的索引状态，仅文本清单会被重新读回。
    入参说明：
        无。
    返回值说明：
        无（副作用：改写 _TEXTS_LOADED、_long_mem_texts、_long_mem_index、_embedding_dim）。
    """
    global _TEXTS_LOADED, _long_mem_index, _embedding_dim
    _TEXTS_LOADED = False
    _long_mem_index = None
    _embedding_dim = None
    _ensure_loaded()
    _ensure_index()


def save_consume_memory(summary_text: str):
    """
    函数功能与逻辑描述：
        月度报表生成后把摘要文本写入长期向量记忆：先做惰性加载/建索引，再求该文本的 embedding。
        取到向量时追加文本并同步把向量加入 FAISS 索引（若索引为空则先按该向量维度建索引）；
        取不到向量时**仅追加文本、不写索引**，该条记忆此后只可能被 jieba 关键词路召回
        （向量路无法命中），这是模型不可用时的降级行为而非异常。
        最后把完整文本清单 pickle 落盘。本函数无异常捕获，磁盘或模型异常会向上抛出。
    入参说明：
        summary_text (str)：待记忆的摘要文本，通常是月度消费报表摘要。
    返回值说明：
        无（副作用：追加 _long_mem_texts、可能向 _long_mem_index 添加向量、并重写 pkl 文件）。
    """
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
    """
    函数功能与逻辑描述：
        根据月度消费习惯聚合数据拼装标准摘要文本（长期记忆语义层的固定格式），
        形如「【2026-08 消费习惯】餐饮12笔1560.00元, 交通8笔420.00元; 总45笔6280.00元」。
        该格式的关键约束：以「【YYYY-MM 」开头，才能被 _MONTH_PREFIX 识别，进而支持
        按月份替换与 12 个月滚动裁剪——修改格式必须同步修改 _MONTH_PREFIX。
        入参不合法时不做兜底拼接，直接返回空串，由调用方据此跳过写入。
    入参说明：
        month (str)：月份，格式 "YYYY-MM"。
        habits (list)：该月聚合结果列表，元素为
            {"category","amount_sum","count","avg_amount"}；空列表直接返回空串。
    返回值说明：
        str：标准摘要文本；habits 为空时返回 ""（不会产生只有表头的残句）。
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
    """
    函数功能与逻辑描述：
        计算滚动保留的边界月份（含当月往前共 keep_months 个自然月的最早一个月），
        用「逐月回退」而非日期加减实现，从而天然处理跨年（如 2026-01 回退得到 2025-01）。
        返回的是 YYYY-MM 字符串，可直接与 _MONTH_PREFIX 提取出的月份做**字符串比较**，
        因为 YYYY-MM 的字典序等价于时间先后序。
        调用方按「月份 >= boundary」保留记忆文本。
    入参说明：
        keep_months (int)：保留的月份数量（含当月），默认 12；传 1 表示只保留当月。
    返回值说明：
        str：边界月份，格式 "YYYY-MM"。
    """
    now = datetime.datetime.now()
    y, m = now.year, now.month
    for _ in range(keep_months - 1):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return f"{y:04d}-{m:02d}"


def _rebuild_index_and_persist():
    """
    函数功能与逻辑描述：
        整体重建 FAISS 索引并把文本清单落盘：先复位索引与维度，再对 _long_mem_texts
        逐条求 embedding 重建 IndexFlatL2，最后 pickle 覆盖写回 LONG_MEM_PATH。
        之所以整体重建而非增量删除：月度粒度下文本量很小（12 个月滚动最多约 60 条，
        每月 5 品类 ≈ 5 条），重建开销可接受，且 FAISS 平铺索引不支持按 id 删除。
        取不到向量的单条文本被静默跳过（保留在文本清单中，仍可被关键词路召回）；
        若全部取不到向量则索引保持 None，检索自动降级为 jieba 关键词路。
        本函数无异常捕获，磁盘或模型异常由调用方处理（upsert_month_habit_memory 会兜底）。
    入参说明：
        无。
    返回值说明：
        无（副作用：改写全局 _long_mem_index / _embedding_dim，并重写 pkl 文件）。
    """
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
    """
    函数功能与逻辑描述：
        记账成功后把某月的消费习惯更新进长期记忆语义层，是「月度习惯」这条记忆的唯一写入口。
        流程：① 惰性加载文本；② 用 build_month_habit_text 生成标准摘要，为空则直接失败返回；
        ③ 按 _MONTH_PREFIX 匹配月份前缀，命中则**原地替换**该月旧摘要，否则追加；
        ④ 12 个月滚动裁剪：丢弃早于边界月份的**带月份前缀**的记忆文本，
        无前缀文本（如整月报表摘要）一律保留；
        ⑤ 调 _rebuild_index_and_persist 整体重建索引并落盘。
        静默失败：任何异常都被吞掉并返回 False，绝不影响主记账流程。
        注意返回 True 只代表本函数流程走完，不代表 embedding 一定成功
        （向量不可用时索引可能为 None，记忆仍以文本形式保留）。
    入参说明：
        month (str)：月份，格式 "YYYY-MM"。
        habits (list)：该月聚合结果列表，元素为 {"category","amount_sum","count","avg_amount"}。
    返回值说明：
        bool：True 表示已成功更新并落盘；False 表示摘要为空（habits 为空）或过程中发生异常。
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
    """
    函数功能与逻辑描述：
        把查询里的汉字数字月份归一化为阿拉伯数字（"去年十二月" → "去年12月"）。
        与 jieba 切碎阿拉伯数字同理："一月" 会被切成 "一"+"月" 两个单字后蒸发，
        且记忆文本里只存在 ISO 格式。这里在时间提取前统一归一化，让后续所有
        `\\d{1,2} 月` 的正则分支直接复用，不必各自再写汉字分支。
        只替换 `[一二三四五六七八九十]{1,2}月` 这一模式，其它文本原样保留。
    入参说明：
        query (str)：用户查询原文。
    返回值说明：
        str：归一化后的查询文本（汉字月份已变为数字月份）；无匹配时返回原文本。
    """
    def repl(m):
        """
        函数功能与逻辑描述：
            re.sub 的替换回调：把捕获到的汉字月份映射为阿拉伯数字月份（如 "十二" → "12月"）。
            映射表未收录的取值保持原样返回（理论上被正则限定在表内，此为防御性兜底）。
        入参说明：
            m：re.Match 对象，group(1) 为捕获到的 1~2 个汉字数字，如 "十二"。
        返回值说明：
            str："{数字}月"；映射缺失时返回整个匹配原文 m.group(0)。
        """
        return f"{_CN_MONTH.get(m.group(1), 0)}月" if m.group(1) in _CN_MONTH else m.group(0)
    return re.sub(r"([一二三四五六七八九十]{1,2})月", repl, query)


def _extract_time_tokens(query: str) -> list:
    """
    函数功能与逻辑描述：
        从用户查询中提取时间表达，并翻译成能匹配记忆文本 ISO 月份（YYYY-MM）的 token，
        专供混合检索的「时间强信号」加权使用；不在本函数内打分，只做翻译与产出 token 列表。
        背景：记忆文本里的月份是 "2026-01" 这类 ISO 格式，而用户口语是 "1月"；
        jieba 会把 "1月" 切成单字 "1" + "月"，单字又被 len>=2 过滤，导致关键词路对月份零贡献；
        向量路对具体数值又不敏感。因此这里用正则把口语时间直接翻译成记忆文本中实际存在的格式。
        覆盖分支：① 带年份「2026年1月」；② 相对年份「去年/今年/前年 X月」；
        ③ 纯数字日期「2026-08」「2026/08」（用户不写"年""月"汉字）；
        ④ 单独月份「1月」（用负向断言排除已带年份的情况，避免与 ① 重复）；
        ⑤ 相对月份「这个月/这月/本月/当月」与「上个月/上上个月/最近N个月」——
        其中 ③⑤ 为 2026-09-10 兼容性增强新增，原实现缺「这个月」导致最高频问法时间加权恒为 0。
        ⑤ 的两条拆为独立 if，使"这个月和上个月对比"能同时命中。
        入参过程中先调用 _cn_month_to_arabic 归一化汉字月份。
    入参说明：
        query (str)：用户查询原文。
    返回值说明：
        list：token 列表（保持按分支的产出顺序，未去重排序），元素为二元组：
            - ("date", "YYYY-MM")：表示该月份在记忆文本中以子串形式出现，直接子串匹配。
            - ("month", int)：1~12 的月份数字，匹配时需要搭配年份前缀
              （记忆文本中表现为 "YYYY-MM"），由调用方按 `\\d{4}-{MM}(?!\\d)` 校验。
            无任何时间表达时返回 []。
    """
    query = _cn_month_to_arabic(query)  # 汉字数字月份先归一化，其余分支全部复用
    tokens = []
    now = datetime.datetime.now()

    # 1) 带年份："2026年1月" / "2026年01月"
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月", query):
        tokens.append(("date", f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"))

    # 2) 去年/今年/前年 X月（隐含年份）
    m = re.search(r"去年\s*(\d{1,2})\s*月", query)
    if m:
        tokens.append(("date", f"{now.year - 1}-{int(m.group(1)):02d}"))
    m = re.search(r"今年\s*(\d{1,2})\s*月", query)
    if m:
        tokens.append(("date", f"{now.year}-{int(m.group(1)):02d}"))
    # 2026-09-10 新增：前年 X月（与去年/今年同源，补齐相对年份覆盖）
    m = re.search(r"前年\s*(\d{1,2})\s*月", query)
    if m:
        tokens.append(("date", f"{now.year - 2}-{int(m.group(1)):02d}"))

    # 2.5) 2026-09-10 新增：纯数字日期 "2026-08" / "2026/08"（用户不写"年""月"汉字）
    for m in re.finditer(r"(?<!\d)(\d{4})[-/](\d{1,2})(?!\d)", query):
        tokens.append(("date", f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"))

    # 3) 单独月份："1月" / "12月"
    #    (?<!\d年)(?<!\d) 排除已带年份的情况（避免与 1) 重复），"去年12月" 仍能命中
    for m in re.finditer(r"(?<!\d年)(?<!\d)(\d{1,2})\s*月", query):
        tokens.append(("month", int(m.group(1))))

    # 4) 相对时间：当月 / 上月 / 最近N个月
    #    2026-09-10 新增：①「这个月/这月/本月/当月」——最高频口语，原实现完全缺失，
    #      导致"这个月花了多少"提取不到时间 token、时间加权恒为 0；
    #      ②「上上个月」表述变体。两者拆为独立 if，使"这个月和上个月对比"可同时命中
    #      （各 +0.15，合计 0.3，仍受既有 min(time_hits,2) 上限约束）。
    if re.search(r"这个月|这月|本月|当月", query):
        tokens.append(("date", f"{now.year:04d}-{now.month:02d}"))
    if re.search(r"上个月|上上月|上上个月|最近.{0,2}个月|近.{0,2}个月", query):
        first = now.replace(day=1)
        prev = first - datetime.timedelta(days=1)
        tokens.append(("date", f"{prev.year:04d}-{prev.month:02d}"))

    return tokens


def search_history_consume(query: str, top_k: int = 3) -> str:
    """
    函数功能与逻辑描述：
        轻量混合检索：向量语义路（权重 0.6）+ jieba 关键词精确匹配路（权重 0.4）
        + 时间表达强信号（每次命中 +0.15，上限 +0.3），三部分**累加**到同一条记忆的得分上。
        向量路：取 top_k*2 条候选，L2 距离按当次最大距离线性归一为相似度再乘 0.6；
        索引或 query 向量不可用时该路整体跳过，自动降级为纯关键词检索。
        关键词路：jieba 切词后过滤长度 <2 的单字，纯数字词用 `(?<!\\d)kw(?!\\d)` 边界匹配
        以免 "12" 误命中金额 "1200"；得分按命中数 / 关键词总数 × 0.4 归一。
        时间信号：按 _extract_time_tokens 产出分别做子串匹配或 `\\d{4}-{MM}(?!\\d)` 匹配，
        兼容「【2026-08 消费习惯】」与「2026-06-01」两种格式（原正则只认后者，恒不命中）。
        排序后按得分取前 top_k 条，**得分为 0 即中断**；若一条都没命中则兜底返回最新的 top_k 条。
    入参说明：
        query (str)：用户查询文本。
        top_k (int)：返回最匹配的条数，默认 3；同时作为向量路候选数（top_k*2）与兜底条数的基数。
    返回值说明：
        str：命中记忆文本以 "\\n" 拼接的结果；无任何记忆时返回固定文案 "暂无历史消费记录"；
            有记忆但无命中时返回最新的 top_k 条文本。
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
                # 2026-09-10：原正则 "-{MM}-" 仅匹配含完整日期的旧格式（2026-06-01），
                #   与生产主格式 "【2026-08 消费习惯】"（08 后是空格）不匹配 → 恒不命中。
                #   放宽为 "\d{4}-{MM}(?!\d)"：兼容两种格式；保留 YYYY- 前缀以
                #   避免误伤 "12笔""1560.00元" 等笔数/金额片段（裸 "08" 会误命中）。
                elif ttype == "month" and re.search(rf"\d{{4}}-{tval:02d}(?!\d)", text):
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
    """
    函数功能与逻辑描述：
        清空全部长期记忆（测试与重置场景专用）：把文本清单、FAISS 索引、向量维度全部复位，
        删除持久化 pkl 文件。为绕过惰性加载的幂等短路，_TEXTS_LOADED 被置为 **True**——
        这样后续调用 _ensure_loaded 不会再去磁盘把刚删掉的文件（或残留旧数据）重新读回来。
        文件不存在时跳过删除，不报错。
    入参说明：
        无。
    返回值说明：
        无（副作用：复位四个模块级全局变量，并删除 LONG_MEM_PATH 文件）。
    """
    global _long_mem_texts, _long_mem_index, _embedding_dim, _TEXTS_LOADED
    _long_mem_texts = []
    _long_mem_index = None
    _embedding_dim = None
    _TEXTS_LOADED = True
    if os.path.exists(LONG_MEM_PATH):
        os.remove(LONG_MEM_PATH)
