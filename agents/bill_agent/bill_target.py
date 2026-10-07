# -*- coding: utf-8 -*-
"""M15（P3a 确定性驱动）账单**目标定位**与**二次确认**的纯函数规则集。

定位背景：用户说"把昨天那笔打车改成 50"时要用哪条规则落到具体 `id`？
- **FC 开启时**（P3b）：定位交给模型在自主循环里完成（先 `bill.recent_list` 再决策）；
- **FC 关闭/不可用时**（P3a，本模块）：定位交给**确定性规则**——这是"功能不丢"的保底路径
  （`设计/12` §4.7 第 3 层降级），也是 `BILL_FC_ENABLED=0` 时的实际执行路径。

★本模块刻意**只做纯计算**（无 I/O、无 DB、无 registry、不 import 任何 agent 节点）：
  ① 便于单测穷举规则优先级（定位错=改错账，属不可逆事故，规则必须可穷举验证）；
  ② 避免 `agents.bill_agent.nodes ↔ 本模块` 的循环依赖（节点侧负责调用与落库）。

★与 `设计/12` §4.2.5 的对齐：可指代范围**只限最近 N 笔**（`RECENT_BILL_WINDOW`），
  窗口外或歧义一律返回"需要追问"而不是猜——猜错的代价是改错/删错别人的账。
"""
import re

# 指代词：用户用"那笔/刚才"这类词指代**最近一笔**（仅当窗口内只有 1 笔时才敢认定）
REFER_WORDS = ("刚才", "刚刚", "这笔", "那笔", "这笔账", "那条", "这条",
               "最近一笔", "最后一笔", "这一笔", "上一笔", "它")

# 确认词表（`设计/12` §4.4）：删除二次确认的第二轮必须命中其一才执行
CONFIRM_KEYWORDS = ("确认", "确定", "是的", "对", "嗯", "可以", "删吧", "删除吧", "同意")

# 定位失败原因（调用方据此决定追问文案：列候选 / 报"没找到"）
REASON_OK = ""
REASON_EMPTY = "empty"            # 最近窗口内一笔账单都没有
REASON_NOT_FOUND = "not_found"    # 有账单，但没有一条能匹配上线索
REASON_AMBIGUOUS = "ambiguous"    # 匹配到多条，需要用户指认


def is_confirm(text: str) -> bool:
    """
    函数功能与逻辑描述：
        判断本轮用户输入是否为"删除确认"（`设计/12` §4.4 的第二轮）。采用**子串命中**而非
        全等匹配——用户实际会说"嗯，删吧""确认删除"这类带修饰的短句，全等匹配会漏。
        注意本函数只回答"这句话像不像确认"，**是否真的执行删除**还要由调用方叠加
        "草稿槽存在同一 id"这个前提（避免用户随口说"可以"就触发删除）。
    入参说明：
        text (str)：用户本轮原始输入（可含多段，调用方自行合并）。
    返回值说明：
        bool：命中确认词表中任一子串返回 True；text 为空或未命中返回 False。
    """
    if not text:
        return False
    return any(word in text for word in CONFIRM_KEYWORDS)


def _text_numbers(text: str) -> list[float]:
    """
    函数功能与逻辑描述：
        提取文本中的全部数字（含小数与负号），供"按金额定位账单"规则使用。
        与节点侧的金额兜底口径保持一致（`-?\\d+(\\.\\d+)?`）：负号保留，便于"改成-30"这类
        非法输入在后续环节被判非法金额，而不是被悄悄当成 30。
    入参说明：
        text (str)：待提取的文本。
    返回值说明：
        list[float]：按出现顺序排列的数字列表；无数字时返回空列表。
    """
    return [float(m.group(0)) for m in re.finditer(r"-?\d+(\.\d+)?", text)]


# 「改成 X」类改写动词：**新值数字**识别器。放在本模块是因为它被两处使用——
#   ① 定位规则（必须把"改成 50"里的 50 从**定位数字**中剔除，否则会拿新值去找账）；
#   ② 节点侧的新值解析（动词后的数字优先于模型抽取）。
#   一张表两处引用，避免"两处各写一份动词表"的漂移。
CHANGE_VERB_RE = re.compile(
    r"(?:改成|改为|变成|变为|调整为|调成|修改为|更新为|设为)\s*(-?\d+(?:\.\d+)?)")

# 日期文本（绝对日期 / X月X日），供定位线索提取与"数字掩码"共用
_DATE_TEXT_RE = r"20\d{2}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}日"


def _mask_dates(text: str) -> str:
    """
    函数功能与逻辑描述：
        把文本中的日期子串替换为等长空白，供"提数字"时把日期排除在外。
        为什么需要它：正则提数字会把 "2026-10-05" 拆成 `2026 / -10 / -05` 三个数，
        这些数会污染定位线索（让规则误以为用户给了金额，进而阻断指代词兜底），
        也让"按金额匹配"规则可能撞上日期片段。掩码后语义清晰：日期归日期、金额归金额。
    入参说明：
        text (str)：用户原文。
    返回值说明：
        str：日期子串被空白替换后的等长文本（长度不变，便于按位置做后续判断）。
    """
    return re.sub(_DATE_TEXT_RE, lambda m: " " * len(m.group(0)), text)


def locate_numbers(text: str) -> list[float]:
    """
    函数功能与逻辑描述：
        提取**定位用**数字：原文全部数字 **减去** 改写动词后的新值数字。
        为什么必须剔除：用户说"把 35 那笔改成 50"时，50 是**要改成的值**而非定位线索；
        不剔除会有两个后果——① 规则拿 50 去匹配账单金额（恰好有 50 元的账就会定位错）；
        ② 把"金额线索对不上"误判成"用户没给线索"，从而错误地走指代词兜底。
        按**值**逐个扣除（同一数值出现多次只扣一次），不做位置分析——"改成 50 的 35 那笔"这类
        倒装会让位置法失效，而值扣除对语序不敏感。
    入参说明：
        text (str)：用户原文。
    返回值说明：
        list[float]：剔除新值后的数字列表（保留原顺序）；全部被剔除或本就无数字时返回空列表。
    """
    nums = _text_numbers(_mask_dates(text))
    for value in [float(m.group(1)) for m in CHANGE_VERB_RE.finditer(text)]:
        if value in nums:
            nums.remove(value)
    return nums


def _explicit_index(text: str, bills: list[dict]) -> dict | None:
    """
    函数功能与逻辑描述：
        规则①：原文显式指认序号或主键——"第 3 条 / 第2笔 / id=5 / 编号 5"。
        序号按**当前清单的下标**解释（清单是 id 倒序，所以"第 1 条"= 最近一笔），
        这与 UI 面板的展示序号同口径（`设计/12` §4.6 扩展路径的 v1 约定）。
        越界一律返回 None（交给后续规则/追问），不猜"最后一条"。
    入参说明：
        text (str)：用户原文。
        bills (list[dict])：`bill.recent_list` 返回的清单（id 倒序）。
    返回值说明：
        dict | None：命中的账单行；无显式指认或越界时返回 None。
    """
    m = re.search(r"第\s*(\d+)\s*(?:条|笔|个)", text)
    if m:
        idx = int(m.group(1)) - 1
        return bills[idx] if 0 <= idx < len(bills) else None
    m = re.search(r"(?:id|ID|编号)\s*[=:：]?\s*(\d+)", text)
    if m:
        bill_id = int(m.group(1))
        for row in bills:
            if int(row.get("id", -1)) == bill_id:
                return row
        return None
    return None


def _locate_by_amount(text: str, bills: list[dict]) -> list[dict]:
    """
    函数功能与逻辑描述：
        规则②：按**金额**定位——原文数字集合与账单金额求交集。
        这条规则在多笔不同金额时最有效（"把 35 那笔改成 50"）；同额多笔时返回多条，
        由调用方判为歧义并追问。比较用 1 分钱容差（浮点等值比较的常规做法）。
    入参说明：
        text (str)：用户原文。
        bills (list[dict])：账单清单。
    返回值说明：
        list[dict]：金额命中的账单行列表；无命中时返回空列表。
    """
    nums = locate_numbers(text)
    if not nums:
        return []
    hits = []
    for row in bills:
        try:
            amount = float(row.get("amount"))
        except (TypeError, ValueError):
            continue
        if any(abs(amount - num) < 0.01 for num in nums):
            hits.append(row)
    return hits


def _relative_date(word: str) -> str | None:
    """
    函数功能与逻辑描述：
        相对日期词（前天 / 昨天 / 今天）→ YYYY-MM-DD；不在表内返回 None。
        相对日期按**运行时当天**推导（不缓存结果，跨天即失效——这也是本模块所有日期函数
        都不接收 today 参数的原因：调用点无法伪造"今天"）。
    入参说明：
        word (str)：相对日期词。
    返回值说明：
        str | None：归一后的日期；词不在表内时返回 None。
    """
    from datetime import date, timedelta
    offsets = {"前天": 2, "昨天": 1, "今天": 0}
    if word not in offsets:
        return None
    return (date.today() - timedelta(days=offsets[word])).isoformat()


def _absolute_date(raw: str) -> str | None:
    """
    函数功能与逻辑描述：
        绝对日期文本（"2026-10-05" / "10月5日"）→ YYYY-MM-DD；无法识别返回 None。
        "X月X日"按**当前年份**解释（与节点侧 `_fix_date` 同口径，不处理跨年表述）。
    入参说明：
        raw (str)：日期文本（允许首尾空白）。
    返回值说明：
        str | None：归一后的日期；无法识别时返回 None。
    """
    from datetime import date
    m = re.fullmatch(r"(20\d{2})-(\d{1,2})-(\d{1,2})", raw.strip())
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})日", raw.strip())
    if m:
        return f"{date.today().year:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def extract_date_hint(text: str) -> str | None:
    """
    函数功能与逻辑描述：
        提取**定位用**日期线索并归一为 YYYY-MM-DD。★优先级刻意是「相对词 > 绝对日期」：
        用户说"把昨天那笔改到 2026-10-05"时，"昨天"在**描述要改哪一笔**，而"2026-10-05"是
        **要改成的新日期**；若绝对日期优先，定位就会拿新日期去找旧账 → 必然找不到。
        （同一条经验：指代描述多用相对词，新值多用精确日期。）
        只处理确定性表述（前天/昨天/今天/绝对日期/X月X日），且**没有"默认取今天"的兜底**——
        无线索时返回 None，否则"没有线索"会被伪装成"定位到今天"，把规则匹配变成瞎猜。
        "上个月/上周"这类区间表述刻意不支持：它们必然对应多笔，属歧义，应走追问
        （`设计/12` §4.2.5：窗口外的指代一律追问）。
    入参说明：
        text (str)：用户原文。
    返回值说明：
        str | None：归一后的定位日期；原文无确定性线索时返回 None。
    """
    for word in ("前天", "昨天", "今天"):
        if word in text:
            return _relative_date(word)
    m = re.search(_DATE_TEXT_RE, text)
    return _absolute_date(m.group(0)) if m else None


def extract_new_date(text: str) -> str | None:
    """
    函数功能与逻辑描述：
        提取**新值**日期：只在出现「改到 / 改在 / 改成 / 换到 …」这类**改写语**时才认，
        且只看改写语之后的那个日期。与 `extract_date_hint` 分开的原因：同一句里
        "定位描述"与"新值"可能各带一个日期（"把昨天那笔改到 2026-10-05"），
        用一个函数必然有一头取错；拆开后各取各的，职责单一、可分别测试。
    入参说明：
        text (str)：用户原文。
    返回值说明：
        str | None：归一后的新日期；无改写语、或改写语后没有可识别日期时返回 None。
    """
    m = re.search(r"(?:改到|改在|改成|改为|调整到|换到|移到|转到)\s*"
                  r"(" + _DATE_TEXT_RE + r"|前天|昨天|今天)", text)
    if not m:
        return None
    token = m.group(1)
    return _relative_date(token) or _absolute_date(token)


def _matches_attribute(text: str, row: dict, *, category: str | None,
                       date_hint: str | None) -> bool:
    """
    函数功能与逻辑描述：
        判定一笔账单是否**同时**满足调用方给出的属性线索（品类 / 日期）。
        两个线索都为 None 时返回 True（表示"该规则不约束任何属性"，由调用方保证不会走到这里）。
        日期比对用前缀匹配（`consume_time` 可能是 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"）。
    入参说明：
        text (str)：用户原文（保留给"备注命中"这类后续扩展，当前未使用）。
        row (dict)：账单行。
        category (str | None)：品类线索（由节点侧的 _fix_category 推断得到）。
        date_hint (str | None)：日期线索（本模块 extract_date_hint 得到）。
    返回值说明：
        bool：线索全部匹配返回 True；任一不符返回 False。
    """
    if category is not None and row.get("category") != category:
        return False
    if date_hint is not None and not str(row.get("consume_time") or "").startswith(date_hint):
        return False
    return True


def locate_bill(text: str, bills: list[dict], *,
                category: str | None = None) -> tuple[dict | None, str]:
    """
    函数功能与逻辑描述：
        **确定性目标定位**（P3a 的核心）。按固定优先级短路，返回
        `(命中的账单行, 失败原因)`；命中时原因为空串，未命中时行为 None。
        规则优先级（先精确后模糊，越靠前越"用户说得越明确"）：
        ① **显式序号/主键**（"第 3 条"/"id=5"）——用户已明确指认，唯一命中即采纳；
        ② **金额**（"35 那笔"）——唯一命中即采纳，多命中转下一步用属性收窄；
        ③ **金额 + 品类/日期属性**——在②的候选里再按属性过滤；
        ④ **品类 + 日期属性**（无金额线索时）；
        ⑤ **仅日期属性**；
        ⑥ **指代词兜底**（"刚才那笔"）——★仅当窗口内只有 1 笔时才采纳，
           否则视为歧义（多笔时"刚才那笔"用户自己都指不清，规则更不该猜）。
        任何一步命中多条 → 立即返回 `ambiguous`（不继续向下稀释线索，避免把
        "两笔都符合"降级成"随便挑一笔"）；全部规则走完仍零命中 → `not_found`。
    入参说明：
        text (str)：用户原文（通常是多段合并后的完整文本）。
        bills (list[dict])：`bill.recent_list` 返回的清单（id 倒序，元素含 id/amount/category/consume_time）。
        category (str | None)：关键字参数，调用方用 `_fix_category` 推断出的品类线索，默认 None。
    返回值说明：
        tuple[dict | None, str]：命中时为 (账单行, "")；失败时为 (None, 原因)，
            原因取值见 REASON_EMPTY / REASON_NOT_FOUND / REASON_AMBIGUOUS。
    """
    if not bills:
        return None, REASON_EMPTY

    # ① 显式序号 / 主键：用户说得最明确，命中即采纳（越界则继续走后面的规则）
    picked = _explicit_index(text, bills)
    if picked is not None:
        return picked, REASON_OK

    date_hint = extract_date_hint(text)
    amount_hits = _locate_by_amount(text, bills)
    locate_nums = locate_numbers(text)      # 定位用数字（已剔除"改成 X"里的新值）

    # ② 仅金额（无任何属性线索时，金额唯一 → 采纳；多笔同额 → 歧义）
    if amount_hits and category is None and date_hint is None:
        return (amount_hits[0], REASON_OK) if len(amount_hits) == 1 else (None, REASON_AMBIGUOUS)

    # ③ 金额 + 属性（在金额命中的候选里按品类/日期收窄）
    if amount_hits:
        narrowed = [r for r in amount_hits
                    if _matches_attribute(text, r, category=category, date_hint=date_hint)]
        if len(narrowed) == 1:
            return narrowed[0], REASON_OK
        return None, REASON_AMBIGUOUS if narrowed else REASON_NOT_FOUND

    # ④ / ⑤ 属性定位（品类 + 日期 / 仅日期）
    if category is not None or date_hint is not None:
        attr_hits = [r for r in bills
                     if _matches_attribute(text, r, category=category, date_hint=date_hint)]
        if len(attr_hits) == 1:
            return attr_hits[0], REASON_OK
        return None, REASON_AMBIGUOUS if attr_hits else REASON_NOT_FOUND

    # ⑥ 指代词兜底：★仅当"既无金额命中、也无定位数字"时才生效——用户明明给了数字却对不上，
    #   那是"没找到"（该列候选让他挑），而不是"指代不清"（别把他的数字线索当成"那笔"处理）。
    #   命中指代词且窗口内只有 1 笔才敢认定"就是它"；多笔时属歧义（用户自己都指不清，规则不猜）。
    if not amount_hits and not locate_nums and any(word in text for word in REFER_WORDS):
        if len(bills) == 1:
            return bills[0], REASON_OK
        return None, REASON_AMBIGUOUS

    return None, REASON_NOT_FOUND


def describe_bill(row: dict) -> str:
    """
    函数功能与逻辑描述：
        把一笔账单渲染成面向用户的一行描述，用于**追问话术**（列出候选让用户指认）与
        **删除二次确认**话术。格式固定为 `09-13 交通 35 元（昨天打车35元）`：
        月份-日 + 品类 + 金额 +（备注），其中备注截断到 20 字以防超长刷屏。
        金额按整数（无小数）或两位小数渲染，避免出现 `35.0 元` 这种不自然的表述。
    入参说明：
        row (dict)：账单行（含 consume_time / category / amount / remark）。
    返回值说明：
        str：单行描述文本；字段缺失时以空串占位而不抛异常。
    """
    consume_time = str(row.get("consume_time") or "")
    day = consume_time[5:10] if len(consume_time) >= 10 else consume_time
    try:
        amount = float(row.get("amount"))
        amount_text = f"{amount:.0f}" if abs(amount - round(amount)) < 0.01 else f"{amount:.2f}"
    except (TypeError, ValueError):
        amount_text = str(row.get("amount"))
    remark = str(row.get("remark") or "")[:20]
    tail = f"（{remark}）" if remark else ""
    return f"{day} {row.get('category') or ''} {amount_text} 元{tail}"


def describe_candidates(bills: list[dict]) -> str:
    """
    函数功能与逻辑描述：
        把候选账单渲染为多行清单（每行带序号），供歧义/未命中时的追问话术使用。
        序号与 `locate_bill` 规则①的下标口径一致（第 1 条 = 最近一笔），
        这样用户回一句"第 2 条"就能被同一套规则接住（闭环）。
    入参说明：
        bills (list[dict])：账单清单（id 倒序）。
    返回值说明：
        str：形如 "1. 09-13 交通 35 元（打车）" 的多行文本；bills 为空时返回空串。
    """
    return "\n".join(f"{i}. {describe_bill(row)}" for i, row in enumerate(bills, start=1))


__all__ = [
    "CONFIRM_KEYWORDS", "REFER_WORDS",
    "REASON_OK", "REASON_EMPTY", "REASON_NOT_FOUND", "REASON_AMBIGUOUS",
    "is_confirm", "locate_bill", "describe_bill", "describe_candidates",
    "extract_date_hint", "extract_new_date", "locate_numbers", "CHANGE_VERB_RE",
]
