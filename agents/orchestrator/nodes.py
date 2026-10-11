# ==============================================
# 编排器（Orchestrator）LangGraph 四节点实现
# 链路：plan_node（LLM 规划 → 清洗 → 启发式裁剪 → 准入拦截）
#       → dispatch_node（就绪任务批量派发）→ wait_result_node（并发等待）
#       → collect_node（确定性汇总 → 数字白名单防幻觉表达）
# 说明：① 本模块属编排层，不直连数据库——所有取数统一走 mcpGateway 事实工具
#          （bill.sum_by_month / city_price.get_avg / habit.list / config.get_latest 等，D15/M9 收口）；
#       ② 溢价阈值口径复用 coreModules/price_compare.py，避免与 price_agent 两处各自维护漂移；
#       ③ 任何降级/回退/兜底必须打 WARNING（`01` R10 禁止静默失败）。
# ==============================================
import secrets
import json
import copy
import time
import asyncio
import re
import calendar
from datetime import date
from typing import Optional
from langgraph.types import Command
from langgraph.graph import END
from agents.orchestrator.state import OrchState
from agentCore.parsers.json_parser import parse_json_output
from mcpGateway.client import mcp_client
from mcpGateway.a2a_queue import a2a_bus
from memory.short_memory import get_session_memory
from memory.long_memory import save_consume_memory, search_history_consume
from utils.common import logger
from config.config import (ASK_GIVEUP_TIP, ASK_ROUND_LIMIT, DYNAMIC_DEPS_ENABLED,
                           MAX_DISPATCH_ROUND, BATCH_DISPATCH_ENABLED,
                           DEFAULT_USER_CITY, EXTERNAL_LLM_ENABLED, USER_DOCS_TOP_K,
                           AGENT_CONTRACTS, INTENT_CONFIDENCE_THRESHOLD, INTENT_OOD_ENABLED,
                           INTENT_INFERABLE_ENABLED, ABORT_ON_TASK_FAILURE, MODEL_TIMEOUT)
from coreModules.price_compare import calc_premium_evaluate
# ContextManager 窗口按 Router 解析的实际后端窗口换算（外部 8k+ / 本地 4k），不写死 4096
from agents.context_manager import ContextManager, estimate_tokens
from modelService.router import context_window_for
from .prompts.prompt_loader import load_system, render_user, load_summarize_system, render_summarize_user


def _log(stage: str, detail: str = ""):
    """
    函数功能与逻辑描述：
        编排链路阶段日志：统一以 `[ORCH] 时:分:秒 阶段 明细` 格式打到标准输出；
        `flush=True` 强制实时刷新，保证长链路（plan→dispatch→wait→collect）排障时
        日志不被缓冲吞掉；真实业务路径专用，无任何持久化。
    入参说明：
        stage (str)：阶段名，如 "plan_node 开始"、"派发任务"、"启发式裁剪后"。
        detail (str)：阶段明细（任务名/任务 id 前 8 位/异常信息等），默认 ""。
    返回值说明：
        无（副作用为向 stdout 打印一行日志）。
    """
    print(f"[ORCH] {time.strftime('%H:%M:%S')} {stage} {detail}", flush=True)

# 当前调度Agent唯一标识
ORCH_AGENT_ID = "orchestrator_agent"

# 模块级 ContextManager 实例（按 orchestrator 实际后端窗口建一次）
_cm_orch = ContextManager(context_window=context_window_for(ORCH_AGENT_ID))


async def _invoke_fact(name: str, args: dict):
    """
    函数功能与逻辑描述：
        编排器取数的唯一出口：统一走网关事实工具（鉴权/审计/超时/权限由执行引擎统一处理），
        避免编排层直连数据库；经 `get_registry().invoke(..., agent_id=ORCH_AGENT_ID)` 以编排器
        身份调用，再 `unwrap()` 拆出工具业务返回值。
        异常口径：不吞异常——失败由 `unwrap()` 抛调用方熟知的原生异常，交由各调用点
        try/except 兜底（多为返回 0 / 空值降级）。
    入参说明：
        name (str)：事实工具名，如 "bill.sum_by_month"、"city_price.get_avg"、"habit.list"、"config.get_latest"。
        args (dict)：工具入参，键值语义由各工具自身约定。
    返回值说明：
        Any：`unwrap()` 后的工具业务返回值；工具未注册、无权限或调用失败时由 `unwrap()` 抛异常。
    """
    from mcpGateway.registry import get_registry
    result = await get_registry().invoke(name, args, agent_id=ORCH_AGENT_ID)
    return result.unwrap()


# 任务名称 -> 对应的worker Agent名称映射
TASK_AGENT_MAP = {
    "bill": "bill_agent",
    "stat": "stat_agent",
    "price": "price_agent",
    "finance": "finance_agent"
}

# 任务名称 -> 默认细分操作（小模型常漏输出 operate_sub_type，规划层兜底填充）
TASK_SUB_TYPE_DEFAULTS = {
    "bill": "add",
    "stat": "query",
    "price": "query",
    "finance": "analyse"
}

# 静态任务依赖：统计/比价等账单记录先完成，理财分析等统计/比价先完成
TASK_CONTEXT_DEPS = {
    "stat": ["bill"],
    "price": ["bill"],
    "finance": ["bill", "stat", "price"]
}


def normalize_keys(data):
    """
    函数功能与逻辑描述：
        递归清洗字典的所有 key（移除全部空格），根治小模型输出 key 带空格
        （如 "operate_sub_type "）导致后续字段取值失败的问题；对 dict 递归处理，
        对 list 逐个元素递归，其余类型原样返回。
    入参说明：
        data (Any)：待清洗数据；通常为解析后的规划 JSON，也兼容 list / 标量输入。
    返回值说明：
        Any：与入参同构的清洗后数据——dict 返回新 dict（key 去空格），list 返回新 list，
            其他类型原样返回。
    """
    if not isinstance(data, dict):
        return data
    new_dict = {}
    for raw_key, value in data.items():
        clean_key = raw_key.replace(" ", "")
        # 递归处理子字典 / 数组
        if isinstance(value, dict):
            new_dict[clean_key] = normalize_keys(value)
        elif isinstance(value, list):
            new_list = []
            for item in value:
                new_list.append(normalize_keys(item))
            new_dict[clean_key] = new_list
        else:
            new_dict[clean_key] = value
    return new_dict


def validate_task_list(raw_tasks: list) -> tuple[bool, str, list]:
    """
    函数功能与逻辑描述：
        【仅格式/字段合法性校验，不做任何业务意图、语义判断】逐条校验规划任务：
        非 dict 条目直接跳过；name 不在 TASK_AGENT_MAP 的条目**直接剔除**（不让整轮规划失败，
        交由下游启发式兜底）；raw_segments 非数组记错；operate_sub_type 缺失/非字符串时按
        TASK_SUB_TYPE_DEFAULTS 填空默认值、不判错（兼容模型笔误键 operate_sub_ype 并就地改名）。

        **同轮重复任务的处理**：
        · `finance` **至多一条**——它产出"综合分析"结论，多条无意义且会重复派发同一分析；
        · `bill` / `stat` / `price` **允许多条**——多笔记账、多个时间范围的统计（"查本月 +
          查上月"）、多个品类的比价（"对比火锅 + 对比奶茶"）都是合理且**互不替代**的诉求，
          按同名去重会把用户诉求吃掉；
        · 重复的 `finance` 保留首条、丢弃多余，**不合并**：同类型出现多条通常是规划越权
          （如把"查开销"塞进 finance，而那是 stat 的职责），合并 raw_segments 等于把错误规划
          固化、让越权诉求真的进入执行。正确做法是**丢弃 + 留痕告警**，并回到规划层修任务边界
          （见 `prompts/system.md`）；
        · 重复**不判整轮失败**：判失败会让 `task_plan` 置空、用户输入被完全丢弃。
    入参说明：
        raw_tasks (list)：LLM 规划出的原始任务列表，元素应为 dict。
    返回值说明：
        tuple[bool, str, list]：(是否合法, 错误汇总信息, 修复后的任务列表)。
            校验通过 → (True, "", 修复后的任务列表)；
            存在错误 → (False, 以 "; " 拼接的错误信息, 已修复/剔除后的任务列表)。
    """
    errors = []
    fixed_tasks = []
    name_check_set = set()

    for idx, task in enumerate(raw_tasks):
        if not isinstance(task, dict):
            errors.append(f"第{idx}条任务不是JSON对象")
            continue

        # 兼容模型笔误 operate_sub_ype
        if "operate_sub_ype" in task and "operate_sub_type" not in task:
            task["operate_sub_type"] = task["operate_sub_ype"]

        t_name = task.get("name")
        segs = task.get("raw_segments")
        sub_type = task.get("operate_sub_type")

        # name 非法/缺失的任务对象直接剔除（不让整轮规划失败），交由下游启发式兜底
        if t_name not in TASK_AGENT_MAP:
            continue

        # 同轮重复：**仅 `finance` 硬约束为至多一条**（保留首条、丢弃多余），其余类型允许多条。
        #   约束内容与理由见本函数 docstring 的"同轮重复任务的处理"。
        if t_name == "finance" and t_name in name_check_set:
            logger.warning(f"[计划] 任务{idx}: 重复任务类型 finance，已丢弃该条"
                           "（finance 同轮至多一条；bill/stat/price 允许多条）")
            continue
        if t_name == "finance":
            name_check_set.add(t_name)

        if not isinstance(segs, list):
            errors.append(f"任务{idx}[{t_name}]: raw_segments必须是数组")
            continue
        # 兜底：小模型经常漏输出 operate_sub_type，按任务类型填默认细分操作，不判错
        if not isinstance(sub_type, str) or len(sub_type.strip()) == 0:
            task["operate_sub_type"] = TASK_SUB_TYPE_DEFAULTS.get(t_name, "query")

        fixed_tasks.append(task)

    if errors:
        return False, "; ".join(errors), fixed_tasks
    return True, "", fixed_tasks


# ========== 启发式业务裁剪：纠正小模型的语义误判 ==========
AMOUNT_REGEX = re.compile(r"\d+(\.\d+)?")
# 统计意图强关键词
STAT_HINT_KEYWORDS = [
    "统计", "查询", "汇总", "这个月", "本月", "上月", "上个月", "上礼拜", "上周",
    "花了多少", "支出", "消费明细", "账单明细", "总共", "多少次", "多少笔", "开销", "余额",
    "多不多",
]
# 统计"兜底补齐"专用词：不含纯时间词，避免"本月省钱建议"被误补 stat
STAT_ENSURE_KEYWORDS = [
    "统计", "查询", "汇总", "开销", "花了多少", "消费了多少", "消费明细", "账单明细",
    "总共", "多少次", "多少笔", "余额", "支出", "总开销", "总支出",
]
# 理财建议意图强关键词
FINANCE_HINT_KEYWORDS = [
    "建议", "怎么", "如何", "优化", "省钱", "理财", "预算", "规划", "意见",
    "分析", "更省", "合理", "省点", "消费习惯",
]
# finance 的「数据依赖词」：
#   finance **自己不查库**，它的分析若涉及真实开销数字，就必须由 stat 提供数据；
#   命中本表即视为"该分析依赖当期账目" → 自动补一个 stat 任务。
#   刻意与 `STAT_ENSURE_KEYWORDS`（统计诉求词）分开：本表只表达"分析内容与账目有关"，
#   不表达"用户要查账"，避免把"给点理财原则"这类无数据诉求也强加统计任务。
FINANCE_DATA_HINT_KEYWORDS = [
    "花了多少", "花了多少钱", "开销", "支出", "花费", "消费了多少", "总花费",
    "剩余预算", "预算还剩", "预算进度", "预算够不够", "够不够", "撑到月底", "撑得住",
    "超支", "超标",
]
# 比价意图强关键词
PRICE_HINT_KEYWORDS = [
    "贵", "便宜", "比价", "对比", "性价比", "哪家", "价格", "划算", "值不值", "贵不贵", "市场价",
]
# 变更既有账单的强关键词（改/删）——与 `agents/bill_agent/bill_target.CHANGE_VERB_RE` **同口径**：
#   出现即代表用户诉求是"改动/删除某条已记账"，既不是新增记账，也不是比价/统计/理财咨询。
#   为什么必须显式成表：① 这类输入常**不带记账动词**（"把 id=5 那笔改成 60"里没有"记"），
#   会被 `_is_consume_statement` 误判成"纯消费陈述"从而补出 price 对标任务——用户说"改"，
#   系统却去做比价；② 模型极易把它规划成 price 任务，规则侧必须能纠偏。
EDIT_HINT_KEYWORDS = [
    "改成", "改为", "变成", "变为", "调整为", "调成", "修改为", "更新为", "设为",
    "改一下", "改到", "修改",
]
DELETE_HINT_KEYWORDS = [
    "删掉", "删除", "去掉", "移除", "撤销", "别记这条", "别记这笔",
]
# 合理性疑问词（"值吗/合理吗/正常吗/超支了吗"）：问该笔消费是否合理，映射为 finance 分析
REVIEW_HINT_KEYWORDS = [
    "合理", "合理吗", "合理不", "合不合理", "值吗", "值不值", "合适吗", "贵吗", "贵不贵", "便宜吗", "划算吗",
    # 纯合理性问句："正常吗/算贵吗"
    "正常吗", "正常不", "算正常吗", "算贵吗",
    # 超支/花多类："是不是花多了/超支了吗"——是分析诉求，不是记账指令
    "花多了", "花超了", "花得值吗", "花超支", "超支了吗", "超支吗", "超预算", "超标了",
]
# 记账后自动预警阈值：预算充足→犒劳提醒，接近/超支→预警，无预算→不打扰
BUDGET_ALERT_THRESHOLDS = {
    "warn_ratio": 0.8,   # 已用比例>=80% 触发预警
    "over_ratio": 1.0,   # 已用比例>=100% 超支预警
    "reward_ratio": 0.5, # 已用比例<=50% 且未预警 → 犒劳提醒
}
# 记账动词：只有明确表达记账指令（记/记账/记录/入账…）才记账；
# "花了/买了"等消费陈述一律按咨询处理，绝不强制记账
BILL_VERB_KEYWORDS = [
    # 单字"记"即记账指令（"记午饭56元"），也是各组合词的子串，置于首位便于阅读
    "记",
    "记录", "记账", "记一下", "记一笔", "帮我记", "帮我记一下", "帮我记一笔",
    "记上", "记到", "记入", "记个账", "记下来", "入账", "登记",
    "写入", "写进", "写进账", "写进账单", "存账", "存账单", "存一笔", "存个账",
]
# 消费实体词：不单独构成记账意图，仅辅助识别消费咨询（金额+实体判定合理性疑问）
CONSUME_NOUNS = [
    "打车", "地铁", "公交", "出租", "网约车", "滴滴", "高铁", "火车", "机票", "飞机", "加油", "停车", "通勤",
    "吃饭", "外卖", "奶茶", "火锅", "晚餐", "午餐", "早餐", "夜宵", "咖啡", "零食", "小吃", "食堂", "聚餐", "烧烤",
    "酒店", "民宿", "住宿", "宾馆", "房租", "旅馆",
    "买", "书", "衣服", "裤子", "鞋", "超市", "商场", "淘宝", "京东", "手机", "电脑", "数码", "日用品", "包包",
    "电影", "KTV", "ktv", "门票", "演出", "游戏", "唱歌",
]
# 收入类词：本系统仅支持支出，收入/退款/报销等必须拦截 bill（bill_agent 侧也有一道拦截）
INCOME_KEYWORDS = [
    "工资", "发薪", "奖金", "收入", "收款", "到账", "收到", "退款", "报销", "分红",
    "中奖", "利息", "红包", "赚了", "挣了", "卖了", "回款", "转入", "发了",
]
# 拒绝记账词：用户明确不记账（"别记/不用记/只是问问"），必须拦截 bill。
# 刻意不含裸"不记"，避免误伤"我不记得了"
REFUSE_BILL_KEYWORDS = [
    "别记", "不用记", "先不记", "先别记", "别记账", "不用记账", "不要记", "不记账",
    "别入账", "不用入账", "只是问问", "就问一下", "随便问问", "随口问问",
]

# 伪记账词：含"记/记录"字串但语义不是记账（记得/日记/行车记录仪/笔记…）。
# 判定记账指令前先剥离这些词，避免"我记得打车花了80"被误判为记账
PSEUDO_RECORD_KEYWORDS = [
    # 含"记录"但指"设备/体裁"而非记账动作
    "行车记录仪", "记录仪", "记录片", "纪录",
    # 含"记"但为记忆/笔记/标记/日记等语义
    "记得", "记住", "记不清", "记不住", "记性", "记忆",
    "日记", "标记", "记号", "记事", "记事本", "笔记", "记者", "记载",
    "铭记", "纪念", "记仇", "记挂", "惦记",
    # 摘录类："抄记""笔录"等非记账表述
    "记单词", "记笔记", "背单词",
]

# 优先剥离的长词在前（子串关系：行车记录仪⊃记录仪⊃记录；记事本⊃记事）
PSEUDO_RECORD_KEYWORDS_SORTED = sorted(PSEUDO_RECORD_KEYWORDS, key=len, reverse=True)


def _strip_pseudo_record(text: str) -> str:
    """
    函数功能与逻辑描述：
        剥离文本中的伪记账词（如"记录仪""记事本""记单词""背单词"等），返回剩余文本，
        供 _has_record_cmd 在剥净后再判定是否含真正的记账动词（记/记账/入账/登记…）。
        为什么必须剥离：记账动词"记"是极高频子串，"行车记录仪""记单词"会让简单的子串匹配误判成记账指令。
        替换顺序按关键词长度倒序（PSEUDO_RECORD_KEYWORDS_SORTED），保证长词优先被切除——
        存在子串包含关系时（行车记录仪 ⊃ 记录仪 ⊃ 记录，记事本 ⊃ 记事），
        若先切短词会在长词中留下残余片段，导致后续仍能拼出误导性子串。
        纯字符串替换，不做分词、不抛异常。
    入参说明：
        text (str)：待清洗的用户原文。
    返回值说明：
        str：剥离全部伪记账词后的剩余文本；无命中时返回与入参内容相同的新字符串。
    """
    cleaned = text
    for w in PSEUDO_RECORD_KEYWORDS_SORTED:
        cleaned = cleaned.replace(w, "")
    return cleaned


def _has_record_cmd(text: str) -> bool:
    """
    函数功能与逻辑描述：
        严格记账指令判定：先剥离伪记账词，再判定是否命中 BILL_VERB_KEYWORDS
        （记/记账/入账/登记…），两者同时成立才算记账指令；用于区分
        "记午饭56元"（指令）与"我记得打车花了80"（陈述）这类易混输入。
    入参说明：
        text (str)：待判定用户原文。
    返回值说明：
        bool：True = 明确记账指令；False = 非记账指令。
    """
    return _has_any_keyword(_strip_pseudo_record(text), BILL_VERB_KEYWORDS)


def _has_any_keyword(text: str, keywords: list) -> bool:
    """
    函数功能与逻辑描述：
        关键词命中判定工具：只要 keywords 中任一关键词作为子串出现在 text 中即视为命中，
        供全部启发式规则（收入/拒绝记账/比价/统计/理财/合理性疑问等）统一复用。
    入参说明：
        text (str)：待判定文本。
        keywords (list)：关键词列表（如 INCOME_KEYWORDS / PRICE_HINT_KEYWORDS）。
    返回值说明：
        bool：True = 至少命中一个关键词；False = 全部未命中（含 text 为空）。
    """
    return any(kw in text for kw in keywords)


def _is_force_bill(user_text: str) -> bool:
    """
    函数功能与逻辑描述：
        强记账指令判定：要求「显式记账词 + 金额 + 消费实体」三者同时成立，
        且**排除**六类干扰意图——收入类、拒绝记账（"别记了"）、比价（"贵吗/值吗"）、
        统计（"这个月花了多少"）、理财（"怎么省"）、合理性疑问（"是不是花多了"）。
        用途：plan_node 的准入拦截分支据此放行——记账指令明确时不做额外拦截，
        交由下游（bill_agent 兜底补 bill 任务）处理，避免把明确的记账诉求误判成咨询。
        判定全部基于子串关键词命中，因此对同义改写敏感度有限；关键词表即本函数的事实定义源。
    入参说明：
        user_text (str)：用户本轮原始输入。
    返回值说明：
        bool：True = 记账指令明确且无干扰意图；False = 任一前置条件不满足或命中任一排除意图。
    """
    return (
        _has_record_cmd(user_text)   # 显式记账指令（剥离伪记账词后，见词表注释）
        and bool(AMOUNT_REGEX.search(user_text))
        and _has_any_keyword(user_text, CONSUME_NOUNS)
        and not _has_any_keyword(user_text, INCOME_KEYWORDS)
        and not _has_any_keyword(user_text, REFUSE_BILL_KEYWORDS)
        and not _has_any_keyword(user_text, PRICE_HINT_KEYWORDS)
        and not _has_any_keyword(user_text, STAT_HINT_KEYWORDS)
        and not _has_any_keyword(user_text, FINANCE_HINT_KEYWORDS)
        # 合理性疑问（"是不是花多了"）≠ 记账指令，不得强制记账
        and not _has_any_keyword(user_text, REVIEW_HINT_KEYWORDS)
    )


def _is_consume_statement(user_text: str) -> bool:
    """
    函数功能与逻辑描述：
        判定是否为"纯消费陈述"：无记账指令，但含金额或消费实体，且不含收入 / 拒绝记账 /
        合理性疑问 / 理财 / 比价 / 统计任一意图信号（即只陈述开销、无任何诉求）。
        这类输入默认按咨询处理——放行并补 price 对标任务，绝不记账；
        带意图词（值吗/贵吗/怎么省）的输入不归本函数，走各自任务分支。
    入参说明：
        user_text (str)：用户本轮原始输入。
    返回值说明：
        bool：True = 纯消费陈述；False = 非纯陈述（含意图信号或连金额/实体都没有）。
    """
    return (
        not _has_record_cmd(user_text)
        and bool(AMOUNT_REGEX.search(user_text) or _has_any_keyword(user_text, CONSUME_NOUNS))
        and not _has_any_keyword(user_text, INCOME_KEYWORDS)
        and not _has_any_keyword(user_text, REFUSE_BILL_KEYWORDS)
        and not _has_any_keyword(user_text, REVIEW_HINT_KEYWORDS)
        and not _has_any_keyword(user_text, FINANCE_HINT_KEYWORDS)
        and not _has_any_keyword(user_text, PRICE_HINT_KEYWORDS)
        and not _has_any_keyword(user_text, STAT_HINT_KEYWORDS)
        # 变更既有账单（"把 id=5 那笔改成 60"/"删掉那笔"）是**写诉求**，不是"纯消费陈述"：
        #   若误判为陈述，兜底会补 price 任务 → 用户说"改"，系统回一段比价结论。
        and not _has_any_keyword(user_text, EDIT_HINT_KEYWORDS)
        and not _has_any_keyword(user_text, DELETE_HINT_KEYWORDS)
    )


# ============ M16（D18）：意图层能力补齐（契约校验 / 落地校验 / 三分支话术）============

# 缺失参数的中文标签（用于追问话术）
_PARAM_LABEL = {"amount": "金额"}

# 任务的中文标签（话术用，★不向用户暴露内部任务名）
_TASK_LABEL = {"bill": "记账", "stat": "统计", "price": "比价", "finance": "消费分析"}

# OOD 兜底：能力清单 + 用法示例（业界 Rich Fallback 做法；`设计/14` §3.6）
_OOD_FALLBACK_DEFAULT = (
    "这个我暂时处理不了。我能帮你：\n"
    "· 记账 → 「记 午饭 35」\n"
    "· 查统计 → 「查本月餐饮」\n"
    "· 比价 → 「火锅人均120贵吗」\n"
    "· 消费分析 → 「给我点省钱建议」"
)

# 低置信澄清的兜底选项
_CLARIFY_DEFAULT = (
    "我不太确定您想做什么。您可以说：\n"
    "· 「记 午饭 35」记账\n"
    "· 「查本月餐饮」查统计\n"
    "· 「火锅120贵吗」比价\n"
    "· 「给我点省钱建议」消费分析"
)


def _contract_required(task_name: str, sub_type: str) -> list:
    """
    函数功能与逻辑描述：
        从契约表（`config/agent_contracts.json`，经 `config.AGENT_CONTRACTS` 载入）取
        某任务在某细分操作下的**必需参数**列表：`required` ∪ `required_by_sub_type[sub_type]`（去重）。
        契约表读失败（AGENT_CONTRACTS 为 {}）时返回空列表 —— 调用方随即退化为"只看 LLM 自报"。
    入参说明：
        task_name (str)：任务名（bill / stat / price / finance）。
        sub_type (str)：细分操作（add / edit / delete / query / analyse）。
    返回值说明：
        list：必需参数名列表（如 ["amount"]）；无必需参数或契约缺失时返回 []。
    """
    contract = AGENT_CONTRACTS.get(task_name) or {}
    req = list(contract.get("required") or [])
    by_sub = (contract.get("required_by_sub_type") or {}).get(sub_type or "") or []
    for p in by_sub:
        if p not in req:
            req.append(p)
    return req


def _check_contract(tasks: list, user_text: str, llm_missing: dict) -> dict:
    """
    函数功能与逻辑描述：
        M16（D18）契约校验：产出 `missing`（任务名 → 缺失参数名列表）。
        **只判"缺不缺"，不填参数** —— 这是"子 Agent 入参契约不变"的前提。
        两个来源取**并集**（保守：宁可多问）：
          ① 代码判定（确定性）：按契约表逐任务取必需参数，`amount` 以 AMOUNT_REGEX 在
             「任务 raw_segments ∪ 用户原文」范围内查找；
          ② LLM 自报（`missing` 字段）：补充代码判不了的参数（如"缺品类"）。
    入参说明：
        tasks (list)：已通过格式校验的任务列表（含 name / raw_segments / operate_sub_type）。
        user_text (str)：用户本轮原文（**OR 范围**：片段与原文都算，避免误判缺参）。
        llm_missing (dict)：规划 LLM 自报的 missing（可能为 {}）。
    返回值说明：
        dict：{任务名: [缺失参数名, ...]}；全部齐备时返回 {}。
    """
    missing: dict = {}

    # ① 代码判定（确定性）
    for t in tasks:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if name not in TASK_AGENT_MAP:
            continue
        sub = t.get("operate_sub_type") or TASK_SUB_TYPE_DEFAULTS.get(name, "")
        req = _contract_required(name, sub)
        if not req:
            continue
        seg_text = " ".join(str(s) for s in (t.get("raw_segments") or []) if isinstance(s, str))
        scope_text = f"{seg_text} {user_text or ''}"
        lack = []
        for param in req:
            if param == "amount" and not AMOUNT_REGEX.search(scope_text):
                lack.append("amount")
        if lack:
            missing.setdefault(name, [])
            for p in lack:
                if p not in missing[name]:
                    missing[name].append(p)

    # ② 并入 LLM 自报（保守取并集：宁可多问）
    if isinstance(llm_missing, dict):
        for k, v in llm_missing.items():
            if not v:
                continue
            missing.setdefault(k, [])
            for p in (v if isinstance(v, list) else [v]):
                p = str(p)
                if p not in missing[k]:
                    missing[k].append(p)

    return missing


def _verify_grounding(segs: list, user_text: str) -> tuple:
    """
    函数功能与逻辑描述：
        M16（D18）参数落地校验（grounding check）：归一化只允许"重述 + 对齐"，**禁止新增事实**。
        规则：片段中的**数值型实体**必须能在用户原文回溯（原文字面命中）；
        命中不了的数字 → 从片段中剔除并记录，避免模型把用户没说的金额写进 raw_segments。
        类别 / 术语型实体不做校验（允许同义对齐，如"吃饭"→"餐饮"）。
    入参说明：
        segs (list)：待校验的文本片段列表（即某任务的 raw_segments）。
        user_text (str)：用户本轮原文（回溯依据）。
    返回值说明：
        tuple：二元组 (过滤后的片段列表, 被剔除的数字列表)。
    """
    if not user_text:
        return list(segs), []
    origin_digits = set(re.findall(r"\d+(?:\.\d+)?", user_text))
    kept, dropped = [], []
    for s in segs:
        s_str = str(s)
        nums = re.findall(r"\d+(?:\.\d+)?", s_str)
        bad = [n for n in nums if n not in origin_digits]
        if bad:
            dropped.extend(bad)
            fixed = s_str
            for n in bad:
                fixed = fixed.replace(n, "")
            kept.append(fixed.strip())
        else:
            kept.append(s_str)
    return kept, dropped


def _preplan_ask_key(scope: str) -> str:
    """
    函数功能与逻辑描述：
        M16（D18）预规划阶段的追问计数 key。
        起因：`_ask_round_key` 依赖已派发结果（子 Agent struct + task_plan），
        而主 Agent 在**派发前**追问时 `task_plan` 为空、无 struct → 拿不到 key。
        故按分支来源生成会话级 key，使主 Agent 追问与子 Agent 追问**共用同一套
        `ASK_ROUND_LIMIT` 上限**（否则形成"主 Agent 无限追问"的新死循环）。
        注：计数落在**会话级**记忆（与子 Agent 追问同口径，`01` §3.4）。
    入参说明：
        scope (str)：分支标识（"missing" / "ood" / "low_confidence"）。
    返回值说明：
        str：会话级追问计数 key，形如 "preplan_missing"。
    """
    return f"preplan_{scope or 'unknown'}"


def _has_text_basis(seg: str, user_text: str, min_len: int = 2) -> bool:
    """
    函数功能与逻辑描述：
        判断片段 seg 与用户原文是否有「连续公共子串」（≥ min_len 字符）——
        用于保守规划的**话语依据校验**（识别"瞎猜"）。轻量实现：滑动窗口子串匹配，
        不引分词依赖；min_len=2 偏保守（宁可放过短对齐），避免误杀同义改写。
    入参说明：
        seg (str)：待校验片段（任务的 raw_segments 元素）。
        user_text (str)：用户本轮原文。
        min_len (int)：最小公共子串长度，默认 2。
    返回值说明：
        bool：True = 存在 ≥ min_len 的连续公共子串（视为有依据）。
    """
    if not seg or not user_text:
        return False
    n = len(seg)
    if n < min_len:
        return seg in user_text
    for i in range(n - min_len + 1):
        if seg[i:i + min_len] in user_text:
            return True
    return False


def _check_inferable_grounding(tasks: list, user_text: str) -> tuple:
    """
    函数功能与逻辑描述：
        M17（D19）**保守规划守门人**：验证 `fit_level == "inferable"` 的规划是否成立。
        两条判据（**缺一即不成立**）：
          ① **参数齐备**——保守规划出的任务**不应缺参**：缺参说明该能力需要用户提供数据，
             不属于"系统自己能查"的 inferable 场景（复用 `_check_contract` 判定）；
          ② **有话语依据**——每个任务的至少一个 raw_segments 片段能与用户原文对齐
             （`_has_text_basis`），找不到依据视为"瞎猜"。
        任一不成立 → 返回 False，调用方**降级为追问**（不得放行瞎猜，`设计/15` §六 R-2）。
    入参说明：
        tasks (list)：规划出的任务列表（含 name / raw_segments）。
        user_text (str)：用户本轮原文（依据来源）。
    返回值说明：
        tuple：(ok: bool, reason: str)；ok=True 表示保守规划成立、可放行。
    """
    # ① 参数齐备性（缺参 = 该能力需要用户数据，不属于 inferable）
    _missing = _check_contract(tasks, user_text, {})
    if _missing:
        return False, f"保守规划任务仍缺参：{_missing}"
    # ② 话语依据（每任务至少一个片段能与原文对齐）
    for t in tasks:
        if not isinstance(t, dict):
            continue
        segs = [str(s) for s in (t.get("raw_segments") or []) if str(s).strip()]
        if not segs:
            return False, f"任务 {t.get('name')} 无 raw_segments 依据"
        if not any(_has_text_basis(s, user_text) for s in segs):
            return False, f"任务 {t.get('name')} 的片段与原文无公共子串（疑似瞎猜）"
    return True, "依据充分"


def _build_ood_fallback(clarify_options: list) -> str:
    """OOD 分支话术：优先用模型给的候选，否则回能力清单 + 用法示例（Rich Fallback）。"""
    if clarify_options:
        return "这个我暂时处理不了。我可以帮你：\n" + "\n".join(f"· {o}" for o in clarify_options)
    return _OOD_FALLBACK_DEFAULT


def _build_missing_ask(missing_map: dict, clarify_options: list) -> str:
    """
    缺参分支话术：优先用模型给的候选选项（贴合用户当前话题）；
    否则回退到「中文任务标签 + 通用引导」，**不得向用户暴露内部任务名**（如 price）。
    """
    if clarify_options:
        return "还需要一点信息：\n" + "\n".join(f"· {o}" for o in clarify_options)
    parts = []
    for task, params in missing_map.items():
        label = _TASK_LABEL.get(task, task)
        labels = "、".join(_PARAM_LABEL.get(p, p) for p in params)
        parts.append(f"{label}还差{labels}")
    detail = "；".join(parts)
    return (
        f"{detail}。您可以直接说，例如：\n"
        "· 比价 → 「唱歌花了 200，贵不贵」\n"
        "· 记账 → 「记 唱歌 200」\n"
        "· 查统计 → 「查本月娱乐花费」"
    )


def _build_clarify(clarify_options: list, user_text: str) -> str:
    """低置信分支话术：优先用模型给的候选选项，否则回通用引导。"""
    if clarify_options:
        return "我不太确定您的意思，您是想要：\n" + "\n".join(f"· {o}" for o in clarify_options)
    return _CLARIFY_DEFAULT


def prune_tasks_by_heuristics(user_text: str, tasks: list) -> list:
    """
    函数功能与逻辑描述：
        按启发式规则裁剪/补齐 LLM 规划出的任务列表，纠正 1.8B 小模型的语义误判，
        是 plan_node 中「LLM 规划 + 规则纠偏」双保险的规则侧实现。
        两类动作：
        - 裁剪：删掉与用户意图明显不符的任务（如无诉求却脑补出 stat/finance 任务、
          把一句建议性表述当成记账任务）；
        - 补齐：模型漏规划但意图明确时补出对应任务（bill / price / finance / stat）。
        ★核心例外：bill 任务**只在用户显式下达记账指令时才保留或补齐**，
        纯消费陈述（如"午饭花了56元"）一律按咨询处理、不补记账——这是防止误记账的关键闸门。
        实现上刻意**不对空任务列表提前 return**，否则记账/收入/拒绝记账三类兜底分支将全部失效。
    入参说明：
        user_text (str)：用户本轮原始输入，全部启发式判定都基于它。
        tasks (list)：LLM 规划出的任务列表（元素为任务名字符串）；允许为空列表。
    返回值说明：
        list：纠偏后的任务列表。可能比入参更短（裁剪）或更长（补齐）；
            用户指令与规则冲突时以规则结论为准。
    """
    # 不能对空任务列表提前 return，否则下方记账/收入/拒绝记账兜底都执行不到。
    # 记账兜底只对"显式记账指令"句生效（如"记打车35元"）；无指令消费陈述不补记账。
    has_amount = bool(AMOUNT_REGEX.search(user_text))
    has_stat_hint = _has_any_keyword(user_text, STAT_HINT_KEYWORDS)
    has_finance_hint = _has_any_keyword(user_text, FINANCE_HINT_KEYWORDS)
    has_price_hint = _has_any_keyword(user_text, PRICE_HINT_KEYWORDS)
    # 合理性疑问（"这笔消费合理吗"）：语义=对比市场水平（溢价率）
    has_review_hint = _has_any_keyword(user_text, REVIEW_HINT_KEYWORDS)
    # 记账指令：剥离伪记账词后仍命中记账动词才算（防"我记得打车花了80"误判）
    has_bill_verb = _has_record_cmd(user_text)
    has_income = _has_any_keyword(user_text, INCOME_KEYWORDS)
    has_consume_noun = _has_any_keyword(user_text, CONSUME_NOUNS)
    # 拒绝记账（"别记这笔/只是问问"）→ 任何情况下不派发 bill
    has_refuse_bill = _has_any_keyword(user_text, REFUSE_BILL_KEYWORDS)
    # ★变更既有账单（改/删）：与收入、拒绝记账互斥（那两类一律不写库）
    has_edit_hint = _has_any_keyword(user_text, EDIT_HINT_KEYWORDS)
    has_delete_hint = _has_any_keyword(user_text, DELETE_HINT_KEYWORDS)
    has_bill_mutation = ((has_edit_hint or has_delete_hint)
                         and not has_income and not has_refuse_bill)
    # 带金额的合理性疑问（"花了300买耳机，值吗"）→ 归 finance"这笔值不值"
    #（REVIEW 词不在 PRICE/FINANCE 词表中，需显式映射，否则会被裁剪丢失）
    has_finance_review = (
        has_review_hint and has_amount and (has_bill_verb or has_consume_noun)
        and not has_income
    )
    pruned = []
    pruned_names = set()  # 用于同类型任务去重：stat/price/finance 单意图只保留第一个
    for task in tasks:
        name = task.get("name")
        if name == "bill":
            # 记账需金额 + 记账动词；收入/拒绝记账/纯分析场景一律不记账
            # ★例外：改/删诉求**常常既无金额也无记账动词**（"删掉那笔"/"把第 2 条改成 50"），
            #   此时按变更意图放行；edit/delete 之分交给 has_delete_hint。
            if has_income or has_refuse_bill:
                continue
            if not has_bill_mutation and (not has_amount or not has_bill_verb):
                continue
            if has_bill_mutation:
                # 模型侧 bill 的 sub_type 兜底是 add（TASK_SUB_TYPE_DEFAULTS），
                # 变更意图下必须就地纠正——否则"改"会被当成"新增"落库（契约 `12` §3.1）。
                task = dict(task)
                task["operate_sub_type"] = "delete" if has_delete_hint else "edit"
        elif name == "stat":
            # **finance 需要数据时不得删 stat**。`has_stat_hint` 的词表（统计/查询/汇总/这个月/
            #   花了多少/支出/开销/余额…）覆盖不了"算算剩下预算够不够撑到月底"这类**隐含查账**的
            #   表述；只据此裁剪会把 LLM 正确规划的 stat 一并裁掉，finance 拿不到当月支出、
            #   只能回"暂缺当月支出数据"。stat↔price 可各自独立裁剪，但 **stat→finance 是真实
            #   数据依赖**（`TASK_CONTEXT_DEPS` 已声明），不能一刀切。
            # stat **允许多条**："查本月 + 查上月"这类不同时间范围的统计互不替代，不做同名去重。
            stat_needed_by_finance = _has_any_keyword(user_text, FINANCE_DATA_HINT_KEYWORDS)
            if not (has_stat_hint or stat_needed_by_finance):
                continue
        elif name == "finance":
            # finance **同轮至多一条**：它产出的是"综合分析"结论，多条无意义且会重复派发
            #   （其余类型 bill / stat / price 均允许多条）。
            if not (has_finance_hint or has_finance_review) or name in pruned_names:
                continue
        elif name == "price":
            # **price 与 finance 不互斥**："贵不贵 / 值不值 / 划算吗"既要比价（拿当地同品类均价
            #   算溢价率）也要合理性分析，两者互补而非二选一。**不可**用 `has_finance_review`
            #   把 price 裁掉，否则"吃火锅花了80，贵不贵"只会派发 finance、比价永远不发生。
            # 比价的归类与计算**只在 price_agent 内完成**，顶层不做比价。
            # price **允许多条**："对比火锅 + 对比奶茶"这类不同品类的比价互不替代，不做同名去重。
            if not has_price_hint:
                continue
        pruned.append(task)
        pruned_names.add(name)

    # 补齐：改/删诉求必须有 bill 任务——模型常把它规划成 price，规则侧在此显式补出，
    #   使其在"pruned 为空"的兜底（消费陈述→补 price）之前就被满足，
    #   否则用户说"改"，系统回一段比价结论。
    if has_bill_mutation and "bill" not in pruned_names:
        pruned.append({"name": "bill", "raw_segments": [user_text],
                       "operate_sub_type": "delete" if has_delete_hint else "edit"})

    # ===== 兜底补齐：意图明确但模型漏规划时补任务（小模型常漏规划 bill/price）=====
    # 即使裁剪后仍有任务也继续补齐，保证多意图不被漏掉
    pruned_names = {t.get("name") for t in pruned}
    seg = [user_text]
    if has_bill_verb and has_amount and not has_income and not has_refuse_bill and "bill" not in pruned_names:
        pruned.append({"name": "bill", "raw_segments": seg, "operate_sub_type": "add"})
    # price 与 finance **不互斥**："贵不贵/值不值"这类带金额的合理性疑问**既要比价也要分析**，
    #   因此合理性疑问（has_finance_review）同样要补 price。若写成
    #   `has_price_hint and not has_finance_review`，合理性疑问下**永远补不出比价**。
    if (has_price_hint or has_finance_review) and "price" not in pruned_names:
        pruned.append({"name": "price", "raw_segments": seg, "operate_sub_type": "query"})
    if (has_finance_hint or has_finance_review) and "finance" not in pruned_names:
        pruned.append({"name": "finance", "raw_segments": seg, "operate_sub_type": "analyse"})
    if has_stat_hint and "stat" not in pruned_names and _has_any_keyword(user_text, STAT_ENSURE_KEYWORDS):
        pruned.append({"name": "stat", "raw_segments": seg, "operate_sub_type": "query"})
    # finance 需要数据支撑时**必须把 stat 一并补上**："算预算够不够 / 撑到月底"这类诉求依赖
    #   **当月支出**，而只有 stat 会查账；否则 finance 拿不到当月支出、只能回"暂缺当月支出数据"。
    #   `TASK_CONTEXT_DEPS["finance"]` 已声明该依赖，此处把它**兑现为实际补齐**。
    #   刻意**不**沿用上游补 stat 的触发条件：那条要求"用户明确要统计"，而本场景用户只是
    #   **要求分析**、并未要求查账；故用独立的数据依赖词表判定（作用域仅限"finance 在列而 stat 缺失"）。
    if ("finance" in pruned_names and "stat" not in pruned_names
            and _has_any_keyword(user_text, FINANCE_DATA_HINT_KEYWORDS)):
        pruned.append({"name": "stat", "raw_segments": seg, "operate_sub_type": "query"})
        print(f"[ORCH] 补齐 stat：finance 分析依赖当期账目（命中文中数据依赖词）", flush=True)

    if not pruned:
        if has_income:
            # 纯收入/退款场景：裁剪后为空 → 返回空任务（不记账）
            print(f"[ORCH] 收入场景拦截：user={user_text[:30]!r} 返回空任务", flush=True)
            return []
        if has_refuse_bill:
            # 拒绝记账场景：裁剪后为空 → 返回空任务（不记账）
            print(f"[ORCH] 拒绝记账拦截：user={user_text[:30]!r} 返回空任务", flush=True)
            return []
        # 无记账指令的消费陈述默认按咨询执行——补 price 对标任务，给"这笔贵不贵"的结论。
        # 选 price 而非 finance：price 只需 city/category/amount，无预算也能跑。
        if _is_consume_statement(user_text):
            print(f"[ORCH] 消费陈述默认咨询：user={user_text[:30]!r} 补 price 对标任务", flush=True)
            return [{"name": "price", "raw_segments": seg, "operate_sub_type": "query"}]
        # 无法确定意图时信任模型原始规划，但被裁掉的 bill 一律不复活
        #（bill 被裁 = 用户未显式下达记账指令）；price/finance/stat 等咨询任务原样保留
        kept = [t for t in tasks if t.get("name") != "bill"]
        if not kept:
            return []
        return kept
    return pruned


def repair_task_names(raw_tasks: list) -> list:
    """
    函数功能与逻辑描述：
        修正 name 非法（不在 TASK_AGENT_MAP）的任务——小模型常把用户原话/测试标记塞进 name。
        推断优先级：operate_sub_type 映射（add→bill / analyse→finance）> raw_segments 关键词
        （bill 只走严格记账指令判定 _has_record_cmd，防"我记得打车花了80"被推断成 bill）；
        仍无法推断则丢弃并通过 _log 告警，避免整轮规划解析失败。
        合法 name 与非 dict 元素分别原样保留 / 直接跳过。
    入参说明：
        raw_tasks (list)：模型输出的原始任务列表，元素应为 dict。
    返回值说明：
        list：name 已修正或原本合法的任务列表；无法推断 name 的任务被剔除。
    """
    sub_type_to_task = {
        "add": "bill",
        "analyse": "finance",
    }
    intent_keywords = [
        # bill 不走关键词匹配，只走 _has_record_cmd（防"我记得打车花了80"被推断成 bill）
        ("price", PRICE_HINT_KEYWORDS),
        ("finance", FINANCE_HINT_KEYWORDS),
        ("stat", STAT_HINT_KEYWORDS),
    ]

    fixed = []
    for task in raw_tasks:
        if not isinstance(task, dict):
            continue
        name = task.get("name")
        if name in TASK_AGENT_MAP:
            fixed.append(task)
            continue

        repaired = sub_type_to_task.get(str(task.get("operate_sub_type", "")).strip().lower())
        if not repaired:
            segs = [str(s) for s in task.get("raw_segments", []) if isinstance(s, str)]
            hint_text = " ".join(segs)
            # 先按严格记账指令判定，再按普通关键词推断其它任务类型
            if _has_record_cmd(hint_text):
                repaired = "bill"
            else:
                for std_name, keywords in intent_keywords:
                    if _has_any_keyword(hint_text, keywords):
                        repaired = std_name
                        break
        if repaired:
            _log("任务name修正", f"name={name!r} -> {repaired!r}")
            task["name"] = repaired
            fixed.append(task)
        else:
            _log("任务name无法推断，丢弃", f"name={name!r}")
    return fixed


def _render_stat_display_text(msg: str, data) -> str:
    """
    函数功能与逻辑描述：
        渲染统计结果文本素材：按 总支出 → 笔数 → 分类明细 → 平均单笔 顺序，
        仅拼接 data 中实际存在的字段（None 跳过）；data 非 dict/为空，或所有字段都缺失时
        优雅降级为 `{msg}：{data}`。
    入参说明：
        msg (str)：统计结果的语义前缀（stat_agent 返回的 msg）。
        data (Any)：统计结构化数据，期望 dict 且含 total_amount / total_count /
            category_summary / avg_amount 等键。
    返回值说明：
        str：以 "；" 拼接的可读统计文本；数据缺失时降级为 `{msg}：{data}`。
    """
    if not isinstance(data, dict) or not data:
        return f"{msg}：{data}"
    parts = [msg]
    if data.get("total_amount") is not None:
        parts.append(f"总支出 {data['total_amount']} 元")
    if data.get("total_count") is not None:
        parts.append(f"共 {data['total_count']} 笔")
    cs = data.get("category_summary")
    if cs:
        cat_str = "，".join(f"{k} {v} 元" for k, v in cs.items())
        parts.append(f"分类明细：{cat_str}")
    if data.get("avg_amount") is not None:
        parts.append(f"平均单笔 {data['avg_amount']} 元")
    return "；".join(parts) if len(parts) > 1 else f"{msg}：{data}"


def _render_price_display_text(msg: str, data) -> str:
    """
    函数功能与逻辑描述：
        渲染物价对标结果文本素材：以溢价率为必选字段，消费档次（consume_level）与
        结论（conclusion）存在时以 "，" 追加；data 非 dict 或 premium_rate 缺失时
        降级为 `{msg}：{data}`。
    入参说明：
        msg (str)：对标结果的语义前缀（price_agent 返回的 msg）。
        data (Any)：对标结构化数据，期望 dict 且含 premium_rate /
            consume_level（可选）/ conclusion（可选）。
    返回值说明：
        str：形如 `{msg}：溢价率 x%，消费档次 y，结论 z` 的可读文本；
            数据缺失时降级为 `{msg}：{data}`。
    """
    # ★M18：价位查询模式（有品类无金额 → 只返回本地参考均价，无溢价率）
    if isinstance(data, dict) and data.get("premium_rate") is None:
        base = data.get("city_base_avg_price")
        if base:
            city = data.get("city") or ""
            cat = data.get("category") or ""
            return f"{msg}：【{city}{cat}】本地参考均价约 {base} 元（告诉我具体金额可判断是否偏高）"
        return f"{msg}：{data}"
    if not isinstance(data, dict):
        return f"{msg}：{data}"
    premium = data["premium_rate"]
    level = data.get("consume_level", "")
    conclusion = data.get("conclusion", "")
    parts = [f"{msg}：溢价率 {premium}%"]
    if level:
        parts.append(f"消费档次 {level}")
    if conclusion:
        parts.append(str(conclusion))
    return "，".join(parts)


def build_user_display_text(struct: dict) -> str:
    """
    函数功能与逻辑描述：
        唯一生成面向用户可读文本的函数：子 Agent 禁止自行拼接展示文案，统一由 Orchestrator
        基于子 Agent 返回的结构化数据渲染，保证数字来源确定性、口径一致。
        按 success 分支：True → 依 agent_type 分派到 bill/stat/price/finance 各自渲染
        （finance 优先取 raw_analysis_text 确定性分析文本，不整字典 dump）；
        False → error=="need_more_info" 时回传 data["prompt"] 作为多轮追问语，
        其余失败统一渲染为 `操作失败：{err}`；success 非 True/False（如旧版兜底的 None）
        → 原样返回 msg。
    入参说明：
        struct (dict)：子 Agent 统一通信结构
            {"success":bool|None, "agent_type":str, "msg":str, "data":dict|None, "error":str|None}。
    返回值说明：
        str：面向用户的可读文本；追问场景为 data["prompt"]，失败场景为 `操作失败：{err}`。
    """
    success = struct.get("success", False)
    agent_type = struct.get("agent_type", "")
    data = struct.get("data")
    err = struct.get("error", "")
    msg = struct.get("msg", "")

    if success is True:
        if agent_type == "bill_agent":
            # 结果字段因操作类型而异：add/edit 带 category/amount/consume_date，**delete 只带
            #   `{"id": ...}`**。不可用硬下标 `data['category']`，否则一次成功的删除会被渲染成
            #   「【系统异常】'category'」、让用户以为删除失败。故：① 优先采用工具循环给出的
            #   reply 话术；② 只拼接实际存在的字段。
            # **分支顺序很重要**：**有 category（add/edit）必须走下方固定句式**——`"记账成功：
            #   消费类目 …"` 是契约句式，本地确定性渲染与下游断言都依赖它。若把 `data.reply`
            #   提前优先返回，reply 是 LLM 生成的**随机措辞**，会让
            #   `tests/integration/test_memory_agent_integration.py::test_multi_round_session_memory`
            #   的 `assert "记账成功" in final_reply` **随机失败**。
            #   只有 delete 载荷（**无 category/amount**）才退化到 reply / id 形态。
            # **多笔必须逐笔列出**：否则多笔成功却只报首笔，用户会以为少记了一笔
            #   （"静默丢账"观感，与真实丢账同样有害）。
            #   单笔仍走下方固定句式（契约句式，下游断言依赖）。
            if (isinstance(data, dict) and isinstance(data.get("items"), list)
                    and len(data["items"]) > 1):
                lines = [f"{msg}（共 {len(data['items'])} 笔）"]
                for i, p in enumerate(data["items"], 1):
                    lines.append(f"{i}. {p.get('category')} {p.get('amount')} 元"
                                 f"（{p.get('consume_date')}）")
                return "；".join(lines)
            if isinstance(data, dict) and data.get("category") is not None:
                return (f"{msg}：消费类目 {data['category']}，金额 {data['amount']} 元，"
                        f"日期 {data['consume_date']}")
            if isinstance(data, dict) and data.get("reply"):
                return str(data["reply"])
            return f"{msg}：已处理账单 id={data.get('id')}"
        elif agent_type == "stat_agent":
            return _render_stat_display_text(msg, data)
        elif agent_type == "price_agent":
            return _render_price_display_text(msg, data)
        elif agent_type == "finance_agent":
            # finance 成功载荷带 raw_analysis_text（确定性分析文本），直接取用，不整字典 dump
            fin_text = (data or {}).get("raw_analysis_text")
            if isinstance(fin_text, str) and fin_text.strip():
                return f"{msg}：{fin_text.strip()}"
            return f"{msg}：{data}"
        return msg
    elif success is False:
        # 多轮追问标记：需要向用户收集信息（统一约定error="need_more_info"）
        if err == "need_more_info":
            return data["prompt"]
        return f"操作失败：{err}"
    return msg


def build_task_plan_prompt(user_input: str, history: str, valid_tasks: list) -> tuple[str, str]:
    """
    函数功能与逻辑描述：
        组装规划提示词：系统提示词取自目录内 system.md（系统角色规则），
        用户提示词用 user.j2（Jinja 模板）渲染，注入用户输入、历史上下文、可选任务名清单，
        以及今天日期（供 LLM 换算"昨天/这个月/上个月"等相对时间）。
    入参说明：
        user_input (str)：用户本轮原始输入。
        history (str)：会话历史 + 长期记忆拼装后的上下文文本。
        valid_tasks (list)：允许的任务名列表，渲染为模板中的 valid_tasks_str。
    返回值说明：
        tuple[str, str]：(系统提示词 system 内容, 渲染后的用户提示词)。
    """
    sys_prompt = load_system()
    valid_tasks_str = ", ".join(valid_tasks)
    user_prompt = render_user(
        user_input=user_input,
        history=history,
        valid_tasks_str=valid_tasks_str,
        # 注入今天日期，供 LLM 换算「昨天/这个月/上个月」等相对时间
        today_date=date.today().isoformat(),
    )
    return sys_prompt, user_prompt


# ===================== 任务依赖动态化：静态表 ∪ LLM deps =====================
def _collect_llm_deps(tasks: list, task_names: set) -> dict:
    """
    函数功能与逻辑描述：
        从 LLM 规划结果提取任务间 deps（依赖），并做引用校验——只保留本批任务内存在、
        且非自依赖的依赖名；非 dict 元素、name 不在本批任务内、deps 非 list 的条目全部跳过。
        刻意从 LLM 原始输出取而非 task_plan：中间链路可能重建任务 dict 丢字段，按任务名回捞更稳。
    入参说明：
        tasks (list)：LLM 原始规划任务列表（含 deps 字段）。
        task_names (set)：本批任务的合法任务名集合，用于引用校验。
    返回值说明：
        dict：{任务名: 依赖任务名集合}；无有效依赖的条目不出现在结果中（可能为空字典）。
    """
    out: dict = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        name = task.get("name")
        if name not in task_names:
            continue
        raw = task.get("deps")
        if not isinstance(raw, list):
            continue
        # 引用校验：只保留本批任务内存在的任务名，排除自依赖，并**剔除与静态表方向相反的依赖**。
        #   `TASK_CONTEXT_DEPS` 已声明 finance 依赖 stat（stat→finance 是真实数据流），
        #   而模型可能把方向写反成 `stat: ["finance"]`。反向边与静态表互指成环 → 触发
        #   `_has_cycle` → 回退**整张**依赖表（把其它**正确**的附加依赖一并丢掉，惩罚过重）。
        #   此处提前剔掉反向边：模型的合理补充得以保留，且不可能再因反向边成环。
        #   判据：若 `d` 在静态表里依赖 `name`，则 `name` 依赖 `d` 属反向，剔除。
        valid = {d for d in raw
                 if isinstance(d, str) and d in task_names and d != name
                 and name not in TASK_CONTEXT_DEPS.get(d, [])}
        if valid:
            out[name] = valid
    return out


def _merge_deps(task_plan: dict, llm_deps: dict) -> tuple:
    """
    函数功能与逻辑描述：
        计算最终依赖 final_deps = 静态表 TASK_CONTEXT_DEPS ∪ LLM deps（并集 + 裁剪），
        就地把每个任务的 deps 覆盖为排序后的并集。约束：LLM 只能增加依赖、不能删除静态依赖
        （最坏退化为纯静态串行，不会更差）；两侧依赖都只保留本批任务内存在的任务名。
    入参说明：
        task_plan (dict)：{任务名: 任务信息 dict}，会被就地更新 deps 字段。
        llm_deps (dict)：LLM 依赖映射 {任务名: 依赖集合}；传 {} 即回退纯静态表。
    返回值说明：
        tuple：(更新后的 task_plan, 本轮新增的动态依赖 {任务名: [依赖名]}；
            无新增时为空 dict)。
    """
    task_names = set(task_plan)
    added = {}
    for tn in task_names:
        static = {d for d in TASK_CONTEXT_DEPS.get(tn, []) if d in task_names}
        dyn = {d for d in llm_deps.get(tn, set()) if d in task_names}
        task_plan[tn]["deps"] = sorted(static | dyn)
        extra = dyn - static
        if extra:
            added[tn] = sorted(extra)
    return task_plan, added


# ===================== 内部工具：检测任务依赖是否存在循环 =====================
def _has_cycle(task_plan: dict) -> bool:
    """
    函数功能与逻辑描述：
        深度优先遍历检测任务依赖是否成环（例：A 依赖 B、B 依赖 A → 成环，无法执行，直接终止）。
        双集合口径：visited 记录已探索节点（避免重复遍历），rec_stack 记录当前递归栈路径，
        邻接节点已在 rec_stack 中即判定成环；从每个未访问节点发起遍历以覆盖非连通图。
    入参说明：
        task_plan (dict)：{任务名: 任务信息 dict}，任务信息须含 "deps"（依赖任务名列表）。
    返回值说明：
        bool：True = 存在循环依赖；False = 无环。
    """
    visited = set()
    rec_stack = set()

    def dfs(node):
        """
        函数功能与逻辑描述：
            _has_cycle 内嵌的深度优先搜索：把 node 标记入 visited 与 rec_stack 后递归其 deps，
            命中 rec_stack 中的节点即回传 True 并立即向上短路；本条路径探索完毕则把 node
            移出 rec_stack，保证 rec_stack 始终等价于"当前递归栈路径"。
        入参说明：
            node：当前访问的任务名（须存在于外层 task_plan 中，否则读取 deps 抛 KeyError）。
        返回值说明：
            bool：True = 以 node 为根的子树中存在环；False = 无环。
        """
        visited.add(node)
        rec_stack.add(node)
        for dep in task_plan[node]["deps"]:
            if dep not in visited:
                if dfs(dep):
                    return True
            elif dep in rec_stack:
                # 在当前递归栈找到依赖，出现循环
                return True
        rec_stack.remove(node)
        return False

    for t in task_plan:
        if t not in visited and dfs(t):
            return True
    return False


# ===================== 读取用户预算目标配置 =====================
async def get_user_target_config(session_id: str) -> dict:
    """
    函数功能与逻辑描述：
        读取用户最新一行的预算目标配置（经 `config.get_latest` 事实工具，不裸连数据库），
        作为 finance_agent 的目标数据来源注入分析提示词。
        返回结构直接以列名取值（因 query_sql 已统一返回字典列表，不能按 tuple 处理）。
        针对旧库缺列的兼容处理：category_budget 为 JSON 字符串，解析失败或为空时退化为空字典；
        quota_mode 缺失时默认 "auto"（自动选择品类额度评估方式）；remind_pref 缺失时为空串。
        ★注意：查询结果为空时返回空字典 {}，由 finance_agent 据此触发"缺少预算"追问，
        因此调用方必须区分"空字典 = 无配置"与"字段值为空"两种情形。
    入参说明：
        session_id (str)：会话标识。当前实现**未使用该形参**（配置按全局最新一行读取，
            而非按会话隔离），保留是为后续多用户隔离预留。
    返回值说明：
        dict：配置字典，含 city、month_budget、consume_mode、budget_type、category_budget（dict）、
            quota_mode、remind_pref；无配置时返回 {}。
    """
    # 走 config.get_latest 工具读取
    rows = await _invoke_fact("config.get_latest", {})
    if not rows:
        return {}
    row = rows[0]
    # query_sql 已统一返回字典列表，按列名取值
    cfg = {
        "city": row["city"],
        "month_budget": row["month_budget"],
        "consume_mode": row["consume_mode"],
        "budget_type": row["budget_type"]
    }
    # 品类预算（JSON 字符串）与额度评估方式（新列兼容旧库缺列场景）
    try:
        cfg["category_budget"] = json.loads(row["category_budget"]) if row.get("category_budget") else {}
    except (TypeError, ValueError):
        cfg["category_budget"] = {}
    cfg["quota_mode"] = row.get("quota_mode") or "auto"
    # 用户个性化提醒偏好（语气/语言/格式，随汇总提示词注入；兼容旧库缺列）
    cfg["remind_pref"] = row.get("remind_pref") or ""
    return cfg


# ===================== 消费习惯读取（供 finance 推算剩余额度/频次） =====================
# habit 写入已下沉 bill_agent：记账成功的副作用由记账方自身收尾，orchestrator 不再写 monthly_habit，
# 此处只保留只读入口（_load_recent_habits）。

async def _load_recent_habits(months_back: int = 3,
                              include_current: bool = True) -> list:
    """
    函数功能与逻辑描述：
        读取最近 N 个自然月的消费习惯聚合数据（走 habit.list 事实工具），供 finance 推算
        剩余额度/频次。以 date.today() 为基准向前逐月生成 "YYYY-MM" 列表（跨年由 m==0 分支处理）。
        异常兜底：任何异常均返回空列表，不打扰主流程。
    入参说明：
        months_back (int)：回溯的自然月数量，默认 3。
        include_current (bool)：是否含当月，默认 True。
            True（finance 上下文注入）：含当月（"累计至今"，供推算剩余天数内频次）。
            False（品类额度月均用）：跳过当月、只取已完成月份——否则月均会被严重低估
                （例：月初首笔 38 元 → 额度=38 → 次笔即误报超支）。
    返回值说明：
        list：习惯聚合行列表（元素含 category、amount_sum 等字段，语义见 habit.list 工具）；
            无数据或调用异常时返回空列表 []。
    """
    try:
        today = date.today()
        month_list = []
        y, m = today.year, today.month
        if not include_current:
            # 从上月开始（跨年回退，与下方循环内 m==0 处理一致）
            m -= 1
            if m == 0:
                m, y = 12, y - 1
        for _ in range(months_back):
            month_list.append(f"{y:04d}-{m:02d}")
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        # 走 habit.list 工具
        return await _invoke_fact("habit.list", {"months": month_list})
    except Exception:
        return []


# ===================== 自动预警链（纯记账场景，确定性计算，不进 LLM） =====================
async def _sum_month_bills(month: str) -> float:
    """
    函数功能与逻辑描述：
        查询指定自然月的账单金额总和（走 bill.sum_by_month 事实工具），供预算预警链做
        确定性计算；无数据与异常统一返回 0，保证调用方无需额外判空。
    入参说明：
        month (str)：自然月，格式 "YYYY-MM"。
    返回值说明：
        float：该月账单金额总和；无数据或调用异常时返回 0.0。
    """
    try:
        # 走 bill.sum_by_month 工具
        return float(await _invoke_fact("bill.sum_by_month", {"month": month}))
    except Exception:
        return 0.0


async def _calc_month_budget_status() -> dict | None:
    """
    函数功能与逻辑描述：
        计算当月预算状态（纯计算，不进 LLM）：读用户配置的月预算 → 查当月已花总额 →
        求已用比例 → 按 BUDGET_ALERT_THRESHOLDS 四档判定。其中 reward 档要求已用比例 > 0，
        即必须当月已有消费才提示犒劳，月首零消费不打扰；days_left 含今天。
        异常兜底：任何异常返回 None（等同于未配置预算，不打扰）。
    入参说明：
        无。
    返回值说明：
        dict | None：None = 未配置预算（不打扰）；否则返回 {level, month, month_spent,
            month_budget, used_ratio, days_left, consume_mode}，其中
            level 取 over=超支 / warn=接近红线 / reward=预算充裕(犒劳) / ok=正常区间(不打扰)。
    """
    try:
        cfg = await get_user_target_config("")
        month_budget = cfg.get("month_budget")
        if not month_budget:
            return None
        month_budget = float(month_budget)
        today = date.today()
        month = today.strftime("%Y-%m")
        month_spent = await _sum_month_bills(month)
        used_ratio = month_spent / month_budget
        # 当月剩余天数（含今天）
        days_left = calendar.monthrange(today.year, today.month)[1] - today.day + 1
        if used_ratio >= BUDGET_ALERT_THRESHOLDS["over_ratio"]:
            level = "over"
        elif used_ratio >= BUDGET_ALERT_THRESHOLDS["warn_ratio"]:
            level = "warn"
        elif 0 < used_ratio <= BUDGET_ALERT_THRESHOLDS["reward_ratio"]:
            level = "reward"  # 必须已消费（>0）才提示犒劳，月首无消费不打扰
        else:
            level = "ok"
        return {
            "level": level,
            "month": month,
            "month_spent": month_spent,
            "month_budget": month_budget,
            "used_ratio": used_ratio,
            "days_left": days_left,
            "consume_mode": cfg.get("consume_mode", ""),
        }
    except Exception:
        return None


def _calc_total_budget_fact(status: dict) -> dict | None:
    """
    函数功能与逻辑描述：
        总量预算维度确定性计算（只产出结构化事实，不进 LLM、不拼文案）：由状态算剩余额度
        remain=max(预算-已花, 0)，并仅在 warn 档且剩余天数 > 0 时给出每日建议额度 daily_limit。
        over_categories 在此只置空列表，超支品类归因由 _calc_alert_facts 回填。
    入参说明：
        status (dict)：_calc_month_budget_status() 的返回结果，须含 level / month_budget /
            month_spent / used_ratio / days_left；调用点已用 `if status` 守卫，
            因此本函数不处理 status 为 None（未配置预算）的情形。
    返回值说明：
        dict | None：level=="ok" → None（正常区间不打扰）；否则返回 {level, month_spent,
            month_budget, used_ratio, days_left, remain, daily_limit, over_categories}，
            daily_limit 仅 warn 档为数值、其余档位为 None。
    """
    level = status["level"]
    if level == "ok":
        return None
    budget = status["month_budget"]
    spent = status["month_spent"]
    remain = max(budget - spent, 0.0)
    days = status["days_left"]
    daily_limit = (remain / days) if (level == "warn" and days > 0) else None
    return {
        "level": level,
        "month_spent": spent,
        "month_budget": budget,
        "used_ratio": round(status["used_ratio"], 4),
        "days_left": days,
        "remain": round(remain, 2),
        "daily_limit": round(daily_limit, 2) if daily_limit is not None else None,
        "over_categories": [],
    }


# ===================== 品类维度评估（均价对比 + 品类进度 + 品类归因） =====================
# 城市物价估算兜底的默认月消费频次（均价 × 频次 = 月度合理额度）
DEFAULT_CATEGORY_FREQ = {"餐饮": 80, "交通": 44, "住宿": 4, "购物": 4, "娱乐": 2}


async def _query_city_avg_price(city: str, category: str) -> float:
    """
    函数功能与逻辑描述：
        查询当地同品类基准均价，经 `city_price.get_avg` 事实工具获取（不裸连数据库）。
        工具侧已实现三级兜底（本城精确命中 → 同级城市均价 → 0），因此本函数只做「调用 + 转型」，
        不再重复兜底逻辑。任何异常（工具不可用、返回不可转 float 等）一律吞掉并返回 0.0，
        由调用方按"无基准价"处理（L1 单笔对比层会因此置为 None）。
    入参说明：
        city (str)：城市名。
        category (str)：消费品类（餐饮/交通/住宿/购物/娱乐）。
    返回值说明：
        float：同品类基准均价（单位元）；无基准或调用异常时返回 0.0。
    """
    try:
        return float(await _invoke_fact("city_price.get_avg",
                                        {"city": city, "category": category}))
    except Exception:
        return 0.0


async def _sum_month_bills_by_category(month: str) -> dict:
    """
    函数功能与逻辑描述：
        查询指定自然月各品类累计金额（走 bill.sum_by_category 事实工具），供品类额度
        进度与超支归因使用；异常兜底空字典。
    入参说明：
        month (str)：自然月，格式 "YYYY-MM"。
    返回值说明：
        dict：{category: 累计金额}；无数据或调用异常时返回空字典 {}。
    """
    try:
        # 走 bill.sum_by_category 工具
        return await _invoke_fact("bill.sum_by_category", {"month": month}) or {}
    except Exception:
        return {}


async def _calc_category_quota(city: str, category: str) -> dict | None:
    """
    函数功能与逻辑描述：
        计算某品类的合理消费需求额度，三级优先级依次尝试：
        ① 用户显式设定（user_config.category_budget）→ source="user"；
        ② 历史习惯月均（仅取已完成月份，排除当月，避免月均被低估）→ source="habit"；
        ③ 城市物价估算（均价 × DEFAULT_CATEGORY_FREQ 默认月频次，品类未配置时频次取 4）→ source="city"。
        各级均受 quota_mode（auto/user/habit/city）开关控制；历史习惯无行时继续落到第 3 级。
        异常兜底：任何异常返回 None。
    入参说明：
        city (str)：城市名（第 3 级城市均价估算用）。
        category (str)：消费品类，如 "餐饮"、"交通"。
    返回值说明：
        dict | None：命中任一基准 → {quota(元, 金额级保留 2 位), source(user/habit/city)}；
            三级皆无或调用异常 → None。
    """
    try:
        cfg = await get_user_target_config("")
        quota_mode = cfg.get("quota_mode") or "auto"
        # 1. 用户显式设定（最高优先级）
        cb = cfg.get("category_budget") or {}
        if quota_mode in ("auto", "user") and cb.get(category):
            return {"quota": float(cb[category]), "source": "user"}
        # 2. 历史习惯月均（仅已完成月份：当月未完成，参与月均会低估额度 → 月初误报超支）
        if quota_mode in ("auto", "habit"):
            habits = await _load_recent_habits(3, include_current=False)
            rows = [h for h in habits if h.get("category") == category]
            if rows:
                total = sum(float(h.get("amount_sum") or 0) for h in rows)
                return {"quota": round(total / len(rows), 2), "source": "habit"}
            # rows 为空（无已完成月份，如新用户首月）→ 落到第 3 级 city 兜底
        # 3. 城市物价估算：均价 × 默认月消费频次
        if quota_mode in ("auto", "city"):
            avg_price = await _query_city_avg_price(city, category)
            if avg_price > 0:
                freq = DEFAULT_CATEGORY_FREQ.get(category, 4)
                return {"quota": round(avg_price * freq, 2), "source": "city"}
        return None
    except Exception:
        return None


def _render_progress_bar(ratio: float, width: int = 16) -> str:
    """
    函数功能与逻辑描述：
        渲染纯文本进度条：█ 表示已用、░ 表示剩余，总长度固定为 width 个字符。
        入参比例会先被裁剪到 [0, 1] 再取整，因此 ratio 超 1 时仅显示满格（不做溢出标记），
        负数按 0 处理。
    入参说明：
        ratio (float)：已用比例（0~1 语义）；越界值会被裁剪到 [0, 1]。
        width (int)：进度条总字符宽度，默认 16。
    返回值说明：
        str：由 █ 与 ░ 组成的定长（width）进度条字符串。
    """
    ratio = max(0.0, min(ratio, 1.0))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


async def _calc_alert_facts(status: dict | None, last_amount: float,
                            category: str = "", city: str = "") -> dict:
    """
    函数功能与逻辑描述：
        预警事实计算（确定性计算层）：只产出结构化事实数据，**不拼文案、不进 LLM**，
        是把「算事实」与「写话术」彻底分离的落点——LLM 只负责把已成事实组织成自然语言，
        从结构上排除模型编造数字的可能。三层事实各自独立、可缺省：
        - l1 单笔均价对比：本笔金额 vs 当地同品类均价（走 city_price.get_avg）→ 溢价率 + 四档；
          仅当 category 非空、last_amount > 0 且均价 > 0 时产出，否则保持 None。
        - l2 品类进度对比：当月该品类累计（bill.sum_by_category）vs 品类合理额度
          （用户设定 / 历史习惯 / 城市估算，由 _calc_category_quota 决定来源并回传 source 标注）；
        - l3 总量对比：当月总支出 vs 月总预算（四档事实 + 超支品类归因）。
        月份取 status.month，缺失时按当天所在月兜底。任一子层数据不足即为 None，
        collect_node 据此决定是否把该层事实交给 LLM 汇总（None 层不渲染、不解释）。
        所有下钻查询均走事实工具，本函数不直接访问数据库。
    入参说明：
        status (dict | None)：预警状态上下文（含 month 等）；None 时月份按当天兜底。
        last_amount (float)：本笔消费金额，用于 L1 对比；<= 0 时 L1 不产出。
        category (str)：消费品类，默认空串；为空时 L1、L2 均不产出。
        city (str)：城市名，默认空串（L1 的 city 字段会回落为"当地"）。
    返回值说明：
        dict：固定三键结构 {"l1": ..., "l2": ..., "l3": ...}，值为该层事实字典或 None。
            l1 含 category / amount / city / avg_price / premium_rate / level；
            l2 含 category / spent / quota / ratio / source / need_quota 等；
            l3 为总量对比事实（当月总支出与月预算对比及超支品类归因）。
    """
    facts = {"l1": None, "l2": None, "l3": None}
    month = (status or {}).get("month") or date.today().strftime("%Y-%m")

    # ---- L1 单笔均价对比：**顶层不做比价** ----
    #   比价的「品类归类 + 溢价率计算」**统一由 price_agent 承担**；顶层不做品类归类，
    #   自行算基准价会在"火锅/奶茶"等子类表述上失真。
    #   本层保留 `l1` 键（恒 None）以保持 alert_facts 结构不变，下游"字段缺失即如实文字说明"
    #   的逻辑照旧；last_amount / city 形参保留，供 L2/L3 与未来扩展使用
    #   （签名不变，避免连带改动调用方与既有单测的桩）。

    # ---- L2 品类进度对比（当月累计 vs 品类额度）----
    if category:
        quota_info = await _calc_category_quota(city, category)
        cat_spent = (await _sum_month_bills_by_category(month)).get(category, 0.0)
        if quota_info:
            quota = quota_info["quota"]
            ratio = cat_spent / quota if quota > 0 else 0.0
            facts["l2"] = {
                "category": category,
                "spent": round(cat_spent, 2),
                "quota": round(quota, 2),
                "ratio": round(ratio, 4),
                "source": quota_info["source"],
                "need_quota": False,
            }
        elif cat_spent > 0:
            # 无任何额度基准（user/habit/city 三级均未命中）但已有消费 → need_quota=True，
            # 由上层引导用户设置品类预算或选择额度评估方式
            facts["l2"] = {
                "category": category,
                "spent": round(cat_spent, 2),
                "quota": None,
                "ratio": None,
                "source": "none",
                "need_quota": True,
            }

    # ---- L3 总量对比（四档事实 + 超支品类归因）----
    if status:
        total_fact = _calc_total_budget_fact(status)
        if total_fact:
            # 超支归因：遍历当月各品类累计消费，凡超过该品类合理额度者收集为 over_items，
            # 再按超出比例从高到低排序，供上层点名主要超支品类
            try:
                cat_spent_map = await _sum_month_bills_by_category(month)
                over_items = []
                for cat, spent in cat_spent_map.items():
                    q = await _calc_category_quota(city, cat)
                    if q and spent > q["quota"]:
                        over_items.append({
                            "category": cat,
                            "spent": round(spent, 2),
                            "quota": round(q["quota"], 2),
                            "exceed_ratio": round((spent - q["quota"]) / q["quota"] if q["quota"] else 0, 4),
                        })
                if over_items:
                    over_items.sort(key=lambda x: x["exceed_ratio"], reverse=True)
                total_fact["over_categories"] = over_items
            except Exception:
                total_fact["over_categories"] = []
            facts["l3"] = total_fact

    return facts


# ===================== 表达层数字安全（防外部 LLM 数字幻觉） =====================
# 汇总提示词示例中的数字会被小模型当事实复用，故设"数字白名单闸门"：LLM 回复中
# 任何不在白名单内的数字（原文/确定性事实/渲染文本）一律判幻觉，回退本地确定性渲染。

# 白名单核心实现抽至 utils/number_whitelist.py 单一来源（finance_agent 防幻觉同源复用）
from utils.number_whitelist import (
    extract_numbers as _extract_numbers,
    numbers_inside_whitelist as _numbers_inside_whitelist,
)


def _whitelist_numbers(user_input: str, alert_facts: dict, text_blocks: list) -> set[float]:
    """
    函数功能与逻辑描述：
        构建表达层数字白名单（防外部 LLM 数字幻觉）：把「用户原文 + 确定性预警事实 JSON +
        确定性渲染文本块」三部分中出现的所有数字取并集，作为 LLM 汇总回复的唯一合法数字来源；
        数字抽取统一复用 utils/number_whitelist（与 finance_agent 防幻觉同源）。
    入参说明：
        user_input (str)：用户本轮原始输入。
        alert_facts (dict)：本轮确定性预警事实（l1/l2/l3）。
        text_blocks (list)：各任务确定性渲染出的文本素材列表。
    返回值说明：
        set[float]：白名单数字集合；三部分均无数字时返回空集合。
    """
    pool = set()
    for t in [user_input, json.dumps(alert_facts, ensure_ascii=False), *text_blocks]:
        pool |= _extract_numbers(t)
    return pool


# ===================== 节点1：任务规划（Plan） =====================
async def plan_node(state: OrchState):
    """
    函数功能与逻辑描述：
        任务规划节点主入口，按序执行：读会话历史 + 长期记忆检索（向量检索为同步 IO，丢线程执行）
        → 分层预算超预算截断 → 组装 system/user 提示词 → parse_json_output 调 LLM 生成规划 JSON
        （失败/缺字段自动重试）→ normalize_keys 清洗 key → 兼容 block_tips 复数键 →
        raw_segments 兜底回退完整原文 → repair_task_names 修正 name → validate_task_list 仅格式校验
        → 顶层准入拦截 → prune_tasks_by_heuristics 启发式裁剪补齐 → 标准化封装 task_plan
        → 静态表 ∪ LLM deps 动态依赖合并 + 环检测。
        顶层准入：pre_check_pass=false 时默认拦截不派发（回 block_tip，缺失则回退咨询引导），
        仅两种情形放行——记账指令明确（补 bill）与纯消费陈述（默认咨询补 price）。
        异常口径：解析失败与静态表成环均以 error_msg 走 collect_node 收口，不抛到图外；
        动态依赖成环则打 WARNING 后回退纯静态表继续执行（`01` R10 禁止静默失败）。
    入参说明：
        state (OrchState)：编排图状态，读取 user_input、session_id、agent_result_cache 等字段。
    返回值说明：
        Command：LangGraph 指令对象——
            成功：update 含 task_plan/current_task/all_task_results/error_msg/current_agent/
              dispatch_round，goto="dispatch_node"；
            失败或拦截：update 含 error_msg 或 final_reply、task_plan={}、dispatch_round=0，
              goto="collect_node"。
    """
    user_text = state["user_input"]
    session_id = state["session_id"]
    _log("plan_node 开始", f"user={user_text[:40]!r} session={session_id[:12]}")

    # 取会话历史用于组装规划提示词（完整内存由 dispatch_node 导出下发，此处无需重复导出）
    mem = get_session_memory(session_id)
    session_history = mem.get_history()

    # 长期记忆检索：跨月历史消费摘要注入规划提示词，与短期历史并列
    long_mem_text = ""
    try:
        # 向量检索是同步 IO（FAISS+jieba），丢线程避免阻塞事件循环
        long_mem_text = await asyncio.to_thread(search_history_consume, user_text)
    except Exception:
        long_mem_text = ""
    history_context = session_history
    _log("长记忆检索完成", f"long_mem={long_mem_text[:60]!r}")
    if long_mem_text and long_mem_text != "暂无历史消费记录":
        history_context = (session_history + "\n" + long_mem_text).strip()

    # 分层预算：历史/长期记忆超预算时按层截断（正常长度输入零改动；异常回退上方原始拼装）
    try:
        _ctx = _cm_orch.build_context(
            system="",
            history=session_history,
            long_memory=(long_mem_text if long_mem_text and long_mem_text != "暂无历史消费记录" else ""),
        )
        if _ctx.truncated:
            history_context = "\n".join(
                p for p in (_ctx.layer("history"), _ctx.layer("long_memory")) if p).strip()
    except Exception:
        pass

    # 组装system+user提示词
    sys_p, usr_p = build_task_plan_prompt(user_text, history_context, list(TASK_AGENT_MAP.keys()))
    try:
        schema_example = ('{"tasks":[{"name":"bill","raw_segments":["xxx"],"operate_sub_type":"add"}],'
                          '"pre_check_pass":true,"block_tip":"","scope_check":"in_scope",'
                          '"confidence":1.0,"missing":{},"clarify_options":[],"boundary_reason":""}')
        # parse_json_output：内部调用 LLM + 清洗提取 JSON，失败/缺字段会自动重试
        plan_data = await parse_json_output(
            mcp_client.call_llm_base,
            sys_p,
            usr_p,
            schema_example,
            agent_tag=ORCH_AGENT_ID,
            # 豁免这几个常漏字段：缺失时不触发 schema 重试，改由业务侧兜底
            # （operate_sub_type 填默认值、block_tip 兼容复数键、raw_segments 回退原文、
            #   name 由 repair_task_names 修复/validate_task_list 剔除），避免小模型漏字段
            #  导致整轮"规划失败"、使下方启发式兜底失去机会。
            ignore_missing_keys=["operate_sub_type", "block_tip", "raw_segments", "name",
                                 "scope_check", "confidence", "missing",
                                 "clarify_options", "boundary_reason"],
            max_self_retry=3
        )
        # 递归清洗所有key，消除空格污染
        plan_data = normalize_keys(plan_data)

        # 兼容模型输出的 block_tips 复数键：缺失 block_tip 时从中取值兜底
        if "block_tip" not in plan_data and "block_tips" in plan_data:
            tip = plan_data.pop("block_tips")
            plan_data["block_tip"] = tip[0] if isinstance(tip, list) and tip else (str(tip) if tip else "")

        raw_tasks = plan_data.get("tasks", [])

        # ===== raw_segments 兜底：模型漏输出或输出空数组时，回退为完整原文 =====
        for _t in raw_tasks:
            if isinstance(_t, dict):
                _segs = _t.get("raw_segments")
                if not (isinstance(_segs, list) and any(str(s).strip() for s in _segs)):
                    _t["raw_segments"] = [user_text]
                    _log("raw_segments兜底", f"task={_t.get('name')} 缺失/为空，回退完整原文")

        # ===== 修正小模型把用户原话/测试标记塞进 name 字段的错误，再走合法性校验 =====
        raw_tasks = repair_task_names(raw_tasks)

        pre_check_pass = bool(plan_data.get("pre_check_pass", False))
        block_tip = str(plan_data.get("block_tip", "")).strip()
        # ===== M16（D18）：解析 5 个新字段（缺省即走旧逻辑，向后兼容）=====
        # 缺省语义见 `09` §2.14：scope_check→in_scope / confidence→1.0 / missing→{} /
        #   clarify_options→[] / boundary_reason→""
        scope_check = str(plan_data.get("scope_check") or "in_scope").strip() or "in_scope"
        # ===== M17（D19）：fit_level 四档（缺省由 scope_check 推导，保证向后兼容）=====
        #   direct / inferable / need_user / out_of_scope —— 见 `设计/15` §3.1
        _fit_raw = str(plan_data.get("fit_level") or "").strip()
        if _fit_raw in ("direct", "inferable", "need_user", "out_of_scope"):
            fit_level = _fit_raw
        else:
            # 兼容：模型未输出 fit_level → 由 scope_check 推导（out_of_scope 同档，其余视为 direct）
            fit_level = "out_of_scope" if scope_check == "out_of_scope" else "direct"
        fit_reason = str(plan_data.get("fit_reason") or "")
        try:
            confidence = float(plan_data.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        llm_missing = plan_data.get("missing") if isinstance(plan_data.get("missing"), dict) else {}
        _clarify_raw = plan_data.get("clarify_options")
        clarify_options = ([str(o) for o in _clarify_raw if str(o).strip()]
                           if isinstance(_clarify_raw, list) else [])
        boundary_reason = str(plan_data.get("boundary_reason") or "")
        _log("LLM规划完成", f"tasks={[t.get('name') for t in raw_tasks]} pre_check={pre_check_pass} "
                            f"scope={scope_check} conf={confidence} missing={llm_missing}")

        # ============ 仅格式校验 ============
        valid, err_msg, fixed_tasks = validate_task_list(raw_tasks)
        if not valid:
            # 抛出异常 → parse_json_output捕获，自动重试LLM，并附带错误信息
            raise ValueError(f"任务规划字段校验失败：{err_msg}")

    except Exception as e:
        err_msg = f"任务规划解析失败：{str(e)}"
        _log("规划失败", err_msg)
        return Command(
            update={
                "error_msg": err_msg,
                "task_plan": {},
                "dispatch_round": 0
            },
            goto="collect_node"
        )

    # ========== M16（D18）+ M17（D19）：意图层分档决策（四档）==========
    _conservative_ok = False      # M17：保守规划依据校验结果（供下方 prune 判断复用）
    # 四档产出统一为 task_plan={} + final_reply → goto collect_node（**不进编排**：
    #   dispatch_node 不会被调用，这是"编排零改动"的结构保证，契约 `09` §2.14 红线 4）。
    # 决策表（`设计/15` §3.3）：
    #   ① fit_level=out_of_scope        → OOD 引导
    #   ② fit_level=inferable           → 保守规划（须过 _check_inferable_grounding 依据校验）
    #   ③ 缺参：写任务(bill) → 一律追问；只读任务 → 依据成立则放行、否则追问 ★M17 细分
    #   ④ confidence < 阈值             → 候选澄清
    # 硬判断优先：`_is_force_bill` 为明确记账指令时**始终放行**，不被软判断误伤（红线 3）。
    if not _is_force_bill(user_text):
        # ① 域外兜底（OOD）：非四基元能力范围 → Rich Fallback（能力清单 + 用法示例）
        if INTENT_OOD_ENABLED and (fit_level == "out_of_scope" or scope_check == "out_of_scope"):
            _log("OOD判定", f"out_of_scope reason={boundary_reason[:40]!r}")
            return Command(
                update={
                    "final_reply": _build_ood_fallback(clarify_options),
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )
        # ② 缺参检查（★M17 按任务类型分两段，**先于**保守规划校验：
        #    写任务缺参语义明确（已授权但缺参数）→ 必须追问；若被保守规划校验先拦会误降级为"澄清"）
        _missing_map = _check_contract(fixed_tasks, user_text, llm_missing)
        _write_missing = {k: v for k, v in _missing_map.items() if k == "bill"}
        _read_missing = {k: v for k, v in _missing_map.items() if k != "bill"}
        # ②-a 写任务缺参 → 一律追问（写操作永不推断，`设计/15` §3.4 第二段）
        if _write_missing:
            _log("写任务缺参追问", f"missing={_write_missing} key={_preplan_ask_key('missing')}")
            return Command(
                update={
                    "final_reply": _build_missing_ask(_write_missing, clarify_options),
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )
        # ③ 保守规划依据校验（仅 inferable 档；此时写任务缺参已排除）
        if INTENT_INFERABLE_ENABLED and fit_level == "inferable":
            _conservative_ok, _conservative_reason = _check_inferable_grounding(fixed_tasks, user_text)
            _log("保守规划校验", f"ok={_conservative_ok} reason={_conservative_reason}")
            # ★依据不成立 → 降级澄清（保守规划前提已被否定，不得放行瞎猜，`设计/15` §六 R-2）
            if not _conservative_ok:
                return Command(
                    update={
                        "final_reply": _build_clarify(clarify_options, user_text),
                        "task_plan": {},
                        "error_msg": None,
                        "dispatch_round": 0
                    },
                    goto="collect_node"
                )
        # ④ 只读任务缺参 → 保守规划依据成立则放行，否则追问
        if _read_missing and not _conservative_ok:
            _log("只读任务缺参追问", f"missing={_read_missing} conservative_ok={_conservative_ok}")
            return Command(
                update={
                    "final_reply": _build_missing_ask(_read_missing, clarify_options),
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )
        if _read_missing:
            _log("保守规划放行", f"只读缺参但依据成立：{_read_missing}")
        # ⑤ 低置信澄清
        if confidence < INTENT_CONFIDENCE_THRESHOLD:
            _log("低置信澄清", f"confidence={confidence} < {INTENT_CONFIDENCE_THRESHOLD}")
            return Command(
                update={
                    "final_reply": _build_clarify(clarify_options, user_text),
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )

    # ========== 顶层准入拦截分支 ==========
    # pre_check_pass=false 时默认拦截、不派发任务（回 block_tip，缺失则回退咨询引导）。
    # 两个放行例外（交给 prune 兜底补任务）：
    #   1) 记账指令明确（"记打车35元"）→ 放行补 bill；
    #   2) 无记账指令的纯消费陈述（"那打车花了80块呢"）→ 放行补 price（默认咨询）。
    # 真正拦截的只剩：既无记账指令、又非消费陈述的其它输入（收入/拒绝记账/闲聊等）。
    if not pre_check_pass and not _is_force_bill(user_text):
        # 无记账指令的消费陈述放行，交由 prune 补 price（默认咨询）
        if not _is_consume_statement(user_text):
            return Command(
                update={
                    "final_reply": block_tip or (
                        "收到。这句话我先按咨询理解，没有记账——记账需要你明确说「记…」"
                        "（如：记打车80元）；想判断某笔消费贵不贵/值不值，直接问「…贵不贵/正常吗」。"
                    ),
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )

    # ========== 启发式业务裁剪：纠正小模型的语义误判 ==========
    # ★M17：inferable 档的保守规划已通过依据校验（参数齐备 + 有话语依据），
    #   跳过启发式裁剪——prune 的关键词表可能误裁受信的只读规划（如"预算"场景的 stat）。
    if not (INTENT_INFERABLE_ENABLED and fit_level == "inferable" and _conservative_ok):
        fixed_tasks = prune_tasks_by_heuristics(user_text, fixed_tasks)
    _log("启发式裁剪后", f"tasks={[t.get('name') for t in fixed_tasks]}")

    # ========== 空任务拦截：收入/拒绝记账场景给确定性文案，不走通用兜底 ==========
    if not fixed_tasks:
        if _has_any_keyword(user_text, INCOME_KEYWORDS):
            return Command(
                update={
                    "final_reply": "收入类内容不予入账，本系统仅支持支出记账。",
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )
        if _has_any_keyword(user_text, REFUSE_BILL_KEYWORDS):
            return Command(
                update={
                    "final_reply": "好的，这笔不记账。",
                    "task_plan": {},
                    "error_msg": None,
                    "dispatch_round": 0
                },
                goto="collect_node"
            )

    # ========== 模型返回合法，开始标准化封装任务 ==========
    task_plan = {}
    task_names = set()
    for task in fixed_tasks:
        t_name = task.get("name")
        segs = list(task.get("raw_segments", []))
        sub_type = task.get("operate_sub_type", "")

        # ===== 金额上下文补全：bill/price/finance 片段缺金额但原文有金额时，追加完整原文 =====
        #（小模型常把金额放进 bill 片段，price/finance 片段只留"值吗"等无参数文本）
        if t_name in ("bill", "price", "finance") and segs:
            seg_text = " ".join(str(s) for s in segs if isinstance(s, str))
            if not re.search(r"\d+(\.\d+)?", seg_text) and re.search(r"\d+(\.\d+)?", user_text):
                segs.append(user_text)
                _log("金额上下文补全", f"task={t_name} 片段无金额，追加完整原文")
        # ===== M16（D18）参数落地校验：数值实体必须能回溯原文，剔除无依据数字 =====
        # （归一化"只重述对齐、不新增事实"——契约 `09` §2.14 红线 1）
        if segs:
            segs, _dropped = _verify_grounding(segs, user_text)
            if _dropped:
                _log("归一化落地校验", f"task={t_name} 剔除无依据数值={_dropped}")

        task_names.add(t_name)
        # 统一标准化任务结构
        task_plan[t_name] = {
            "status": "pending",       # pending待执行 / running执行中 / done完成
            "deps": [],                 # 当前任务依赖的前置任务列表
            "raw_segments": segs,       # 用户原始对话片段，传给子agent
            "operate_sub_type": sub_type,# 细分操作 add/edit/query/analyse
            "result": None,             # 保存任务执行完毕后的返回【存储结构化dict】
            "agent_id": TASK_AGENT_MAP[t_name]
        }

    # 任务依赖动态化：静态表 ∪ LLM 输出的 deps
    llm_deps_map = (_collect_llm_deps(fixed_tasks, task_names)
                    if DYNAMIC_DEPS_ENABLED else {})
    task_plan, dyn_added = _merge_deps(task_plan, llm_deps_map)
    if dyn_added:
        _log("动态依赖生效", f"{dyn_added}（静态表之外的补充顺序约束）")

    # 环检测：成环即丢弃 LLM deps、回退纯静态表（不终止执行）；成环只可能来自 LLM 补充
    if _has_cycle(task_plan):
        logger.warning(f"[M6] 动态依赖成环，回退纯静态表：{dyn_added}"
                       "（`01` R10 禁止静默失败）")
        task_plan, _ = _merge_deps(task_plan, {})
    # 兜底：静态表仍成环说明依赖表被改坏 → 终止执行
    if _has_cycle(task_plan):
        logger.error(f"[M6] 静态依赖表成环，终止执行："
                     f"{ {k: v['deps'] for k, v in task_plan.items()} }")
        return Command(
            update={
                "error_msg": "任务依赖存在死循环，终止执行",
                "task_plan": {},
                "dispatch_round": 0
            },
            goto="collect_node"
        )

    # 所有变更打包提交（禁止就地修改 state）；session_history 不跨节点传递
    update_payload = {
        "task_plan": task_plan,
        "current_task": None,
        "all_task_results": [],
        "error_msg": None,
        "current_agent": ORCH_AGENT_ID,
        "dispatch_round": 0
    }
    return Command(update=update_payload, goto="dispatch_node")


async def _build_task_data(task_name: str, task_plan: dict, state: OrchState,
                           mem_export: dict, cache: dict) -> dict:
    """
    函数功能与逻辑描述：
        按任务类型组装下发给子 agent 的 task_data（由原 dispatch_node 内联逻辑抽取而来）：
        finance → 额外带 target_control（预算目标配置）、habit_data（近 3 个月习惯聚合）、
          按 finance 原始输入检索一次的 user_docs（检索失败/无结果降级为 []），以及前置
          bill/stat/price 的结构化结果；stat → 强制用完整原文（小模型常把 stat 片段改写成臆想值，
          导致 stat_agent 无法理解意图）；price → 原样透传片段与细分操作；
        bill → 额外注入 city（取用户配置，缺失回落 DEFAULT_USER_CITY）与当前记账 month。
    入参说明：
        task_name (str)：任务名，取值 bill / stat / price / finance。
        task_plan (dict)：{任务名: 任务信息 dict}，读取其中的 raw_segments 与 operate_sub_type。
        state (OrchState)：编排图状态，读取 session_id、user_input。
        mem_export (dict)：会话内存快照（历史对话 + 草稿），随任务下发给 worker。
        cache (dict)：前置 agent 结构化结果缓存 {agent_type: struct}，供 finance 复用。
    返回值说明：
        dict：对应类型的 task_data，四类均含 raw_segments、operate_sub_type、session_memory，
            其余为各类型专属字段；task_name 不在四类内时返回空字典 {}。
    """
    if task_name == "finance":
        target_control_data = await get_user_target_config(state["session_id"])
        # 历史消费习惯（近 3 个月聚合），支撑"剩余预算能否撑到月底"推算
        habit_data = await _load_recent_habits(3)
        # 用户资料：按 finance 原始输入检索一次下发，finance 侧只做渲染；失败/无结果 → []
        user_docs_data: list = []
        try:
            from memory.user_docs import search_user_docs
            q = "\n".join(task_plan[task_name].get("raw_segments") or [])
            if not q:
                q = state.get("user_input", "") or ""
            if q.strip():
                user_docs_data = search_user_docs(q, top_k=USER_DOCS_TOP_K)
        except Exception:  # noqa: BLE001  检索失败降级为空，主流程不受影响
            user_docs_data = []
        return {
            "raw_segments": task_plan[task_name]["raw_segments"],
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
            "target_control": target_control_data,
            "bill_result": cache.get("bill_agent"),
            "stat_result": cache.get("stat_agent"),
            "price_result": cache.get("price_agent"),
            "habit_data": habit_data,
            "user_docs": user_docs_data
        }
    if task_name == "stat":
        # stat 强制用完整原文：小模型常把 stat 片段改写成臆想值，导致 stat_agent 无法理解意图
        return {
            "raw_segments": [state.get("user_input", "")],
            "user_input": state.get("user_input", ""),
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
        }
    if task_name == "price":
        # price 同样**强制使用完整原文**：模型切片常把"对比餐饮物价"这类子句单独切出来、
        #   **丢掉同句里的金额**（如"午饭32元记一下…，对比餐饮物价"），导致 price_agent
        #   拿不到消费金额、只能回"缺少相关消费数据，无法进行餐饮物价的对比分析"。
        #   price 侧本就会从原文自行提取品类与金额，无需精简片段，故与 stat 同口径下发完整输入
        #   （并附 user_input 便于 price_agent 定位消费描述）。
        return {
            "raw_segments": [state.get("user_input", "")],
            "user_input": state.get("user_input", ""),
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
        }
    if task_name == "bill":
        # 注入城市与记账月份：city 取用户配置（缺失用默认），month 用当前月
        cfg = await get_user_target_config(state["session_id"])
        return {
            # bill 也**强制使用完整原文**：模型切片常只截到**第一笔**（"记午饭30，记打车20"
            #   只下发过"午饭30"），后一笔根本没进抽取器 → 只能落一笔。而"金额上下文补全"
            #   仅在片段**无金额**时才追加完整原文，对这种"已有金额但只有一笔"的切片无效。
            #   bill 侧本就会逐笔抽取，无需精简片段，故与 stat / price 同口径下发完整输入
            #   （并附 user_input，便于子层定位）。
            "raw_segments": [state.get("user_input", "")],
            "user_input": state.get("user_input", ""),
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
            "city": (cfg.get("city") or DEFAULT_USER_CITY),
            "month": date.today().strftime("%Y-%m")
        }
    return {}


# ===================== 节点2：任务调度（Dispatch） =====================
async def dispatch_node(state: OrchState):
    """
    函数功能与逻辑描述：
        任务调度节点：筛出就绪任务（status=="pending" 且 deps 全部落在 done_tasks 内）后
        批量派发本波全部就绪任务（并行只发生在同波内互不依赖的任务之间）。
        两种收口：无就绪且任务已全部完成 → collect_node；无就绪但仍有 pending
        （依赖永远无法满足）→ 记 error_msg="任务依赖死锁，执行终止" 转 collect_node。
        派发细节：deepcopy task_plan 后逐任务生成 task_id（毫秒时间戳 + 6 位随机十六进制，
        写入 bill.task_id 便于对账）、置 status="running" 并写入 task_id；恢复链路的
        run_id/parent_span_id 随 task_data 下发（contextvars 不跨协程传播）；
        派发前经 startup.task_manager 建任务单（DB 是任务事实唯一来源），建单失败仅告警不阻断；
        最后经 a2a_bus 投递。
        批量开关：BATCH_DISPATCH_ENABLED=False 时回退单派发 ready[:1]（回滚与 A/B 对比用）。
    入参说明：
        state (OrchState)：编排图状态，读取 task_plan、session_id、agent_result_cache、dispatch_round。
    返回值说明：
        Command：派发分支 update 的键为 **task_plan**（值为本地 deepcopy 出来的 new_task_plan，
            已写回各就绪任务的 status="running" 与 task_id）与 current_task（仅兼容/日志用，
            批量等待一律以 status=="running" 集合为准），goto="wait_result_node"；
            收口分支 goto="collect_node"（死锁时附带 error_msg）。
    """
    task_plan = state["task_plan"]
    done_tasks = {k for k, v in task_plan.items() if v["status"] == "done"}

    # ★M16：本轮已被用户取消 → 不再派发任何新任务，直接收口进 collect（由 collect 决定不产出业务回复）。
    #   取消状态来自 wait_result_node 识别 worker 回传的取消结果（数据驱动，不查 DB，避免历史取消污染）。
    if state.get("cancelled"):
        _log("用户已取消", "跳过派发，直接进入汇总")
        return Command(update={}, goto="collect_node")

    # 就绪任务：状态 pending 且依赖都在 done_tasks 内
    ready = []
    for name, info in task_plan.items():
        if info["status"] == "pending" and set(info["deps"]).issubset(done_tasks):
            ready.append(name)

    # 没有就绪任务：所有任务全部执行完毕，进入汇总节点
    if not ready and len(done_tasks) == len(task_plan):
        return Command(update={}, goto="collect_node")

    # 存在依赖死锁：有pending任务，但依赖永远无法满足
    if not ready:
        return Command(
            update={"error_msg": "任务依赖死锁，执行终止"},
            goto="collect_node"
        )

    # 批量派发本波全部就绪任务（并行只发生在同波内互不依赖的任务之间）
    new_task_plan = copy.deepcopy(task_plan)
    # 读取完整会话内存，随任务下发给 worker（历史对话+草稿）——本波共用同一份快照
    mem = get_session_memory(state["session_id"])
    mem_export = mem.export_all()
    cache = copy.deepcopy(state.get("agent_result_cache", {}))

    # 回滚开关：BILLAGENT_BATCH_DISPATCH=0 退回单派发 ready[0]
    wave = ready if BATCH_DISPATCH_ENABLED else ready[:1]

    dispatched: list = []
    for current_task in wave:
        # task_id = 毫秒时间戳 + 6 位随机十六进制（可排序、可读、写进 bill.task_id 便于对账）
        task_id = f"t{int(time.time() * 1000)}{secrets.token_hex(3)}"
        target_agent = new_task_plan[current_task]["agent_id"]
        task_data = await _build_task_data(current_task, new_task_plan, state, mem_export, cache)
        # run_id + parent span 随 task_data 下发 → worker 恢复同一链路（contextvars 不跨协程传播）
        try:
            from utils.tracer import current_run_id, current_span_id
            _rid = current_run_id()
            _pid = current_span_id()
        except Exception:  # noqa: BLE001
            _rid, _pid = None, None
        if _rid:
            task_data = {**task_data, "run_id": _rid}
            if _pid:
                task_data["parent_span_id"] = _pid

        new_task_plan[current_task]["status"] = "running"
        new_task_plan[current_task]["task_id"] = task_id

        # 派发前建任务单（DB 是任务事实唯一来源）；建单失败只告警不阻断派发
        try:
            from startup.task_manager import get_task_manager  # noqa: PLC0415 —— 延迟导入避免 import 期循环

            get_task_manager().create_task(
                task_id=task_id,
                session_id=state["session_id"],
                agent_id=target_agent,
                payload=task_data,
            )
        except Exception as e:  # noqa: BLE001
            _log("M5 任务建单失败（降级：不阻断派发）", str(e))

        # ★M20（D21）：send_task 现返回 bool —— 队列满时被**拒绝**（False）。
        #   **不阻断**本轮：任务已在 DB 侧建单（status=submitted），交 wait_result 超时兜底
        #   （拒绝不改变任务最终状态——队列只是下游投影，`04` §3.1）。此处只记日志便于排障。
        if a2a_bus.send_task(
            target_agent=target_agent,
            session_id=state["session_id"],
            task_id=task_id,
            task_data=task_data
        ):
            dispatched.append(current_task)
            _log("派发任务", f"{current_task} -> {target_agent} task_id={task_id[:8]}")
        else:
            _log("派发被拒（队列满）",
                 f"{current_task} -> {target_agent} task_id={task_id[:8]}（交超时兜底）")

    _log("本波派发完成", f"波次={state.get('dispatch_round', 0) + 1} 并行任务={dispatched}")

    update_payload = {
        "task_plan": new_task_plan,
        # current_task 仅作兼容/日志（批量等待以 status=="running" 集合为准）
        "current_task": dispatched[-1] if dispatched else None,
    }
    return Command(update=update_payload, goto="wait_result_node")


# ===================== 节点3：等待子任务结果 =====================
# ★M19：**任务级失败**的可识别标记（`msg` 字段精确匹配）。
#   为什么必须用 `msg` 白名单而不是"`success is False` 就判失败"：
#   `success=False` 在系统里有**三类**语义完全不同的来源——
#     ① `track()` 代传的失败载荷（`msg="任务执行失败"`）→ **任务级失败，应终止本轮**；
#     ② `_wait_one` 的等待兜底载荷（`msg="任务执行异常"`）→ **任务级失败，应终止本轮**；
#     ③ 子 Agent 的**业务性拒绝**（追问 `need_more_info` / 收入不入账拦截等）→
#        `success=False` 但属**正常业务交互**，`collect_node` 依此追加追问或如实告知，
#        **绝不能终止**——否则"信息不全时追问"这一核心交互会被打成"执行失败"。
#   故判据只认前两者（由平台层产生），业务拒绝一律放行。
_TASK_FAILURE_MSGS = ("任务执行失败", "任务执行异常")


async def _abort_inflight(session_id: str) -> dict:
    """
    函数功能与逻辑描述：
        M19（D20）：请求级终止时中断本会话**仍在途**的任务（`submitted` / `running`）。
        直接复用 `TaskManager.cancel_session`——它已实现正确的两步时序（**先落 `cancelled`
        终态、再 cancel 内层 runner Task**），这正是本动作成立的前提：若不先落库，`track()`
        捕获 `CancelledError` 后会查 DB 发现"非用户取消"，误走 `interrupted` 分支
        （`mark_interrupted` + 继续抛），导致任务在下次启动时被重投——而"超时 / 失败终止"
        的任务**不应被重投**（与 M16 用户取消同为终态语义）。
        边界：只影响 `submitted` / `running`，已 `completed` 的任务**不受影响**（部分成功
        必须保留，由 `collect_node` 在回执中如实告知用户）；本函数吞掉一切异常——中断失败
        不得阻断收口链路。
    入参说明：
        session_id (str)：要终止的会话标识。
    返回值说明：
        dict：`cancel_session` 的结果 {"cancelled": [...], "interrupted": [...]}；
            异常时返回空字典（已吞掉，不向上抛）。
    """
    try:
        from startup.task_manager import get_task_manager  # noqa: PLC0415 —— 延迟导入避免循环
        return get_task_manager().cancel_session(session_id)
    except Exception as e:  # noqa: BLE001 —— 中断失败只告警，不阻断收口
        logger.warning(f"[M19] 在途任务中断失败（不影响收口）: {e}")
        return {}


async def wait_result_node(state: OrchState):
    """
    函数功能与逻辑描述：
        批量等待本波全部 running 任务的结果，四条口径：
        ① 并发等待：asyncio.gather 并发收集，整波耗时 = max(各任务)，每个任务各自 240s 超时
           （对齐 MODEL_TIMEOUT）；
        ② 结果乱序无害：按任务名归位写入 task_plan 与 agent_result_cache，后续按 deps 取用，
           不依赖返回顺序；
        ③ 结果兜底：等待异常与旧版非结构化文本均在 _wait_one 内转成标准失败结构，不向上抛；
        ④ 调度轮次按波次 +1，超过 MAX_DISPATCH_ROUND 即终止执行。
        无任何 running 任务时打 ERROR 直接终止（不可回 dispatch，否则 dispatch↔wait 死循环）。
    入参说明：
        state (OrchState)：编排图状态，读取 task_plan、agent_result_cache、dispatch_round。
    返回值说明：
        Command：正常分支 update 含 task_plan、current_sub_agent_struct（本波最后一个结果）、
            agent_result_cache、dispatch_round，goto="dispatch_node"；
            无 running 任务或波次超限分支带 error_msg，goto="collect_node"。
    """
    task_plan = copy.deepcopy(state["task_plan"])
    # 本波在跑的任务（dispatch_node 已标为 running 并写入 task_id）
    running = {name: info["task_id"] for name, info in task_plan.items()
               if info.get("status") == "running" and info.get("task_id")}
    if not running:
        # 兜底：无任何 running 任务时不能回 dispatch，否则 dispatch↔wait 死循环
        logger.error("[M6] wait_result_node 未找到 running 任务（状态异常），终止本轮")
        return Command(
            update={"error_msg": "未找到执行中的任务，执行终止"},
            goto="collect_node"
        )

    _log("等待本波结果", f"任务={list(running)} 并发等待（各自 {MODEL_TIMEOUT}s 超时）...")

    async def _wait_one(name: str, tid: str) -> tuple:
        """
        函数功能与逻辑描述：
            等待单个任务结果并解析：await a2a_bus.wait_result(tid, timeout=MODEL_TIMEOUT)
            取回原始字符串，再 json.loads 解析为结构化 dict。
            两处兜底：等待抛异常 → 组装 success=False + error=str(e) 的标准失败结构；
            解析抛 JSONDecodeError → 组装 success=None + error=原始文本 的"旧版非结构化文本"结构。
            两种兜底均不向上抛，保证整波 gather 不被单个任务拖垮。
        入参说明：
            name (str)：任务名（bill/stat/price/finance），用于回填 agent_type。
            tid (str)：本任务的 task_id，用于从消息总线取结果。
        返回值说明：
            tuple：二元组 (name, 结构化结果 dict)，dict 含 success / agent_type / msg / data / error。
        """
        try:
            # 等待 worker 返回结果（★M22：引用 MODEL_TIMEOUT 常量，单一来源）
            raw_result_str = await a2a_bus.wait_result(tid, timeout=MODEL_TIMEOUT)
            _log("收到子任务结果", f"{name} 返回前{len(raw_result_str)}字符")
        except Exception as e:
            raw_result_str = json.dumps({
                "success": False,
                "agent_type": TASK_AGENT_MAP.get(name, "unknown"),
                "msg": "任务执行异常",
                "data": None,
                "error": str(e)
            }, ensure_ascii=False)

        # JSON解析，向下兼容兜底旧文本格式
        try:
            return name, json.loads(raw_result_str)
        except json.JSONDecodeError:
            return name, {
                "success": None,
                "agent_type": TASK_AGENT_MAP.get(name, "unknown"),
                "msg": "旧版非结构化文本",
                "data": None,
                "error": raw_result_str
            }

    # 并发收集：整波耗时 = max(各任务)
    results = await asyncio.gather(*(_wait_one(n, t) for n, t in running.items()))

    # 写入全局缓存（供后续finance agent调用）
    cache = copy.deepcopy(state.get("agent_result_cache", {}))
    last_payload = None
    cancelled_now = bool(state.get("cancelled"))     # ★M16：本轮是否已被取消
    abort_reason = state.get("abort_reason")         # ★M19：终止原因（None/user_cancel/task_timeout/task_error）
    for name, struct_payload in results:
        task_plan[name]["status"] = "done"
        task_plan[name]["result"] = struct_payload
        cache[struct_payload.get("agent_type", "unknown")] = struct_payload
        last_payload = struct_payload
        # ★M16：识别 worker 回传的取消结果（track 在用户取消时代传，格式与常规结果同构）
        err = str(struct_payload.get("error") or "")
        if err == "用户取消":
            cancelled_now, abort_reason = True, "user_cancel"
        # ★M19：识别任务级失败（超时 / 执行异常）→ 本轮终止（不再派发下游波次）。
        #   为什么必须终止：① 失败任务已被标 `done`，下游会误判"依赖已满足"而照常派发；
        #   ② 失败结果会溜进 collect 的 `struct_success_list` 被当作汇总素材 → 产出
        #   "看似正常但无数据"的回复（静默降级，比明确报错更危险，见 `设计/17` D20）。
        #   ★判据用 `_TASK_FAILURE_MSGS` 白名单（只认平台层产生的失败载荷），
        #   从而**放行业务的拒绝**（追问 `need_more_info` / 收入不入账拦截）——它们的
        #   `success` 同样是 False，但属正常交互，终止会直接破坏"信息不全时追问"。
        #   ★首个失败决定原因（abort_reason is None 才赋值），避免后续失败覆盖前者。
        elif (ABORT_ON_TASK_FAILURE and struct_payload.get("success") is False
              and str(struct_payload.get("msg") or "") in _TASK_FAILURE_MSGS
              and abort_reason is None):
            cancelled_now = True
            abort_reason = "task_timeout" if "超时" in err else "task_error"

        # habit 沉淀已下沉 bill_agent，编排器不再写 monthly_habit

    # ★M16/M19：本轮终止（用户取消 / 任务级失败）→ 不再进入下一波派发，直接收口；
    #   把 cancelled + abort_reason 写回 state，供 collect_node 决定"不产出业务回复"与回执文案。
    if cancelled_now:
        reason = abort_reason or "user_cancel"
        # ★M19：任务级失败时中断本会话**仍在跑的 runner**。要收拾的正是"wait_result 已超时
        #   放弃、但 worker 侧仍在跑"的任务——不中断的话，它们会继续占住该 Agent 的
        #   单协程消费循环（worker 是串行消费）。用户取消路径已由 WebUI 的 cancel_session
        #   处理过，故此处不重复调用。
        if reason != "user_cancel":
            await _abort_inflight(state["session_id"])
        _log("本轮终止", f"reason={reason} → 终止后续波次派发")
        return Command(
            update={
                "task_plan": task_plan,
                "current_sub_agent_struct": last_payload,
                "agent_result_cache": cache,
                "cancelled": True,
                "abort_reason": reason,
            },
            goto="collect_node"
        )

    # 调度轮次自增 + 防无限循环：按波次计数
    new_round = state["dispatch_round"] + 1
    if new_round > MAX_DISPATCH_ROUND:
        return Command(
            update={
                "error_msg": f"调度波次超过{MAX_DISPATCH_ROUND}上限，终止执行",
                "task_plan": task_plan,
                "agent_result_cache": cache
            },
            goto="collect_node"
        )

    return Command(
        update={
            "task_plan": task_plan,
            "current_sub_agent_struct": last_payload,
            "agent_result_cache": cache,
            "dispatch_round": new_round
        },
        goto="dispatch_node"
    )

# ===================== 长期记忆沉淀：stat 聚合结果 → 月度消费摘要 =====================
def _build_monthly_summary(struct: dict) -> Optional[str]:
    """
    函数功能与逻辑描述：
        把 stat_agent 的成功统计结果浓缩为一行摘要文本，供长期记忆沉淀（跨月消费习惯检索）。
        仅保留聚合信息：时间范围 / 总支出 / 分类明细 / 笔数 / 单笔最大；明细类查询不沉淀。
        time_range 缺失、或最终有效信息不足 2 段时返回 None，由调用方跳过写入。
    入参说明：
        struct (dict)：stat_agent 返回的结构化结果；读取 data.time_range、data.total_amount、
            data.category_summary、data.total_count、data.max_record。
    返回值说明：
        Optional[str]：形如 "【起~止 消费摘要】; 总支出: x元; 分类: ...; 共n笔; 单笔最大: x元(类目)"
            的摘要文本；无有效信息时返回 None。
    """
    data = struct.get("data") or {}
    time_range = data.get("time_range") or []
    if not time_range:
        return None
    parts = [f"【{time_range[0]}~{time_range[1]} 消费摘要】"]
    if data.get("total_amount") is not None:
        parts.append(f"总支出: {data['total_amount']}元")
    cs = data.get("category_summary")
    if cs:
        cat_str = ", ".join(f"{k}: {v}元" for k, v in cs.items())
        parts.append(f"分类: {cat_str}")
    if data.get("total_count") is not None:
        parts.append(f"共{data['total_count']}笔")
    if data.get("max_record"):
        parts.append(f"单笔最大: {data['max_record'].get('amount')}元({data['max_record'].get('category')})")
    if len(parts) < 2:
        return None
    return "; ".join(parts)


def _render_local_final(text_blocks: list, alert_facts: dict, remind_pref: str) -> str:
    """
    函数功能与逻辑描述：
        本地模式的确定性汇总渲染：先把各任务结果文本块按序号拼接，再依 l1/l2/l3 预警事实
        追加提醒行（单笔溢价对比、品类额度进度、总量预算与超支品类归因、每日建议额度），
        最后附个性化提醒偏好；全文数字均取自确定性计算层，杜绝小模型数字幻觉。
        外部模型可用时仍走 LLM 汇总（表达更自然），本函数仅作本地模式与 LLM 失败的兜底。
    入参说明：
        text_blocks (list)：各任务确定性渲染出的文本素材列表，按序号拼接进最终回复。
        alert_facts (dict)：预警事实 {l1, l2, l3}；空字典或缺失的层自动跳过对应提醒。
        remind_pref (str)：用户个性化提醒偏好（语气/语言/格式）；为空时不追加偏好行。
    返回值说明：
        str：以 "\n" 拼接的最终回复文本。
    """
    parts = []
    for idx, block in enumerate(text_blocks, 1):
        parts.append(f"{idx}. {block}")
    alert_lines = []
    l1 = alert_facts.get("l1") or {}
    if l1.get("amount") is not None:
        alert_lines.append(
            f"提醒：这笔{l1['category']}消费 {l1['amount']} 元，比{l1['city']}同品类均价 "
            f"{l1['avg_price']} 元高约 {l1['premium_rate']}%，{l1['level']}。"
        )
    l2 = alert_facts.get("l2") or {}
    if l2.get("spent") is not None:
        if l2.get("quota"):
            ratio_pct = int(round((l2.get("ratio") or 0) * 100))
            alert_lines.append(
                f"{l2['category']}本月累计消费 {l2['spent']} 元，已达合理额度 {l2['quota']} 元的 {ratio_pct}%。"
            )
        elif l2.get("need_quota"):
            alert_lines.append(
                f"{l2['category']}本月已累计消费 {l2['spent']} 元，建议设置品类预算以便更好地控制消费。"
            )
    l3 = alert_facts.get("l3") or {}
    if l3.get("month_spent") is not None:
        level_cn = {"over": "已超支", "warn": "接近预算红线", "reward": "预算还很充裕"}.get(l3.get("level"), "预算使用正常")
        used_pct = int(round((l3.get("used_ratio") or 0) * 100))
        alert_lines.append(
            f"本月总支出 {l3['month_spent']} 元，预算 {l3['month_budget']} 元（{level_cn}，"
            f"已使用 {used_pct}%，剩余 {l3.get('remain', 0)} 元，本月还剩 {l3.get('days_left', 0)} 天）。"
        )
        if l3.get("daily_limit") is not None:
            alert_lines.append(f"建议未来每天消费控制在 {l3['daily_limit']} 元以内。")
        for oc in l3.get("over_categories") or []:
            alert_lines.append(
                f"其中「{oc['category']}」已超合理额度 {oc['quota']} 元，超出约 {int(round((oc.get('exceed_ratio') or 0) * 100))}%。"
            )
    if alert_lines:
        parts.append("\n".join(alert_lines))
    if remind_pref:
        parts.append(f"（已按你的偏好：{remind_pref}）")
    return "\n".join(parts)


def _ask_round_key(struct: dict, task_plan: dict) -> str:
    """
    函数功能与逻辑描述：
        生成追问计数的会话级 key：先按结果的 agent_type 在 task_plan 中反查对应任务，
        bill 类按草稿类型细分为 bill_add / bill_edit / bill_delete，其余 agent 直接用 agent_id
        （与清草稿用的 key 保持一致，确保"计数重置"与"清草稿"命中同一槽位）。
    入参说明：
        struct (dict)：子 agent 结构化结果，取其 agent_type 作为定位依据。
        task_plan (dict)：{任务名: 任务信息 dict}，用于按 agent_id 反查 operate_sub_type。
    返回值说明：
        str：追问计数 key；无法定位到任务时兜底返回 "bill_add"（与 bill_agent 默认子类型一致）。
    """
    agent_id = struct.get("agent_type") or ""
    for info in task_plan.values():
        if info.get("agent_id") == agent_id:
            if agent_id == "bill_agent":
                return f"bill_{info.get('operate_sub_type') or 'add'}"
            return agent_id
    return "bill_add"      # 兜底：无法定位任务时按新增记账计（与 bill_agent 默认子类型一致）


# ===================== 节点4：结果汇总（Collect） =====================
async def collect_node(state: OrchState):
    """
    函数功能与逻辑描述：
        结果汇总节点（链路终点）：所有任务执行完成或流程终止后统一产出最终回复。分支口径——
        ★M16/M19：本轮若已终止（用户取消 / 任务级失败）→ **不产出业务回复**，按 `abort_reason`
        出对应回执（失败时**如实附上"已完成部分"**）后直接 END；
        ① state.error_msg 存在 → 直接输出 `执行终止：{错误}`；
        ② 无 done 结果 → 优先用 plan_node 下发的确定性拦截文案（final_reply），否则回退通用引导语；
        ③ 有结果 → 逐个分流：need_more_info 按会话级 key 计数，达 ASK_ROUND_LIMIT 即清草稿放弃（give_up），
        其余计入成功结果（计数清零，stat 成功结果经 _build_monthly_summary 沉淀长期记忆）；
        随后把成功结果确定性渲染为素材，纯记账单结果时额外计算 l1/l2/l3 预警事实；
        EXTERNAL_LLM_ENABLED 时按汇总提示词生成回复并过数字白名单闸门（出现白名单外数字 →
        修正重试一次 → 仍编造则回退本地确定性渲染），否则直接 _render_local_final；
        有成功结果时追问以附加形式追加。副作用：最终回复写入会话历史，并清空 agent_result_cache。
    入参说明：
        state (OrchState)：编排图状态，读取 session_id、error_msg、task_plan、final_reply、user_input。
    返回值说明：
        Command：update 含 final_reply 与 agent_result_cache={}，goto=END（本轮对话结束）。
    """
    session_id = state["session_id"]
    mem = get_session_memory(session_id)
    _log("collect_node 开始汇总", f"session={session_id[:12]}")

    # ★M16/M19：本轮已终止 → **不产出任何业务回复**，按 `abort_reason` 出对应回执。
    #   M16（用户取消）：用户已明确要求停止，若此时再回一条"已为您记账…"，等于系统无视了
    #     停止指令（且会误导用户以为操作已完成）→ 只给取消回执，跳过全部汇总逻辑
    #     （含 LLM 汇总、白名单闸门、追问追加）。
    #   M19（任务级失败）：超时 / 异常终止。★必须如实说明**已完成的部分**——记账场景下失败前
    #     可能已落库，只说"执行失败"会让用户以为什么都没发生（认知错位）。这与 M16 遇到的
    #     "账单已写、但回执说已停止"是同一类问题（`设计/17` §六 R-3）。
    if state.get("cancelled"):
        reason = state.get("abort_reason") or "user_cancel"
        if reason == "user_cancel":
            _log("用户已取消", "跳过业务回复生成，仅回执取消提示")
            reply = "已按您的要求停止本次请求。"
        else:
            head = ("本次请求执行超时，已终止（未完成的后续任务不再执行）。"
                    if reason == "task_timeout" else
                    "本次请求执行失败，已终止。")
            done_blocks = []
            for info in (state.get("task_plan") or {}).values():
                payload = info.get("result") if isinstance(info, dict) else None
                if isinstance(payload, dict) and payload.get("success") is True:
                    try:
                        text = build_user_display_text(payload)
                    except Exception:  # noqa: BLE001 —— 渲染异常不得阻断终止回执（否则用户拿不到任何反馈）
                        text = ""
                    if text:
                        done_blocks.append(text)
            _log("任务级失败中止", f"reason={reason} 已完成 {len(done_blocks)} 项")
            reply = head + (f"\n已完成部分：{'；'.join(done_blocks)}" if done_blocks else "")
        mem.add_msg("agent", reply)
        return Command(
            update={
                "final_reply": reply,
                "agent_result_cache": {},
                "cancelled": True,
                "abort_reason": reason,
            },
            goto=END
        )

    # 如果流程出现顶层错误，直接输出错误信息
    if state.get("error_msg"):
        final = f"执行终止：{state['error_msg']}"
    else:
        task_plan = state["task_plan"]
        all_struct_results = [v["result"] for v in task_plan.values() if v["status"] == "done"]

        if not all_struct_results:
            # 优先用 plan_node 下发的确定性拦截文案（收入不入账/准入拦截），无则回退通用兜底
            final = state.get("final_reply") or (
                "没听懂这条的需求。记账请明确说「记…」（如：记打车80元）；想查开销、"
                "对比价格或分析某笔消费是否合理，直接描述即可。"
            )
        else:
            prompt_list = []    # 需要向用户追问的内容
            struct_success_list = []  # 正常结构化结果

            give_up = False      # 追问超 ASK_ROUND_LIMIT 轮 → 放弃
            for struct in all_struct_results:
                # 安全判断：防止struct字段缺失崩溃
                if struct.get("success") is False and struct.get("error") == "need_more_info":
                    # 追问计数必须会话级（graph state 每次 ainvoke 会重建；dispatch_round 是
                    # 调度轮次、每次新输入重置 0，不能当追问轮次用）
                    ask_key = _ask_round_key(struct, task_plan)
                    if mem.incr_ask_round(ask_key) >= ASK_ROUND_LIMIT:
                        # 放弃即清草稿，防止半成品账单污染下一次记账
                        mem.clear_draft(ask_key)
                        mem.reset_ask_round(ask_key)
                        give_up = True
                        _log("追问超限放弃", f"key={ask_key} 已达 {ASK_ROUND_LIMIT} 轮")
                        break
                    prompt_text = struct["data"]["prompt"]
                    prompt_list.append(prompt_text)
                else:
                    struct_success_list.append(struct)
                    # 非追问结果（成功 / 业务性拒绝）→ 该 key 计数清零，下次记账重新计数
                    mem.reset_ask_round(_ask_round_key(struct, task_plan))
                    # 长期记忆沉淀：stat 成功的聚合结果 → 月度消费摘要（同步 IO 丢线程）
                    if struct.get("success") and struct.get("agent_type") == "stat_agent":
                        summary = _build_monthly_summary(struct)
                        if summary:
                            try:
                                await asyncio.to_thread(save_consume_memory, summary)
                            except Exception:
                                # 长期记忆写入失败不影响主流程回复
                                pass

            # 汇总策略：成功结果确定性渲染为素材，最终回复由外部 LLM 按汇总提示词生成
            if give_up:
                # 超限放弃：固定提示语，不再追加追问
                final = ASK_GIVEUP_TIP
            elif struct_success_list:
                # 确定性渲染各任务结果为文本素材（保证数字准确，不直接拼给用户）
                text_blocks = [build_user_display_text(s) for s in struct_success_list]

                # 预警事实（确定性计算层，不进 LLM）：纯记账成功且无追问时计算 L1/L2/L3
                alert_facts = {}
                remind_pref = ""
                solo = struct_success_list[0]
                if (
                    len(struct_success_list) == 1
                    and not prompt_list
                    and solo.get("success") is True
                    and solo.get("agent_type") == "bill_agent"
                ):
                    try:
                        status = await _calc_month_budget_status()
                        data = solo.get("data") or {}
                        last_amount = float(data.get("amount") or 0)
                        category = str(data.get("category") or "")
                        # 城市取用户配置，缺省用全局默认（与 price_agent 兜底一致）
                        cfg = await get_user_target_config("")
                        city = cfg.get("city") or DEFAULT_USER_CITY
                        remind_pref = str(cfg.get("remind_pref") or "")
                        alert_facts = await _calc_alert_facts(status, last_amount, category, city)
                    except Exception:
                        alert_facts = {}

                # 表达层：外部模型可用时按汇总提示词生成最终回复；本地模式用确定性渲染防幻觉
                if EXTERNAL_LLM_ENABLED:
                    # 任务结果层预算：多任务结果超长时保序截尾（异常回退原始块）
                    try:
                        if sum(estimate_tokens(b) for b in text_blocks) > _cm_orch.budget_for("task_results"):
                            text_blocks, _ = _cm_orch.budget_blocks(
                                [(b, False) for b in text_blocks])
                    except Exception:
                        pass
                    sys_p = load_summarize_system()
                    usr_p = render_summarize_user(
                        user_input=state.get("user_input", ""),
                        task_results_json=json.dumps(text_blocks, ensure_ascii=False),
                        alert_facts_json=json.dumps(alert_facts, ensure_ascii=False),
                        remind_pref=remind_pref,
                    )
                    whitelist = _whitelist_numbers(
                        state.get("user_input", ""), alert_facts, text_blocks
                    )
                    try:
                        final = str(await mcp_client.call_llm_base(sys_p, usr_p, agent_tag=ORCH_AGENT_ID) or "").strip()
                        if not final:
                            raise ValueError("汇总模型返回空结果")
                        # 数字白名单闸门：LLM 回复出现白名单外数字（幻觉）→ 修正重试一次，
                        # 重试仍编造则回退本地确定性渲染
                        if whitelist and not _numbers_inside_whitelist(final, whitelist):
                            hallucinated = sorted(
                                {v for v in _extract_numbers(final)
                                 if not any(abs(v - w) < 0.5 for w in whitelist)}
                            )
                            _log("collect_node 数字幻觉拦截",
                                 f"LLM编造数字{hallucinated} → 修正重试1次")
                            fix_p = (
                                f"{usr_p}\n\n"
                                "【数字修正要求】你上一版回复中出现了以下输入中不存在的数字："
                                f"{hallucinated}。这些是编造的，必须全部删除或替换为任务执行结果/预警事实中"
                                "真实提供的数字。除任务结果与预警事实中明确给出的数字外，禁止出现任何其他"
                                "数字（不得假设、推算、估算）。请基于同一批任务结果与预警事实，重写一版"
                                "自然、简洁的回复。"
                            )
                            final = str(await mcp_client.call_llm_base(sys_p, fix_p, agent_tag=ORCH_AGENT_ID) or "").strip()
                            if not final:
                                raise ValueError("汇总模型修正重试返回空结果")
                            if whitelist and not _numbers_inside_whitelist(final, whitelist):
                                raise ValueError(
                                    "修正重试仍包含白名单外数字，回退本地确定性渲染"
                                )
                    except Exception as e:
                        _log("collect_node LLM汇总失败", str(e))
                        final = _render_local_final(text_blocks, alert_facts, remind_pref)
                else:
                    final = _render_local_final(text_blocks, alert_facts, remind_pref)
                # 有成功结果时，追问以补充形式附加（如"另外，请提供…"）
                if prompt_list:
                    final = final + "\n\n" + prompt_list[0]
            elif prompt_list:
                # 全部任务都在追问、无成功结果：直接输出追问
                final = prompt_list[0]

    # 将最终回复写入会话历史
    mem.add_msg("agent", final)

    # 本轮对话结束清空缓存，降低内存占用
    return Command(
        update={
            "final_reply": final,
            "agent_result_cache": {}
        },
        goto=END
    )


# 对外导出所有节点，供LangGraph构建图使用
__all__ = ["plan_node", "dispatch_node", "wait_result_node", "collect_node", "ORCH_AGENT_ID"]