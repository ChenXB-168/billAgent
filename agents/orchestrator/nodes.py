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
                           DEFAULT_USER_CITY, EXTERNAL_LLM_ENABLED, USER_DOCS_TOP_K)
from coreModules.price_compare import calc_premium_evaluate
# M14（D17）：ContextManager 窗口接管——orchestrator_agent 默认走外部（EXTERNAL_LLM_AGENTS），
#   分层预算按 Router 解析的实际后端窗口（Capabilities.context_window）换算，而非写死 4096
#   （`08` §8.3 铁律3；ContextManager docstring 预留的 M14 接管点）
from agents.context_manager import ContextManager, estimate_tokens
from modelService.router import context_window_for
from .prompts.prompt_loader import load_system, render_user, load_summarize_system, render_summarize_user


def _log(stage: str, detail: str = ""):
    """编排链路阶段日志（真实业务路径，带时间戳，实时刷新）"""
    print(f"[ORCH] {time.strftime('%H:%M:%S')} {stage} {detail}", flush=True)

# 当前调度Agent唯一标识
ORCH_AGENT_ID = "orchestrator_agent"

# M14（D17）：带路由窗口的 ContextManager 实例（模块级建一次；budget 层比例预算同共享实例，
#   仅窗口刻度按实际后端换算——外部 8k+ / 本地 4k，见 `10` §5.6.3.1 分层预算表）
_cm_orch = ContextManager(context_window=context_window_for(ORCH_AGENT_ID))


async def _invoke_fact(name: str, args: dict):
    """M9（D15）：编排器取数统一走网关事实工具（`fact:read`），替代裸连 `utils.common.db`。

    引擎统一鉴权 / 审计 / 超时 / 打点自动生效（读操作后置审计全量，满足 M9 验收
    「审计出现编排器只读记录」）；失败抛原生异常（`ToolResult.unwrap` 类型映射），
    由调用点既有的 `try/except` 静默兜底——与原裸连行为的异常语义一致。
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

# 任务名称 -> 默认细分操作（1.8B小模型经常漏输出 operate_sub_type，规划层兜底填充）
TASK_SUB_TYPE_DEFAULTS = {
    "bill": "add",
    "stat": "query",
    "price": "query",
    "finance": "analyse"
}

# 任务依赖定义：key=任务名，value=该任务需要等待哪些前置任务执行完成
# 有账单写入(bill)，统计/物价必须等账单先执行
# 理财分析必须等统计先执行
TASK_CONTEXT_DEPS = {
    "stat": ["bill"],
    "price": ["bill"],
    "finance": ["bill", "stat", "price"]
}


def normalize_keys(data):
    """递归清洗字典所有key，移除全部空格，根治模型输出key带空格问题"""
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
    【仅格式/字段合法性校验，不做任何业务意图、语义判断】
    只校验：字段存在、类型合法、任务名称在允许列表、禁止同轮重复任务
    返回 (是否合法, 错误汇总信息, 修复后的任务列表)
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

        # 字段合法性校验
        # ★2026-08-31：外部 hy3 偶发输出 name 非法/缺失的任务对象（如 {}）。
        #   此类无效项静默剔除，避免整轮任务规划失败；全部剔除时返回空任务列表，
        #   由下游 prune_tasks_by_heuristics 的 force_bill / 空任务兜底接管
        #   （先记账或提示"未识别到有效业务需求"），比直接"规划失败"更可控。
        if t_name not in TASK_AGENT_MAP:
            continue
        # 检测重复任务
        if t_name in name_check_set:
            errors.append(f"任务{idx}: 存在重复任务名称 {t_name}，不允许同一轮多次派发")
            continue
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


# ========== 启发式业务裁剪：纠正1.8B小模型的语义误判（脑补任务/误判任务类型） ==========
AMOUNT_REGEX = re.compile(r"\d+(\.\d+)?")
# 统计意图强关键词
STAT_HINT_KEYWORDS = [
    "统计", "查询", "汇总", "这个月", "本月", "上月", "上个月", "上礼拜", "上周",
    "花了多少", "支出", "消费明细", "账单明细", "总共", "多少次", "多少笔", "开销", "余额",
    "多不多",
]
# 统计任务"兜底补齐"用强意图词：排除"本月/这个月"等纯时间词，
# 避免"给我点本月省钱建议"这类finance输入被误补stat任务
STAT_ENSURE_KEYWORDS = [
    "统计", "查询", "汇总", "开销", "花了多少", "消费了多少", "消费明细", "账单明细",
    "总共", "多少次", "多少笔", "余额", "支出", "总开销", "总支出",
]
# 理财建议意图强关键词
FINANCE_HINT_KEYWORDS = [
    "建议", "怎么", "如何", "优化", "省钱", "理财", "预算", "规划", "意见",
    "分析", "更省", "合理", "省点", "消费习惯",
]
# 理财"强"意图词（不含"合理"）：用于区分"纯比价合理性疑问"与"真理财分析"
FINANCE_STRONG_KEYWORDS = [
    "建议", "怎么", "如何", "优化", "省钱", "理财", "预算", "规划", "意见",
    "分析", "更省", "省点", "消费习惯",
]
# 比价意图强关键词
PRICE_HINT_KEYWORDS = [
    "贵", "便宜", "比价", "对比", "性价比", "哪家", "价格", "划算", "值不值", "贵不贵", "市场价",
]
# 合理性疑问关键词（"这笔消费合理吗""吃饭多少钱合理吗"）：
# 语义=对比该笔消费的市场水平（溢价率），与纯"比价"同源，但独立成词便于区分记账后分析与纯比价
REVIEW_HINT_KEYWORDS = [
    "合理", "合理吗", "合理不", "合不合理", "值吗", "值不值", "合适吗", "贵吗", "贵不贵", "便宜吗", "划算吗",
    # 纯合理性问句（2026-09-05）："正常吗/算贵吗"与"值不值"同属合理性咨询，
    # 让"吃饭600块正常吗"这类无记账指令的消费咨询在启发式层可映射为 finance 分析
    "正常吗", "正常不", "算正常吗", "算贵吗",
    # 超支/花多类合理性疑问（"是不是花多了""花超了""超支了吗"）：语义=询问该笔/该月消费是否超合理范围，
    # 属于合理性分析而非记账指令，必须拦截 force_bill 兜底，防止误记账
    "花多了", "花超了", "花得值吗", "花超支", "超支了吗", "超支吗", "超预算", "超标了",
]
# 记账后自动预警链触发：记账成功且为纯记账(无finance)时，主Agent自主计算预算状态
# 预算充足 → 犒劳提醒；预算不足/超支 → 预警；无预算配置 → 不打扰
BUDGET_ALERT_THRESHOLDS = {
    "warn_ratio": 0.8,   # 已用比例>=80% 触发预警
    "over_ratio": 1.0,   # 已用比例>=100% 超支预警
    "reward_ratio": 0.5, # 已用比例<=50% 且未预警 → 犒劳提醒
}
# 记账动词（2026-09-05 产品规则收紧）：只有用户**明确表达记账指令**（记/记账/记录/写入/
# 入账/帮我记…等）才允许生成 bill 任务并写入账单；"花了/买了/用了/付了/消费了"等纯消费
# 陈述一律视为**咨询**（如"吃一顿饭花了600块正常吗""那打车花了80块呢"），由规划模型结合
# 上下文决策 price/finance，启发式层绝不为这类无指令句强制记账。
BILL_VERB_KEYWORDS = [
    # 单字"记"即记账指令（"记午饭56元"），也是各组合词的子串，置于首位便于阅读
    "记",
    "记录", "记账", "记一下", "记一笔", "帮我记", "帮我记一下", "帮我记一笔",
    "记上", "记到", "记入", "记个账", "记下来", "入账", "登记",
    "写入", "写进", "写进账", "写进账单", "存账", "存账单", "存一笔", "存个账",
]
# 消费实体词（2026-09-05 起不再单独构成记账意图，仅用于辅助消费咨询识别）：
# finance/price 咨询兜底需"金额+消费实体"判定合理性疑问；纯价格比价等无金额场景识别实体
CONSUME_NOUNS = [
    "打车", "地铁", "公交", "出租", "网约车", "滴滴", "高铁", "火车", "机票", "飞机", "加油", "停车", "通勤",
    "吃饭", "外卖", "奶茶", "火锅", "晚餐", "午餐", "早餐", "夜宵", "咖啡", "零食", "小吃", "食堂", "聚餐", "烧烤",
    "酒店", "民宿", "住宿", "宾馆", "房租", "旅馆",
    "买", "书", "衣服", "裤子", "鞋", "超市", "商场", "淘宝", "京东", "手机", "电脑", "数码", "日用品", "包包",
    "电影", "KTV", "ktv", "门票", "演出", "游戏", "唱歌",
]
# 收入类关键词：本系统仅支持支出记账，收入/退款/报销等场景必须拦截 bill 任务
# （与 bill_agent INCOME_KEYWORDS 双保险：即使任务派发下来，bill_agent 也会拦截不入库）
INCOME_KEYWORDS = [
    "工资", "发薪", "奖金", "收入", "收款", "到账", "收到", "退款", "报销", "分红",
    "中奖", "利息", "红包", "赚了", "挣了", "卖了", "回款", "转入", "发了",
]
# 拒绝记账关键词：用户明确表示不记账（"别记这笔""不用记""只是问问"等）。
# 即使句子含记账动词/金额（1.8B 常误规划 bill），也必须拦截 bill 任务，防止误记账。
# 注意刻意排除裸"不记"，避免误伤"我不记得了"这类陈述。
REFUSE_BILL_KEYWORDS = [
    "别记", "不用记", "先不记", "先别记", "别记账", "不用记账", "不要记", "不记账",
    "别入账", "不用入账", "只是问问", "就问一下", "随便问问", "随口问问",
]

# 伪记账词（2026-09-05 修复胡乱记账）：文本含"记/记录"字串但**语义不是记账指令**的常见词。
# 背景：BILL_VERB_KEYWORDS 含裸单字"记"与"记录"，而 _has_any_keyword 是子串匹配，导致
# "我记得打车花了80""买了行车记录仪300块""日记里写了买书80""记住这个月别乱花"等句子
# 只要带金额+消费实体词，就会被误判成记账指令 → force_bill/启发式兜底补 bill → 误落库。
# 修复：判定记账指令前先剥离这些伪记账词（先删长词后删短词，避免子串互相掩盖），
# 剩余文本再查 BILL_VERB_KEYWORDS。剥离后若句中仍有独立记账动词（如"记得把打车80记上"），
# 仍会正确命中——伪词剥离只去除"记"作为词内语素的情形，不误伤"记上/记一下"等真实指令。
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
    """剥离文本中的伪记账词，返回剩余文本（避免子串匹配误判记账指令）"""
    cleaned = text
    for w in PSEUDO_RECORD_KEYWORDS_SORTED:
        cleaned = cleaned.replace(w, "")
    return cleaned


def _has_record_cmd(text: str) -> bool:
    """
    严格记账指令判定（2026-09-05 修复胡乱记账）：
    剥离伪记账词（记得/日记/标记/行车记录仪/笔记…）后，剩余文本命中 BILL_VERB_KEYWORDS
    才算"用户明确表达记账动作"。替换 prune_tasks_by_heuristics / _is_force_bill /
    repair_task_names / plan_node 中所有裸 `_has_any_keyword(text, BILL_VERB_KEYWORDS)` 判定。
    """
    return _has_any_keyword(_strip_pseudo_record(text), BILL_VERB_KEYWORDS)


def _has_any_keyword(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)


def _is_force_bill(user_text: str) -> bool:
    """
    记账优先判定（2026-09-05 收紧）：**显式记账指令词** + 金额 + 消费实体词 + 非收入 +
    非拒绝记账 + 非比价/统计/理财/合理性疑问。

    用途：用户已明确下达记账指令（"记打车35元""帮我把奶茶18块记一下"），但模型偶发
    漏规划/误规划（外部 LLM 常判为其它任务或输出空任务）时，兜底补 bill 任务。
    两处复用：
      1) prune_tasks_by_heuristics：模型漏规划时补 bill 任务；
      2) plan_node 的 pre_check_pass=false 拦截分支：记账指令明确时放行，不让准入
         拦截抢在兜底之前终止流程。

    ★规则边界（用户投诉修复）：无记账指令的消费陈述（"奶茶15块""那打车花了80块呢"）
      一律视为咨询，不得强制记账——哪怕含金额与消费实体词。
      2026-09-05 追加：记账指令判定必须先剥离伪记账词（记得/日记/行车记录仪/笔记等），
      否则"我记得打车花了80"这类含"记"字但非记账指令的句子会误触发兜底记账。
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


def prune_tasks_by_heuristics(user_text: str, tasks: list) -> list:
    """
    业务启发式裁剪+兜底补充：
    - 裁剪：删除与用户输入意图不符的任务（小模型常脑补 stat/finance、误把建议当记账）
    - 补充：全部被裁剪且输入意图明确时，补充正确任务（小模型常漏规划 stat/price）
    若完全无法判断，则信任模型原始规划——例外：bill 只在用户**显式下达记账指令**时
    才保留/补齐/复活（2026-09-05 产品规则），无指令词的消费陈述一律按咨询处理。
    """
    # ★2026-08-31 修复（M1 实测）：原实现对空任务列表在此提前 return，导致下方
    #   force_bill / 收入拦截 / 拒绝记账兜底（依赖 has_amount 等变量）永远执行不到。
    #   移除提前返回，让空列表也进入完整裁剪+兜底流程。
    #   （2026-09-05 收紧后：force_bill 仅对"显式记账指令"句生效，如"记打车35元"模型
    #   漏规划时兜底补 bill；"住酒店一晚388元""那打车花了80块呢"等无指令消费陈述不再
    #   补记账——返回空任务，由 plan_node 消费咨询引导或模型规划的 price/finance 承接。）
    has_amount = bool(AMOUNT_REGEX.search(user_text))
    has_stat_hint = _has_any_keyword(user_text, STAT_HINT_KEYWORDS)
    has_finance_hint = _has_any_keyword(user_text, FINANCE_HINT_KEYWORDS)
    has_price_hint = _has_any_keyword(user_text, PRICE_HINT_KEYWORDS)
    # 合理性疑问（"这笔消费合理吗""吃饭多少钱合理吗"）：语义=对比市场水平/溢价率
    has_review_hint = _has_any_keyword(user_text, REVIEW_HINT_KEYWORDS)
    # 记账指令：剥离伪记账词（记得/日记/行车记录仪/笔记等）后仍命中记账动词才算，
    # 防止"我记得打车花了80"这类含"记"字但非记账指令的句子触发记账（2026-09-05）
    has_bill_verb = _has_record_cmd(user_text)
    has_income = _has_any_keyword(user_text, INCOME_KEYWORDS)
    has_consume_noun = _has_any_keyword(user_text, CONSUME_NOUNS)
    # 拒绝记账：用户明确表示不记账（"别记这笔""不用记""只是问问"等）→ 任何情况下不派发 bill
    has_refuse_bill = _has_any_keyword(user_text, REFUSE_BILL_KEYWORDS)
    # 纯比价合理性疑问：无记账动词 + 有合理性疑问 + 无理财强意图词 → 只需 price（算溢价率）
    # 区分"记一笔打车35元，合理吗"（记账+合理性 → bill+finance）
    # 与 "吃饭多少钱合理吗"（纯比价 → 只需 price）
    pure_price_review = (
        has_review_hint and not has_bill_verb and not has_amount
        and not _has_any_keyword(user_text, FINANCE_STRONG_KEYWORDS)
    )
    # 记账上下文合理性疑问（"花了300买耳机，值吗""记一笔打车35元，合理吗"）：
    # 带金额 + 记账动词/消费实体词 + 合理性疑问 → 理财分析"这笔值不值"（finance 消费合理性分析）
    # REVIEW_HINT 词不在 PRICE_HINT/FINANCE_HINT 中（如"值吗""合理吗"），此前会被裁剪丢失，
    # 这里显式映射为 finance 意图，确保"记账+问值不值"多意图完整落地
    has_finance_review = (
        has_review_hint and has_amount and (has_bill_verb or has_consume_noun)
        and not has_income
    )
    force_bill = _is_force_bill(user_text)

    pruned = []
    pruned_names = set()  # 用于同类型任务去重：stat/price/finance 单意图只保留第一个
    for task in tasks:
        name = task.get("name")
        if name == "bill":
            # 记账必须有金额 + 记账动词；收入/退款、拒绝记账场景坚决不记账；
            # 纯分析/比价场景（哪怕带金额）不记账
            if has_income or has_refuse_bill or not has_amount or not has_bill_verb:
                continue
        elif name == "stat":
            # 模型常把单条统计意图拆成多个 stat 任务且改写 raw_segments，
            # 只保留第一个，避免重复统计
            if not has_stat_hint or name in pruned_names:
                continue
        elif name == "finance":
            if not (has_finance_hint or has_finance_review) or name in pruned_names:
                continue
        elif name == "price":
            if not has_price_hint or name in pruned_names:
                continue
        pruned.append(task)
        pruned_names.add(name)

    # ===== 兜底补齐：强意图存在但模型漏规划时，补充对应任务（1.8B常漏规划 bill/price）=====
    # 即使裁剪后仍有剩余任务也继续补齐，确保用户显式表达的多意图不被漏掉
    pruned_names = {t.get("name") for t in pruned}
    seg = [user_text]
    if has_bill_verb and has_amount and not has_income and not has_refuse_bill and "bill" not in pruned_names:
        pruned.append({"name": "bill", "raw_segments": seg, "operate_sub_type": "add"})
    if has_price_hint and "price" not in pruned_names:
        pruned.append({"name": "price", "raw_segments": seg, "operate_sub_type": "query"})
    if (has_finance_hint or has_finance_review) and "finance" not in pruned_names:
        pruned.append({"name": "finance", "raw_segments": seg, "operate_sub_type": "analyse"})
    if has_stat_hint and "stat" not in pruned_names and _has_any_keyword(user_text, STAT_ENSURE_KEYWORDS):
        pruned.append({"name": "stat", "raw_segments": seg, "operate_sub_type": "query"})

    if not pruned:
        if has_income:
            # 纯收入/退款场景：模型误规划了 bill，启发式裁剪后为空 → 返回空任务（不记账）
            print(f"[ORCH] 收入场景拦截：user={user_text[:30]!r} 返回空任务", flush=True)
            return []
        if has_refuse_bill:
            # 拒绝记账场景：模型误规划了 bill（"别记这笔"含"记"字常被1.8B误判为记账），
            # 裁剪后为空 → 返回空任务（不记账），防止误记账
            print(f"[ORCH] 拒绝记账拦截：user={user_text[:30]!r} 返回空任务", flush=True)
            return []
        if force_bill:
            # 记账优先兜底：用户已明确下达记账指令（"记打车35元"）但模型漏规划/误规划为
            # stat/finance 或输出空任务 → 强制补记账任务。
            # （2026-09-05 收紧：无指令词的消费陈述已不满足 force_bill，不会走到此分支）
            print(f"[ORCH] 记账优先兜底：user={user_text[:30]!r} 含记账指令且模型未规划bill，强制补记账任务", flush=True)
            return [{"name": "bill", "raw_segments": seg, "operate_sub_type": "add"}]
        # 兜底：启发式识别不出可确认意图时信任模型原始规划，但**被裁剪掉的 bill 一律不复活**
        # ——bill 被裁剪即"用户未显式下达记账指令"（见上方保留条件）。复活会导致"那打车花了
        # 80块呢""奶茶15块"这类咨询/陈述被误写账单（2026-09-05 产品规则）；模型规划中的
        # price/finance/stat 等咨询任务原样保留，交由子 agent 分析
        kept = [t for t in tasks if t.get("name") != "bill"]
        if not kept:
            return []
        return kept
    return pruned


def repair_task_names(raw_tasks: list) -> list:
    """
    修正 name 字段不在 TASK_AGENT_MAP 的任务（1.8B小模型常把用户原话/测试标记塞进 name，
    例如 name="[测试标记]" 而 operate_sub_type="add" 实际是记账任务）。
    推断优先级：operate_sub_type 强映射(add->bill / analyse->finance) > raw_segments 意图关键词。
    仍无法推断的任务丢弃并告警，避免整个规划解析失败。
    """
    sub_type_to_task = {
        "add": "bill",
        "analyse": "finance",
    }
    intent_keywords = [
        # bill 排除：记账指令需经 _has_record_cmd（剥离记得/日记/行车记录仪等伪记账词后判定），
        # 不能走子串匹配——"我记得打车花了80"这类片段不得被推断成 bill 任务
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
            # 记账指令片段经严格判定（剥离伪记账词），再按普通关键词推断其它任务类型
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
    """渲染统计结果：总支出/笔数/分类明细，数据缺失时优雅降级"""
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
    """渲染物价对标结果：溢价率/消费档次/结论"""
    if not isinstance(data, dict) or data.get("premium_rate") is None:
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
    唯一生成面向用户可读文本的函数
    子Agent禁止拼接展示文案，统一由Orchestrator基于结构化数据渲染
    通信标准：{"success":bool, "agent_type":str, "msg":str, "data":dict|None, "error":str|None}
    """
    success = struct.get("success", False)
    agent_type = struct.get("agent_type", "")
    data = struct.get("data")
    err = struct.get("error", "")
    msg = struct.get("msg", "")

    if success is True:
        if agent_type == "bill_agent":
            return f"{msg}：消费类目 {data['category']}，金额 {data['amount']} 元，日期 {data['consume_date']}"
        elif agent_type == "stat_agent":
            return _render_stat_display_text(msg, data)
        elif agent_type == "price_agent":
            return _render_price_display_text(msg, data)
        elif agent_type == "finance_agent":
            # D16（M11）：finance 成功载荷携带 raw_analysis_text（已过白名单闸门的确定性分析文本）
            # 与 fact_sheet / fallback_used（审计/校验字段）。渲染直接取 raw_analysis_text，
            # 不再整字典 dump（旧实现会把 Python dict 直接拼进面向 LLM 的文本块）。
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
    组装规划提示词
    使用目录内 system.md（系统角色规则） + user.j2(Jinja模板)
    返回：系统提示词、渲染后的用户提示词
    """
    sys_prompt = load_system()
    valid_tasks_str = ", ".join(valid_tasks)
    user_prompt = render_user(
        user_input=user_input,
        history=history,
        valid_tasks_str=valid_tasks_str
    )
    return sys_prompt, user_prompt


# ===================== M6（P0-2）：任务依赖动态化（`03` §9.8.5 / D-b）=====================
def _collect_llm_deps(tasks: list, task_names: set) -> dict:
    """从 LLM 规划结果中提取 `deps`（步骤④：丢弃引用了不存在任务的依赖）。

    ★从 **LLM 原始输出**取而非 `task_plan`：中间链路（`repair_task_names` /
      `validate_task_list` / `prune_tasks_by_heuristics`）可能重建任务 dict 而丢字段
      （如 force_bill 兜底直接 new 一个 dict），按任务名从原始 tasks 回捞最稳。
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
        # 引用校验：只保留本批任务内存在的任务名，且排除自依赖
        valid = {d for d in raw
                 if isinstance(d, str) and d in task_names and d != name}
        if valid:
            out[name] = valid
    return out


def _merge_deps(task_plan: dict, llm_deps: dict) -> tuple:
    """`final_deps = 静态表 ∪ LLM deps`（步骤①②：并集 + 裁剪）。

    ★安全不变式：LLM **只能增加**依赖、**不能删除**静态表依赖 ⇒ 动态性只往"更保守
      （更串行）"方向走，最坏退化为改造前的串行，**绝不比现状更差**（`03` §9.8.5）。
    ★裁剪：仅保留**本批任务内存在**的依赖（沿用现状逻辑，避免"只问统计却强制先记账"）。

    :param llm_deps: LLM 输出的依赖映射；传 `{}` 即**回退纯静态表**
    :return: (task_plan, 本轮相对静态表**新增**的动态依赖)
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
    深度优先遍历检测循环依赖
    例：A依赖B，B依赖A → 循环，无法执行，直接终止
    """
    visited = set()
    rec_stack = set()

    def dfs(node):
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


# ===================== 工具：读取用户预算目标配置（消除硬编码） =====================
async def get_user_target_config(session_id: str) -> dict:
    """
    从 user_config 表读取用户最新预算目标配置
    无数据返回空字典，由 finance_agent 触发"缺少预算"追问引导用户设置
    """
    # M9（D15）：收口走 config.get_latest 工具（FACT_READ）
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


# ===================== 长期记忆沉淀：记账成功 → 月度消费习惯 =====================
# M9（D15 P0-5）：`habit.upsert` 下沉 bill_agent——记账成功的副作用由记账方自身收尾
#（bill_agent `execute_node` 成功分支沉淀 monthly_habit + pkl 语义层），
# orchestrator 不再写 monthly_habit（原 `_persist_bill_habit` 已删除，零写路径）。

async def _load_recent_habits(months_back: int = 3) -> list:
    """读取最近 N 个自然月（含当月）的消费习惯聚合数据，供 finance 推算剩余额度/频次"""
    try:
        today = date.today()
        month_list = []
        y, m = today.year, today.month
        for _ in range(months_back):
            month_list.append(f"{y:04d}-{m:02d}")
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        # M9（D15）：收口走 habit.list 工具（FACT_READ）
        return await _invoke_fact("habit.list", {"months": month_list})
    except Exception:
        return []


# ===================== 需求1/4 自动预警链（纯记账场景，确定性计算，不进LLM） =====================
async def _sum_month_bills(month: str) -> float:
    """查询指定自然月（YYYY-MM）账单金额总和，无数据返回 0（异常兜底 0）"""
    try:
        # M9（D15）：收口走 bill.sum_by_month 工具（FACT_READ）
        return float(await _invoke_fact("bill.sum_by_month", {"month": month}))
    except Exception:
        return 0.0


async def _calc_month_budget_status() -> dict | None:
    """
    计算当月预算状态（需求1/4预警链核心，纯计算不进LLM）。
    返回 None = 未配置预算（不打扰，与现状完全一致）；
    否则返回 {level, month, month_spent, month_budget, used_ratio, days_left, consume_mode}
    level: over=超支 / warn=接近红线 / reward=预算充裕(犒劳) / ok=正常区间(不打扰)
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
    总量预算维度确定性计算（需求1修正：超预算评估须逐品类后再汇总）。
    只产出结构化事实（不进LLM，也不拼文案）：
      level=ok 或无预算 → 返回 None（不打扰）；
      否则返回 {level, month_spent, month_budget, used_ratio, days_left, remain, daily_limit, over_categories}
    daily_limit：warn 档剩余预算/剩余天数的每日建议额度；其余档位为 None。
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


# ===================== 需求1修正：品类维度评估（均价对比 + 品类进度 + 品类归因） =====================
# 品类合理消费需求额度的默认月消费频次（城市物价估算兜底：均价 × 频次 = 月度合理额度）
DEFAULT_CATEGORY_FREQ = {"餐饮": 30, "交通": 44, "住宿": 4, "购物": 4, "娱乐": 2}


async def _query_city_avg_price(city: str, category: str) -> float:
    """查询当地同品类基准均价（零 LLM，走 `city_price.get_avg` 工具）。
    兜底逻辑（本城无 → 同级城市均价 → 0）已整体迁入工具侧 `mcpGateway/fact_tools.py`。
    """
    try:
        return float(await _invoke_fact("city_price.get_avg",
                                        {"city": city, "category": category}))
    except Exception:
        return 0.0


async def _sum_month_bills_by_category(month: str) -> dict:
    """当月各品类累计金额 {category: total}；异常兜底空字典"""
    try:
        # M9（D15）：收口走 bill.sum_by_category 工具（FACT_READ）
        return await _invoke_fact("bill.sum_by_category", {"month": month}) or {}
    except Exception:
        return {}


async def _calc_category_quota(city: str, category: str) -> dict | None:
    """
    计算品类合理消费需求额度（需求1修正：超预算评估须逐品类）。
    优先级：用户设定(category_budget) > 历史习惯月均(monthly_habit近3月) > 城市物价估算(均价×默认月频次)。
    返回 {"quota": float, "source": "user"/"habit"/"city"}；三者皆无 → None。
    """
    try:
        cfg = await get_user_target_config("")
        quota_mode = cfg.get("quota_mode") or "auto"
        # 1. 用户显式设定（最高优先级）
        cb = cfg.get("category_budget") or {}
        if quota_mode in ("auto", "user") and cb.get(category):
            return {"quota": float(cb[category]), "source": "user"}
        # 2. 历史消费习惯月均（近3个月该品类月均金额）
        if quota_mode in ("auto", "habit"):
            habits = await _load_recent_habits(3)
            rows = [h for h in habits if h.get("category") == category]
            if rows:
                total = sum(float(h.get("amount_sum") or 0) for h in rows)
                return {"quota": round(total / len(rows), 2), "source": "habit"}
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
    """渲染纯文本进度条：█ 已用 / ░ 剩余；ratio 超 1 显示满格 + 溢出"""
    ratio = max(0.0, min(ratio, 1.0))
    filled = int(round(ratio * width))
    return "█" * filled + "░" * (width - filled)


async def _calc_alert_facts(status: dict | None, last_amount: float,
                            category: str = "", city: str = "") -> dict:
    """
    需求1修正后的预警事实计算（确定性计算层，只产出结构化数据，不拼文案、不进LLM）：
      l1 单笔均价对比：本笔金额 vs 当地同品类均价 → 溢价率 + 四档（偏低/正常/偏高/过高）
      l2 品类进度对比：当月该品类累计 vs 品类合理消费需求额度（用户设定/历史习惯/城市估算）
      l3 总量对比：当月总支出 vs 月总预算（四档总量事实 + 超支品类归因列表）
    返回 dict（无任何可评估数据 → 对应层为 None，collect_node 据此决定是否把事实交给 LLM 汇总提示词）。
    """
    facts = {"l1": None, "l2": None, "l3": None}
    month = (status or {}).get("month") or date.today().strftime("%Y-%m")

    # ---- L1 单笔均价对比（每笔都对比）----
    if category and last_amount > 0:
        avg_price = await _query_city_avg_price(city, category)
        if avg_price > 0:
            ev = calc_premium_evaluate(last_amount, avg_price)
            facts["l1"] = {
                "category": category,
                "amount": round(last_amount, 2),
                "city": city or "当地",
                "avg_price": round(avg_price, 2),
                "premium_rate": round(ev["premium_rate"], 2),
                "level": ev["level"],
            }

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
            # 无任何额度基准 → 标记需引导用户设置/选择评估方式（需求1修正决策点②）
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
            # 超支归因：点名超出品类合理额度的主要品类（任一品类超支即预警，需求1修正决策点③）
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


# ===================== 表达层数字安全（防外部LLM数字幻觉） =====================
# 背景：混元 hy3 在纯记账+预警场景会编造"假设24元/317%/756.8元/预算5998.8元"等
# 幻觉数字（见 eval_report_20260816.json single_bill_001/anti_hallucination_012）。
# 根因：①汇总提示词示例自带具体数字被模型当作事实复用；②alert_facts 缺基准时模型
#       自行"假设/计算"补全。对策：①提示词示例占位符化；②数字白名单闸门——LLM 回复
#       中任何不在白名单内的数字（原文/确定性事实/渲染文本）一律判幻觉，回退本地渲染。

# 剔除日期整体，防止月/日被误判为金额数字
# D16/M11（2026-09-04）：白名单核心实现抽至 utils/number_whitelist.py 单一来源
# （finance_agent D16 防幻觉同源复用；`nodes._extract_numbers` 等私有名保持可用，既有单测引用不破）
from utils.number_whitelist import (
    extract_numbers as _extract_numbers,
    numbers_inside_whitelist as _numbers_inside_whitelist,
)


def _whitelist_numbers(user_input: str, alert_facts: dict, text_blocks: list) -> set[float]:
    """构建数字白名单：用户原文 + 确定性预警事实 + 确定性渲染文本 中的所有数字"""
    pool = set()
    for t in [user_input, json.dumps(alert_facts, ensure_ascii=False), *text_blocks]:
        pool |= _extract_numbers(t)
    return pool


# ===================== 节点1：任务规划节点（Plan） =====================
async def plan_node(state: OrchState):
    """
    作用：
    1、读取会话上下文
    2、调用大模型，让秘书Agent生成任务规划JSON
    3、解析、清洗任务，构建标准化task_plan
    4、顶层准入判断：pre_check_pass=false 直接拦截，不派发任务
    """
    user_text = state["user_input"]
    session_id = state["session_id"]
    _log("plan_node 开始", f"user={user_text[:40]!r} session={session_id[:12]}")

    # 获取会话内存对象（对话历史+草稿缓存）
    # 注意：完整内存（历史+草稿）由 dispatch_node 每轮自行 export_all 后随 task_data 下发，
    # 此处仅取 history 用于组装规划提示词，无需（也不应）重复导出
    mem = get_session_memory(session_id)
    session_history = mem.get_history()

    # 长期记忆检索：跨月/历史消费摘要注入规划提示词，与短期历史并列
    long_mem_text = ""
    try:
        # 向量检索是同步IO（FAISS+jieba），丢线程避免阻塞事件循环
        long_mem_text = await asyncio.to_thread(search_history_consume, user_text)
    except Exception:
        long_mem_text = ""
    history_context = session_history
    _log("长记忆检索完成", f"long_mem={long_mem_text[:60]!r}")
    if long_mem_text and long_mem_text != "暂无历史消费记录":
        history_context = (session_history + "\n" + long_mem_text).strip()

    # M12（D6）：分层预算——历史/长期记忆超预算按层截断（预算内逐字节零改动 = 常规输入零回归；
    # 组件异常回退上方原始拼装，不阻断规划链路）
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
        schema_example = '{"tasks":[{"name":"bill","raw_segments":["xxx"],"operate_sub_type":"add"}],"pre_check_pass":true,"block_tip":""}'
        # parse_json_output内部自动调用LLM + 清洗提取JSON + 带重试
        # ignore_missing_keys: operate_sub_type 缺失时由 validate_task_list 兜底默认值，避免无效重试
        plan_data = await parse_json_output(
            mcp_client.call_llm_base,
            sys_p,
            usr_p,
            schema_example,
            agent_tag=ORCH_AGENT_ID,
            # ★2026-08-31 name/raw_segments 加入豁免（M1 实测）：二者是 tasks 数组
            #   内嵌套对象的示例字段，外部 hy3 偶发输出空 tasks 或漏字段时，通用
            #   schema 校验（递归查找）会误判"缺失"→ 重试3次仍失败 → 整轮"规划失败"，
            #   使已有的 force_bill / 空任务兜底完全没有机会执行。改为在业务侧兜底：
            #   raw_segments 下方回退完整原文、非法 name 由 validate_task_list 剔除。
            #   （与 operate_sub_type 的"默认值兜底"策略一致，避免无效重试）
            ignore_missing_keys=["operate_sub_type", "block_tip", "raw_segments", "name"],
            max_self_retry=3
        )
        # 递归清洗所有key，消除空格污染
        plan_data = normalize_keys(plan_data)

        # 兼容1.8B模型输出的 block_tips 复数键：缺失 block_tip 时从复数键取值兜底
        if "block_tip" not in plan_data and "block_tips" in plan_data:
            tip = plan_data.pop("block_tips")
            plan_data["block_tip"] = tip[0] if isinstance(tip, list) and tip else (str(tip) if tip else "")

        raw_tasks = plan_data.get("tasks", [])

        # ===== raw_segments 兜底：外部 hy3 偶发漏输出该字段或输出空数组 =====
        # raw_segments 语义为"用户原始语句片段"，缺失时回退为完整原文总是安全的
        # （子agent 内部会自行做关键词/金额抽取），优于让校验失败导致整轮规划终止。
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
        _log("LLM规划完成", f"tasks={[t.get('name') for t in raw_tasks]} pre_check={pre_check_pass}")

        # ============ 仅格式校验，移除所有业务意图判断 ============
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

    # ========== 顶层准入拦截分支 ==========
    # 规则：bill不满足准入条件，pre_check_pass=false，直接回复提示，不生成、不派发任何任务
    # ★2026-08-31 例外（M1 实测）：外部 LLM 对显式记账指令句（如"记打车35元"）偶发误判
    #   pre_check_pass=false，直接拦截会让 force_bill 兜底失去执行机会，故记账指令明确的
    #   句式放行，交给 prune_tasks_by_heuristics 补 bill 任务。
    #   （2026-09-05 收紧：force_bill 要求"记/记账/记录"等显式指令词，"奶茶18块""那打车
    #   花了80块呢"等无指令词的消费陈述不满足 → 走拦截，且 block_tip 为空时回退咨询引导）
    if not pre_check_pass and not _is_force_bill(user_text):
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

    # ========== 启发式业务裁剪：纠正小模型语义误判（脑补任务/误判任务类型） ==========
    fixed_tasks = prune_tasks_by_heuristics(user_text, fixed_tasks)
    _log("启发式裁剪后", f"tasks={[t.get('name') for t in fixed_tasks]}")

    # ========== 空任务确定性拦截：收入不入账 / 拒绝记账，直接给拦截文案，不走通用兜底 ==========
    # 启发式裁剪在收入/拒绝记账场景返回空任务列表（防止误记账），此处转换为对用户可见的
    # 确定性回复（与 bill_agent 收入拦截文案一致），避免 collect_node 输出"未识别到有效业务需求"
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
        # ===== 消费陈述但无记账指令（2026-09-05 产品规则）：一律视为咨询，给确定性引导 =====
        # 触发：含金额或消费实体词、但**不含显式记账指令**的文本（"那打车花了80块呢"
        # "奶茶15块"），启发式裁剪后为空任务。此时绝不能落库，直接回复引导文案，
        # 避免 collect_node 输出"未识别到有效业务需求"这种让用户困惑的通用兜底。
        # 记账指令判定用 _has_record_cmd（剥离伪记账词），"我记得打车花了80"不会命中"记"
        if not _has_record_cmd(user_text) and (
            AMOUNT_REGEX.search(user_text) or _has_any_keyword(user_text, CONSUME_NOUNS)
        ):
            return Command(
                update={
                    "final_reply": (
                        "收到。这句话我先按咨询理解，没有记账——只有你明确说「记…」我才会把"
                        "消费写入账单（例如：记打车80元）。如果想判断这笔贵不贵/值不值，可以"
                        "补一句「…贵不贵/正常吗」，我再帮你分析。"
                    ),
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

        # ===== 金额上下文补全：bill/price/finance 任务的片段缺数字但原文有金额时，附上完整原文 =====
        # （1.8B模型分段规划时，常把金额放进 bill 片段，price/finance 片段只留"对比xx物价""值吗"等
        # 无金额文本；多意图场景必须保证 finance 也携带金额/品类参数，否则推算缺失上下文）
        if t_name in ("bill", "price", "finance") and segs:
            seg_text = " ".join(str(s) for s in segs if isinstance(s, str))
            if not re.search(r"\d+(\.\d+)?", seg_text) and re.search(r"\d+(\.\d+)?", user_text):
                segs.append(user_text)
                _log("金额上下文补全", f"task={t_name} 片段无金额，追加完整原文")

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

    # ★M6（P0-2）：任务依赖动态化（`03` §9.8.5 / D-b）——静态表 ∪ LLM 输出的 deps
    llm_deps_map = (_collect_llm_deps(fixed_tasks, task_names)
                    if DYNAMIC_DEPS_ENABLED else {})
    task_plan, dyn_added = _merge_deps(task_plan, llm_deps_map)
    if dyn_added:
        _log("动态依赖生效", f"{dyn_added}（静态表之外的补充顺序约束）")

    # 环检测（步骤③）：**成环即丢弃 LLM 的 deps、回退纯静态表**，**不终止执行**
    # （成环只可能来自 LLM 的补充依赖，静态表本身无环）
    if _has_cycle(task_plan):
        logger.warning(f"[M6] 动态依赖成环，回退纯静态表：{dyn_added}"
                       "（`01` R10 禁止静默失败）")
        task_plan, _ = _merge_deps(task_plan, {})
    # 兜底：静态表仍成环说明依赖表被改坏 → 保留原"终止执行"行为
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

    # 所有变更打包提交，禁止就地修改state
    # 注：session_history 仅用于组装规划提示词，不跨节点传递；
    # dispatch_node 每轮自行 get_session_memory() 导出最新内存后随 task_data 下发
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
    """按任务类型组装下发给子 Agent 的 `task_data`（`03` §3.2 下发契约）。

    ★M6（P1）：由原 `dispatch_node` 内联逻辑**逐字抽取**为函数，供批量派发循环复用——
      抽取前后各分支字段完全一致，是「依赖链场景结果逐字一致」红线的基础。
    """
    if task_name == "finance":
        target_control_data = await get_user_target_config(state["session_id"])
        # 历史消费习惯（近3个月聚合：品类/笔数/金额/均额），支撑"剩余预算能否撑到月底"推算
        habit_data = await _load_recent_habits(3)
        # M8：用户资料（`memory/user_docs.py` 唯一检索入口）——按 finance 原始输入检索一次下发，
        #     finance 侧只做渲染（标注「用户资料」来源）。检索失败/无结果 → []，不阻断理财分析。
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
        # 统计任务必须使用用户原始输入作为 raw_segments / user_input：
        # 1.8B 模型常把 stat 任务 raw_segments 改写成臆想值（如"总支出: 2901.48元"），
        # 导致 stat_agent 无法理解真实意图（"这个月一共有几笔消费"），count算子兜底失效
        return {
            "raw_segments": [state.get("user_input", "")],
            "user_input": state.get("user_input", ""),
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
        }
    if task_name == "price":
        return {
            "raw_segments": task_plan[task_name]["raw_segments"],
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
        }
    if task_name == "bill":
        # 注入用户城市与记账月份：city 取用户配置，缺失时用全局默认城市；month 用当前月
        cfg = await get_user_target_config(state["session_id"])
        return {
            "raw_segments": task_plan[task_name]["raw_segments"],
            "operate_sub_type": task_plan[task_name]["operate_sub_type"],
            "session_memory": mem_export,
            "city": (cfg.get("city") or DEFAULT_USER_CITY),
            "month": date.today().strftime("%Y-%m")
        }
    return {}


# ===================== 节点2：任务调度节点（Dispatch） =====================
async def dispatch_node(state: OrchState):
    """
    核心调度逻辑：
    1、遍历task_plan，筛选【所有依赖已经全部执行完成】的任务（就绪任务）
    2、★M6：**批量派发本波全部就绪任务**（原：只派 ready[0]），通过A2A队列发送给对应worker执行
    3、按需组装参数；finance任务额外携带预算目标 + 前置agent结构化结果
    """
    task_plan = state["task_plan"]
    # 收集已经执行完成的任务名称
    done_tasks = {k for k, v in task_plan.items() if v["status"] == "done"}

    # 筛选就绪任务：状态pending，并且所有依赖都在done_tasks内
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

    # ★M6（P1）：**批量派发本波全部就绪任务**（`03` §9.8.4 ②）
    #   就绪检查 `deps ⊆ done_tasks` **原样保留、一字未改**——依赖链上的任务本就不会
    #   同时就绪，故并行只发生在同波内互不依赖的任务之间，依赖语义零变化。
    new_task_plan = copy.deepcopy(task_plan)
    # 读取完整会话内存，随任务下发给worker（历史对话+草稿）——本波共用同一份快照
    mem = get_session_memory(state["session_id"])
    mem_export = mem.export_all()
    cache = copy.deepcopy(state.get("agent_result_cache", {}))

    # ★回滚开关（`11` M6）：`BILLAGENT_BATCH_DISPATCH=0` 退回单派发 `ready[0]`
    wave = ready if BATCH_DISPATCH_ENABLED else ready[:1]

    dispatched: list = []
    for current_task in wave:
        # ★M5（D2）：task_id 生成规则改为 `04` §3.2 规定格式——毫秒时间戳 + 6 位随机十六进制
        #   （如 t1756857000123a1b2c）：① 含时间可排序 ② ~20 字符可读（会写进 `bill.task_id` 列，
        #   便于崩溃对账时人工排查）③ 单用户量级全局唯一。
        #   原 `uuid.uuid4()` 不可排序、不可读，且无法支撑对账场景的肉眼定位。
        task_id = f"t{int(time.time() * 1000)}{secrets.token_hex(3)}"
        target_agent = new_task_plan[current_task]["agent_id"]
        task_data = await _build_task_data(current_task, new_task_plan, state, mem_export, cache)
        # M7（D3）：run_id + root span_id 随 task_data 下发 → worker 恢复同一 run 上下文
        #   （contextvars 不跨协程传播，span 栈亦不共享，故显式指定父 span 使树合并为一棵根，
        #   见 `11` §3 M7）；worker 的 init_state 为白名单式组装，附加键不进入 Agent 图状态
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

        # ★M5（D2）：派发前建任务单（`submitted`）——DB 是任务事实的唯一来源（`04` §3.1 补注）；
        #   队列只是 DB 状态的下游投影（重启即失），故崩溃恢复时由 DB 重建队列。
        #   ★建单失败只告警不阻断派发：状态库是"加固"，不能成为记账主流程的单点故障。
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

        a2a_bus.send_task(
            target_agent=target_agent,
            session_id=state["session_id"],
            task_id=task_id,
            task_data=task_data
        )
        dispatched.append(current_task)
        _log("派发任务", f"{current_task} -> {target_agent} task_id={task_id[:8]}")

    _log("本波派发完成", f"波次={state.get('dispatch_round', 0) + 1} 并行任务={dispatched}")

    update_payload = {
        "task_plan": new_task_plan,
        # `current_task` 仅保留"本波最后派发的一个"作兼容/日志语义；
        # 批量等待以 `status == "running"` 集合为准，不再依赖单值
        "current_task": dispatched[-1] if dispatched else None,
    }
    return Command(update=update_payload, goto="wait_result_node")


# ===================== 节点3：等待子任务结果节点 =====================
async def wait_result_node(state: OrchState):
    """
    ★M6（P1）：**批量**等待本波全部 `running` 任务的结果（`03` §9.8.4 ②）

    - 并发等待：整波耗时 = `max(各任务)`，而非累加 `ΣT`；每个任务仍是**各自** 240s 超时；
    - 结果乱序无害：按任务名归位写入 `task_plan` 与 `agent_result_cache`，后续上下文注入
      按 `deps` 精确取用，不依赖返回顺序；
    - 调度轮次按**波次** +1（上限由 20 下调至 `MAX_DISPATCH_ROUND`=8）。
    """
    task_plan = copy.deepcopy(state["task_plan"])
    # 本波在跑的任务：`dispatch_node` 已把它们标为 running 并写入 task_id
    running = {name: info["task_id"] for name, info in task_plan.items()
               if info.get("status") == "running" and info.get("task_id")}
    if not running:
        # 兜底：无任何 running 任务时不能回 dispatch，否则 dispatch↔wait 死循环
        logger.error("[M6] wait_result_node 未找到 running 任务（状态异常），终止本轮")
        return Command(
            update={"error_msg": "未找到执行中的任务，执行终止"},
            goto="collect_node"
        )

    _log("等待本波结果", f"任务={list(running)} 并发等待（各自 240s 超时）...")

    async def _wait_one(name: str, tid: str) -> tuple:
        """等待单个任务结果并解析（异常与旧文本格式均在内部兜底，不向上抛）"""
        try:
            # 等待worker返回结果：CPU推理单次LLM需30-90s，超时必须对齐 MODEL_TIMEOUT(240s)
            raw_result_str = await a2a_bus.wait_result(tid, timeout=240)
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

    # ★并发收集：整波耗时 = max(各任务)，非累加
    results = await asyncio.gather(*(_wait_one(n, t) for n, t in running.items()))

    # 写入全局缓存（供后续finance agent调用）
    cache = copy.deepcopy(state.get("agent_result_cache", {}))
    last_payload = None
    for name, struct_payload in results:
        task_plan[name]["status"] = "done"
        task_plan[name]["result"] = struct_payload
        cache[struct_payload.get("agent_type", "unknown")] = struct_payload
        last_payload = struct_payload

        # M9（D15 P0-5）：habit 沉淀已下沉 bill_agent（记账成功侧自身收尾），
        # 编排器不再写 monthly_habit（原 `_persist_bill_habit` 调用已删除）

    # 调度轮次自增 + 防无限循环保护：★M6 起按**波次**计数（上限 8）
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
    把 stat_agent 的成功统计结果浓缩为摘要文本，供长期记忆沉淀
    仅保留聚合性信息（总支出/分类/笔数/单笔最大），明细类查询不沉淀
    :return: 摘要文本；无有效聚合信息返回 None
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
    本地模式下的确定性汇总渲染：直接拼接各任务确定性结果与预警事实，
    杜绝 1.8B 小模型按汇总提示词生成时的数字幻觉（M5 本地基线专用）。
    外部模型可用时仍走 LLM 汇总（表达更自然），本函数仅作本地兜底。
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
    """追问计数的会话级 key：bill 类按草稿类型区分（`bill_add`/`bill_edit`/`bill_delete`），
    其余 Agent 用 `agent_id`。

    ★必须与管理草稿的 key 保持一致：`01` §3.4 规定放弃时清空的是
    `bill_add` / `bill_edit` / `bill_delete` 三类半成品草稿（T3）。
    """
    agent_id = struct.get("agent_type") or ""
    for info in task_plan.values():
        if info.get("agent_id") == agent_id:
            if agent_id == "bill_agent":
                return f"bill_{info.get('operate_sub_type') or 'add'}"
            return agent_id
    return "bill_add"      # 兜底：无法定位任务时按新增记账计（与 bill_agent 默认子类型一致）


# ===================== 节点4：结果汇总节点（Collect） =====================
async def collect_node(state: OrchState):
    """
    所有任务执行完成/流程终止后，统一汇总输出给用户
    基于结构化数据生成回复，不再依赖原始文本字符串
    """
    session_id = state["session_id"]
    mem = get_session_memory(session_id)
    _log("collect_node 开始汇总", f"session={session_id[:12]}")

    # 如果流程出现顶层错误，直接输出错误信息
    if state.get("error_msg"):
        final = f"执行终止：{state['error_msg']}"
    else:
        task_plan = state["task_plan"]
        all_struct_results = [v["result"] for v in task_plan.values() if v["status"] == "done"]

        if not all_struct_results:
            # 优先采用 plan_node 已下发的确定性拦截文案（如收入不入账、准入拦截 block_tip），
            # 无文案时才回退到通用兜底提示，防止拦截文案被覆盖丢失
            final = state.get("final_reply") or (
                "没听懂这条的需求。记账请明确说「记…」（如：记打车80元）；想查开销、"
                "对比价格或分析某笔消费是否合理，直接描述即可。"
            )
        else:
            prompt_list = []    # 需要向用户追问的内容
            struct_success_list = []  # 正常结构化结果

            give_up = False      # ★M5（D5）：追问超 `ASK_ROUND_LIMIT` 轮 → 放弃
            for struct in all_struct_results:
                # 安全判断：防止struct字段缺失崩溃
                if struct.get("success") is False and struct.get("error") == "need_more_info":
                    # ★M5（D5 T2）：**会话级**追问计数——graph state 每次 `ainvoke` 会重建，
                    #   不能放 state；而 `dispatch_round` 是调度轮次（防 DAG 死循环、每次新
                    #   用户输入即重置 0），**不是**追问轮次 → 这正是"无限追问"的根因（`02` D5）
                    ask_key = _ask_round_key(struct, task_plan)
                    if mem.incr_ask_round(ask_key) >= ASK_ROUND_LIMIT:
                        # T3：放弃即清草稿——防止半成品账单污染下一次记账（`01` §3.4）
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
                    # 长期记忆沉淀：stat 成功的聚合统计结果 → 月度消费摘要（同步FAISS+pkl写入丢线程，避免阻塞事件循环）
                    if struct.get("success") and struct.get("agent_type") == "stat_agent":
                        summary = _build_monthly_summary(struct)
                        if summary:
                            try:
                                await asyncio.to_thread(save_consume_memory, summary)
                            except Exception:
                                # 长期记忆写入失败不影响主流程回复
                                pass

            # 汇总策略：成功结果确定性渲染为文本素材，最终回复/提醒统一由外部LLM按汇总提示词生成
            if give_up:
                # ★M5（D5）：超限放弃——**固定提示语、不再追加追问**（`01` §3.4 / §8 Q5）
                final = ASK_GIVEUP_TIP
            elif struct_success_list:
                # 确定性渲染各任务结果文本（保证数字准确，作为LLM素材，不直接拼给用户）
                text_blocks = [build_user_display_text(s) for s in struct_success_list]

                # 预警事实（确定性计算层，不进LLM）：纯记账成功且无追问时计算 L1/L2/L3
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

                # 表达层：外部模型可用时按汇总提示词生成最终回复（含提醒，表达更自然）；
                # 本地模式（无外部LLM）用确定性渲染，杜绝 1.8B 小模型按模板生成时的数字幻觉
                if EXTERNAL_LLM_ENABLED:
                    # M12（D6）：任务结果层预算——多任务结果拼接超长时保序截尾（预算内零改动；
                    # 组件异常回退原始块，不阻断汇总；数字白名单随治理后块计算，头部结论保留）
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
                        # 数字白名单闸门：LLM 回复中出现白名单外数字（幻觉）
                        # ① 第一次：构造修正指令重试一次（保留混元表达力）
                        # ② 重试仍编造：回退本地确定性渲染（数字 100% 来自确定性计算层）
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