# ==============================================
# bill_agent 节点实现 - 记账抽取、代码层兜底修正与落库
# 说明：全部兜底规则（收入拦截、金额/类别/日期修正）都在本文件用代码实现，
#       目的是一旦 1.8B 小模型判错，由确定性规则把落库字段拉回用户原意，而非只靠提示词约束。
# ==============================================
import asyncio
import json
import re
import time
from datetime import date, timedelta
from langgraph.types import Command
from langgraph.graph import END
from agents.bill_agent.state import BillState
from agents.bill_agent import bill_target
from agents.bill_agent.prompts.prompt_loader import (
    load_system_prompt, render_full_template, render_simple_template, load_fc_system_prompt,
)
from agentCore.parsers.json_parser import parse_json_output
from config.config import BILL_FC_ENABLED, FC_MAX_TURNS, RECENT_BILL_MAX, RECENT_BILL_WINDOW
from mcpGateway.client import mcp_client
from mcpGateway.a2a_queue import a2a_bus

# 常量定义
BILL_AGENT_ID = "bill_agent"
ORCH_ID = "orchestrator_agent"
DATE_FULL = r"^\d{4}-\d{2}-\d{2}$"
DATE_YM = r"^\d{4}-\d{2}$"

# 收入类内容关键词：本系统仅支持支出记账，收入/退款/报销等一律不入库
# 覆盖常见表达："工资发了8000元""老板发工资""收到退款""报销餐费"等
INCOME_KEYWORDS = [
    "发奖金", "奖金到账", "发工资", "工资到账", "发薪", "收入", "收到转账", "收到款项",
    "退款", "报销", "分红", "中奖", "利息", "红包到账", "抢到红包", "赚了", "挣了",
    "卖了", "回款", "工资", "收款", "入账金额", "转账收入",
]

# 消费分类白名单：与 bill 表 CHECK 约束一致，禁止自创分类
VALID_CATEGORIES = {"餐饮", "交通", "住宿", "购物", "娱乐"}

# 类别语义关键词（与 system.md 语义绑定规则一致，代码化兜底：
# 1.8B 小模型常把"打车/买书/买衣服"等误判为"餐饮"，必须用原文关键词强制修正）
CATEGORY_KEYWORDS = {
    "餐饮": ["吃饭", "外卖", "奶茶", "火锅", "就餐", "零食", "早餐", "午餐", "晚餐", "夜宵",
              "咖啡", "小吃", "食堂", "聚餐", "下馆子", "快餐", "烧烤", "饮品", "甜品"],
    "交通": ["打车", "公交", "地铁", "加油", "停车", "出租车", "网约车", "滴滴", "打的士",
              "高铁", "火车", "机票", "飞机", "高速", "骑车", "单车", "出行", "通勤"],
    "住宿": ["酒店", "民宿", "住宿", "宾馆", "房费", "租房", "公寓", "青旅", "旅馆", "开房"],
    "购物": ["衣服", "裤子", "裙子", "鞋子", "鞋", "数码", "日用品", "百货", "采购", "买书",
              "书本", "文具", "化妆品", "超市", "商场", "淘宝", "京东", "网购", "包包", "包",
              "手机", "电脑", "家电", "日杂", "日化"],
    "娱乐": ["电影", "KTV", "ktv", "游乐园", "演出", "桌游", "游戏", "门票", "唱歌",
              "蹦迪", "剧本杀", "看戏", "影院", "游乐"],
}


def _has_income_keyword(text: str) -> bool:
    """
    函数功能与逻辑描述：
        收入类内容的前置启发式拦截：只要原文命中 INCOME_KEYWORDS 中任一子串即判定为收入类内容。
        采用**子串包含**而非分词匹配，因此"退款""报销""工资"等词出现在任何位置都会命中；
        这是刻意的宽松策略——本系统只支持支出记账，宁可对个别"我报销一下这顿饭花了50"
        这类混合表述误拦，也不让收入进账（调用方在拦截时只做提示、不写库，代价可控）。
        该判定在 execute_node 内于调用 LLM 之前执行，用于省掉一次无效模型调用。
    入参说明：
        text (str)：待判定的原始文本。
    返回值说明：
        bool：True 表示命中收入类关键词，应拒绝入账；False 表示未命中。
    """
    return any(k in text for k in INCOME_KEYWORDS)


def _fix_category(raw_text: str, llm_category: str) -> str:
    """
    函数功能与逻辑描述：
        类别代码层兜底：优先用原文关键词规则推断类别，覆盖 LLM 误判。
        优先级为「规则命中（如"打车"→交通）> LLM 合法类别 > 默认餐饮」。
        规则匹配按 CATEGORY_KEYWORDS 的字典插入顺序（餐饮/交通/住宿/购物/娱乐）短路返回，
        因此同一句话命中多类关键词时以字典序靠前者为准（如"打车去吃饭"取餐饮）。
        每次发生与 LLM 判断不一致的修正、或最终落到默认值时，都会打印一条带原文前 30 字的
        诊断日志，便于线上回溯误判样本。
    入参说明：
        raw_text (str)：用户原始消费文本，用于关键词匹配。
        llm_category (str)：LLM 输出的类别；可能为 None、带空白或不在白名单内。
    返回值说明：
        str：最终采用的类别，取值必在 VALID_CATEGORIES 内（餐饮/交通/住宿/购物/娱乐）。
    """
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in raw_text for k in keywords):
            if cat != str(llm_category or "").strip():
                print(f"[BILL] 类别修正：LLM={llm_category!r} 规则={cat!r} raw={raw_text[:30]!r}", flush=True)
            return cat
    llm_cat = str(llm_category or "").strip()
    if llm_cat in VALID_CATEGORIES:
        return llm_cat
    print(f"[BILL] 类别兜底：LLM={llm_cat!r} 非法且无关键词命中，默认餐饮 raw={raw_text[:30]!r}", flush=True)
    return "餐饮"


def _fix_date(raw_text: str, llm_date: str) -> str:
    """
    函数功能与逻辑描述：
        日期代码层兜底：以原文时间线索为准，杜绝 LLM 照抄 schema 示例里的旧日期。
        按固定优先级短路匹配：原文绝对日期（20xx-x-x）> "前天" > "昨天" > "X月X日" >
        "今天/现在/刚刚/今天早上/今天下午/今天中午" > 默认今天。
        已知近似：检查顺序上"前天"先于"昨天"，而"大前天"包含子串"前天"，
        因此"大前天"会被算作 2 天前（而非 3 天前）；这是当前的已知取舍，未做额外分支处理。
        "X月X日"分支固定使用**当前年份**，不处理跨年表述（如 12 月说"1月5日"）。
        原文完全没有时间线索时一律取今天，避免 LLM 输出的示例日期被写入。
        当结果与 LLM 给出的日期不一致、或走"今天"兜底时打印诊断日志。
    入参说明：
        raw_text (str)：用户原始消费文本，用于提取时间线索。
        llm_date (str)：LLM 输出的日期；可能为 None 或空串。
    返回值说明：
        str：最终采用的消费日期，格式固定为 YYYY-MM-DD（由 date.isoformat 或手工补零保证）。
    """
    today = date.today()
    m = re.search(r"(20\d{2})-(\d{1,2})-(\d{1,2})", raw_text)
    if m:
        fixed = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        if fixed != str(llm_date or ""):
            print(f"[BILL] 日期修正：LLM={llm_date!r} 原文绝对日期={fixed!r}", flush=True)
        return fixed
    if "前天" in raw_text:
        return (today - timedelta(days=2)).isoformat()
    if "昨天" in raw_text:
        return (today - timedelta(days=1)).isoformat()
    m = re.search(r"(\d{1,2})月(\d{1,2})日", raw_text)
    if m:
        return f"{today.year:04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    if any(k in raw_text for k in ("今天", "现在", "刚刚", "今天早上", "今天下午", "今天中午")):
        return today.isoformat()
    # 原文无任何时间线索 → 一律基准今天（防止 LLM 照抄 schema_example 里的示例日期）
    if str(llm_date or "") != today.isoformat():
        print(f"[BILL] 日期修正：LLM={llm_date!r} 原文无时间词 取今天={today.isoformat()}", flush=True)
    return today.isoformat()

def build_bill_prompt(state: BillState) -> tuple[str, str]:
    """
    函数功能与逻辑描述：
        组装记账链路的系统提示词与用户提示词。用户提示词按上下文完整度二选一：
        当 state 同时提供 city 与 month 时用 user_full.j2（含城市与月份，支持物价对标与归属月），
        否则退化为 user_simple.j2（仅原文与今天日期）。两个模板均注入 today_date，
        避免模型自己猜日期。raw_segments 为空时以空串作为原文，不抛异常。
    入参说明：
        state (BillState)：记账状态字典，读取 raw_segments（取首段作为原文）、
            city（城市，可为 None）、month（月份，可为 None）。
    返回值说明：
        tuple[str, str]：二元组 (系统提示词, 用户提示词)，系统提示词经 load_system_prompt 渲染过 today_date。
    """
    sys_prompt = load_system_prompt()
    today_str = date.today().isoformat()

    raw_segments = state.get("raw_segments", [])
    raw_text_val = raw_segments[0] if raw_segments else ""
    city = state.get("city")
    month = state.get("month")

    if city and month:
        user_prompt = render_full_template(
            raw_text=raw_text_val,
            today_date=today_str,
            city=city,
            month=month
        )
    else:
        user_prompt = render_simple_template(
            raw_text=raw_text_val,
            today_date=today_str
        )
    return sys_prompt, user_prompt


def bill_validate(parsed: dict):
    """
    函数功能与逻辑描述：
        记账链路的业务规则校验回调，交由 parse_json_output 在每轮解析后调用，
        返回错误文案即触发模型自我修正重试。
        两条规则：① 若 need_more_info 为 True，只要求 prompt 追问话术非空（其余字段允许缺失）；
        ② 否则 consume_date 必须存在且**精确到日**——仅输出年月（YYYY-MM）会被拒绝，
        并明确要求模型改为设置 need_more_info=true 向用户询问，禁止自行补齐残缺日期
        （这是为了不让"这个月"这类表述被落库成某个月的 1 号或当天）。
        注意本校验**不校验** amount 与 category 的合法性，那两项由 execute_node 的代码层兜底处理。
    入参说明：
        parsed (dict)：已解析并做过键名去空格的模型输出字典。
    返回值说明：
        Optional[str]：None 表示校验通过；返回字符串表示不通过，内容为面向模型的修正提示。
    """
    # 兼容"模型输出多个裸对象"：`_try_loads_merged` 对同构并列记录返回 **list**，
    #   语义等同 items 数组，此处先归一，避免 list 上没有 .get 而报错。
    if isinstance(parsed, list):
        parsed = {"need_more_info": False, "prompt": "", "items": parsed}

    # 追问分支：只要求有 prompt（其余字段允许缺失）
    if parsed.get("need_more_info") is True:
        if not parsed.get("prompt"):
            return "need_more_info=true 时必须输出 prompt 追问话术"
        return None
    # 完整记账分支：**逐笔**校验 consume_date（必须存在且精确到日）。
    #   注意 consume_date 位于 `items[].consume_date`，只读顶层会恒报"缺少 consume_date"
    #   并触发重试；同时兼容旧扁平结构（模型偶尔仍按旧格式输出）。
    items = parsed.get("items")
    if not isinstance(items, list) or not items:
        items = [parsed]
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            return f"items 第{i}条不是JSON对象"
        consume_date = it.get("consume_date", "")
        if not consume_date:
            return f"第{i}笔缺少 consume_date 字段，必须输出精确到 YYYY-MM-DD 的消费日期。"
        if re.fullmatch(DATE_YM, consume_date):
            return (f"第{i}笔日期禁止仅输出年月YYYY-MM，必须精确到 YYYY-MM-DD。"
                    "若用户没有提供具体几号，请设置 need_more_info=true 向用户询问准确日期，"
                    "不要自行填充残缺日期。")
    return None


# ===================== 节点1：核心执行节点【改造完成】 =====================
async def _settle_month_habit(data: dict) -> None:
    """
    函数功能与逻辑描述：
        记账成功后的月度习惯沉淀（**静默失败**，不影响主流程）。
        习惯沉淀原在 orchestrator 的 `_persist_bill_habit`（裸连 db ×2），M9 收口后
        **下沉到记账方自身**：先经 `habit.upsert` 工具（需 HABIT_WRITE 权限）写入 monthly_habit
        并读回当月聚合，再把聚合结果交给 memory.long_memory.upsert_month_habit_memory
        更新 pkl 语义层（用 asyncio.to_thread 卸载，避免磁盘/编码阻塞事件循环）。
        三重短路：amount/category/consume_date 任一缺失直接返回；工具调用失败（res.ok 为假）返回；
        读回的 habits 为空则不更新语义层。整个函数体被 try/except 包裹并吞掉一切异常——
        习惯沉淀是记账的附属动作，绝不能因其失败而让已成功的记账对外报错。
    入参说明：
        data (dict)：记账成功载荷的 data 字段，需含 amount、category、consume_date；
            月份由 consume_date 前 7 位截取得到。
    返回值说明：
        无（协程；副作用为写 monthly_habit 表并更新长期记忆 pkl；任何失败都静默忽略）。
    """
    try:
        amount = data.get("amount")
        category = data.get("category")
        consume_date = data.get("consume_date")
        if amount is None or not category or not consume_date:
            return
        month = str(consume_date)[:7]
        from mcpGateway.registry import get_registry
        res = await get_registry().invoke(
            "habit.upsert",
            {"month": month, "category": category, "amount": float(amount)},
            agent_id=BILL_AGENT_ID)
        if not res.ok:
            return
        habits = (res.data or {}).get("habits") or []
        if habits:
            from memory.long_memory import upsert_month_habit_memory
            await asyncio.to_thread(upsert_month_habit_memory, month, habits)
    except Exception:
        pass


async def execute_node(state: BillState):
    """
    函数功能与逻辑描述：
        记账链路的核心执行节点（图中第 1 步），负责「抽取 → 多层兜底修正 → 落库」全流程，
        所有出口都通过 Command(goto="reply_node") 进入收尾节点，不存在条件边。
        执行顺序：① 组装提示词并合并全部 raw_segments（不合并会丢失 orchestrator 追加的金额上下文）；
        ② 三层前置拦截——空输入、收入类内容、无金额数字（edit/delete 除外），均在调用 LLM 前完成，
        命中即直接返回标准载荷以省掉无效模型调用；
        ③ 调 parse_json_output 抽取，异常统一兜底转追问（1.8B 输出不稳定，硬失败体验更差）；
        ④ 分派 need_more_info / valid=false 两个分支；
        ⑤ 金额正则兜底——原文有数字而 LLM 金额不在原文数字集合内（含漏提返回 0）时回退原文首个数字，
        并保留负号（"-30元"是非法金额，不能被"修正"为正数）；
        ⑥ 类别与日期代码层兜底（_fix_category / _fix_date）；
        ⑦ 经 MCP 写入 bill 表，**代码层强制写入 task_id** 作为崩溃恢复对账依据（不依赖 LLM 生成 SQL）；
        ⑧ 落库成功则调用 _settle_month_habit 沉淀月度习惯（静默失败）。
        state 的读取面：raw_segments、operate_sub_type、task_id。数据不来自 A2A 消息拉取——
        外层 worker 组装 state 时已注入。
    入参说明：
        state (BillState)：记账状态字典。raw_segments 为待记账文本片段列表；
            operate_sub_type 取值 "edit"/"delete" 等，用于放宽"必须有金额"的拦截；
            task_id 用于写入 bill.task_id 供对账。
    返回值说明：
        Command：统一为 Command(update={"result": <JSON 字符串>}, goto="reply_node")。
            result 载荷固定五键 success / agent_type / msg / data / error：
            - success=True：msg="记账成功"，data 含 amount、category、consume_date、remark。
            - success=False：msg 取"输入内容为空"/"收入不记账"/"需要向用户补充询问信息"/
              "账单信息校验不通过"/"金额非法"/"数据库调用异常"/"账单入库操作失败"之一，
              追问场景的 data.prompt 为面向用户的追问话术，error 为 "need_more_info"。
    """
    # ⭐重点：数据直接从state读取，不再从a2a消息拉取
    # 外层worker组装state时已经注入：session_id, task_id, raw_segments, city, month
    sys_p, user_p = build_bill_prompt(state)
    raw_segments = state.get("raw_segments", [])
    # 合并全部片段：orchestrator可能为缺金额的片段追加了完整原文，仅取首段会丢失金额上下文
    raw_text = "\n".join(str(s) for s in raw_segments) if raw_segments else ""
    agent_id = BILL_AGENT_ID
    operate_sub_type = state.get("operate_sub_type") or ""
    print(f"[BILL] {time.strftime('%H:%M:%S')} execute_node 开始 raw={raw_text[:40]!r} op={operate_sub_type}", flush=True)

    # 空输入拦截
    if not raw_text.strip():
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "输入内容为空",
            "data": None,
            "error": "未获取到有效的账单描述信息"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # 前置启发式拦截：收入类内容（奖金/工资/退款/报销等）不入账，直接提示，省去无效LLM调用
    if _has_income_keyword(raw_text):
        print(f"[BILL] {time.strftime('%H:%M:%S')} 检测到收入类内容，不入账 raw={raw_text[:30]!r}", flush=True)
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "收入不记账",
            "data": None,
            "error": "收入类内容不予入账，本系统仅支持支出记账"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # 前置启发式拦截：新增记账必须有金额数字；无数字且非修改/删除操作 → 直接追问，省去无效LLM调用
    if not re.search(r"\d+(\.\d+)?", raw_text) and operate_sub_type not in ("edit", "delete"):
        print(f"[BILL] {time.strftime('%H:%M:%S')} 无金额数字，直接追问", flush=True)
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "需要向用户补充询问信息",
            "data": {"prompt": "没有识别到可记账的金额信息，请提供消费金额、内容和日期，例如：打车35元。"},
            "error": "need_more_info"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # ★M15（P3a 确定性驱动）：修改 / 删除走独立分支——**add 路径的后续代码一行不动**。
    #   为什么在这里分派而不是继续往下走：下面整段是"记账抽取 → INSERT"的确定性 workflow，
    #   edit/delete 需要的是"定位目标 → 部分更新 / 删除"，两者共享的只有提示词组装与金额兜底。
    #   ★操作类型由 orchestrator 判定后经 A2A 下发（`operate_sub_type`），子层不自行推断——
    #     这是"提议-执行分离"的落点（`设计/12` §2.2）。
    #   两条路径由 `BILL_FC_ENABLED` 选择：=1（默认）走**子层自主循环**（P3b：模型在裁剪过的
    #     工具可见面内自主定位与决策）；=0 走**确定性驱动**（P3a：规则定位 + 代码确认）。
    #     两者共用同一批工具、同一份落库语义，切换开关**不改变数据行为**（`设计/12` §4.7 三层降级）。
    # 显式 id（UI 点选产出"把 id=515 那笔改成 …"）→ 走 **P3a 确定性路径**：
    #   目标已由主键唯一确定，**不需要也不该**再交给模型"自主定位"——FC 路径下模型会把
    #   `bill.update` 的 tool_call 形态当**正文**输出（`{"name":"bill.update","parameters":{…}}`），
    #   工具并未真正执行：用户看到一段 JSON 而数据没改。确定性路径"给定 id 即改该笔"，
    #   行为可预期、可验收。自然语言指代（"把昨天那笔打车改成 50"）仍走 FC，自主定位的范式不丢。
    _explicit_id = bool(_EXPLICIT_ID_RE.search(raw_text))
    if operate_sub_type == "edit":
        if BILL_FC_ENABLED and not _explicit_id:
            return await _execute_fc(state, raw_text, agent_id, operate_sub_type)
        return await _execute_edit(state, raw_text, agent_id)
    if operate_sub_type == "delete":
        # 删除**恒走 P3a 确定性路径**，不受 BILL_FC_ENABLED 影响：
        #   删除不可逆，契约 R2 要求"先确认后删"；而子层自主循环把"要不要先确认"交给模型判断，
        #   模型会**跳过确认直接调 bill.delete**，真库被删且用户没机会反悔——安全优先，
        #   删除确认流程必须确定性执行。（edit 可逆、且需要模型在候选里自主定位，仍保留 FC 路径。）
        return await _execute_delete(state, raw_text, agent_id)

    try:
        parse_res = await parse_json_output(
            mcp_client.call_llm_base,
            sys_p,
            user_p,
            # schema 为 **`items` 数组**，与提示词 `system.md` 的"多条独立消费"承诺对齐：
            #   单笔时 `items` 恰好一个元素；`items: []` 表示待追问（need_more_info=true）。
            #   本示例同时是**必需字段的来源**（`parse_json_output` 会正则抽键名做校验），
            #   故 items 内层的 amount/category/consume_date/remark 必须出现在示例中。
            schema_example=json.dumps(
                {"need_more_info": False, "prompt": "", "valid": True,
                 "items": [{"amount": 200, "category": "餐饮",
                            "consume_date": date.today().isoformat(), "remark": "晚饭200"}]},
                ensure_ascii=False),
            validate=bill_validate,
            max_self_retry=3,
            agent_tag=BILL_AGENT_ID,
            ignore_missing_keys=["need_more_info"]
        )
    except Exception as e:
        # 兜底：LLM无法从文本中提取有效记账信息时，转追问而不是硬失败（1.8B模型输出不稳定）
        print(f"[BILL] {time.strftime('%H:%M:%S')} 解析失败，兜底转追问: {str(e)[:80]}", flush=True)
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "需要向用户补充询问信息",
            "data": {"prompt": "暂时没有识别到可记账的信息，请告诉我消费的金额、内容和日期，例如：打车35元。"},
            "error": "need_more_info"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # 用户信息不足，需要追问
    if parse_res.get("need_more_info"):
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "需要向用户补充询问信息",
            "data": {
                "prompt": parse_res["prompt"]
            },
            "error": "need_more_info"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # 兼容"模型输出**多个裸对象**"的形态：`_try_loads_merged` 对**同构并列记录**会返回
    #   **list**（刻意的，防止浅合并丢数据）。此时模型其实是"把两笔各写成一个对象、没套
    #   `items` 外壳"，语义上就等于 items 列表，故在此归一为标准结构。
    #   若不做这层兼容，`parse_json_output` 的必需字段校验会在 list 上找不到 `need_more_info`
    #   等键 → 触发重试 → 模型会**退化为追问日期**，两笔账一笔都落不了。
    if isinstance(parse_res, list):
        parse_res = {"need_more_info": False, "prompt": "", "valid": True, "items": parse_res}

    # 解析字段合法性校验
    if not parse_res.get("valid"):
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "账单信息校验不通过",
            "data": None,
            "error": "valid标记为false，无法录入账单"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # 抽取结果按 `items` **逐笔**处理。
    #   兼容策略：若模型仍按旧格式输出**裸单笔对象**（顶层直接含 amount/category），
    #   自动包成单元素列表 —— 模型不一定严格遵守 schema，不能让旧格式直接失败。
    raw_items = parse_res.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        if parse_res.get("amount") is not None or parse_res.get("category"):
            raw_items = [{
                "amount": parse_res.get("amount", 0),
                "category": parse_res.get("category", ""),
                "consume_date": parse_res.get("consume_date", ""),
                "remark": parse_res.get("remark") or raw_text,
            }]
        else:
            payload = {
                "success": False,
                "agent_type": BILL_AGENT_ID,
                "msg": "需要向用户补充询问信息",
                "data": {"prompt": "暂时没有识别到可记账的信息，请告诉我消费的金额、内容和日期，例如：打车35元。"},
                "error": "need_more_info"
            }
            return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # ===== 逐笔：片段化兜底 + 校验 =====
    # 各处兜底都传入**本笔片段**而非整句：金额兜底若取"整句首个数字"、类目/日期兜底若按
    #   "整句 → 唯一值"，多笔时会把第一笔的金额/类目串给第二笔；按笔作用域后天然隔离。
    prepared = []
    for idx, it in enumerate(raw_items):
        if not isinstance(it, dict):
            continue
        seg = str(it.get("remark") or "").strip() or raw_text   # 该笔片段；缺失时退回整句
        amount = it.get("amount", 0)
        category = it.get("category", "")
        consume_date = it.get("consume_date", "")

        # 金额正则兜底：LLM 可能编造原文没有的金额（如把45幻觉成200），以**本笔片段**数字为准
        try:
            llm_amount = float(amount)
        except (TypeError, ValueError):
            llm_amount = 0.0
        # 负号需保留："-30元"表示非法金额，不能被正则"修正"为正数30
        text_nums = [float(m.group(0)) for m in re.finditer(r"-?\d+(\.\d+)?", seg)]
        if text_nums and llm_amount not in text_nums:
            amount = text_nums[0]
            print(f"[BILL] {time.strftime('%H:%M:%S')} 第{idx + 1}笔金额修正："
                  f"LLM={llm_amount} 片段数字={text_nums} 取 {amount}", flush=True)

        if amount <= 0:
            # 任一笔非法 → **整批拒绝**（全回滚语义，避免"半笔账"）。
            # 文案按笔数区分：单笔沿用原文案（对用户更自然，也是既有测试契约）；
            # 多笔才说明"第几笔 + 共几笔 + 均未入库"，让用户明确整批都没有落库。
            total = len(raw_items)
            err_msg = ("消费金额必须大于0" if total == 1
                       else f"第{idx + 1}笔消费金额必须大于0（本次共{total}笔，均未入库）")
            payload = {
                "success": False,
                "agent_type": BILL_AGENT_ID,
                "msg": "金额非法",
                "data": {"amount": amount, "category": category},
                "error": err_msg
            }
            return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

        # 类别/日期代码层兜底：修正小模型语义误判（打车→交通、买书→购物）与日期照抄 schema 示例的问题
        category = _fix_category(seg, category)
        consume_date = _fix_date(seg, consume_date)

        prepared.append({"amount": amount, "category": category,
                         "consume_date": consume_date, "remark": seg})

    if not prepared:
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "需要向用户补充询问信息",
            "data": {"prompt": "暂时没有识别到可记账的信息，请告诉我消费的金额、内容和日期，例如：打车35元。"},
            "error": "need_more_info"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # 数据库写入
    # 用**单条多值 INSERT**（= 天然原子，全成功或全回滚）：`call_bill_sql` 每次都是独立的
    #   MCP 调用，跨调用无法共享事务，循环写会出现"第一笔成功、第二笔失败"的半笔账；
    #   单条多值语句由数据库保证原子性。
    # task_id 语义：**同一次请求的多笔共用同一个 task_id**（task = 一次用户请求）。
    #   `state["task_id"]` 由 worker 从 A2A 消息注入（`startup/bootstrap.py`），代码层保证写入，
    #   崩溃恢复按 task_id 反查这批账单即可。
    #   ⚠️ 请勿改为"每笔一个新 task_id"——那会破坏崩溃恢复的对账依据。
    try:
        placeholders = ", ".join(["(?, ?, ?, ?, ?)"] * len(prepared))
        sql = ("INSERT INTO bill (amount, category, consume_time, remark, task_id) "
               f"VALUES {placeholders}")
        params = []
        for p in prepared:
            params += [p["amount"], p["category"], p["consume_date"],
                       p["remark"], state.get("task_id")]
        sql_res = await mcp_client.call_bill_sql(agent_id, sql, params)
    except Exception as e:
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "数据库调用异常",
            "data": {"items": prepared},
            "error": str(e)
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # ========= 输出标准结构化JSON =========
    # ★多笔：`data.items` 为完整列表；同时保留 `amount`/`category` 等单笔字段（取首笔）
    #   以兼容尚未改造的消费方（如 collect_node 的既有渲染分支），并由 `count` 标明笔数。
    if "成功" in sql_res:
        payload = {
            "success": True,
            "agent_type": BILL_AGENT_ID,
            "msg": "记账成功",
            "data": {
                "items": prepared,
                "amount": prepared[0]["amount"],
                "category": prepared[0]["category"],
                "consume_date": prepared[0]["consume_date"],
                "remark": raw_text,
                "count": len(prepared),
            },
            "error": None
        }
    else:
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "账单入库操作失败",
            "data": {"items": prepared},
            "error": sql_res
        }

    # M9（D15 P0-5）：记账成功 → 月度习惯沉淀（静默失败，不影响主流程）
    # 多笔时**逐笔**沉淀：各笔可能跨月，需各自重算对应月份。
    if payload.get("success") is True:
        for p in payload["data"]["items"]:
            await _settle_month_habit(p)

    return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

# ===================== M15（P3a）：修改 / 删除的确定性驱动分支 =====================
# 设计依据：`设计/12` §4.2.4（工具契约）、§4.4（删除二次确认）、§4.6（失败矩阵）、
#          §4.7 第 3 层（FC 不通时的降级路径 = 本分支）。
# 一句话：**定位目标与二次确认用代码做（确定性），落库用工具做（SQL 写死在工具内）**。
#   FC 开启时（P3b）定位交给模型自主循环；本分支是 `BILL_FC_ENABLED=0` 时的保底执行路径，
#   两条路径**共用同一批工具与同一份落库语义**，所以切换开关不会改变数据行为。

# 「改成 X」类改写动词：用于从原文**确定性**提取新金额。
# ★为何优先于 LLM 抽取：实测 LLM 会把定位数字当成新值（"把 35 那笔改成 50" → 返回 35），
#   而"动词后的第一个数字"在中文里就是用户要改成的值，确定性规则在此更可靠。
# ★表的定义在 `bill_target`（定位规则同样要用它把新值数字从"定位线索"里剔除）——此处只引用，
#   避免出现两份动词表而漂移。
_CHANGE_VERB_RE = bill_target.CHANGE_VERB_RE


def _category_hint(text: str) -> str | None:
    """
    函数功能与逻辑描述：
        从原文提取**品类线索**，供目标定位使用。与 `_fix_category` 的关键差异：本函数
        **没有默认兜底**——`_fix_category` 在无线索时会返回"餐饮"（落库需要合法值），
        而定位场景下把"餐饮"当成线索会让"把昨天那笔改成 50"错误地去找餐饮账。
        故此处只用 `CATEGORY_KEYWORDS` 做纯关键词命中，命中不了就返回 None。
    入参说明：
        text (str)：用户原文。
    返回值说明：
        str | None：命中的品类（餐饮/交通/住宿/购物/娱乐）；无关键词命中时返回 None。
    """
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return cat
    return None


def _ask_payload(prompt: str) -> Command:
    """
    函数功能与逻辑描述：
        构造「向用户追问」的标准出口载荷（`success=False` + `error=need_more_info` +
        `data.prompt`），供 edit/delete 分支的所有"信息不足"出口复用，避免各处手写导致
        字段口径漂移（`build_user_display_text` 依赖 `data.prompt` 渲染追问）。
    入参说明：
        prompt (str)：面向用户的追问话术（可含候选清单换行）。
    返回值说明：
        Command：Command(update={"result": <JSON 串>}, goto="reply_node")。
    """
    payload = {
        "success": False,
        "agent_type": BILL_AGENT_ID,
        "msg": "需要向用户补充询问信息",
        "data": {"prompt": prompt},
        "error": "need_more_info"
    }
    return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")


def _fail_payload(msg: str, error: str, data: dict | None = None) -> Command:
    """
    函数功能与逻辑描述：
        构造「业务性失败」标准出口载荷（`success=False`、`error` 为具体原因、不含 prompt），
        供"目标不存在 / 金额非法 / 写库失败"等出口复用。与 `_ask_payload` 的区别是语义：
        追问是"请再给我信息"，失败是"这件事做不成"——编排侧对两者的后续处理不同
        （追问计轮次、失败不计数）。
    入参说明：
        msg (str)：面向用户的一句话结论（如"账单更新失败"）。
        error (str)：结构化失败原因（进载荷 error 字段，供编排/日志归因）。
        data (dict | None)：附带的上下文数据，默认 None。
    返回值说明：
        Command：Command(update={"result": <JSON 串>}, goto="reply_node")。
    """
    payload = {
        "success": False,
        "agent_type": BILL_AGENT_ID,
        "msg": msg,
        "data": data,
        "error": error
    }
    return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")


def _locate_failed_payload(reason: str, bills: list[dict], action: str) -> Command:
    """
    函数功能与逻辑描述：
        把目标定位的失败原因翻译成面向用户的出口：**窗口为空**→ 失败（无账可操作）；
        **歧义**→ 追问并列出候选清单；**未命中**→ 追问并列出最近账单（提示用户换一种说法）。
        三种出口都给出可执行的下一步，而不是笼统的"我不明白"（`设计/12` §4.6）。
    入参说明：
        reason (str)：`bill_target.locate_bill` 返回的失败原因常量。
        bills (list[dict])：当前可操作窗口内的账单清单（用于渲染候选）。
        action (str)：动作名（"修改"/"删除"），用于组织失败文案。
    返回值说明：
        Command：失败或追问的标准载荷 Command。
    """
    if reason == bill_target.REASON_EMPTY:
        return _fail_payload("没有可操作的账单", f"最近没有任何账单记录，无法{action}")
    if reason == bill_target.REASON_AMBIGUOUS:
        return _ask_payload(
            f"找到多笔可能的账单，您是指哪一笔？\n{bill_target.describe_candidates(bills)}\n"
            "（可直接说「第 2 条」）")
    return _ask_payload(
        f"没找到符合条件的账单，请说得更具体一些（例如：把昨天那笔打车{action}…）。\n"
        f"最近的账单：\n{bill_target.describe_candidates(bills)}")


# 显式 id 线索（"id=9 那笔" / "编号 9"）：命中即视为**用户已明确指定目标**（UI 点选也产出这种输入），
# 此时定位不受"最近 N 笔"窗口约束——窗口约束的本意是防"自然语言指代"的歧义，而非限制显式指定。
_EXPLICIT_ID_RE = re.compile(r"(?:id|ID|编号)\s*[=:：]?\s*\d+")


async def _fetch_recent_bills(limit: int | None = None) -> list[dict] | None:
    """
    函数功能与逻辑描述：
        读账单清单（经 `bill.recent_list` 工具，而不是裸连 DB——保持"Agent 只经网关访问数据"的红线）。
        工具返回 `ok=False`（权限/超时/远端错误）或抛异常时统一返回 None，由调用方转成"稍后再试"的追问。
        `limit` 默认 `RECENT_BILL_WINDOW`（自然语言指代的窗口）；当调用方检测到**显式 id** 线索时，
        传 `RECENT_BILL_MAX` 拉宽清单——否则"把 id=9 那笔改成 50"会在窗口外查不到目标而误报"没找到"。
    入参说明：
        limit (int | None)：本次取数条数，默认 None 表示取窗口下限 `RECENT_BILL_WINDOW`
            （工具侧仍会 clamp 到 `[RECENT_BILL_WINDOW, RECENT_BILL_MAX]`）。
    返回值说明：
        list[dict] | None：账单清单（id 倒序）；通道不可用时返回 None（**与"清单为空"区分**）。
    """
    try:
        from mcpGateway.registry import get_registry
        res = await get_registry().invoke(
            "bill.recent_list", {"limit": limit or RECENT_BILL_WINDOW}, agent_id=BILL_AGENT_ID)
        if not res.ok:
            return None
        return list(res.data or [])
    except Exception:
        return None


async def _fetch_bill_by_id(bill_id: int) -> dict | None:
    """
    函数功能与逻辑描述：
        按主键直取单行账单（经 `bill_tools.get_bill_by_id`，**不经工具注册**——这是节点内部
        定位用的读数，模型无需也不该看见）。用于「显式 id 指认」的定位：用户给出主键即
        **唯一无歧义的目标**，不应受"最近 N 笔"窗口约束——否则 UI 面板（窗口 200）里选中
        #245 再发"删掉 id=245 那笔"时，定位侧只拉 20 笔会把目标落在窗口外，回答"没找到
        符合条件的账单"。
        异常/查无此行一律返回 None，由调用方回落窗口规则（仍可能靠金额/日期命中）。
    入参说明：
        bill_id (int)：账单主键 id。
    返回值说明：
        dict | None：命中返回该行（不含 task_id）；查无此行或读数异常返回 None。
    """
    try:
        from mcpGateway import bill_tools
        return bill_tools.get_bill_by_id(int(bill_id))
    except Exception:
        return None


async def _recalc_habits(months) -> None:
    """
    函数功能与逻辑描述：
        改删成功后重算受影响月份的月度习惯（**静默失败**，照抄 `_settle_month_habit` 的
        「不阻断主流程」口径）。为什么必须整月重算而不是增量：`habit.upsert` 是累加语义，
        表达不了"改小/删除"（`设计/12` §4.5）。
        调用方传入一个月份集合即可（edit 改期跨月时会同时含新旧两个月；None/空串自动剔除）。
    入参说明：
        months：受影响月份集合（任意可迭代，元素形如 "2026-09"；None/空串会被忽略）。
    返回值说明：
        无（副作用：逐月调用 `habit.recalc`；任何失败都被吞掉，仅影响派生数据新鲜度）。
    """
    try:
        targets = sorted({str(m) for m in months if m})
        if not targets:
            return
        from mcpGateway.registry import get_registry
        reg = get_registry()
        for month in targets:
            await reg.invoke("habit.recalc", {"month": month}, agent_id=BILL_AGENT_ID)
    except Exception:
        pass


# ===================== M15（P3b）：子层自主循环（bounded autonomy）=====================
# 设计依据：`设计/12` §4.2.1–§4.2.3（循环形态 / 可见面 / 熔断 / 写幂等闸门）、§4.3（FC 通道）。
# 一句话：**操作类型决定工具可见面（顶层裁决），可见面之内由模型自主（子层决策），
#          而落库仍是确定性 workflow（SQL 写死在工具内）**。
# 与 P3a 的关系：本循环与 `_execute_edit` / `_execute_delete` **共用同一批工具、同一份落库语义**，
# 差异只在"谁决定改哪一条、改成什么"——模型（本循环）还是规则（P3a）。
# 开关：`BILL_FC_ENABLED=0` 时退回 P3a，功能不丢（`设计/12` §4.7 三层降级）。

_FC_WRITE_TOOLS = frozenset({"bill.add", "bill.update", "bill.delete"})
# observation 截断上限：工具结果要回填给模型，无节制会把上下文预算（D6）吃光
_FC_OBSERVATION_MAX_CHARS = 2000


def _tool_tags(operate_sub_type: str) -> tuple[str, ...]:
    """
    函数功能与逻辑描述：
        把操作类型映射为工具标签，是**运行时可见面裁剪**的输入（`设计/12` §4.2.2）。
        映射表决定"模型在本类任务里能看到什么"：
        - `edit`   → 读 + 改（**看不到 `bill.delete`**）
        - `delete` → 读 + 删（**看不到 `bill.update`**）
        - `add`    → 只有新增（选无可选）
        兜底返回 `("op:read",)`：未知操作类型只给只读面——宁可让模型多查一次，也不敢放开写面。
    入参说明：
        operate_sub_type (str)：操作类型（add / edit / delete）。
    返回值说明：
        tuple[str, ...]：标签元组，直接喂给 `registry.to_openai_schema(agent_id, tags=...)`。
    """
    return {
        "add": ("op:add",),
        "edit": ("op:read", "op:edit"),
        "delete": ("op:read", "op:delete"),
    }.get(operate_sub_type, ("op:read",))


def _tool_observation(call_id: str, payload) -> dict:
    """
    函数功能与逻辑描述：
        把工具结果包装成 `role=tool` 的 observation 消息：JSON 序列化 + 截断 + 关联 `tool_call_id`。
        ★工具**失败也走这里**（以 `{"ok": false, "error": ...}` 回填）而不是抛异常终止循环——
        这是"失败作为 observation"的口径（`设计/12` §4.2.3）：让模型看到失败原因并自行修正
        （改参数 / 换目标 / 转追问），比直接把循环崩掉更符合自主循环的设计意图。
        `default=str` 兜底防非 JSON 类型（如 sqlite Row）导致序列化失败而中断循环。
    入参说明：
        call_id (str)：该次工具调用的 id（与 assistant 消息里的 tool_calls 对应）。
        payload：工具返回数据（任意可 JSON 化对象）。
    返回值说明：
        dict：形如 {"role": "tool", "tool_call_id": ..., "content": <JSON 串>} 的消息字典。
    """
    try:
        content = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        content = str(payload)
    return {"role": "tool", "tool_call_id": call_id,
            "content": content[:_FC_OBSERVATION_MAX_CHARS]}


def _fc_finalize(text: str, write_result: dict | None) -> Command:
    """
    函数功能与逻辑描述：
        循环终态（模型不再产生 `tool_calls`）的出口。★**唯一判定成功的依据是"写工具真的成功过"**
        （`write_result` 非空）——绝不允许凭模型文本报成功（`设计/12` §6 红线 7：实测模型在没有
        对应工具时会回答"已记录"，属谎报成功；若这里信了文本，用户会以为账改了其实没改）。
        - 有成功写操作 → 成功载荷（`delete` / `update` / `add` 各自组织 data 字段），
          并把模型的一句话总结挂到 `data.reply`（供 UI 展示，不影响结构化字段的消费方）；
        - 无成功写操作 → **追问**（用模型的话术当 prompt；模型没说话时给兜底话术）。
    入参说明：
        text (str)：模型本轮正文（可能为空串）。
        write_result (dict | None)：最近一次**成功**的写工具结果
            （形如 {"tool": "bill.update", "data": {...}, "args": {...}}）；从未成功则传 None。
    返回值说明：
        Command：成功或追问的标准出口 Command。
    """
    if write_result is None:
        return _ask_payload((text or "").strip()
                            or "请把要修改或删除的内容说得更具体一点。")
    tool = write_result["tool"]
    data = write_result.get("data") or {}
    args = write_result.get("args") or {}
    if tool == "bill.delete":
        msg, out = "账单删除成功", {"id": data.get("id"), "deleted": data.get("deleted")}
    elif tool == "bill.update":
        msg, out = "账单修改成功", {
            "id": data.get("id"), "changed": data.get("changed") or {},
            "amount": args.get("amount"), "category": args.get("category"),
            "consume_date": args.get("consume_date")}
    else:
        msg, out = "记账成功", {
            "id": data.get("id"), "amount": args.get("amount"),
            "category": args.get("category"), "consume_date": args.get("consume_date")}
    if (text or "").strip():
        out["reply"] = text.strip()
    payload = {"success": True, "agent_type": BILL_AGENT_ID, "msg": msg,
               "data": out, "error": None}
    return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")


async def _recalc_after_fc(tool: str, data: dict, args: dict) -> None:
    """
    函数功能与逻辑描述：
        自主循环里写操作成功后的派生数据修复（与 P3a 共用 `_recalc_habits`，静默失败）。
        月份来源按可用信息分三种情形：
        ① `update` 改了日期 → `changed["consume_date"]` 的 `[旧, 新]` 直接给出**两个月**；
        ② `delete` → 用返回快照里的 `consume_time`；
        ③ `update` 未改日期 → 拿 id 回清单里查它当前所属月份（其余信息无从得知原月份）。
    入参说明：
        tool (str)：写工具名。
        data (dict)：工具返回数据。
        args (dict)：本次工具入参（当前未直接使用，保留以便将来扩展）。
    返回值说明：
        无（副作用：调用 `habit.recalc`；任何失败被静默吞掉）。
    """
    months: set[str] = set()
    changed = data.get("changed") or {}
    if "consume_date" in changed:
        months |= {str(v)[:7] for v in (changed.get("consume_date") or []) if v}
    elif tool == "bill.delete":
        months.add(str((data.get("deleted") or {}).get("consume_time") or "")[:7])
    else:
        for row in (await _fetch_recent_bills() or []):
            if str(row.get("id")) == str(data.get("id")):
                months.add(str(row.get("consume_time") or "")[:7])
                break
    await _recalc_habits(months)


async def _execute_fc(state: BillState, raw_text: str, agent_id: str,
                      operate_sub_type: str) -> Command:
    """
    函数功能与逻辑描述：
        子层目标消解循环（P3b）。形态 = 标准 tool loop：
        `模型 → tool_calls → 引擎执行 → 回填 observation → 再决策`，**无 tool_calls 即结束**
        （追问也因此不需要专门工具：模型不调工具、直接输出话术即可）。
        四道护栏（缺一不可，`设计/12` §4.2.3）：
        ① **可见面裁剪**：工具清单由 `operate_sub_type` 经 tags 过滤得到，"改着改着顺手删了"
           在能力层面即不可能（代价是"改→删"切换须交回顶层重规划）；
        ② **有界**：最多 `FC_MAX_TURNS` 轮，达上限转追问（防无限试探）；
        ③ **写幂等闸门**：单次任务内同一写工具只允许成功一次，重复调用立即拦截并终止
           （防模型重复调 `bill.add` 造成重复记账，破坏 D2 成果）；
        ④ **失败即 observation**：工具报错回填给模型而非抛出，连续失败自然走到熔断转追问。
        另：每轮 tool 调用由引擎自动落 `tool.{name}` span（现成能力），失败可定位到"第几轮调了什么"。
        通道异常（`call_llm_fc` 抛 RuntimeError 等）不向图外抛，一律转追问（把"通道坏了"如实告知用户）。
    入参说明：
        state (BillState)：需含 task_id（工具对账依据）与 session_id。
        raw_text (str)：用户原文。
        agent_id (str)：调用主体（bill_agent）。
        operate_sub_type (str)：操作类型（edit / delete）。
    返回值说明：
        Command：成功载荷（含 `data.reply`）或追问载荷；任何异常都收敛为追问，不抛到图外。
    """
    from mcpGateway.registry import get_registry
    reg = get_registry()
    tools = reg.to_openai_schema(agent_id, tags=_tool_tags(operate_sub_type))

    sys_p = load_fc_system_prompt()
    user_p = f"操作类型：{operate_sub_type}\n用户原话：{raw_text}"
    messages: list[dict] = []
    executed_write: set[str] = set()
    write_result: dict | None = None

    for _turn in range(FC_MAX_TURNS):
        try:
            out = await mcp_client.call_llm_fc(agent_id, sys_p, user_p,
                                               tools=tools, extra_messages=messages or None)
        except Exception as e:
            print(f"[BILL] FC 通道异常，转追问: {str(e)[:120]}", flush=True)
            return _ask_payload("改删通道暂时不可用，请稍后再试，或换一种说法。")

        text = str(out.get("text") or "")
        calls = out.get("tool_calls") or []
        if not calls:
            # 模型不再调工具 → 终态（成功与否只看"写工具是否真的成功过"）
            return _fc_finalize(text, write_result)

        # 回填 assistant 消息（含本轮 tool_calls），供下一轮模型看到自己刚才调了什么
        messages.append({
            "role": "assistant", "content": text,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"],
                                         "arguments": json.dumps(c["arguments"],
                                                                 ensure_ascii=False)}}
                           for c in calls],
        })

        for call in calls:
            name, args = call["name"], call["arguments"]
            if name in _FC_WRITE_TOOLS and name in executed_write:
                print(f"[BILL] 写幂等闸门拦截重复调用: {name}", flush=True)
                return _fail_payload("操作未重复执行",
                                     f"模型重复调用写工具 {name}，已拦截",
                                     {"tool": name})
            try:
                res = await reg.invoke(name, args, agent_id=agent_id,
                                       task_id=state.get("task_id"))
                if res.ok:
                    if name in _FC_WRITE_TOOLS:
                        executed_write.add(name)
                        write_result = {"tool": name, "data": res.data, "args": args}
                        await _recalc_after_fc(name, res.data or {}, args)
                    messages.append(_tool_observation(call["id"], res.data))
                else:
                    err = str(getattr(res.error, "message", "") or "工具执行失败")
                    messages.append(_tool_observation(call["id"], {"ok": False, "error": err}))
            except Exception as e:
                # 工具异常同样回填为 observation：循环内不崩，让模型看到失败并自行调整
                messages.append(_tool_observation(call["id"], {"ok": False, "error": str(e)}))

    print("[BILL] FC 循环达上限，转追问", flush=True)
    return _ask_payload("请把要修改或删除的内容说得更具体一点，例如：把昨天那笔打车改成 50。")


async def _execute_edit(state: BillState, raw_text: str, agent_id: str) -> Command:
    """
    函数功能与逻辑描述：
        修改账单的**确定性驱动**执行路径（P3a）。四步：
        ① **定位目标**：读最近 N 笔 → `bill_target.locate_bill` 用确定性规则定位
           （序号/主键 > 金额 > 金额+属性 > 属性 > 指代词），歧义或未命中都转追问；
        ② **解析新值**——★政策：**只有金额可能来自模型，品类与日期一律只来自原文线索**。
           金额优先取「改写动词 + 数字」（确定性，防模型把定位数字当新值），拿不到才花一次
           模型调用抽取；品类取 `_category_hint`（纯关键词）、日期取 `date_hint`（确定性线索）。
           这样做的原因：模型对"改金额"的话里同时输出的品类/日期是**顺带推断**的噪声，
           一旦采纳就会把用户没提到的字段改掉，属不可逆误伤；
        ③ **落库**：经 `bill.update` 工具（部分更新，SQL 写死在工具内），`task_id` 由引擎注入；
        ④ **派生一致性**：成功后重算受影响月份（含改期跨月的新月），静默失败。
        三个"没有新值"出口、金额≤0 出口、目标不存在出口都按 `设计/12` §4.6 矩阵返回。
    入参说明：
        state (BillState)：需含 task_id（对账）与 session_id。
        raw_text (str)：合并后的用户原文（定位与新值都从这里提取）。
        agent_id (str)：调用主体（bill_agent），用于工具鉴权。
    返回值说明：
        Command：统一 Command(update={"result": <JSON 串>}, goto="reply_node")，
            载荷语义见模块头 execute_node 的说明（成功 data 含 id/changed）。
    """
    # ★显式 id（"id=245 那笔"）**按主键直取该行**，完全不受窗口约束（与 delete 侧同一口径）：
    #   主键是无歧义指认；窗口只用于防止"自然语言指代"命中错行。
    id_m = _EXPLICIT_ID_RE.search(raw_text)
    target: dict | None = None
    reason = ""
    bills = await _fetch_recent_bills(RECENT_BILL_MAX if id_m else None)
    if bills is None:
        return _ask_payload("暂时查不到最近的账单，请稍后再说一次，例如：把昨天那笔打车改成 50。")
    if id_m:
        target = await _fetch_bill_by_id(int(re.findall(r"\d+", id_m.group(0))[0]))
    if target is None:
        target, reason = bill_target.locate_bill(raw_text, bills, category=_category_hint(raw_text))
    if target is None:
        return _locate_failed_payload(reason, bills, "修改")

    # ② 新值解析（金额：动词后数字 > 模型抽取；品类/日期：只认原文线索）
    new_amount: float | None = None
    verb_match = _CHANGE_VERB_RE.search(raw_text)
    if verb_match:
        new_amount = float(verb_match.group(1))
    new_category = _category_hint(raw_text)
    new_date = bill_target.extract_new_date(raw_text)

    if new_amount is None:
        # 确定性路径拿不到金额 → 才花一次模型调用（此处沿用 add 路径的提示词与校验口径）
        try:
            sys_p, user_p = build_bill_prompt(state)
            parsed = await parse_json_output(
                mcp_client.call_llm_base, sys_p, user_p,
                # schema 与提示词对齐为 `items` 数组：本路径是 **edit 的单笔兜底**（只关心
                #   "要改成的新值"），故取 items 首笔消费。若沿用旧的顶层 amount schema，
                #   模型按新提示词输出 `{"items":[...]}` 时下面的取值会拿不到 → 新金额静默失效。
                schema_example=json.dumps(
                    {"need_more_info": False, "prompt": "", "valid": True,
                     "items": [{"amount": 200, "category": "餐饮",
                                "consume_date": date.today().isoformat(), "remark": "改成200"}]},
                    ensure_ascii=False),
                validate=bill_validate, max_self_retry=3, agent_tag=BILL_AGENT_ID,
                ignore_missing_keys=["need_more_info", "items"])
        except Exception:
            parsed = {}
        # ★兼容两种形态：模型输出多个裸对象时 `_try_loads_merged` 会返回 list（语义等同 items）
        if isinstance(parsed, list):
            parsed = {"need_more_info": False, "prompt": "", "items": parsed}
        if parsed.get("need_more_info"):
            return _ask_payload(parsed.get("prompt") or "请说明要改成什么，例如：金额改成 50。")
        # edit 只取**首笔**的新值；同时兼容旧扁平格式（顶层直接含 amount）
        _items = parsed.get("items") if isinstance(parsed.get("items"), list) else []
        _first = _items[0] if _items and isinstance(_items[0], dict) else {}
        try:
            llm_amount = float(_first.get("amount") if _first else parsed.get("amount"))
        except (TypeError, ValueError):
            llm_amount = None
        # ★防"把定位数字当新值"：模型给出的金额若与目标行原金额相同，视为未提供新值
        if llm_amount is not None and llm_amount != float(target.get("amount") or 0):
            new_amount = llm_amount

    if new_amount is None and new_category is None and new_date is None:
        return _ask_payload("没听清要改成什么，请说明新值，例如：金额改成 50。")
    if new_amount is not None and new_amount <= 0:
        return _fail_payload("金额非法", "消费金额必须大于0", {"id": target.get("id")})

    # ③ 落库（部分更新；只带**真正要改**的字段）
    #   ★与目标行原值相同的线索视为"指代描述"而非"要改成的值"——例如"把昨天那笔打车改成 50"里
    #   的"打车"只是用来指认账目，不该被当成"把品类改成交通"。故同值不下发，入参保持最小。
    args: dict = {"id": int(target["id"])}
    if new_amount is not None:
        args["amount"] = new_amount
    if new_category is not None and new_category != target.get("category"):
        args["category"] = new_category
    if new_date is not None and new_date != str(target.get("consume_time") or "")[:10]:
        args["consume_date"] = new_date
    try:
        from mcpGateway.registry import get_registry
        res = await get_registry().invoke("bill.update", args, agent_id=agent_id,
                                          task_id=state.get("task_id"))
    except Exception as e:
        return _fail_payload("账单更新失败", str(e), {"id": target.get("id")})
    if not res.ok:
        err = str(getattr(res.error, "message", "") or "")
        return _fail_payload("账单更新失败", err or "更新未生效", {"id": target.get("id")})

    changed = (res.data or {}).get("changed") or {}
    # ④ 派生一致性：重算旧月与新日期的月份（改期跨月时是两个月）
    await _recalc_habits({str(target.get("consume_time") or "")[:7], str(new_date or "")[:7]})

    data = {
        "id": target.get("id"),
        "amount": new_amount if new_amount is not None else target.get("amount"),
        "category": new_category or target.get("category"),
        "consume_date": new_date or str(target.get("consume_time") or "")[:10],
        "changed": changed,
        "remark": target.get("remark"),
    }
    payload = {
        "success": True,
        "agent_type": BILL_AGENT_ID,
        "msg": "账单修改成功",
        "data": data,
        "error": None
    }
    return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")


async def _execute_delete(state: BillState, raw_text: str, agent_id: str) -> Command:
    """
    函数功能与逻辑描述：
        删除账单的**确定性驱动**执行路径（P3a），核心是**二次确认**（`设计/12` §4.4）：
        第 1 轮（"删掉昨天那笔打车"）：定位目标 → 把待删行快照写入 `bill_delete` 草稿槽 →
                                     **追问确认**（列出账单摘要让人核对），本轮绝不删除；
        第 2 轮（"确认/嗯/删吧"）：命中确认词**且**草稿槽存在 → 才真正执行 `bill.delete`。
        安全要点：① 确认判断叠加"草稿存在"这一前提，避免用户随口一句"可以"触发删除；
        ② 未确认就换话题 → 走定位轮并**覆盖**草稿，不会误删（失败矩阵"未确认就换话题 → 不删"）；
        ③ 执行前不信任草稿、由工具做"先读存在性"，目标已消失则失败且清空草稿。
        草稿的读面：优先用下发的 `session_memory.draft_store` 快照；写入与清空走
        `get_session_memory(...).set_draft/clear_draft`（快照是浅拷贝，就地改不会写回全局）。
    入参说明：
        state (BillState)：需含 session_id（草稿归属）与 task_id（对账）。
        raw_text (str)：本轮用户原文（可能只是"确认"两个字）。
        agent_id (str)：调用主体（bill_agent）。
    返回值说明：
        Command：统一 Command(update={"result": <JSON 串>}, goto="reply_node")；
            追问轮为 need_more_info + data.prompt（含待删账单摘要）。
    """
    from memory.short_memory import get_session_memory
    mem = get_session_memory(state.get("session_id"))
    draft = ((state.get("session_memory") or {}).get("draft_store") or {}).get("bill_delete") \
        or mem.get_draft("bill_delete")

    # ① 确认轮：草稿在 + 本轮命中确认词 → 执行删除（唯一会真正删除的出口）
    if draft and bill_target.is_confirm(raw_text):
        bill_id = draft.get("id")
        try:
            from mcpGateway.registry import get_registry
            res = await get_registry().invoke("bill.delete", {"id": int(bill_id)},
                                              agent_id=agent_id, task_id=state.get("task_id"))
        except Exception as e:
            mem.clear_draft("bill_delete")
            return _fail_payload("账单删除失败", str(e), {"id": bill_id})
        mem.clear_draft("bill_delete")     # 不论成败都清草稿：避免陈旧草稿被下一句"确认"复用
        if not res.ok:
            err = str(getattr(res.error, "message", "") or "")
            return _fail_payload("账单删除失败", err or "删除未生效", {"id": bill_id})
        await _recalc_habits({str(draft.get("consume_time") or "")[:7]})
        payload = {
            "success": True,
            "agent_type": BILL_AGENT_ID,
            "msg": "账单删除成功",
            "data": {"id": bill_id,
                     "deleted": (res.data or {}).get("deleted") or draft},
            "error": None
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # ② 定位轮：定位目标 → 写草稿 → 追问确认（本轮绝不删）
    # 显式 id（"id=245 那笔"）**按主键直取该行**，完全不受窗口约束：用户给出主键即
    #   唯一无歧义的目标；窗口约束的本意是防"自然语言指代"的歧义，而非限制显式指定。
    #   若仍走窗口，UI 面板（窗口 200）里选中 #245 后发送本句、定位侧只拉 20 笔，
    #   目标会落在窗口外 → 回答"没找到符合条件的账单"。
    id_m = _EXPLICIT_ID_RE.search(raw_text)
    target: dict | None = None
    reason = ""
    if id_m:
        target = await _fetch_bill_by_id(int(re.findall(r"\d+", id_m.group(0))[0]))
    bills = await _fetch_recent_bills(RECENT_BILL_MAX if id_m else None)
    if bills is None:
        return _ask_payload("暂时查不到最近的账单，请稍后再说一次，例如：删掉昨天那笔打车。")
    if target is None:
        target, reason = bill_target.locate_bill(raw_text, bills, category=_category_hint(raw_text))
    if target is None:
        return _locate_failed_payload(reason, bills, "删除")

    mem.set_draft("bill_delete", {
        "id": target.get("id"), "amount": target.get("amount"),
        "category": target.get("category"), "consume_time": target.get("consume_time"),
        "remark": target.get("remark"),
    })
    return _ask_payload(f"确认删除这笔吗？{bill_target.describe_bill(target)}（回复「确认」即删除）")


# ===================== 节点2：回传结果给调度【完全无需改动】 =====================
async def reply_node(state: BillState):
    """
    函数功能与逻辑描述：
        记账链路的收尾节点（图中第 2 步）：把 execute_node 写入 state["result"] 的 JSON 字符串
        原样经 A2A 总线回传给编排器——回传键用 task_id（send_result 的唯一路由键），
        使等待中的编排器被唤醒并领取结果。
        本节点为**同步投递、不 await**（send_result 是同步方法），且不做任何结果加工：
        序列化在 execute_node 已完成，这里只做搬运。
        执行后无条件 goto=END 结束子图。
    入参说明：
        state (BillState)：记账状态字典，需含 task_id 与 result（由 execute_node 写入）。
    返回值说明：
        Command：Command(update={}, goto=END)。不带任何 state 更新——结果已通过 A2A 旁路回传，
            不走 graph state 参与后续节点。
    """
    a2a_bus.send_result(
        task_id=state["task_id"],
        result=state["result"]
    )
    return Command(update={}, goto=END)


# 导出节点：删除recv_task_node
__all__ = [
    "execute_node",
    "reply_node",
    "BILL_AGENT_ID"
]
