import asyncio
import json
import re
import time
from datetime import date, timedelta
from langgraph.types import Command
from langgraph.graph import END
from agents.bill_agent.state import BillState
from agents.bill_agent.prompts.prompt_loader import load_system_prompt, render_full_template, render_simple_template
from agentCore.parsers.json_parser import parse_json_output
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
    "交通": ["打车", "公交", "地铁", "加油", "停车", "出租", "网约车", "滴滴", "打的",
              "高铁", "火车", "机票", "飞机", "高速", "骑车", "单车", "出行", "通勤"],
    "住宿": ["酒店", "民宿", "住宿", "宾馆", "房费", "租房", "公寓", "青旅", "旅馆", "开房"],
    "购物": ["衣服", "裤子", "裙子", "鞋子", "鞋", "数码", "日用品", "百货", "采购", "买书",
              "书本", "文具", "化妆品", "超市", "商场", "淘宝", "京东", "网购", "包包", "包",
              "手机", "电脑", "家电", "日杂", "日化"],
    "娱乐": ["电影", "KTV", "ktv", "游乐园", "演出", "桌游", "游戏", "门票", "唱歌",
              "蹦迪", "剧本杀", "看戏", "影院", "游乐"],
}


def _has_income_keyword(text: str) -> bool:
    return any(k in text for k in INCOME_KEYWORDS)


def _fix_category(raw_text: str, llm_category: str) -> str:
    """
    类别代码层兜底：优先用原文关键词规则推断类别，覆盖 LLM 误判。
    规则命中（如"打车"→交通）> LLM 合法类别 > 默认餐饮。
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
    日期代码层兜底：以原文时间线索为准，杜绝 LLM 照抄 schema 示例旧日期。
    优先级：原文绝对日期 > 前天/昨天 > X月X日 > 今天等相对词 > 默认今天。
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
    # 追问分支：只要求有 prompt（其余字段允许缺失）
    if parsed.get("need_more_info") is True:
        if not parsed.get("prompt"):
            return "need_more_info=true 时必须输出 prompt 追问话术"
        return None
    # 完整记账分支：consume_date 必须存在且精确到日
    consume_date = parsed.get("consume_date", "")
    if not consume_date:
        return "缺少 consume_date 字段，必须输出精确到 YYYY-MM-DD 的消费日期。"
    if re.fullmatch(DATE_YM, consume_date):
        return "日期禁止仅输出年月YYYY-MM，必须精确到 YYYY-MM-DD。若用户没有提供具体几号，请设置 need_more_info=true 向用户询问准确日期，不要自行填充残缺日期。"
    return None


# ===================== 节点1：核心执行节点【改造完成】 =====================
async def _settle_month_habit(data: dict) -> None:
    """M9（D15 P0-5）：记账成功 → 月度习惯沉淀（**静默失败**，不影响主流程）。

    habit 沉淀原在 orchestrator `_persist_bill_habit`（裸连 db ×2），M9 收口后
    **下沉到记账方自身**：经 `habit.upsert` 工具（HABIT_WRITE，写 monthly_habit +
    读回当月聚合）→ 更新 pkl 语义层（`upsert_month_habit_memory`，to_thread）。
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

    try:
        parse_res = await parse_json_output(
            mcp_client.call_llm_base,
            sys_p,
            user_p,
            schema_example=json.dumps(
                {"need_more_info": False, "prompt": "", "valid": True, "amount": 200,
                 "category": "餐饮", "consume_date": date.today().isoformat()},
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

    amount = parse_res.get("amount", 0)
    category = parse_res.get("category", "")
    consume_date = parse_res.get("consume_date", "")

    # ===== 金额正则兜底：LLM可能编造原文没有的金额（如把45幻觉成200），以原文数字为准 =====
    # 规则：原文存在数字时，若LLM金额不在原文数字集合中（含LLM漏提取返回0），回退为原文首个数字
    try:
        llm_amount = float(amount)
    except (TypeError, ValueError):
        llm_amount = 0.0
    # 负号需保留："-30元"表示非法金额，不能被正则"修正"为正数30
    text_nums = [float(m.group(0)) for m in re.finditer(r"-?\d+(\.\d+)?", raw_text)]
    if text_nums and llm_amount not in text_nums:
        amount = text_nums[0]
        print(f"[BILL] {time.strftime('%H:%M:%S')} 金额修正：LLM={llm_amount} 原文数字={text_nums} 取 {amount}", flush=True)

    if amount <= 0:
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "金额非法",
            "data": {"amount": amount, "category": category},
            "error": "消费金额必须大于0"
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # ===== 类别/日期代码层兜底：修正 1.8B 小模型的语义误判（打车→交通、买书→购物）与
    # 日期照抄 schema 示例（07-14）问题，确保落库字段 100% 符合用户原意 =====
    category = _fix_category(raw_text, category)
    consume_date = _fix_date(raw_text, consume_date)

    # 数据库写入
    try:
        # ★M5（D2）：写入 task_id——崩溃恢复时按此反查账单对账（`04` §3.1 修订版）。
        #   `state["task_id"]` 由 worker 从 A2A 消息注入（`startup/bootstrap.py`），
        #   **代码层保证写入**，不依赖 LLM 生成 SQL → 对账依据 100% 可靠。
        sql = "INSERT INTO bill (amount, category, consume_time, remark, task_id) VALUES (?, ?, ?, ?, ?)"
        params = [amount, category, consume_date, raw_text, state.get("task_id")]
        sql_res = await mcp_client.call_bill_sql(agent_id, sql, params)
    except Exception as e:
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "数据库调用异常",
            "data": {"amount": amount, "category": category, "consume_date": consume_date},
            "error": str(e)
        }
        return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

    # ========= 输出标准结构化JSON =========
    if "成功" in sql_res:
        payload = {
            "success": True,
            "agent_type": BILL_AGENT_ID,
            "msg": "记账成功",
            "data": {
                "amount": amount,
                "category": category,
                "consume_date": consume_date,
                "remark": raw_text
            },
            "error": None
        }
    else:
        payload = {
            "success": False,
            "agent_type": BILL_AGENT_ID,
            "msg": "账单入库操作失败",
            "data": {"amount": amount, "category": category, "consume_date": consume_date},
            "error": sql_res
        }

    # M9（D15 P0-5）：记账成功 → 月度习惯沉淀（静默失败，不影响主流程）
    if payload.get("success") is True:
        await _settle_month_habit(payload["data"])

    return Command(update={"result": json.dumps(payload, ensure_ascii=False)}, goto="reply_node")

# ===================== 节点2：回传结果给调度【完全无需改动】 =====================
async def reply_node(state: BillState):
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