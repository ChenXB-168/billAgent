# ==============================================
# finance_agent 提示词加载与渲染
# 注意：本文件用 os.path 拼路径（与 bill/stat/price/orchestrator 的 pathlib 写法不同），
#       且加载/渲染失败时返回错误文案或抛 RuntimeError，与其它 agent 的抛裸异常行为不同。
# ==============================================
import json
from jinja2 import Template
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system.md")
USER_TPL_PATH = os.path.join(BASE_DIR, "user.j2")


def load_finance_sys() -> str:
    """
    函数功能与逻辑描述：
        读取理财助手系统提示词 system.md。与其它 agent 不同，本函数**吞掉全部异常**，
        读取失败时返回 "【提示词加载失败】{异常}" 文案而不是抛错，
        以保证理财链路在提示词缺失时仍能走通（模型会收到该错误文案而非崩溃）。
    入参说明：
        无。
    返回值说明：
        str：system.md 全文（**未做 strip**）；读取失败时返回 "【提示词加载失败】..." 文案。
    """
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"【提示词加载失败】{str(e)}"


def render_finance_user(user_query: str, target_data: dict, stat_data: dict, price_data: dict,
                        habit_data: list = None, user_docs_data: list = None,
                        fact_sheet: dict = None) -> str:
    """
    函数功能与逻辑描述：
        用 user.j2 渲染理财链路的用户侧提示词。除用户问题外，把三类数据块
        （目标数据 / 统计数据 / 物价数据）与三类可选上下文（消费习惯 / 用户资料 / 事实底稿）
        一并注入：其中 target/stat/price/habit/fact_sheet 走 json.dumps（ensure_ascii=False、indent=2）
        以便模型阅读，user_docs_data 则原样作为列表传入由模板自行遍历。
        异常不吞：模板读取或渲染失败一律包装为 RuntimeError 上抛，避免模型在缺数据的情况下瞎编。
    入参说明：
        user_query (str)：用户原始提问。
        target_data (dict)：目标（目标消费额/预算）数据块。
        stat_data (dict)：统计数据块（当月支出等聚合结果）。
        price_data (dict)：物价对标数据块。
        habit_data (list)：消费习惯数据，默认 None，渲染时为空则等价传入 []。
        user_docs_data (list)：M8 用户资料检索结果（由 `memory/user_docs.py` 下发），
            list[dict]，元素形如 {"doc_id","text","score"}；None/空 时模板渲染为"无相关用户笔记"。
        fact_sheet (dict)：D16 防幻觉（M11）确定性规则事实底稿（预算/当月支出/余额等，白名单来源）；
            为 None 时渲染为「无底稿」提示，禁止编造，默认 None。
    返回值说明：
        str：渲染后的用户提示词全文。
    异常说明：
        RuntimeError：模板读取或渲染失败时抛出，文案为 "模板渲染异常：{异常}"。
    """
    try:
        with open(USER_TPL_PATH, "r", encoding="utf-8") as f:
            tpl = Template(f.read())
        return tpl.render(
            user_query=user_query,
            target_data=json.dumps(target_data, ensure_ascii=False, indent=2),
            stat_data=json.dumps(stat_data, ensure_ascii=False, indent=2),
            price_data=json.dumps(price_data, ensure_ascii=False, indent=2),
            habit_data=json.dumps(habit_data or [], ensure_ascii=False, indent=2),
            user_docs_data=user_docs_data or [],
            fact_sheet=json.dumps(fact_sheet or {}, ensure_ascii=False, indent=2)
        )
    except Exception as e:
        raise RuntimeError(f"模板渲染异常：{str(e)}")
