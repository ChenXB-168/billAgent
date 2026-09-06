import json
from jinja2 import Template
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system.md")
USER_TPL_PATH = os.path.join(BASE_DIR, "user.j2")


def load_finance_sys() -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"【提示词加载失败】{str(e)}"


def render_finance_user(user_query: str, target_data: dict, stat_data: dict, price_data: dict,
                         habit_data: list = None, user_docs_data: list = None,
                         fact_sheet: dict = None) -> str:
    """渲染 finance 用户侧提示词。

    user_docs_data: M8 用户资料检索结果（`memory/user_docs.py` 下发），list[dict]：
        {"doc_id","text","score"}；None/空 渲染为"无相关用户笔记"。
    fact_sheet: D16 防幻觉（M11）确定性规则事实底稿（预算/当月支出/余额等，白名单来源）；
        为 None 时模板渲染为「无底稿」提示，禁止编造。
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