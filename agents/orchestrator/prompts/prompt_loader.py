# ==============================================
# orchestrator 提示词加载与渲染
# 每套提示词分「系统」（任务规划）/「汇总」（最终回复与提醒）两条链路，各自 md + j2 成对存在
# ==============================================
from datetime import date
from pathlib import Path
from jinja2 import Template

BASE_DIR = Path(__file__).parent


def load_system() -> str:
    """
    函数功能与逻辑描述：
        读取编排器任务规划链路的系统提示词 system.md 并去除首尾空白。
        纯文件读取，不做模板渲染（该文件不含 Jinja 变量），也不做缓存——
        每次调用都会重新读盘，便于调试期直接改 md 生效。
    入参说明：
        无。
    返回值说明：
        str：system.md 的全文（已 strip）；文件不存在时由 read_text 抛 FileNotFoundError。
    """
    file_path = BASE_DIR / "system.md"
    return file_path.read_text(encoding="utf-8").strip()


def render_user(user_input: str, history: str, valid_tasks_str: str,
                today_date: str | None = None) -> str:
    """
    函数功能与逻辑描述：
        用 user.j2 渲染任务规划链路的用户侧提示词，注入用户输入、历史对话与合法任务清单。
        today_date 为「今天日期」注入（与 bill_agent / stat_agent 同一范式），
        供 LLM 换算「昨天 / 这个月 / 上个月」等相对时间；**缺省自动取当天**，
        保证既有调用方与单测零改动。该参数于 2026-09-10 新增。
    入参说明：
        user_input (str)：用户本轮原始输入。
        history (str)：历史对话文本，通常由上下文管理器压缩后传入。
        valid_tasks_str (str)：合法任务清单文本，用于约束模型只能规划受支持的任务。
        today_date (str | None)：今天日期字符串（YYYY-MM-DD）；默认 None 表示取 date.today()。
    返回值说明：
        str：渲染后的用户提示词全文。
    """
    tpl_file = BASE_DIR / "user.j2"
    tpl = Template(tpl_file.read_text(encoding="utf-8"))
    return tpl.render(
        user_input=user_input,
        history=history,
        valid_tasks_str=valid_tasks_str,
        today_date=today_date or date.today().isoformat(),
    )


def load_summarize_system() -> str:
    """
    函数功能与逻辑描述：
        读取「汇总」链路的系统提示词 summarize.md 并去除首尾空白，用于最终回复与提醒生成。
        与 load_system 是两套完全独立的提示词，互不复用。
    入参说明：
        无。
    返回值说明：
        str：summarize.md 的全文（已 strip）；文件不存在时抛 FileNotFoundError。
    """
    file_path = BASE_DIR / "summarize.md"
    return file_path.read_text(encoding="utf-8").strip()


def render_summarize_user(user_input: str, task_results_json: str,
                          alert_facts_json: str, remind_pref: str) -> str:
    """
    函数功能与逻辑描述：
        用 summarize_user.j2 渲染汇总链路的用户侧提示词，把各子任务结果、预警事实与
        用户提醒偏好一并注入，供 LLM 生成面向用户的最终回复。
        四个参数均为已序列化好的字符串或原始偏好文本，本函数不做序列化与校验。
    入参说明：
        user_input (str)：用户本轮原始输入，用于让汇总话术呼应提问。
        task_results_json (str)：全部子任务结果的 JSON 字符串。
        alert_facts_json (str)：预警事实底稿的 JSON 字符串（确定性的预算/支出事实）。
        remind_pref (str)：用户个性化提醒偏好（语气/语言/格式）。
    返回值说明：
        str：渲染后的汇总用户提示词全文；模板中缺少对应变量时由 Jinja2 渲染为空白。
    """
    tpl_file = BASE_DIR / "summarize_user.j2"
    tpl = Template(tpl_file.read_text(encoding="utf-8"))
    return tpl.render(
        user_input=user_input,
        task_results_json=task_results_json,
        alert_facts_json=alert_facts_json,
        remind_pref=remind_pref
    )
