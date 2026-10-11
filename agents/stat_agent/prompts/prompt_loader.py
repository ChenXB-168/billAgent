# ==============================================
# stat_agent 提示词加载与渲染
# system.md 用 Jinja FileSystemLoader 加载，并渲染 today_date / month_start / month_end 三个占位符
# ==============================================
from pathlib import Path
from datetime import date
import jinja2

BASE_PATH = Path(__file__).parent


def load_system() -> str:
    """
    函数功能与逻辑描述：
        用 Jinja FileSystemLoader 加载 system.md 并渲染三个日期占位符：today_date（今天）、
        month_start（本月 1 号）、month_end（今天，作为本月区间右界）。
        必须渲染的原因：杜绝 LLM 照抄提示词里的硬编码示例日期（典型故障是把「这个月」
        解析成示例中的 7 月）。每次调用都新建 Environment，不缓存模板。
    入参说明：
        无。
    返回值说明：
        str：渲染后的系统提示词全文（**未做 strip**，由模板自身控制首尾）。
    """
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(BASE_PATH))
    template = env.get_template("system.md")
    today = date.today()
    # 渲染 today_date / month_start / month_end，杜绝 LLM 照抄硬编码示例日期（如把"这个月"解析成7月）
    return template.render(
        today_date=today.isoformat(),
        month_start=today.replace(day=1).strftime("%Y-%m-%d"),
        month_end=today.strftime("%Y-%m-%d"),
    )


def render_user(user_input: str, history: str) -> str:
    """
    函数功能与逻辑描述：
        用 user.j2 渲染统计链路的用户侧提示词，注入本轮用户输入与历史对话。
        每次调用新建 Environment 与模板对象，不缓存。
    入参说明：
        user_input (str)：用户本轮原始输入。
        history (str)：历史对话文本，通常由上下文管理器压缩后传入。
    返回值说明：
        str：渲染后的用户提示词全文。
    """
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(BASE_PATH))
    template = env.get_template("user.j2")
    return template.render(user_input=user_input, history=history)


__all__ = ["load_system", "render_user"]
