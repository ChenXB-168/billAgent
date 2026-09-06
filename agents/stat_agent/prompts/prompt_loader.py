from pathlib import Path
from datetime import date
import jinja2

BASE_PATH = Path(__file__).parent

def load_system() -> str:
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
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(BASE_PATH))
    template = env.get_template("user.j2")
    return template.render(user_input=user_input, history=history)

__all__ = ["load_system", "render_user"]