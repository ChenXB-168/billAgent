from pathlib import Path
from jinja2 import Template

BASE_DIR = Path(__file__).parent

def load_system() -> str:
    """加载system.md系统提示词"""
    file_path = BASE_DIR / "system.md"
    return file_path.read_text(encoding="utf-8").strip()

def render_user(user_input: str, history: str, valid_tasks_str: str) -> str:
    """渲染user.j2模板"""
    tpl_file = BASE_DIR / "user.j2"
    tpl = Template(tpl_file.read_text(encoding="utf-8"))
    return tpl.render(
        user_input=user_input,
        history=history,
        valid_tasks_str=valid_tasks_str
    )


def load_summarize_system() -> str:
    """加载汇总提示词 system（第二套：最终回复/提醒生成）"""
    file_path = BASE_DIR / "summarize.md"
    return file_path.read_text(encoding="utf-8").strip()


def render_summarize_user(user_input: str, task_results_json: str,
                          alert_facts_json: str, remind_pref: str) -> str:
    """渲染汇总用户模板 summarize_user.j2"""
    tpl_file = BASE_DIR / "summarize_user.j2"
    tpl = Template(tpl_file.read_text(encoding="utf-8"))
    return tpl.render(
        user_input=user_input,
        task_results_json=task_results_json,
        alert_facts_json=alert_facts_json,
        remind_pref=remind_pref
    )