from pathlib import Path
from datetime import date
from jinja2 import Template

PROMPT_DIR = Path(__file__).parent

def load_system_prompt() -> str:
    tpl_text = (PROMPT_DIR / "system.md").read_text(encoding="utf-8")
    # 渲染 {{today_date}}，避免字面量 {{today_date}} 传给 LLM 导致模型不知道今天日期
    return Template(tpl_text).render(today_date=date.today().isoformat()).strip()

def render_full_template(**kwargs) -> str:
    tpl_text = (PROMPT_DIR / "user_full.j2").read_text("utf-8")
    return Template(tpl_text).render(**kwargs)

def render_simple_template(**kwargs) -> str:
    tpl_text = (PROMPT_DIR / "user_simple.j2").read_text("utf-8")
    return Template(tpl_text).render(**kwargs)

__all__ = ["load_system_prompt", "render_full_template", "render_simple_template"]