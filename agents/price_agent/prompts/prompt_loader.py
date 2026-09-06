from pathlib import Path
from jinja2 import Template

PROMPT_DIR = Path(__file__).parent

def load_extract_system() -> str:
    file = PROMPT_DIR / "system.md"
    return file.read_text(encoding="utf-8").strip()

def render_extract_template(**kwargs) -> str:
    tpl_text = (PROMPT_DIR / "user.j2").read_text("utf-8")
    return Template(tpl_text).render(**kwargs)

__all__ = ["load_extract_system", "render_extract_template"]