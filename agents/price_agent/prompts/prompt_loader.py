# ==============================================
# price_agent 提示词加载与渲染
# 仅一套提示词（system.md + user.j2），不做日期占位符渲染
# ==============================================
from pathlib import Path
from jinja2 import Template

PROMPT_DIR = Path(__file__).parent


def load_extract_system() -> str:
    """
    函数功能与逻辑描述：
        读取物价抽取链路的系统提示词 system.md 并去除首尾空白。
        纯文件读取，不做 Jinja 渲染（该文件不含变量占位符），也不做缓存，每次调用重新读盘。
    入参说明：
        无。
    返回值说明：
        str：system.md 的全文（已 strip）。
    """
    file = PROMPT_DIR / "system.md"
    return file.read_text(encoding="utf-8").strip()


def render_extract_template(**kwargs) -> str:
    """
    函数功能与逻辑描述：
        用 user.j2 渲染物价抽取链路的用户侧提示词，模板变量由调用方以关键字参数传入。
        不做变量名白名单校验：多传的变量被 Jinja2 忽略，少传的变量渲染为空白。
    入参说明：
        **kwargs：模板变量键值对，由调用点按 user.j2 的占位符提供。
    返回值说明：
        str：渲染后的用户提示词全文。
    """
    tpl_text = (PROMPT_DIR / "user.j2").read_text("utf-8")
    return Template(tpl_text).render(**kwargs)


__all__ = ["load_extract_system", "render_extract_template"]
