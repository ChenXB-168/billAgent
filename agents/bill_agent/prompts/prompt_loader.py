# ==============================================
# bill_agent 提示词加载与渲染
# system.md 内含 {{today_date}} 占位符，必须在加载时渲染，否则字面量会原样传给模型
# ==============================================
from pathlib import Path
from datetime import date
from jinja2 import Template

PROMPT_DIR = Path(__file__).parent


def load_system_prompt() -> str:
    """
    函数功能与逻辑描述：
        读取 system.md 并即时渲染其中的 {{today_date}} 占位符为当天日期（YYYY-MM-DD），
        最后 strip 首尾空白。必须渲染的原因：若把字面量 {{today_date}} 传给 LLM，
        模型将不知道今天日期，进而无法正确换算「昨天 / 本月」等相对时间。
    入参说明：
        无。
    返回值说明：
        str：渲染并 strip 后的系统提示词全文。
    """
    tpl_text = (PROMPT_DIR / "system.md").read_text(encoding="utf-8")
    # 渲染 {{today_date}}，避免字面量 {{today_date}} 传给 LLM 导致模型不知道今天日期
    return Template(tpl_text).render(today_date=date.today().isoformat()).strip()


def render_full_template(**kwargs) -> str:
    """
    函数功能与逻辑描述：
        用 user_full.j2（完整版：含历史对话、长期记忆等全量上下文）渲染用户侧提示词。
        模板变量全部由调用方经关键字参数传入，本函数不做变量名白名单校验，
        多传的变量被 Jinja2 忽略，少传的变量渲染为空白。
    入参说明：
        **kwargs：模板变量键值对，由各调用点按 user_full.j2 的占位符提供。
    返回值说明：
        str：渲染后的用户提示词全文。
    """
    tpl_text = (PROMPT_DIR / "user_full.j2").read_text("utf-8")
    return Template(tpl_text).render(**kwargs)


def render_simple_template(**kwargs) -> str:
    """
    函数功能与逻辑描述：
        用 user_simple.j2（精简版：仅保留本次记账必需字段）渲染用户侧提示词，
        用于上下文受限或仅需抽取单笔账单的场景。变量处理方式同 render_full_template。
    入参说明：
        **kwargs：模板变量键值对，由各调用点按 user_simple.j2 的占位符提供。
    返回值说明：
        str：渲染后的用户提示词全文。
    """
    tpl_text = (PROMPT_DIR / "user_simple.j2").read_text("utf-8")
    return Template(tpl_text).render(**kwargs)


def load_fc_system_prompt() -> str:
    """
    函数功能与逻辑描述：
        读取 fc_system.md（M15 P3b：**工具使用说明**版系统提示词）并渲染其中的 {{today_date}}。
        为什么另起一份而不改 system.md：两者服务的目标不同且**同时存在**——
        system.md 是"记账字段抽取器"的提示词（要求只输出 JSON，供 add 路径与 edit 的新值抽取使用），
        fc_system.md 是"改删助手"的提示词（要求先调工具、按工具返回值说话）。
        把工具语义写进 system.md 会让 add 路径的模型在"该输出 JSON"与"该调工具"之间摇摆，
        直接损害 add 的确定性（`设计/12` §4.2.1 的"分工不同、提示词分开"）。
    入参说明：
        无。
    返回值说明：
        str：渲染并 strip 后的 FC 通道系统提示词全文。
    """
    tpl_text = (PROMPT_DIR / "fc_system.md").read_text(encoding="utf-8")
    return Template(tpl_text).render(today_date=date.today().isoformat()).strip()


__all__ = ["load_system_prompt", "load_fc_system_prompt",
           "render_full_template", "render_simple_template"]
