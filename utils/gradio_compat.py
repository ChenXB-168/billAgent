# -*- coding: utf-8 -*-
"""gradio 运行时兼容补丁（gradio 4.44.x + gradio_client 1.3.0 配套环境）。

问题背景（已在本仓库 venv 实测复现）：
  gradio_client 1.3.0 的 ``_json_schema_to_python_type`` 无法处理"布尔型 JSON-Schema"。
  JSON Schema 规范允许以 bool 直接表达 schema（如 ``additionalProperties: true``），
  而本项目 Chatbot(type="messages") 生成的嵌套消息 schema 恰好含该写法，
  导致 gradio 每次生成 API 信息（页面加载 /config 时）走到递归深处时执行
  ``"const" in True`` 抛 ``TypeError: argument of type 'bool' is not iterable``。

修复方式：
  单点替换模块级 ``_json_schema_to_python_type`` —— 原函数内部递归通过模块全局
  名查找自身，因此一次替换即覆盖所有嵌套调用。布尔 schema 按语义映射为 Python
  类型提示（true=任意值合法 -> Any），不改动业务 schema 数据本身。
"""
from __future__ import annotations


def apply_gradio_patch() -> None:
    import gradio_client.utils as _cu

    # 幂等：避免重复包裹
    if getattr(_cu, "_BOOL_SCHEMA_PATCHED", False):
        return

    _orig = _cu._json_schema_to_python_type

    def _patched(schema, defs):
        if not isinstance(schema, dict):
            # JSON-Schema 布尔形式（additionalProperties: true/false 等）。
            # 此处仅用于生成客户端类型提示，无精确 Python 等价 -> Any 即可。
            return "Any"
        return _orig(schema, defs)

    _cu._json_schema_to_python_type = _patched
    _cu._BOOL_SCHEMA_PATCHED = True
