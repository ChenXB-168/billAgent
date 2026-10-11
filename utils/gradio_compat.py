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
    """
    函数功能与逻辑描述：
        对 gradio_client 打一次模块级单点补丁，修复布尔型 JSON-Schema 导致的启动期崩溃。
        做法是把 gradio_client.utils 模块全局名 `_json_schema_to_python_type` 整体替换为
        一个先判类型的包装函数——原函数内部递归时是通过模块全局名查找自身，因此一次替换即覆盖
        所有嵌套调用层级，无需改动 gradio 源码。
        以模块属性 `_BOOL_SCHEMA_PATCHED` 作为幂等标记：已打过补丁则直接返回，
        避免多次 import/多次调用时重复包裹导致调用链膨胀。
        本函数不修改业务 schema 数据本身，只影响客户端类型提示的生成结果。
    入参说明：
        无。
    返回值说明：
        无（副作用为改写 gradio_client.utils 的模块级属性）。
    """
    import gradio_client.utils as _cu

    # 幂等：避免重复包裹
    if getattr(_cu, "_BOOL_SCHEMA_PATCHED", False):
        return

    _orig = _cu._json_schema_to_python_type

    def _patched(schema, defs):
        """
        函数功能与逻辑描述：
            替换 gradio_client 原 _json_schema_to_python_type 的包装实现：先拦截布尔型 schema
            （JSON Schema 规范允许 additionalProperties: true/false 等直接以 bool 表达），
            否则原样委托给被保存的原始函数 _orig，保证非布尔场景行为完全不变。
            不拦截时原实现会执行 `"const" in True`，触发
            TypeError: argument of type 'bool' is not iterable。
        入参说明：
            schema：JSON-Schema 片段，正常情况下为 dict；布尔形式时为 bool。
            defs：JSON-Schema 的 $defs 定义表，原样透传给 _orig 用于解析 $ref。
        返回值说明：
            str：Python 类型提示字符串。布尔型 schema 统一返回 "Any"
                （该结果仅用于生成客户端类型提示，布尔 schema 无精确 Python 等价类型）；
                非布尔 schema 返回 _orig 的原始结果。
        """
        if not isinstance(schema, dict):
            # JSON-Schema 布尔形式（additionalProperties: true/false 等）。
            # 此处仅用于生成客户端类型提示，无精确 Python 等价 -> Any 即可。
            return "Any"
        return _orig(schema, defs)

    _cu._json_schema_to_python_type = _patched
    _cu._BOOL_SCHEMA_PATCHED = True
