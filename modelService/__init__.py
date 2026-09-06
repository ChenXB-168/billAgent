# modelService/__init__.py
"""
模型服务统一导出入口
职责：统一封装LLM/Embedding/Rerank底层模型调用，对外屏蔽文件层级
上层所有模块仅需 import modelService 即可获取模型客户端，无需深层导入

★2026-08-30 改为惰性加载（M1 动工实测发现的性能 + 稳定性缺陷）：

  原实现在模块级 `from .embedding_loader import embedding_client` 会**立即加载**向量模型，
  `from .rerank_loader import rerank_client` 会**立即加载**重排模型。
  而**导入任何子模块都会先执行本文件**——于是 `from modelService.llm_loader import ...`
  这种只要 LLM 的导入，也会被迫把向量/重排模型一起加载起来。后果：

  ① MCP server_llm_base 子进程**每次启动**多花 26~29s
     （印证 `client.py` L56-57 的注释；`05` §9.8 曾误判该注释"已过时"，实为真）；
  ② 重排模型加载失败 → 页面文件不足（os error 1455）→ 子进程崩溃
     → `McpError: Connection closed` / `unhandled errors in a TaskGroup`。

  实测全项目**无一处**使用 `from modelService import embedding_client`——
  真实调用方（memory/user_docs.py M8 起 / memory/long_memory.py，旧 rag_service/rag_builder
  已于 2026-09-04 随 M8 删除）**全部**直接导入子模块，
  故改为 PEP 562 惰性加载，对现有代码零影响。
"""

# LLM 基础调用函数（纯 HTTP 请求，**不加载任何本地模型**，可安全在模块级导入）
from .llm_loader import ollama_base_call, external_llm_call

# 对外暴露的全部API，限制import * 时的导出范围
__all__ = [
    # LLM大模型调用
    "ollama_base_call",
    "external_llm_call",
    # 向量Embedding（惰性）
    "EmbeddingService",
    "embedding_client",
    # 检索重排Rerank（惰性）
    "RerankService",
    "rerank_client",
]


def __getattr__(name):
    """PEP 562 惰性属性：仅在真正被访问时才加载向量 / 重排模型。"""
    if name in ("EmbeddingService", "embedding_client"):
        from . import embedding_loader
        return getattr(embedding_loader, name)
    if name in ("RerankService", "rerank_client"):
        from . import rerank_loader
        return getattr(rerank_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
